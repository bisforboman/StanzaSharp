using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.DataIngestion;
using Microsoft.ML.Tokenizers;

namespace StanzaSharp.DataIngestion;

/// <summary>
/// Splits a document into chunks of whole sentences, found by StanzaSharp's tokenizer and sentence splitter, each at most
/// <see cref="IngestionChunkerOptions.MaxTokensPerChunk"/> tokens of <see cref="IngestionChunkerOptions.Tokenizer"/>
/// (your embedding model's tokenizer).
/// </summary>
/// <remarks>
/// <para>Headers work as in <see cref="HeaderChunker"/>: a header starts a new chunk and becomes the
/// <see cref="IngestionChunk{T}.Context"/> of the chunks under it (with its parent headers, joined by spaces), which
/// also start with it. Paragraphs and the text of images (alternative text, else OCR text) are split into sentences;
/// a table is kept whole. Chunks never cross a header.</para>
/// <para>A chunk holds the sentences as written: sentences of one element keep the text between them, elements are
/// joined by a newline (<see cref="Environment.NewLine"/>, like <see cref="HeaderChunker"/>), so a section with room to
/// spare gives <see cref="HeaderChunker"/>'s chunk when its elements' text starts and ends at a sentence (the space before
/// the first sentence and after the last is dropped). <see cref="IngestionChunkerOptions.OverlapTokens"/> is filled
/// with whole sentences: a chunk starts with the last sentences of the one before it (under the same header) whose
/// tokens fit.</para>
/// <para>Only a single sentence or table longer than a chunk is cut inside, at token boundaries, into chunks of their
/// own (without overlap).</para>
/// <para>The pipeline needs only the tokenizer (<c>Processors = "tokenize"</c> is the fastest); its language decides
/// the sentence splitting. One chunker may serve concurrent documents: <see cref="Pipeline"/>'s Process is
/// thread-safe.</para>
/// </remarks>
public sealed class SentenceChunker : IngestionChunker<string>
{
    private const int MaxHeaderLevel = 10;
    private static readonly string ElementSeparator = Environment.NewLine;

    private readonly Pipeline _pipeline;
    private readonly Tokenizer _tokenizer;
    private readonly int _maxTokens, _overlapTokens;

    /// <summary>Creates a chunker that splits sentences with <paramref name="pipeline"/>.</summary>
    /// <param name="pipeline">A loaded pipeline of the documents' language. The chunker doesn't dispose it.</param>
    /// <param name="options">The tokenizer that counts chunk sizes, the most tokens per chunk and the overlap.</param>
    public SentenceChunker(Pipeline pipeline, IngestionChunkerOptions options)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(options);
        _pipeline = pipeline;
        _tokenizer = options.Tokenizer;
        _maxTokens = options.MaxTokensPerChunk;
        _overlapTokens = options.OverlapTokens;
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(IngestionDocument document,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(document);
        var sections = Sections(document, cancellationToken);

        // One pipeline call for every text of the document.
        var texts = sections.SelectMany(s => s.Pieces).Where(p => p.Split).Select(p => p.Text).ToList();
        var documents = texts.Count == 0 ? [] : _pipeline.Process(texts, cancellationToken);
        var next = 0;
        foreach (var section in sections)
        {
            var units = new List<Unit>();
            foreach (var piece in section.Pieces)
            {
                if (!piece.Split)
                {
                    units.Add(new Unit(piece.Text, ElementSeparator, IsSentence: false));
                    continue;
                }

                var first = true;
                var end = 0;
                foreach (var sentence in documents[next++].Sentences)
                {
                    if (sentence.Tokens.Count == 0 || sentence.Tokens[0].StartChar is not int from || sentence.Tokens[^1].EndChar is not int to)
                        continue;
                    units.Add(new Unit(piece.Text[from..to], first ? ElementSeparator : piece.Text[end..from], IsSentence: true));
                    first = false;
                    end = to;
                }
            }

            foreach (var chunk in Pack(document, section.Context, units, cancellationToken))
                yield return chunk;
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    // The document's elements grouped under their headers, as HeaderChunker groups them.
    private static List<Section> Sections(IngestionDocument document, CancellationToken cancellationToken)
    {
        var sections = new List<Section>();
        var headers = new string?[MaxHeaderLevel + 1];
        var pieces = new List<Piece>();
        foreach (var element in document.EnumerateContent())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (element is IngestionDocumentHeader header)
            {
                Close();
                var level = Math.Clamp(header.Level.GetValueOrDefault(), 0, MaxHeaderLevel);
                headers[level] = header.GetMarkdown();
                headers.AsSpan(level + 1).Clear();
                continue;
            }

            var text = element switch
            {
                IngestionDocumentImage image => image.AlternativeText ?? image.Text,
                _ => element.GetMarkdown(),
            };
            if (!string.IsNullOrWhiteSpace(text))
                pieces.Add(new Piece(text, Split: element is not IngestionDocumentTable));
        }

        Close();
        return sections;

        void Close()
        {
            if (pieces.Count == 0)
                return;
            sections.Add(new Section(string.Join(" ", headers.Where(h => !string.IsNullOrEmpty(h))), [.. pieces]));
            pieces.Clear();
        }
    }

    // Packs a section's units into chunks, greedily, each checked against the tokenizer as a whole.
    private List<IngestionChunk<string>> Pack(IngestionDocument document, string context, List<Unit> units,
        CancellationToken cancellationToken)
    {
        var chunks = new List<IngestionChunk<string>>();
        if (Count(context) >= _maxTokens)
            throw new InvalidOperationException($"The headers \"{context}\" alone take {_maxTokens} tokens or more; raise MaxTokensPerChunk.");

        var current = new List<Unit>();
        foreach (var unit in units)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (Fits(current, unit))
            {
                current.Add(unit);
                continue;
            }

            if (current.Count > 0)
            {
                chunks.Add(Chunk(current));
                current = Overlap(current);
                while (current.Count > 0 && !Fits(current, unit))
                    current.RemoveAt(0);
                if (Fits(current, unit))
                {
                    current.Add(unit);
                    continue;
                }
            }

            // Longer than a chunk alone: cut it at token boundaries.
            var rest = unit.Text.AsSpan();
            var budget = _maxTokens - Count(Build([new Unit("", "", false)]));
            while (!rest.IsEmpty)
            {
                var index = _tokenizer.GetIndexByTokenCount(rest, budget, out _, out _, considerNormalization: false);
                if (index <= 0)
                    throw new InvalidOperationException("Not even one token fits in a chunk after the headers; raise MaxTokensPerChunk.");
                chunks.Add(Chunk([new Unit(rest[..index].ToString(), "", false)]));
                rest = rest[index..];
            }
        }

        if (current.Count > 0)
            chunks.Add(Chunk(current));
        return chunks;

        IngestionChunk<string> Chunk(List<Unit> content) => new(Build(content), document, context);

        bool Fits(List<Unit> content, Unit unit) => Count(Build([.. content, unit])) <= _maxTokens;

        // The last sentences of a chunk whose tokens fit in the overlap (none past a table or a cut).
        List<Unit> Overlap(List<Unit> chunk)
        {
            var overlap = new List<Unit>();
            if (_overlapTokens <= 0)
                return overlap;
            for (var i = chunk.Count - 1; i >= 0 && chunk[i].IsSentence; i--)
            {
                var candidate = chunk[i..];
                if (Count(Join(candidate)) > _overlapTokens)
                    break;
                overlap = candidate;
            }

            return overlap;
        }

        string Build(List<Unit> content)
        {
            var body = Join(content);
            return context.Length == 0 ? body : context + ElementSeparator + body;
        }
    }

    // The units' text, each after its separator but the first.
    private static string Join(List<Unit> units)
    {
        var builder = new StringBuilder();
        for (var i = 0; i < units.Count; i++)
        {
            if (i > 0)
                builder.Append(units[i].Before);
            builder.Append(units[i].Text);
        }

        return builder.ToString();
    }

    private int Count(string text) => _tokenizer.CountTokens(text, considerNormalization: false);

    private sealed record Piece(string Text, bool Split);

    private sealed record Section(string Context, Piece[] Pieces);

    // A sentence, or a whole table; Before is what joins it to the unit before it.
    private sealed record Unit(string Text, string Before, bool IsSentence);
}
