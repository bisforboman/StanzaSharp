using System.Text;
using System.Text.RegularExpressions;
using StanzaSharp.Nn;
using static TorchSharp.torch;

namespace StanzaSharp.Tokenize;

/// <summary>
/// Splits raw text into sentences and tokens, following Stanza's TokenizeProcessor
/// (stanza/models/tokenization/data.py and utils.py).
/// </summary>
/// <remarks>
/// The model sees one unit per Unicode code point, like Python's <c>str</c>. Offsets
/// (<c>StartChar</c>/<c>EndChar</c>) are UTF-16 indices into the input string, so they differ from
/// Stanza's code-point offsets only after characters outside the BMP (e.g. emoji).
/// </remarks>
internal sealed class Tokenizer : IDisposable
{
    // TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT; output_predictions uses max(1000, max_seqlen).
    private const int MaxSeqLen = 1000;
    private const string LongTokenReplacement = "<UNK>";

    // data.py: NEWLINE_WHITESPACE_RE, WHITESPACE_RE. Python's \s also matches \x1c-\x1f; .NET's doesn't.
    private static readonly Regex ParagraphBreak = new(@"\n[\s\x1c-\x1f\u0080-\u009f]*\n");

    // data.py: STRUCTURAL_FEATURES, in the order features are appended.
    private static readonly (string Name, Regex Re)[] StructuralFeatures =
    [
        ("labeled_field", PyRegex(@"\b[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*){0,2}\s*:")),
        ("phone_id", PyRegex(@"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b|\bx\d{3,5}\b")),
        ("date_pattern", PyRegex(@"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b")),
        ("currency", PyRegex(@"\$\s?\d[\d,]*(?:\.\d+)?")),
    ];

    // Python's \b: its word characters are str.isalnum() plus '_', so superscript digits and other
    // No/Nl characters count (3/14/2023² has no boundary after 2023) and combining marks do not,
    // unlike .NET's \w (L, Mn, Nd, Pc, and ZWJ/ZWNJ for \b).
    private const string PyWord = @"[\p{L}\p{N}_]";
    private static Regex PyRegex(string pattern) =>
        new(pattern.Replace(@"\b", $"(?:(?<={PyWord})(?!{PyWord})|(?<!{PyWord})(?={PyWord}))"));

    // utils.py: EMAIL_RAW_RE and URL_RAW_RE, combined as MASK_RE. Matches are forced into one token.
    private const string EmailRe = """"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""";
    private const string UrlRe = """"(?:https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s"]{2,}|www\.[a-zA-Z0-9]+\.[^\s"]{2,})|[a-zA-Z0-9]+\.(?:gov|org|edu|net|com|co)(?:\.[^\s"]{2,})"""";
    private static readonly Regex MaskRe = new($"(?:{EmailRe}|{UrlRe})");

    private readonly ITokenizerNet _net;
    private readonly Dictionary<string, int> _vocab;
    private readonly int _unkId, _padId, _batchSize, _featDim;
    private readonly string[] _featFuncs;

    private Tokenizer(Checkpoint ckpt, Backend backend)
    {
        _net = backend == Backend.Managed ? new ManagedTokenizerNet(ckpt) : new TokenizerNet(ckpt);
        var config = ckpt.Root["config"]!;
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        _unkId = _vocab["<UNK>"];
        _padId = _vocab["<PAD>"];
        _batchSize = config["batch_size"]!.GetValue<int>();
        _featDim = config["feat_dim"]!.GetValue<int>();

        // data.py para_to_sentences: per-character features in config order, then the
        // position features, then the structural features in STRUCTURAL_FEATURES order.
        var funcs = config["feat_funcs"]!.AsArray().Select(f => f!.GetValue<string>()).ToHashSet();
        string[] perChar = ["space_before", "capitalized", "numeric"];
        _featFuncs =
        [
            .. config["feat_funcs"]!.AsArray().Select(f => f!.GetValue<string>()).Where(perChar.Contains),
            .. new[] { "end_of_para", "start_of_para" }.Where(funcs.Contains),
            .. StructuralFeatures.Select(s => s.Name).Where(funcs.Contains),
        ];
        var unknown = funcs.Except(_featFuncs).ToList();
        if (unknown.Count > 0)
            throw new NotSupportedException($"Unknown tokenizer feature functions: {string.Join(", ", unknown)}");
        if (_featFuncs.Length != _featDim)
            throw new InvalidOperationException($"feat_dim is {_featDim} but {_featFuncs.Length} features are configured");
    }

    /// <summary>Loads <c>basePath.json</c> + <c>basePath.safetensors</c>, e.g. <c>models/converted/en/tokenize/combined_nocharlm</c>.</summary>
    /// <param name="device">Where the model runs; CPU by default (TorchSharp only).</param>
    public static Tokenizer Load(string basePath, Device? device = null, Backend backend = Backend.TorchSharp) =>
        Weights.On(device, () => new Tokenizer(Checkpoint.Load(basePath), backend));

    /// <param name="splitSentences">False is Stanza's <c>tokenize_no_ssplit</c>: each paragraph is one sentence.</param>
    public Document Process(string text, bool splitSentences = true, CancellationToken cancellationToken = default)
    {
        var paragraphs = SplitParagraphs(text);
        var preds = Predict(paragraphs, cancellationToken);

        var doc = new Document { Text = text };
        for (int i = 0; i < paragraphs.Count; i++)
            Decode(paragraphs[i], preds[i], doc, splitSentences);
        MarkWhitespace(doc);
        return doc;
    }

    /// <summary>
    /// TokenizeProcessor.bulk_process: tokenizes <paramref name="texts"/> joined by <c>"\n\n"</c> in one call, then
    /// gives each text its own document, with offsets relative to that text.
    /// </summary>
    public List<Document> Process(IReadOnlyList<string> texts, bool splitSentences = true, CancellationToken cancellationToken = default) =>
        Split(Process(string.Join("\n\n", texts), splitSentences, cancellationToken), texts);

    /// <summary>
    /// The second half of bulk_process: deals out <paramref name="combined"/>'s sentences (tokenized from
    /// <paramref name="texts"/> joined by <c>"\n\n"</c>) to one document per text, as Stanza does: a sentence goes to the
    /// current text while its last token ends inside it. Sentence ids continue across documents, as in Stanza.
    /// </summary>
    internal static List<Document> Split(Document combined, IReadOnlyList<string> texts)
    {
        var docs = new List<Document>(texts.Count);
        int offset = 0, next = 0;
        foreach (var text in texts)
        {
            var doc = new Document { Text = text };
            while (next < combined.Sentences.Count && combined.Sentences[next].Tokens[^1].EndChar - offset <= text.Length)
                doc.Sentences.Add(combined.Sentences[next++]);
            foreach (var token in doc.Sentences.SelectMany(s => s.Tokens))
            {
                token.StartChar -= offset;
                token.EndChar -= offset;
                foreach (var word in token.Words)
                {
                    word.StartChar -= offset;
                    word.EndChar -= offset;
                }
            }
            // The joining "\n\n" is not part of either text.
            if (doc.Sentences.Count > 0)
            {
                var last = doc.Sentences[^1].Tokens[^1];
                last.SpaceAfter = text[last.EndChar!.Value..];
                var first = doc.Sentences[0].Tokens[0];
                first.SpacesBefore = text[..first.StartChar!.Value];
            }
            docs.Add(doc);
            offset += text.Length + 2;
        }
        return docs;
    }

    /// <summary>
    /// TokenizeProcessor.process_pre_tokenized_text: one sentence per list, the tokens kept as given. The document's
    /// text is every token joined by single spaces, and the offsets point into it. Nothing is marked for MWT expansion.
    /// </summary>
    public static Document Pretokenized(IReadOnlyList<IReadOnlyList<string>> sentences)
    {
        var doc = new Document { Text = string.Join(" ", sentences.Select(s => string.Join(" ", s))) };
        int start = 0;
        foreach (var tokens in sentences)
        {
            var sent = NewSentence(doc);
            foreach (var text in tokens)
            {
                int end = start + text.Length;
                var token = new Token { Text = text, StartChar = start, EndChar = end };
                token.Words.Add(new Word { Id = sent.Tokens.Count + 1, Text = text, StartChar = start, EndChar = end });
                sent.Tokens.Add(token);
                start = end + 1;
            }
            FinishSentence(sent, doc);
        }
        MarkWhitespace(doc);
        return doc;
    }

    /// <summary>Document.mark_whitespace: the text between tokens, from the original string.</summary>
    private static void MarkWhitespace(Document doc)
    {
        var text = doc.Text!;
        var tokens = doc.Sentences.SelectMany(s => s.Tokens).ToList();
        for (int i = 0; i < tokens.Count; i++)
        {
            int end = tokens[i].EndChar!.Value;
            int next = i + 1 < tokens.Count ? tokens[i + 1].StartChar!.Value : text.Length;
            tokens[i].SpaceAfter = text[end..next];
        }
        if (tokens.Count > 0)
            tokens[0].SpacesBefore = text[..tokens[0].StartChar!.Value];
    }

    /// <summary>Log-probabilities [len + 1, 5] for one paragraph run alone, as Stanza pads it. For tests.</summary>
    internal float[] ParagraphLogits(string paragraph, out long[] shape)
    {
        var para = SplitParagraphs(paragraph).Single();
        shape = [para.Length + 1, Classes];
        return Run([para], [0], para.Length + 1, [para.Length + 1], default);
    }

    // ----- input: paragraphs of code-point units -----

    private sealed class Paragraph
    {
        public required string[] Units;   // one code point each, whitespace normalized to " "
        public required int[] Starts;     // UTF-16 offset of each unit in the original text
        public required int[] Ends;
        public required long[] Ids;
        public required float[] Feats;    // [Length * featDim]
        public int Length => Units.Length;
    }

    private List<Paragraph> SplitParagraphs(string text)
    {
        var result = new List<Paragraph>();
        int pos = 0;
        foreach (Match m in ParagraphBreak.Matches(text))
        {
            AddParagraph(text, pos, m.Index, result);
            pos = m.Index + m.Length;
        }
        AddParagraph(text, pos, text.Length, result);
        return result;
    }

    private void AddParagraph(string text, int start, int end, List<Paragraph> result)
    {
        while (end > start && PyString.IsSpace(text[end - 1]))
            end--;
        if (end == start)
            return;

        var units = new List<string>();
        var starts = new List<int>();
        var ends = new List<int>();
        int pos = start;
        foreach (var rune in text.AsSpan(start, end - start).EnumerateRunes())
        {
            int len = rune.Utf16SequenceLength;
            bool space = rune.IsBmp && (PyString.IsSpace((char)rune.Value) || rune.Value is >= 0x80 and <= 0x9f);
            // filter_consecutive_whitespaces: keep the first of a run of spaces.
            if (!(space && units.Count > 0 && units[^1] == " "))
            {
                units.Add(space ? " " : rune.ToString());
                starts.Add(pos);
                ends.Add(pos + len);
            }
            pos += len;
        }

        var unitArray = units.ToArray();
        result.Add(new Paragraph
        {
            Units = unitArray,
            Starts = starts.ToArray(),
            Ends = ends.ToArray(),
            Ids = unitArray.Select(u => (long)_vocab.GetValueOrDefault(u, _unkId)).ToArray(),
            Feats = Features(unitArray),
        });
    }

    private float[] Features(string[] units)
    {
        int n = units.Length;
        var feats = new float[n * _featDim];
        var (joined, unitAt) = Join(units);
        for (int k = 0; k < _featDim; k++)
        {
            var name = _featFuncs[k];
            var structural = StructuralFeatures.FirstOrDefault(s => s.Name == name).Re;
            if (structural != null)
            {
                foreach (Match m in structural.Matches(joined))
                    for (int p = m.Index; p < m.Index + m.Length; p++)
                        feats[unitAt[p] * _featDim + k] = 1;
                continue;
            }
            for (int i = 0; i < n; i++)
            {
                var rune = Rune.GetRuneAt(units[i], 0);
                bool on = name switch
                {
                    "space_before" => units[i] == " ",
                    "capitalized" => PyString.IsUpper(rune),
                    "numeric" => Rune.IsDigit(rune), // NUMERIC_RE on a single code point
                    "end_of_para" => i == n - 1,
                    "start_of_para" => i == 0,
                    _ => throw new NotSupportedException(name),
                };
                if (on)
                    feats[i * _featDim + k] = 1;
            }
        }
        return feats;
    }

    // ----- prediction, batched exactly like utils.predict -----

    private int[][] Predict(List<Paragraph> paragraphs, CancellationToken ct)
    {
        // SortedDataset sorts by length, longest first (stable), then batches of batch_size.
        // SortedDataset.collate appends one <PAD> to each row's raw units, and the model packs each
        // row at that length (trainer.predict: lengths = len(raw)), so a row's own length + 1.
        var order = Enumerable.Range(0, paragraphs.Count).OrderByDescending(i => paragraphs[i].Length).ToArray();
        var preds = new int[paragraphs.Count][];
        for (int b = 0; b < order.Length; b += _batchSize)
        {
            ct.ThrowIfCancellationRequested();
            var rows = order[b..Math.Min(b + _batchSize, order.Length)].Select(i => paragraphs[i]).ToList();
            int maxLen = rows.Max(r => r.Length);
            int[][] rowPreds;
            if (maxLen + 1 <= MaxSeqLen)
            {
                var p = Argmax(Run(rows, new int[rows.Count], maxLen + 1, rows.Select(r => r.Length + 1).ToArray(), ct), rows.Count);
                rowPreds = rows.Select((r, j) => p[j][..r.Length]).ToArray();
            }
            else
            {
                rowPreds = PredictWindowed(rows, ct);
            }
            for (int j = 0; j < rows.Count; j++)
            {
                FixLabels(rows[j], rowPreds[j]);
                preds[order[b + j]] = rowPreds[j];
            }
        }
        return preds;
    }

    /// <summary>Paragraphs over MaxSeqLen run in windows that restart after the last predicted sentence end.</summary>
    private int[][] PredictWindowed(List<Paragraph> rows, CancellationToken ct)
    {
        var idx = new int[rows.Count];
        var output = rows.Select(_ => new List<int>()).ToArray();
        for (bool first = true; ; first = false)
        {
            ct.ThrowIfCancellationRequested();
            var ens = rows.Select((r, j) => Math.Min(r.Length - idx[j], MaxSeqLen)).ToArray();
            int width = ens.Max();
            // The first window keeps collate's raw units (own length + 1, cut to the window);
            // advance_old_batch pads every row's raw units to the batch width, so later windows
            // run every row at full width and padding reaches the backward LSTM.
            var lengths = rows.Select(r => first ? Math.Min(r.Length + 1, width) : width).ToArray();
            var p = Argmax(Run(rows, idx, width, lengths, ct), rows.Count);
            for (int j = 0; j < rows.Count; j++)
            {
                int lastBreak = Array.FindLastIndex(p[j], x => x is 2 or 4);
                int advance = lastBreak < 0 || idx[j] >= rows[j].Length - MaxSeqLen ? ens[j] : lastBreak + 1;
                output[j].AddRange(p[j][..advance]);
                idx[j] += advance;
            }
            if (rows.Select((r, j) => idx[j] >= r.Length).All(x => x))
                return output.Select(o => o.ToArray()).ToArray();
        }
    }

    /// <summary>Runs rows[j] from unit offsets[j], padded/truncated to width, packed at lengths[j]. Returns [rows, width, 5].</summary>
    private float[] Run(List<Paragraph> rows, int[] offsets, int width, int[] lengths, CancellationToken ct)
    {
        var ids = new long[rows.Count * width];
        Array.Fill(ids, _padId);
        var feats = new float[rows.Count * width * _featDim];
        for (int j = 0; j < rows.Count; j++)
        {
            int count = Math.Clamp(rows[j].Length - offsets[j], 0, width);
            Array.Copy(rows[j].Ids, offsets[j], ids, j * width, count);
            Array.Copy(rows[j].Feats, offsets[j] * _featDim, feats, j * width * _featDim, count * _featDim);
        }
        return _net.Forward(ids, feats, rows.Count, width, lengths.Select(l => (long)l).ToArray(), ct);
    }

    private const int Classes = 5;

    /// <summary>Per row and position, the class with the highest log-probability (the first on a tie, like torch.argmax).</summary>
    private static int[][] Argmax(float[] logits, int rows)
    {
        int width = logits.Length / (rows * Classes);
        var result = new int[rows][];
        for (int j = 0; j < rows; j++)
        {
            result[j] = new int[width];
            for (int i = 0; i < width; i++)
            {
                var v = logits.AsSpan((j * width + i) * Classes, Classes);
                int best = 0;
                for (int c = 1; c < Classes; c++)
                    if (v[c] > v[best])
                        best = c;
                result[j][i] = best;
            }
        }
        return result;
    }

    /// <summary>The paragraph always ends a sentence; e-mail addresses and URLs are kept whole.</summary>
    private static void FixLabels(Paragraph para, int[] pred)
    {
        int last = para.Length - 1;
        pred[last] = pred[last] < 2 ? 2 : pred[last] > 2 ? 4 : 2;

        var (joined, unitAt) = Join(para.Units);
        foreach (Match m in MaskRe.Matches(joined))
        {
            int first = unitAt[m.Index], end = unitAt[m.Index + m.Length - 1];
            for (int i = first; i < end; i++)
                pred[i] = 0;
            if (pred[end] == 0)
                pred[end] = 1;
        }
    }

    // ----- output: utils.decode_predictions -----

    private static void Decode(Paragraph para, int[] pred, Document doc, bool splitSentences)
    {
        Sentence? sent = null;
        int tokStart = 0;
        for (int i = 0; i < para.Length; i++)
        {
            if (pred[i] < 1)
                continue;

            // The token is units [tokStart, i] minus leading spaces; all-space tokens are dropped.
            int first = tokStart;
            while (first <= i && para.Units[first] == " ")
                first++;
            int label = pred[i];
            tokStart = i + 1;
            if (first > i)
                continue;

            var text = string.Concat(para.Units[first..(i + 1)]);
            if (text.Length > MaxSeqLen)
                text = LongTokenReplacement;
            int start = para.Starts[first], end = para.Ends[i];

            sent ??= NewSentence(doc);
            var token = new Token { Text = text, StartChar = start, EndChar = end, IsMwtCandidate = label >= 3 };
            token.Words.Add(new Word { Id = sent.Tokens.Count + 1, Text = text, StartChar = start, EndChar = end });
            sent.Tokens.Add(token);

            // no_ssplit: only the paragraph's end (below) ends the sentence.
            if (label is 2 or 4 && splitSentences)
            {
                FinishSentence(sent, doc);
                sent = null;
            }
        }
        if (sent != null)
            FinishSentence(sent, doc);
    }

    private static Sentence NewSentence(Document doc)
    {
        var sent = new Sentence { SentId = doc.Sentences.Count.ToString() };
        doc.Sentences.Add(sent);
        return sent;
    }

    private static void FinishSentence(Sentence sent, Document doc) =>
        sent.Text = doc.Text![sent.Tokens[0].StartChar!.Value..sent.Tokens[^1].EndChar!.Value];

    // ----- helpers -----

    /// <summary>Concatenates units and maps each UTF-16 index of the result back to its unit.</summary>
    private static (string Joined, int[] UnitAt) Join(string[] units)
    {
        var sb = new StringBuilder();
        var unitAt = new List<int>();
        for (int i = 0; i < units.Length; i++)
        {
            sb.Append(units[i]);
            for (int k = 0; k < units[i].Length; k++)
                unitAt.Add(i);
        }
        return (sb.ToString(), unitAt.ToArray());
    }

    public void Dispose() => _net.Dispose();
}
