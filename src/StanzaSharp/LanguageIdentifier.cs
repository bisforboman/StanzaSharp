using System.Buffers;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp;

/// <summary>
/// Stanza's language identifier (the <c>langid</c> processor, <c>multilingual/langid/ud.pt</c>): which of 68 languages
/// a text is in, as UD codes (<c>en</c>, <c>de</c>, <c>fr</c>, <c>zh-hans</c>, ...). It reads the text's characters only,
/// so it is meant for whole documents or sentences rather than single words.
/// </summary>
/// <example>
/// <code>
/// await ModelDownloader.DownloadLanguageIdentifierAsync("models/stanza/multilingual");
/// using var identifier = LanguageIdentifier.Load("models/stanza/multilingual");
/// string language = identifier.Identify("Das ist ein deutscher Satz."); // "de"
/// </code>
/// </example>
public sealed class LanguageIdentifier : IDisposable
{
    // The model's path in its folder (models/stanza/multilingual), without the extension.
    internal const string ModelPath = "langid/ud";

    private readonly LangIdModel _model;
    private readonly BackendModels _backend;

    private LanguageIdentifier(LangIdModel model, BackendModels backend) => (_model, _backend) = (model, backend);

    /// <summary>
    /// Loads the identifier from <paramref name="modelDir"/>, Stanza's <c>multilingual</c> folder (e.g.
    /// <c>models/stanza/multilingual</c>, holding <c>langid/ud.pt</c>; download it with
    /// <see cref="ModelDownloader.DownloadLanguageIdentifierAsync"/>).
    /// </summary>
    /// <exception cref="ArgumentException">A code in <see cref="LanguageIdentifierOptions.Languages"/> the model doesn't know.</exception>
    /// <exception cref="FileNotFoundException">The model file is missing.</exception>
    /// <exception cref="InvalidDataException">With <see cref="LanguageIdentifierOptions.VerifyChecksums"/>: the file's checksum differs from Stanza's.</exception>
    public static LanguageIdentifier Load(string modelDir, LanguageIdentifierOptions? options = null)
    {
        options ??= new LanguageIdentifierOptions();
        if (options.Threads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Threads must be at least 1, or null for the default");
        var basePath = Path.Combine(modelDir, ModelPath);
        if (!File.Exists(basePath + ".pt") && !File.Exists(basePath + ".json"))
        {
            throw new FileNotFoundException($"Model file not found: {basePath}.pt. Download it with ModelDownloader.DownloadLanguageIdentifierAsync(\"{modelDir}\") "
                + $"or \"stanzasharp download {modelDir} --lang multilingual\".", basePath + ".pt");
        }

        if (options.VerifyChecksums)
            ModelDownloader.VerifyLanguageIdentifier(modelDir);
        var backend = options.Backend.Models();
        try
        {
            backend.Configure(new PipelineOptions { Backend = options.Backend, Threads = options.Threads });
            var model = backend.Load(() => new LangIdModel(Checkpoint.Load(basePath), backend.LangId, options.Languages));
            return new LanguageIdentifier(model, backend);
        }
        catch
        {
            backend.Dispose();
            throw;
        }
    }

    /// <summary>The model, for tests (its scores).</summary>
    internal LangIdModel Model => _model;

    /// <summary>The languages the model knows, as UD codes in the model's order.</summary>
    public IReadOnlyList<string> Languages => _model.Languages;

    /// <summary>The language of <paramref name="text"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="text"/> is empty (Stanza fails on it too).</exception>
    public string Identify(string text) => Identify([text])[0];

    /// <summary>The language of each text, in order; texts of one length are scored in one batch, as in Stanza.</summary>
    /// <exception cref="ArgumentException">A text is empty (Stanza fails on it too).</exception>
    public IReadOnlyList<string> Identify(IEnumerable<string> texts, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        return _model.Identify([.. texts], out _, cancellationToken);
    }

    /// <summary>Frees the model.</summary>
    public void Dispose()
    {
        _model.Dispose();
        _backend.Dispose();
    }
}

/// <summary>Settings for <see cref="LanguageIdentifier.Load"/>.</summary>
public sealed class LanguageIdentifierOptions
{
    /// <summary>
    /// The only languages the identifier may answer, Stanza's <c>langid_lang_subset</c> (e.g. <c>["en", "de"]</c>); every
    /// other language's score is set to minus infinity, so a French text then comes out as English or German. Null or
    /// empty: all of them. An unknown code throws <see cref="ArgumentException"/> at <see cref="LanguageIdentifier.Load"/>.
    /// </summary>
    public IReadOnlyCollection<string>? Languages { get; init; }

    /// <summary>The backend the model runs on: <see cref="PipelineBackend.Managed"/> (the default) or one of StanzaSharp.Cuda's.</summary>
    public PipelineBackend Backend { get; init; } = PipelineBackend.Managed;

    /// <summary>The threads the model's math runs on, as <see cref="PipelineOptions.Threads"/>; null for the default.</summary>
    public int? Threads { get; init; }

    /// <summary>Checks the model file against Stanza's MD5 before loading, as <see cref="PipelineOptions.VerifyChecksums"/> does.</summary>
    public bool VerifyChecksums { get; init; }
}

/// <summary>The language identifier's network per backend: summed tag scores of equal-length rows of character ids.</summary>
internal interface ILangIdNet : IDisposable
{
    /// <summary>[rows, Tags]: the Linear's output at every position of each row, summed over the positions (model.py forward).</summary>
    float[] Forward(long[] ids, int rows, int width, CancellationToken ct);
}

/// <summary>
/// stanza/pipeline/langid_processor.py + models/langid/model.py: characters (code points; <c>UNK</c> for unknown ones) to
/// ids, texts grouped by length (one batch per length, so no row is padded), the summed scores, the language subset's
/// mask, argmax. <c>clean_text</c> (off by default; it needs the emoji package's tables) is not ported.
/// </summary>
internal sealed class LangIdModel : IDisposable
{
    private readonly Dictionary<string, int> _charToId;
    private readonly int _unkId;
    private readonly string[] _tags;
    private readonly bool[]? _allowed;
    private readonly ILangIdNet _net;

    public LangIdModel(Checkpoint ckpt, Func<Checkpoint, ILangIdNet> net, IReadOnlyCollection<string>? subset)
    {
        var root = ckpt.Root;
        _charToId = Checkpoint.StringToId(root["char_to_idx"] ?? throw new InvalidDataException("No char_to_idx in the langid model"));
        _unkId = _charToId.TryGetValue("UNK", out int unk) ? unk : throw new InvalidDataException("No UNK in the langid model's characters");
        var tags = Checkpoint.StringToId(root["tag_to_idx"] ?? throw new InvalidDataException("No tag_to_idx in the langid model"));
        _tags = [.. tags.OrderBy(t => t.Value).Select(t => t.Key)]; // idx_to_tag: sorted by index
        if (subset is { Count: > 0 })
        {
            var unknown = subset.Where(code => !tags.ContainsKey(code)).ToList();
            if (unknown.Count > 0)
            {
                throw new ArgumentException($"Unknown language {string.Join(", ", unknown)} for the language identifier. Known: "
                    + string.Join(", ", _tags.Order(StringComparer.Ordinal)), "options");
            }

            _allowed = [.. _tags.Select(subset.Contains)];
        }

        _net = net(ckpt);
    }

    public IReadOnlyList<string> Languages => _tags;

    /// <summary>Each text's language; <paramref name="scores"/> gets the summed scores before the subset's mask (for tests).</summary>
    public string[] Identify(IReadOnlyList<string> texts, out float[][] scores, CancellationToken ct = default)
    {
        var units = texts.Select(t => t.EnumerateRunes().Select(r => r.ToString()).ToArray()).ToList();
        if (units.FindIndex(u => u.Length == 0) is int empty and >= 0)
            throw new ArgumentException($"Text {empty} is empty: the language identifier needs at least one character", nameof(texts));
        var result = new string[texts.Count];
        scores = new float[texts.Count][];
        foreach (var group in Enumerable.Range(0, texts.Count).GroupBy(i => units[i].Length))
        {
            ct.ThrowIfCancellationRequested();
            var rows = group.ToArray();
            int width = group.Key;
            var ids = new long[rows.Length * width];
            for (int r = 0; r < rows.Length; r++)
            {
                for (int t = 0; t < width; t++)
                    ids[r * width + t] = _charToId.GetValueOrDefault(units[rows[r]][t], _unkId);
            }

            var sums = _net.Forward(ids, rows.Length, width, ct);
            for (int r = 0; r < rows.Length; r++)
            {
                var row = sums.AsSpan(r * _tags.Length, _tags.Length);
                scores[rows[r]] = row.ToArray();

                // prediction_scores: + the mask (-inf off the subset), torch.argmax (the first maximum).
                int best = -1;
                for (int j = 0; j < row.Length; j++)
                {
                    if ((_allowed == null || _allowed[j]) && (best < 0 || row[j] > row[best]))
                        best = j;
                }

                result[rows[r]] = _tags[best];
            }
        }

        return result;
    }

    public void Dispose() => _net.Dispose();
}

/// <summary>The language identifier's network on the managed backend: embedding, a biLSTM (<see cref="ManagedLstm"/>), a Linear per position, the sum.</summary>
internal sealed class ManagedLangIdNet : ILangIdNet
{
    private readonly float[] _embeddings; // [chars, emb]
    private readonly int _emb, _tags;
    private readonly ManagedLstm _lstm;
    private readonly PackedMatrix _toTag;
    private readonly float[] _toTagBias;

    public ManagedLangIdNet(Checkpoint ckpt)
    {
        var root = ckpt.Root;
        var model = root["model_state_dict"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        _emb = root["embedding_dim"]!.GetValue<int>();
        int hidden = root["hidden_dim"]!.GetValue<int>();
        _embeddings = T("char_embeds.weight");
        _lstm = new ManagedLstm(ckpt, model, "lstm.", _emb, hidden, root["num_layers"]!.GetValue<int>(), bidirectional: true);
        var bias = T("hidden_to_tag.bias");
        _tags = bias.Length;
        _toTag = new PackedMatrix(T("hidden_to_tag.weight"), _tags, 2 * hidden);
        _toTagBias = new float[_toTag.PaddedN];
        bias.CopyTo(_toTagBias, 0);
    }

    public float[] Forward(long[] ids, int rows, int width, CancellationToken ct)
    {
        int n = rows * width, h = _lstm.OutputSize;
        var x = ArrayPool<float>.Shared.Rent(n * _emb);
        var encoded = ArrayPool<float>.Shared.Rent(n * h);
        try
        {
            for (int i = 0; i < n; i++)
                _embeddings.AsSpan((int)ids[i] * _emb, _emb).CopyTo(x.AsSpan(i * _emb));
            _lstm.ForwardPadded(x, rows, width, [.. Enumerable.Repeat((long)width, rows)], encoded, ct);
            var tags = Gemm.Run(encoded, n, _toTag, _toTagBias); // [n, PaddedN]

            // torch.sum(dim=1) over the positions, in position order (in double: closer to Stanza's float64 sums).
            var sums = new float[rows * _tags];
            for (int r = 0; r < rows; r++)
            {
                for (int j = 0; j < _tags; j++)
                {
                    double sum = 0;
                    for (int t = 0; t < width; t++)
                        sum += tags[(r * width + t) * _toTag.PaddedN + j];
                    sums[r * _tags + j] = (float)sum;
                }
            }

            return sums;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(encoded);
        }
    }

    public void Dispose()
    {
    }
}
