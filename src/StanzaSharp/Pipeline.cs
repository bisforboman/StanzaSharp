using StanzaSharp.Constituency;
using StanzaSharp.Depparse;
using StanzaSharp.Lemma;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;
using TorchSharp;

namespace StanzaSharp;

/// <summary>
/// Runs Stanza's English default pipeline: tokenize → mwt → pos → lemma → depparse → constituency.
/// </summary>
/// <example>
/// <code>
/// using var nlp = Pipeline.Load("models/stanza/en");
/// var doc = nlp.Process("Barack Obama was born in Hawaii.");
/// Console.WriteLine(doc.Sentences[0].Constituency);
/// </code>
/// </example>
public sealed class Pipeline : IDisposable
{
    public const string AllProcessors = "tokenize,mwt,pos,lemma,depparse,constituency";

    // Stanza's processor dependencies (REQUIRES_DEFAULT); English pos also needs mwt to have run.
    private static readonly Dictionary<string, string[]> Requires = new()
    {
        ["tokenize"] = [],
        ["mwt"] = ["tokenize"],
        ["pos"] = ["tokenize", "mwt"],
        ["lemma"] = ["tokenize", "mwt", "pos"], // Stanza only requires tokenize, but the model reads UPOS
        ["depparse"] = ["tokenize", "mwt", "pos", "lemma"],
        ["constituency"] = ["tokenize", "mwt", "pos"],
    };

    private readonly Tokenizer _tokenizer;
    private readonly MwtExpander? _mwt;
    private readonly PosTagger? _pos;
    private readonly Lemmatizer? _lemma;
    private readonly DependencyParser? _depparse;
    private readonly ConstituencyParser? _parser;
    private readonly Pretrain? _pretrain;
    private readonly CharLanguageModel? _charlmForward, _charlmBackward;
    private readonly CharlmCacheOptions _cacheOptions;

    private Pipeline(string modelDir, HashSet<string> processors, CharlmCacheOptions cacheOptions)
    {
        _cacheOptions = cacheOptions;
        string Model(string relative) => Path.Combine(modelDir, relative);

        _tokenizer = Tokenizer.Load(Model("tokenize/combined_nocharlm"));
        if (processors.Contains("mwt"))
            _mwt = MwtExpander.Load(Model("mwt/combined"));
        if (processors.Contains("pos") || processors.Contains("depparse") || processors.Contains("constituency"))
        {
            _pretrain = Pretrain.Load(Model("pretrain/conll17"));
            _charlmForward = CharLanguageModel.Load(Model("forward_charlm/1billion"));
            _charlmBackward = CharLanguageModel.Load(Model("backward_charlm/1billion"));
        }
        if (processors.Contains("pos"))
            _pos = PosTagger.Load(Model("pos/combined_charlm"), _pretrain!, _charlmForward!, _charlmBackward!);
        if (processors.Contains("lemma"))
            _lemma = Lemmatizer.Load(Model("lemma/combined_nocharlm"));
        if (processors.Contains("depparse"))
            _depparse = DependencyParser.Load(Model("depparse/combined_charlm"), _pretrain!, _charlmForward!, _charlmBackward!);
        if (processors.Contains("constituency"))
            _parser = ConstituencyParser.Load(Model("constituency/ptb3-revised_charlm"), _pretrain!, _charlmForward!, _charlmBackward!);
    }

    /// <summary>
    /// Loads the English models from <paramref name="modelDir"/>: either converted ones (e.g. <c>models/converted/en</c>)
    /// or Stanza's own download with its <c>.pt</c> files (e.g. <c>models/stanza/en</c>), chosen per file.
    /// </summary>
    /// <param name="options">Processors, device and cache settings; defaults to all six processors on the CPU.</param>
    public static Pipeline Load(string modelDir, PipelineOptions? options = null)
    {
        options ??= new PipelineOptions();
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir} (download them with ModelDownloader.DownloadAsync or \"StanzaSharp.Cli download\")");
        var set = options.Processors.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(p => p.ToLowerInvariant()).ToHashSet();
        foreach (var p in set)
        {
            if (!Requires.TryGetValue(p, out var needs))
                throw new ArgumentException($"Unknown processor '{p}'. Available: {string.Join(",", Requires.Keys)}", nameof(options));
            var missing = needs.Where(n => !set.Contains(n)).ToList();
            if (missing.Count > 0)
                throw new ArgumentException($"Processor '{p}' requires {string.Join(", ", missing)}", nameof(options));
        }
        if (set.Count == 0)
            throw new ArgumentException("No processors given", nameof(options));
        if (options.CharlmCache.IsEnabled && options.CharlmCache.MaxWords <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "CharlmCache.MaxWords must be positive; set IsEnabled = false to turn the cache off");

        if (options.DisableTf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = false;
        return Weights.On(options.Device, () => new Pipeline(modelDir, set, options.CharlmCache));
    }

    public Document Process(string text)
    {
        var doc = _tokenizer.Process(text);
        _mwt?.Process(doc);
        // The parser reuses the tagger's charlm outputs instead of computing them again.
        using var charlms = _pos != null && _parser != null && _cacheOptions.IsEnabled ? new CharlmCache(_cacheOptions.MaxWords) : null;
        _pos?.Process(doc, charlms);
        _lemma?.Process(doc);
        // No CharlmCache: depparse runs the charlms with a ROOT word in front, so the tagger's outputs don't apply.
        _depparse?.Process(doc);
        _parser?.Process(doc, charlms);
        return doc;
    }

    public void Dispose()
    {
        _parser?.Dispose();
        _depparse?.Dispose();
        _pos?.Dispose();
        _lemma?.Dispose();
        _mwt?.Dispose();
        _tokenizer.Dispose();
        _charlmForward?.Dispose();
        _charlmBackward?.Dispose();
        _pretrain?.Dispose();
    }
}
