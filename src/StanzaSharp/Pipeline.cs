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
/// Runs Stanza's English default pipeline: tokenize → mwt → pos → constituency.
/// </summary>
/// <example>
/// <code>
/// using var nlp = Pipeline.Load("models/converted/en");
/// var doc = nlp.Process("Barack Obama was born in Hawaii.");
/// Console.WriteLine(doc.Sentences[0].Constituency);
/// </code>
/// </example>
public sealed class Pipeline : IDisposable
{
    public const string AllProcessors = "tokenize,mwt,pos,constituency";

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

    private Pipeline(string modelDir, HashSet<string> processors)
    {
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
    /// <param name="processors">Comma-separated processors: those in <see cref="AllProcessors"/>, plus <c>lemma</c> and <c>depparse</c>; each needs the ones before it.</param>
    /// <param name="device">Where the models run: CPU by default, or e.g. <c>torch.CUDA</c> with a <c>TorchSharp-cuda-*</c> package.
    /// Only the CPU gives output identical to Python Stanza on CPU; see the README's GPU section.</param>
    public static Pipeline Load(string modelDir, string processors = AllProcessors, torch.Device? device = null)
    {
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir} (download them with ModelDownloader.DownloadAsync or \"StanzaSharp.Cli download\")");
        var set = processors.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(p => p.ToLowerInvariant()).ToHashSet();
        foreach (var p in set)
        {
            if (!Requires.TryGetValue(p, out var needs))
                throw new ArgumentException($"Unknown processor '{p}'. Available: {string.Join(",", Requires.Keys)}", nameof(processors));
            var missing = needs.Where(n => !set.Contains(n)).ToList();
            if (missing.Count > 0)
                throw new ArgumentException($"Processor '{p}' requires {string.Join(", ", missing)}", nameof(processors));
        }
        if (set.Count == 0)
            throw new ArgumentException("No processors given", nameof(processors));
        return Weights.On(device, () => new Pipeline(modelDir, set));
    }

    public Document Process(string text)
    {
        var doc = _tokenizer.Process(text);
        _mwt?.Process(doc);
        // The parser reuses the tagger's charlm outputs instead of computing them again.
        using var charlms = _pos != null && _parser != null ? new CharlmCache() : null;
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
