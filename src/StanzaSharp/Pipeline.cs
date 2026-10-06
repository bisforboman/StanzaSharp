using StanzaSharp.Constituency;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;

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
        ["constituency"] = ["tokenize", "mwt", "pos"],
    };

    private readonly Tokenizer _tokenizer;
    private readonly MwtExpander? _mwt;
    private readonly PosTagger? _pos;
    private readonly ConstituencyParser? _parser;
    private readonly Pretrain? _pretrain;
    private readonly CharLanguageModel? _charlmForward, _charlmBackward;

    private Pipeline(string modelDir, HashSet<string> processors)
    {
        string Model(string relative) => Path.Combine(modelDir, relative);

        _tokenizer = Tokenizer.Load(Model("tokenize/combined_nocharlm"));
        if (processors.Contains("mwt"))
            _mwt = MwtExpander.Load(Model("mwt/combined"));
        if (processors.Contains("pos") || processors.Contains("constituency"))
        {
            _pretrain = Pretrain.Load(Model("pretrain/conll17"));
            _charlmForward = CharLanguageModel.Load(Model("forward_charlm/1billion"));
            _charlmBackward = CharLanguageModel.Load(Model("backward_charlm/1billion"));
        }
        if (processors.Contains("pos"))
            _pos = PosTagger.Load(Model("pos/combined_charlm"), _pretrain!, _charlmForward!, _charlmBackward!);
        if (processors.Contains("constituency"))
            _parser = ConstituencyParser.Load(Model("constituency/ptb3-revised_charlm"), _pretrain!, _charlmForward!, _charlmBackward!);
    }

    /// <summary>
    /// Loads the converted English models from <paramref name="modelDir"/> (e.g. <c>models/converted/en</c>).
    /// </summary>
    /// <param name="processors">Comma-separated subset of <see cref="AllProcessors"/>; each needs the ones before it.</param>
    public static Pipeline Load(string modelDir, string processors = AllProcessors)
    {
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir} (run setup.ps1 -Models)");
        var set = processors.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(p => p.ToLowerInvariant()).ToHashSet();
        foreach (var p in set)
        {
            if (!Requires.TryGetValue(p, out var needs))
                throw new ArgumentException($"Unknown processor '{p}'. Available: {AllProcessors}", nameof(processors));
            var missing = needs.Where(n => !set.Contains(n)).ToList();
            if (missing.Count > 0)
                throw new ArgumentException($"Processor '{p}' requires {string.Join(", ", missing)}", nameof(processors));
        }
        if (set.Count == 0)
            throw new ArgumentException("No processors given", nameof(processors));
        return new Pipeline(modelDir, set);
    }

    public Document Process(string text)
    {
        var doc = _tokenizer.Process(text);
        _mwt?.Process(doc);
        // The parser reuses the tagger's charlm outputs instead of computing them again.
        using var charlms = _pos != null && _parser != null ? new CharlmCache() : null;
        _pos?.Process(doc, charlms);
        _parser?.Process(doc, charlms);
        return doc;
    }

    public void Dispose()
    {
        _parser?.Dispose();
        _pos?.Dispose();
        _mwt?.Dispose();
        _tokenizer.Dispose();
        _charlmForward?.Dispose();
        _charlmBackward?.Dispose();
        _pretrain?.Dispose();
    }
}
