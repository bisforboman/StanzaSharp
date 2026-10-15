using System.Diagnostics;
using System.Runtime.ExceptionServices;
using Microsoft.Extensions.Logging;
using StanzaSharp.Constituency;
using StanzaSharp.Depparse;
using StanzaSharp.Lemma;
using StanzaSharp.Mwt;
using StanzaSharp.Ner;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Sentiment;
using StanzaSharp.Tokenize;

namespace StanzaSharp;

/// <summary>
/// Runs Stanza's English pipeline in its order: tokenize → mwt → pos → lemma → constituency → depparse → sentiment → ner
/// (by default all those of the selected package).
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
    /// <summary>Every processor, in the order Stanza runs them (PIPELINE_NAMES): Stanza's English default.</summary>
    public const string AllProcessors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos},{Processor.Lemma},{Processor.Constituency},{Processor.Depparse},{Processor.Sentiment},{Processor.Ner}";

    /// <summary>The package <see cref="PipelineOptions.Package"/> selects by default: Stanza's English <see langword="default"/>.</summary>
    public const string DefaultPackage = "default";

    internal const string English = "en";

    // Stanza 1.15.0's packages per language (resources.json): each processor's model. Their files need MD5s in
    // ModelDownloader.Files, and the shared files they read an entry in Dependencies.
    internal static readonly Dictionary<string, Dictionary<string, Dictionary<string, string>>> Packages = new()
    {
        [English] = new()
        {
            [DefaultPackage] = new()
            {
                ["tokenize"] = "combined_nocharlm",
                ["mwt"] = "combined",
                ["pos"] = "combined_charlm",
                ["lemma"] = "combined_nocharlm",
                ["constituency"] = "ptb3-revised_charlm",
                ["depparse"] = "combined_charlm",
                ["sentiment"] = "sstplus_charlm",
                ["ner"] = "ontonotes-ww-multi_charlm",
            },

            // Smaller and faster: pos, depparse and ner have their own character LSTMs instead of the charlms. No constituency.
            ["default_fast"] = new()
            {
                ["tokenize"] = "combined_nocharlm",
                ["mwt"] = "combined",
                ["pos"] = "combined_nocharlm",
                ["lemma"] = "combined_nocharlm",
                ["depparse"] = "combined_nocharlm",
                ["sentiment"] = "sstplus_charlm",
                ["ner"] = "ontonotes-ww-multi_nocharlm",
            },

            // The treebank packages: tokenize to depparse only. combined has default's models; eslspok has no mwt and
            // Stanza's identity lemmatizer (no model).
            ["combined"] = Treebank("combined"),
            ["ewt"] = Treebank("ewt"),
            ["gum"] = Treebank("gum"),
            ["lines"] = Treebank("lines"),
            ["partut"] = Treebank("partut"),
            ["atis"] = Treebank("atis"),
            ["childes"] = Treebank("childes"),
            ["eslspok"] = new()
            {
                ["tokenize"] = "eslspok_nocharlm",
                ["pos"] = "eslspok_charlm",
                ["lemma"] = IdentityLemma,
                ["depparse"] = "eslspok_charlm",
            },
        },
        ["de"] = new()
        {
            [DefaultPackage] = new()
            {
                ["tokenize"] = "combined_nocharlm",
                ["mwt"] = "combined",
                ["pos"] = "combined_charlm",
                ["lemma"] = "combined_nocharlm",
                ["constituency"] = "spmrl_charlm",
                ["depparse"] = "combined_charlm",
                ["sentiment"] = "sb10k_charlm",
                ["ner"] = "germeval2014",
            },

            // Like English's, but ner and sentiment keep their charlm models.
            ["default_fast"] = new()
            {
                ["tokenize"] = "combined_nocharlm",
                ["mwt"] = "combined",
                ["pos"] = "combined_nocharlm",
                ["lemma"] = "combined_nocharlm",
                ["depparse"] = "combined_nocharlm",
                ["sentiment"] = "sb10k_charlm",
                ["ner"] = "germeval2014",
            },

            // The treebank packages: tokenize to depparse only.
            ["combined"] = Treebank("combined"),
            ["gsd"] = Treebank("gsd"),
            ["hdt"] = Treebank("hdt"),
        },
    };

    /// <summary>Stanza's lemma "model" that sets each word's lemma to its text (LemmaProcessor's use_identity); it has no file.</summary>
    internal const string IdentityLemma = "identity";

    // A treebank package: <name>_nocharlm tokenizer and lemmatizer, <name> MWT, <name>_charlm tagger and parser.
    private static Dictionary<string, string> Treebank(string name) => new()
    {
        ["tokenize"] = $"{name}_nocharlm",
        ["mwt"] = name,
        ["pos"] = $"{name}_charlm",
        ["lemma"] = $"{name}_nocharlm",
        ["depparse"] = $"{name}_charlm",
    };

    private static readonly string[] Conll17 = ["pretrain/conll17"],
        EnglishCharlm = ["pretrain/conll17", "forward_charlm/1billion", "backward_charlm/1billion"],
        GermanCharlm = ["pretrain/conll17", "forward_charlm/newswiki", "backward_charlm/newswiki"];

    // The shared files (pretrains, charlms) each model reads, by "<processor>/<model>": resources.json's "dependencies",
    // which the model's name doesn't tell (germeval2014 reads charlms and its own pretrain). Models without an entry
    // (tokenize, mwt, lemma) read none.
    internal static readonly Dictionary<string, Dictionary<string, string[]>> Dependencies = new()
    {
        [English] = new()
        {
            ["pos/combined_charlm"] = EnglishCharlm,
            ["constituency/ptb3-revised_charlm"] = EnglishCharlm,
            ["depparse/combined_charlm"] = EnglishCharlm,
            ["sentiment/sstplus_charlm"] = EnglishCharlm,
            ["ner/ontonotes-ww-multi_charlm"] = EnglishCharlm,
            ["pos/combined_nocharlm"] = Conll17,
            ["depparse/combined_nocharlm"] = Conll17,
            ["ner/ontonotes-ww-multi_nocharlm"] = Conll17,
            ["pos/ewt_charlm"] = EnglishCharlm,
            ["depparse/ewt_charlm"] = EnglishCharlm,
            ["pos/gum_charlm"] = EnglishCharlm,
            ["depparse/gum_charlm"] = EnglishCharlm,
            ["pos/lines_charlm"] = EnglishCharlm,
            ["depparse/lines_charlm"] = EnglishCharlm,
            ["pos/partut_charlm"] = EnglishCharlm,
            ["depparse/partut_charlm"] = EnglishCharlm,
            ["pos/atis_charlm"] = EnglishCharlm,
            ["depparse/atis_charlm"] = EnglishCharlm,
            ["pos/childes_charlm"] = EnglishCharlm,
            ["depparse/childes_charlm"] = EnglishCharlm,
            ["pos/eslspok_charlm"] = EnglishCharlm,
            ["depparse/eslspok_charlm"] = EnglishCharlm,
        },
        ["de"] = new()
        {
            ["pos/combined_charlm"] = GermanCharlm,
            ["constituency/spmrl_charlm"] = GermanCharlm,
            ["depparse/combined_charlm"] = GermanCharlm,
            ["sentiment/sb10k_charlm"] = GermanCharlm,
            ["ner/germeval2014"] = ["pretrain/fasttextwiki", "forward_charlm/newswiki", "backward_charlm/newswiki"],
            ["pos/combined_nocharlm"] = Conll17,
            ["depparse/combined_nocharlm"] = Conll17,
            ["pos/gsd_charlm"] = GermanCharlm,
            ["depparse/gsd_charlm"] = GermanCharlm,
            ["pos/hdt_charlm"] = GermanCharlm,
            ["depparse/hdt_charlm"] = GermanCharlm,
        },
    };

    // Stanza's processor dependencies (REQUIRES_DEFAULT); English pos also needs mwt to have run.
    internal static readonly Dictionary<string, string[]> Requires = new()
    {
        ["tokenize"] = [],
        ["mwt"] = ["tokenize"],
        ["pos"] = ["tokenize", "mwt"],
        ["lemma"] = ["tokenize", "mwt", "pos"], // Stanza only requires tokenize, but the model reads UPOS
        ["depparse"] = ["tokenize", "mwt", "pos", "lemma"],
        ["ner"] = ["tokenize"], // reads only the tokens' text
        ["constituency"] = ["tokenize", "mwt", "pos"],
        ["sentiment"] = ["tokenize"], // the model reads only the tokens' text
    };

    private readonly Tokenizer _tokenizer;
    private readonly MwtExpander? _mwt;
    private readonly PosTagger? _pos;
    private readonly Lemmatizer? _lemma;
    private readonly DependencyParser? _depparse;
    private readonly NerTagger? _ner;
    private readonly ConstituencyParser? _parser;

    /// <summary>The loaded parser, for studies (samples/StanzaSharp.Benchmark constituency-divergence).</summary>
    internal ConstituencyParser? Parser => _parser;

    private readonly SentimentClassifier? _sentiment;
    private readonly Pretrain[] _pretrains = [];
    private readonly BackendModels _models;
    private readonly CharlmCacheOptions _cacheOptions;
    private readonly bool _splitSentences, _trimNativeHeap, _depparseReadsCache, _identityLemma;
    private readonly ILogger? _logger;
    private readonly SemaphoreSlim? _gate;

    private Pipeline(string modelDir, Dictionary<string, string> models, PipelineOptions options, BackendModels backend)
    {
        _models = backend;
        _cacheOptions = options.CharlmCache;
        _splitSentences = options.SplitSentences;
        _depparseReadsCache = backend.DepparseReadsCache;
        _trimNativeHeap = options.TrimNativeHeap && NativeHeap.CanTrim;
        _logger = options.Logger;
        _gate = options.MaxConcurrentCalls is int max ? new SemaphoreSlim(max, max) : null;
        string Model(string processor) => Path.Combine(modelDir, processor, models[processor]);
        string Name(string processor) => $"{processor}/{models[processor]}";

        // Managed: two models load at a time, on the thread pool (round 17: the load took 23-40% less time on the
        // runners, its peak 22-72 MB more; all at once took 28-46% less, but 95-152 MB more). The pretrain and the
        // charlms go first; their readers start once they are loaded. TorchSharp: one after another on this thread, in
        // this order (its device and dispose scopes are per thread).
        bool parallel = backend.LoadInParallel;
        using var lanes = new SemaphoreSlim(Math.Min(LoadLanes, Environment.ProcessorCount));
        T Timed<T>(string name, Func<T> load)
        {
            if (_logger?.IsEnabled(LogLevel.Information) != true)
                return Collected(load());
            long start = Stopwatch.GetTimestamp();
            var model = Collected(load());
            _logger.LogInformation("Loaded {Model} in {Milliseconds:F0} ms", name, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            return model;
        }

        T Collected<T>(T model)
        {
            if (backend.CollectAfterEachModel)
                GC.Collect();
            return model;
        }

        var started = new List<Task>();
        Task<T> Start<T>(string name, Func<T> load)
        {
            var task = parallel ? Task.Run(() => Laned(name, load)) : Task.FromResult(Timed(name, load));
            started.Add(task);
            return task;
        }

        T Laned<T>(string name, Func<T> load)
        {
            lanes.Wait();
            try
            {
                return Timed(name, load);
            }
            finally
            {
                lanes.Release();
            }
        }

        object Charlm(string path, bool forward)
        {
            backend.Charlm(Path.Combine(modelDir, path), forward);
            return path;
        }

        // Every pretrain the models read (German NER has its own), and one pair of charlms.
        var shared = SharedModels(models, options.Language);
        string? forwardPath = shared.SingleOrDefault(p => p.StartsWith("forward_charlm/", StringComparison.Ordinal));
        string? backwardPath = shared.SingleOrDefault(p => p.StartsWith("backward_charlm/", StringComparison.Ordinal));
        var pretrains = shared.Where(p => p.StartsWith("pretrain/", StringComparison.Ordinal))
            .ToDictionary(p => p, p => Start(p, () => backend.Pretrain(Path.Combine(modelDir, p))));
        var forward = forwardPath != null ? Start(forwardPath, () => Charlm(forwardPath, forward: true)) : null;
        var backward = backwardPath != null ? Start(backwardPath, () => Charlm(backwardPath, forward: false)) : null;

        // The pretrain a processor's model reads (Dependencies), once loaded.
        Pretrain PretrainOf(string processor) =>
            pretrains[Dependencies[options.Language][Name(processor)].Single(p => p.StartsWith("pretrain/", StringComparison.Ordinal))].Result;

        var tokenizer = Start(Name("tokenize"), () => backend.Tokenizer(Model("tokenize")));
        var mwt = models.ContainsKey("mwt") ? Start(Name("mwt"), () => backend.Mwt(Model("mwt"))) : null;
        _identityLemma = models.GetValueOrDefault("lemma") == IdentityLemma;
        var lemma = models.ContainsKey("lemma") && !_identityLemma ? Start(Name("lemma"), () => backend.Lemma(Model("lemma"))) : null;
        try
        {
            forward?.GetAwaiter().GetResult();
            backward?.GetAwaiter().GetResult();
            foreach (var p in pretrains.Values)
                p.GetAwaiter().GetResult();
            var pos = models.ContainsKey("pos") ? Start(Name("pos"), () => backend.Pos(Model("pos"), PretrainOf("pos"))) : null;
            var depparse = models.ContainsKey("depparse") ? Start(Name("depparse"), () => backend.Depparse(Model("depparse"), PretrainOf("depparse"))) : null;
            var ner = models.ContainsKey("ner") ? Start(Name("ner"), () => backend.Ner(Model("ner"), PretrainOf("ner"))) : null;
            var parser = models.ContainsKey("constituency") ? Start(Name("constituency"), () => backend.Constituency(Model("constituency"), PretrainOf("constituency"))) : null;
            var sentiment = models.ContainsKey("sentiment") ? Start(Name("sentiment"), () => backend.Sentiment(Model("sentiment"), PretrainOf("sentiment"))) : null;
            Task.WaitAll(started);
            _tokenizer = tokenizer.Result;
            _mwt = mwt?.Result;
            _lemma = lemma?.Result;
            _pos = pos?.Result;
            _depparse = depparse?.Result;
            _ner = ner?.Result;
            _parser = parser?.Result;
            _sentiment = sentiment?.Result;
            _pretrains = [.. pretrains.Values.Select(p => p.Result)];
        }
        catch (Exception e)
        {
            // Every load finishes before the first failure is thrown, as its own exception (not an AggregateException).
            try
            {
                Task.WaitAll(started);
            }
            catch (AggregateException)
            {
            }

            ExceptionDispatchInfo.Throw(started.FirstOrDefault(t => t.IsFaulted)?.Exception?.InnerException ?? e);
        }
    }

    /// <summary>
    /// Loads the models of <see cref="PipelineOptions.Language"/> (English by default) from <paramref name="modelDir"/>,
    /// that language's folder: either converted ones (e.g. <c>models/converted/en</c>) or Stanza's own download with its
    /// <c>.pt</c> files (e.g. <c>models/stanza/en</c>), chosen per file.
    /// </summary>
    /// <param name="options">Language, package, processors, backend, threads and cache settings; defaults to all eight
    /// processors of the English <see langword="default"/> package on <see cref="PipelineBackend.Managed"/>.</param>
    /// <exception cref="ArgumentException">An unknown language, package or processor, a processor the package lacks
    /// (e.g. constituency in <c>default_fast</c>), or a processor without the ones it requires.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="PipelineOptions.Threads"/> below 1, or a
    /// non-positive <see cref="CharlmCacheOptions.MaxWords"/>.</exception>
    /// <exception cref="InvalidDataException">With <see cref="PipelineOptions.VerifyChecksums"/>: a model file whose
    /// checksum differs from Stanza's.</exception>
    /// <exception cref="InvalidOperationException">With <see cref="PipelineOptions.VerifyChecksums"/>: a model that
    /// would be read from converted files, which have no published checksum.</exception>
    public static Pipeline Load(string modelDir, PipelineOptions? options = null)
    {
        options ??= new PipelineOptions();
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir} (download them with ModelDownloader.DownloadAsync or \"stanzasharp download\")");
        var models = SelectModels(options.Package, options.Processors, addRequired: false, nameof(options), options.Language);
        if (options.CharlmCache.IsEnabled && options.CharlmCache.MaxWords <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "CharlmCache.MaxWords must be positive; set IsEnabled = false to turn the cache off");
        if (options.Threads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Threads must be at least 1, or null for the default");
        if (options.MaxConcurrentCalls < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentCalls must be at least 1, or null for no limit");
        var logger = options.Logger;
        long start = Stopwatch.GetTimestamp();
        if (options.VerifyChecksums)
        {
            ModelDownloader.Verify(modelDir, models, options.Language);
            logger?.LogInformation("Verified the model checksums in {Milliseconds:F0} ms", Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        }

        var backend = options.Backend.Models();
        Pipeline pipeline;
        try
        {
            backend.Configure(options);
            pipeline = backend.Load(() => new Pipeline(modelDir, models, options, backend));
        }
        catch
        {
            backend.Dispose();
            throw;
        }

        logger?.LogInformation("Loaded the {Package} pipeline ({Processors}) in {Milliseconds:F0} ms", options.Package,
            string.Join(",", AllProcessors.Split(',').Where(models.ContainsKey)), Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return pipeline;
    }

    // How many models a managed pipeline loads at a time (round 17).
    private const int LoadLanes = 2;

    // English's shared files, for the samples that load them by hand.
    internal const string PretrainPath = "pretrain/conll17", ForwardCharlmPath = "forward_charlm/1billion", BackwardCharlmPath = "backward_charlm/1billion";

    /// <summary>
    /// The processors to run and each one's model: <paramref name="processors"/> (null: all of the package's), checked
    /// against <paramref name="package"/> of <paramref name="language"/>. With <paramref name="addRequired"/> the
    /// processors they require are added (what to download); without, a missing one throws (what to load).
    /// </summary>
    internal static Dictionary<string, string> SelectModels(string package, string? processors, bool addRequired, string paramName, string language = English)
    {
        if (!Packages.TryGetValue(language ?? "", out var packages))
            throw new ArgumentException($"Unknown language '{language}'. Available: {string.Join(", ", Packages.Keys)}", paramName);
        if (!packages.TryGetValue(package ?? "", out var models))
        {
            string why = package == "default_accurate" ? " (default_accurate needs transformer models, which StanzaSharp doesn't run)" : "";
            throw new ArgumentException($"Unknown package '{package}' for language '{language}'{why}. Available: {string.Join(", ", packages.Keys)}", paramName);
        }

        var set = processors == null ? models.Keys.ToHashSet() : ParseProcessors(processors, paramName);
        foreach (var p in set.ToList())
        {
            // What the package lacks isn't required (eslspok has no mwt); the identity lemmatizer reads no UPOS.
            var needs = p == "lemma" && models.GetValueOrDefault(p) == IdentityLemma ? ["tokenize"] : Requires[p];
            var missing = needs.Where(n => !set.Contains(n) && models.ContainsKey(n)).ToList();
            if (missing.Count > 0 && !addRequired)
                throw new ArgumentException($"Processor '{p}' requires {string.Join(", ", missing)}", paramName);
            set.UnionWith(missing);
        }

        foreach (var p in set)
        {
            if (!models.ContainsKey(p))
            {
                var others = packages.Where(kv => kv.Value.ContainsKey(p)).Select(kv => $"\"{kv.Key}\"");
                throw new ArgumentException($"Package '{package}' has no {p} model; use Package = {string.Join(" or ", others)} for {p}", paramName);
            }
        }

        return set.ToDictionary(p => p, p => models[p]);
    }

    /// <summary>
    /// The shared files <paramref name="models"/> of <paramref name="language"/> read (paths without extension), from
    /// <see cref="Dependencies"/>: pretrains first, then the forward and the backward charlm.
    /// </summary>
    internal static List<string> SharedModels(IReadOnlyDictionary<string, string> models, string language = English) =>
        models.SelectMany(m => Dependencies[language].GetValueOrDefault($"{m.Key}/{m.Value}", []))
            .Distinct()
            .OrderBy(p => p.StartsWith("pretrain/", StringComparison.Ordinal) ? 0 : p.StartsWith("forward_charlm/", StringComparison.Ordinal) ? 1 : 2)
            .ThenBy(p => p, StringComparer.Ordinal)
            .ToList();

    /// <summary>Every file <paramref name="models"/> read (paths without extension): their own, but none for the identity lemmatizer, then <see cref="SharedModels"/>.</summary>
    internal static IEnumerable<string> ModelFiles(IReadOnlyDictionary<string, string> models, string language) =>
        models.Where(m => m.Value != IdentityLemma).Select(m => $"{m.Key}/{m.Value}").Concat(SharedModels(models, language));

    /// <summary>A comma-separated processor list as a set, checking every name.</summary>
    internal static HashSet<string> ParseProcessors(string processors, string paramName)
    {
        var set = processors.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(p => p.ToLowerInvariant())
            .ToHashSet();
        foreach (var p in set)
        {
            if (!Requires.ContainsKey(p))
                throw new ArgumentException($"Unknown processor '{p}'. Available: {AllProcessors}", paramName);
        }

        if (set.Count == 0)
            throw new ArgumentException("No processors given", paramName);
        return set;
    }

    /// <summary>
    /// Runs the loaded processors on <paramref name="text"/>: it is split into sentences and tokens, then
    /// annotated by each processor in Stanza's order. Blank lines separate paragraphs, which never share a
    /// sentence (with <see cref="PipelineOptions.SplitSentences"/> false, each paragraph is one sentence).
    /// </summary>
    /// <remarks>
    /// Thread-safe: one pipeline can serve concurrent calls from many threads, sharing its models, and each call's
    /// output is the same as it would be alone. Throughput is bounded by the CPU, not by the number of callers: on
    /// <see cref="PipelineBackend.Managed"/> concurrent calls share one pool of <see cref="PipelineOptions.Threads"/>
    /// threads; on TorchSharp (<c>StanzaSharp.Cuda</c>) every call runs libtorch's operations on up to that many
    /// threads of its own, so the calls compete for the same cores.
    /// </remarks>
    public Document Process(string text) => Process(text, CancellationToken.None);

    /// <summary>
    /// <see cref="Process(string)"/> that can be canceled. The token is checked between processors and between the
    /// batches inside each processor, so a long text stops within about one batch's work.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was canceled. No document is returned, and the
    /// pipeline stays usable.</exception>
    public Document Process(string text, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Gated(() =>
        {
            var doc = Step("tokenize", () => _tokenizer.Process(text, _splitSentences, cancellationToken), cancellationToken);
            return Annotate(doc, mwt: true, cancellationToken);
        }, cancellationToken);
    }

    /// <summary>
    /// Runs the loaded processors on text that is already split into sentences and tokens, like Stanza's
    /// <c>tokenize_pretokenized=True</c> with a list of token lists: the tokenizer model is skipped and the tokens are
    /// kept exactly as given. No token is expanded into several words (Stanza's MWT stage only expands what the
    /// tokenizer model marks), so <c>"don't"</c> stays one word. <see cref="Document.Text"/> is all the tokens joined by
    /// single spaces, and the offsets point into it.
    /// </summary>
    /// <param name="sentences">One list of tokens per sentence.</param>
    /// <exception cref="ArgumentException">A sentence without tokens, or an empty or whitespace-only token
    /// (Stanza fails on both).</exception>
    /// <example>
    /// <code>
    /// var doc = nlp.Process(new[] { new[] { "Hello", "world", "." }, new[] { "Bye", "." } });
    /// // doc.Text == "Hello world . Bye ."
    /// </code>
    /// </example>
    public Document Process(IEnumerable<IEnumerable<string>> sentences) => Process(sentences, CancellationToken.None);

    /// <summary>
    /// <see cref="Process(IEnumerable{IEnumerable{string}})"/> that can be canceled, like
    /// <see cref="Process(string, CancellationToken)"/>.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was canceled; no document is returned.</exception>
    public Document Process(IEnumerable<IEnumerable<string>> sentences, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(sentences);
        var list = sentences.Select(s => (IReadOnlyList<string>)(s ?? throw new ArgumentNullException(nameof(sentences), "A sentence is null")).ToList()).ToList();
        foreach (var tokens in list)
        {
            if (tokens.Count == 0)
                throw new ArgumentException("A sentence has no tokens", nameof(sentences));
            if (tokens.Any(string.IsNullOrWhiteSpace))
                throw new ArgumentException("Tokens must not be null, empty or whitespace", nameof(sentences));
        }

        return Gated(() => Annotate(Tokenizer.Pretokenized(list), mwt: true, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// Runs the loaded processors on many texts at once, like Stanza's <c>Pipeline.bulk_process</c>: one
    /// <see cref="Document"/> per text, in order, each with offsets relative to its own text. Sentences from all texts
    /// are batched together, which is much faster than one <see cref="Process(string)"/> call per text when the texts
    /// are short.
    /// </summary>
    /// <param name="texts">The texts; an empty text gives an empty document.</param>
    /// <remarks>
    /// The output equals Stanza's <c>bulk_process</c> for the same texts. It can differ slightly from processing each
    /// text alone, as it does in Stanza: the sentiment classifier sees the padding of its batches, so a sentence's
    /// label can depend on the other sentences batched with it (on the golden validation texts, 172 of 854 labels
    /// change). The dependency parser's scores depend on its batches too, though no parse changed there. Everything
    /// else is the same as processing each text alone, except that sentence ids (<see cref="Sentence.SentId"/>)
    /// continue across the documents, as in Stanza.
    /// </remarks>
    /// <example>
    /// <code>
    /// List&lt;Document&gt; docs = nlp.Process(new[] { "First text.", "Second one." });
    /// </code>
    /// </example>
    public List<Document> Process(IEnumerable<string> texts) => Process(texts, CancellationToken.None);

    /// <summary><see cref="Process(IEnumerable{string})"/> that can be canceled, like <see cref="Process(string, CancellationToken)"/>.</summary>
    /// <exception cref="OperationCanceledException">The token was canceled; no documents are returned.</exception>
    public List<Document> Process(IEnumerable<string> texts, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var list = texts.ToList();
        if (list.Contains(null!))
            throw new ArgumentNullException(nameof(texts), "A text is null");
        if (list.Count == 0)
            return [];
        var docs = Gated(() =>
        {
            var docs = Step("tokenize", () => _tokenizer.Process(list, _splitSentences, cancellationToken), cancellationToken);

            // The other processors run on all sentences as one document (UDProcessor.bulk_process).
            var combined = new Document();
            combined.Sentences.AddRange(docs.SelectMany(d => d.Sentences));
            Annotate(combined, mwt: true, cancellationToken);
            return docs;
        }, cancellationToken);

        // NERProcessor.bulk_process: entities again, with each document's own text.
        if (_ner != null)
        {
            foreach (var doc in docs)
            {
                foreach (var sentence in doc.Sentences)
                    Entity.Build(sentence, doc.Text);
            }
        }

        return docs;
    }

    /// <summary>
    /// Runs the loaded processors on a document that already has its sentences, tokens and words, such as one from
    /// <see cref="Conllu.Read"/> or <see cref="Json.Read"/>, like Stanza's <c>nlp.process(doc, processors=...)</c>
    /// without tokenize and mwt (<c>Pipeline.process_conllu</c>): tokenize and mwt are skipped, and every other
    /// processor of the pipeline runs in Stanza's order and overwrites what it sets (UPOS, XPOS and feats; lemmas;
    /// the tree; heads and relations; sentiment; NER tags and entities). Everything else is kept, including
    /// annotations of processors the pipeline lacks. The document is changed in place and returned.
    /// </summary>
    /// <param name="document">The document to annotate. Its offsets and texts are kept as they are; none are needed.</param>
    /// <example>
    /// <code>
    /// var doc = nlp.Process(Conllu.Read(File.ReadAllText("tokens.conllu")));
    /// </code>
    /// </example>
    public Document Process(Document document) => Process(document, CancellationToken.None);

    /// <summary><see cref="Process(Document)"/> that can be canceled, like <see cref="Process(string, CancellationToken)"/>.</summary>
    /// <exception cref="OperationCanceledException">The token was canceled. The document may be partly annotated.</exception>
    public Document Process(Document document, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(document);
        return Gated(() => Annotate(document, mwt: false, cancellationToken), cancellationToken);
    }

    /// <summary>
    /// Runs <paramref name="run"/>, waiting first while <see cref="PipelineOptions.MaxConcurrentCalls"/> other calls are
    /// running (no wait without a limit). The wait honours <paramref name="ct"/>.
    /// </summary>
    private T Gated<T>(Func<T> run, CancellationToken ct)
    {
        if (_gate == null)
            return run();
        _gate.Wait(ct);
        try
        {
            return run();
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Everything after the tokenizer, in Stanza's order. All state is per call (the document and the charlm cache);
    /// the models are only read, which is what makes Process thread-safe.
    /// </summary>
    private Document Annotate(Document doc, bool mwt, CancellationToken ct)
    {
        try
        {
            return AnnotateWithModels(doc, mwt, ct);
        }
        finally
        {
            // Linux/glibc: give the call's freed tensor memory back to the OS, or it stays in the RSS (NativeHeap).
            // Smaller calls free little (about 15 MB at 700 words) and the next call reuses it; trimming costs 2–12 ms.
            if (_trimNativeHeap && doc.Sentences.Sum(s => s.Words.Count()) >= NativeHeap.TrimMinWords)
            {
                long start = Stopwatch.GetTimestamp();
                NativeHeap.Trim();
                _logger?.LogDebug("malloc_trim took {Milliseconds:F1} ms", Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            }
        }
    }

    private Document AnnotateWithModels(Document doc, bool mwt, CancellationToken ct)
    {
        if (mwt && _mwt != null)
            Step("mwt", () => _mwt.Process(doc), ct); // one batch, and only for the words the dictionary lacks

        // NER, the parser and the sentiment classifier reuse the tagger's charlm outputs instead of computing them again.
        // A _nocharlm tagger (default_fast) has no charlm outputs to share.
        using var charlms = _pos is { UsesCharlm: true } && (_ner != null || _parser != null || _sentiment != null || (_depparse != null && _depparseReadsCache))
            && _cacheOptions.IsEnabled ? new CharlmCache(_cacheOptions.MaxWords) : null;
        if (_pos != null)
            Step("pos", () => _pos.Process(doc, charlms, ct), ct);
        if (_lemma != null)
            Step("lemma", () => _lemma.Process(doc, ct), ct);
        else if (_identityLemma)
            Step("lemma", () => Lemmatizer.Identity(doc), ct);

        // Stanza's order (PIPELINE_NAMES). Only the CoNLL-U comment order shows it, and Conllu.Write fixes that.
        if (_parser != null)
            Step("constituency", () => _parser.Process(doc, charlms, ct), ct);

        // Depparse runs the charlms with a ROOT word in front, so of the tagger's outputs only the backward ones apply, with
        // ROOT run on from the tagger's final state (managed; DependencyParser.Process).
        if (_depparse != null)
            Step("depparse", () => _depparse.Process(doc, charlms, ct), ct);
        if (_sentiment != null)
            Step("sentiment", () => _sentiment.Process(doc, charlms, ct), ct);
        if (_ner != null)
            Step("ner", () => _ner.Process(doc, charlms, ct), ct);
        return doc;
    }

    /// <summary>Runs one processor after checking for cancellation, timing it when debug logging is on.</summary>
    private T Step<T>(string processor, Func<T> run, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (_logger?.IsEnabled(LogLevel.Debug) != true)
            return run();
        long start = Stopwatch.GetTimestamp();
        var result = run();
        _logger.LogDebug("{Processor} took {Milliseconds:F1} ms", processor, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return result;
    }

    private void Step(string processor, Action run, CancellationToken ct) => Step(processor, () => { run(); return 0; }, ct);

    /// <summary>Frees the models' memory (CPU or GPU).</summary>
    public void Dispose()
    {
        _parser?.Dispose();
        _sentiment?.Dispose();
        _depparse?.Dispose();
        _ner?.Dispose();
        _pos?.Dispose();
        _lemma?.Dispose();
        _mwt?.Dispose();
        _tokenizer.Dispose();
        _models.Dispose();
        foreach (var pretrain in _pretrains)
            pretrain.Dispose();
        _gate?.Dispose();
    }
}
