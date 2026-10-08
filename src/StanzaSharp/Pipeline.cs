using System.Diagnostics;
using Microsoft.Extensions.Logging;
using StanzaSharp.Constituency;
using StanzaSharp.Depparse;
using StanzaSharp.Lemma;
using StanzaSharp.Mwt;
using StanzaSharp.Ner;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using StanzaSharp.Pos;
using StanzaSharp.Sentiment;
using StanzaSharp.Tokenize;
using TorchSharp;

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

    /// <summary>The package <see cref="PipelineOptions.Package"/> selects by default: Stanza's English <c>default</c>.</summary>
    public const string DefaultPackage = "default";

    // Stanza 1.15.0's English packages (resources.json): each processor's model. Their files need MD5s in ModelDownloader.Files.
    internal static readonly Dictionary<string, Dictionary<string, string>> Packages = new()
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
    };

    // Processors with a managed network (issue #29, docs/backends.md); the others run on TorchSharp on either backend.
    internal static readonly HashSet<string> ManagedProcessors = ["tokenize", "mwt", "pos", "ner"];

    // Processors whose models (every package's) read the shared pretrained word vectors.
    private static readonly HashSet<string> UsesPretrain = ["pos", "depparse", "ner", "constituency", "sentiment"];

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
    private readonly SentimentClassifier? _sentiment;
    private readonly Pretrain? _pretrain;
    private readonly CharLanguageModel? _charlmForward, _charlmBackward;
    private readonly ManagedCharLanguageModel? _managedCharlmForward, _managedCharlmBackward;
    private readonly CharlmCacheOptions _cacheOptions;
    private readonly TorchSharp.torch.Device? _device;
    private readonly bool _splitSentences, _trimNativeHeap;
    private readonly ILogger? _logger;

    private Pipeline(string modelDir, Dictionary<string, string> models, PipelineOptions options)
    {
        _cacheOptions = options.CharlmCache;
        _device = options.Device;
        bool Managed(string processor) => options.Backend == Backend.Managed && ManagedProcessors.Contains(processor);
        _splitSentences = options.SplitSentences;
        _trimNativeHeap = options.TrimNativeHeap && NativeHeap.CanTrim;
        _logger = options.Logger;
        string Model(string processor) => Path.Combine(modelDir, processor, models[processor]);
        string Name(string processor) => $"{processor}/{models[processor]}";
        T Timed<T>(string name, Func<T> load)
        {
            if (_logger?.IsEnabled(LogLevel.Information) != true)
                return load();
            long start = Stopwatch.GetTimestamp();
            var model = load();
            _logger.LogInformation("Loaded {Model} in {Milliseconds:F0} ms", name, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            return model;
        }

        var shared = SharedModels(models);
        if (shared.Contains(PretrainPath))
            _pretrain = Timed(PretrainPath, () => Pretrain.Load(Path.Combine(modelDir, PretrainPath)));
        if (shared.Contains(ForwardCharlmPath))
        {
            // Each backend's charlms, if a processor on it reads them.
            var users = models.Where(kv => kv.Value.EndsWith("_charlm", StringComparison.Ordinal)).Select(kv => kv.Key).ToList();
            if (users.Any(p => !Managed(p)))
            {
                _charlmForward = Timed(ForwardCharlmPath, () => CharLanguageModel.Load(Path.Combine(modelDir, ForwardCharlmPath)));
                _charlmBackward = Timed(BackwardCharlmPath, () => CharLanguageModel.Load(Path.Combine(modelDir, BackwardCharlmPath)));
            }
            if (users.Any(Managed))
            {
                _managedCharlmForward = Timed(ForwardCharlmPath + " (managed)", () => ManagedCharLanguageModel.Load(Path.Combine(modelDir, ForwardCharlmPath)));
                _managedCharlmBackward = Timed(BackwardCharlmPath + " (managed)", () => ManagedCharLanguageModel.Load(Path.Combine(modelDir, BackwardCharlmPath)));
            }
        }

        _tokenizer = Timed(Name("tokenize"), () => Tokenizer.Load(Model("tokenize"), backend: options.Backend));
        if (models.ContainsKey("mwt"))
            _mwt = Timed(Name("mwt"), () => MwtExpander.Load(Model("mwt"), backend: options.Backend));
        if (models.ContainsKey("pos"))
            _pos = Timed(Name("pos"), () => Managed("pos")
                ? PosTagger.LoadManaged(Model("pos"), _pretrain!, _managedCharlmForward, _managedCharlmBackward)
                : PosTagger.Load(Model("pos"), _pretrain!, _charlmForward, _charlmBackward));
        if (models.ContainsKey("lemma"))
            _lemma = Timed(Name("lemma"), () => Lemmatizer.Load(Model("lemma")));
        if (models.ContainsKey("depparse"))
            _depparse = Timed(Name("depparse"), () => DependencyParser.Load(Model("depparse"), _pretrain!, _charlmForward, _charlmBackward));
        if (models.ContainsKey("ner"))
            _ner = Timed(Name("ner"), () => Managed("ner")
                ? NerTagger.LoadManaged(Model("ner"), _pretrain!, _managedCharlmForward, _managedCharlmBackward)
                : NerTagger.Load(Model("ner"), _pretrain!, _charlmForward, _charlmBackward));
        if (models.ContainsKey("constituency"))
            _parser = Timed(Name("constituency"), () => ConstituencyParser.Load(Model("constituency"), _pretrain!, _charlmForward!, _charlmBackward!));
        if (models.ContainsKey("sentiment"))
            _sentiment = Timed(Name("sentiment"), () => SentimentClassifier.Load(Model("sentiment"), _pretrain!, _charlmForward!, _charlmBackward!));
    }

    /// <summary>
    /// Loads the English models from <paramref name="modelDir"/>: either converted ones (e.g. <c>models/converted/en</c>)
    /// or Stanza's own download with its <c>.pt</c> files (e.g. <c>models/stanza/en</c>), chosen per file.
    /// </summary>
    /// <param name="options">Package, processors, device, threads and cache settings; defaults to all eight processors
    /// of the <c>default</c> package on the CPU.</param>
    /// <exception cref="ArgumentException">An unknown package or processor, a processor the package lacks (e.g.
    /// constituency in <c>default_fast</c>), or a processor without the ones it requires.</exception>
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
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir} (download them with ModelDownloader.DownloadAsync or \"StanzaSharp.Cli download\")");
        var models = SelectModels(options.Package, options.Processors, addRequired: false, nameof(options));
        if (options.CharlmCache.IsEnabled && options.CharlmCache.MaxWords <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "CharlmCache.MaxWords must be positive; set IsEnabled = false to turn the cache off");
        if (options.Threads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Threads must be at least 1, or null for at most Environment.ProcessorCount");

        var logger = options.Logger;
        long start = Stopwatch.GetTimestamp();
        if (options.VerifyChecksums)
        {
            ModelDownloader.Verify(modelDir, models);
            logger?.LogInformation("Verified the model checksums in {Milliseconds:F0} ms", Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        }
        // libtorch defaults to the host's physical cores; ProcessorCount respects a container's CPU quota.
        int threads = options.Threads ?? Math.Min(torch.get_num_threads(), Environment.ProcessorCount);
        if (threads != torch.get_num_threads())
            torch.set_num_threads(threads);
        logger?.LogInformation("Using {Threads} torch intra-op threads (Environment.ProcessorCount is {ProcessorCount})", threads, Environment.ProcessorCount);
        if (options.Backend == Backend.Managed)
        {
            // The same semantics for the managed pool: null caps its current count (ProcessorCount unless set lower).
            ManagedThreads.Count = options.Threads ?? Math.Min(ManagedThreads.Count, Environment.ProcessorCount);
            logger?.LogInformation("Using {Threads} managed threads", ManagedThreads.Count);
        }

        if (options.DisableTf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = false;
        var pipeline = Weights.On(options.Device, () => new Pipeline(modelDir, models, options));
        logger?.LogInformation("Loaded the {Package} pipeline ({Processors}) in {Milliseconds:F0} ms", options.Package,
            string.Join(",", AllProcessors.Split(',').Where(models.ContainsKey)), Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return pipeline;
    }

    internal const string PretrainPath = "pretrain/conll17", ForwardCharlmPath = "forward_charlm/1billion", BackwardCharlmPath = "backward_charlm/1billion";

    /// <summary>
    /// The processors to run and each one's model: <paramref name="processors"/> (null: all of the package's), checked
    /// against <paramref name="package"/>. With <paramref name="addRequired"/> the processors they require are added
    /// (what to download); without, a missing one throws (what to load).
    /// </summary>
    internal static Dictionary<string, string> SelectModels(string package, string? processors, bool addRequired, string paramName)
    {
        if (!Packages.TryGetValue(package ?? "", out var models))
            throw new ArgumentException($"Unknown package '{package}'. Available: {string.Join(", ", Packages.Keys)}", paramName);
        var set = processors == null ? models.Keys.ToHashSet() : ParseProcessors(processors, paramName);
        foreach (var p in set.ToList())
        {
            var missing = Requires[p].Where(n => !set.Contains(n)).ToList();
            if (missing.Count > 0 && !addRequired)
                throw new ArgumentException($"Processor '{p}' requires {string.Join(", ", missing)}", paramName);
            set.UnionWith(missing);
        }
        foreach (var p in set)
            if (!models.ContainsKey(p))
            {
                var others = Packages.Where(kv => kv.Value.ContainsKey(p)).Select(kv => $"\"{kv.Key}\"");
                throw new ArgumentException($"Package '{package}' has no {p} model; use Package = {string.Join(" or ", others)} for {p}", paramName);
            }
        return set.ToDictionary(p => p, p => models[p]);
    }

    /// <summary>
    /// The shared files <paramref name="models"/> read (paths without extension): the pretrained word vectors for pos,
    /// depparse, ner, constituency and sentiment, and the charlms for every <c>_charlm</c> model (Stanza's naming).
    /// </summary>
    internal static List<string> SharedModels(IReadOnlyDictionary<string, string> models)
    {
        var shared = new List<string>();
        if (models.Keys.Any(UsesPretrain.Contains))
            shared.Add(PretrainPath);
        if (models.Values.Any(m => m.EndsWith("_charlm", StringComparison.Ordinal)))
            shared.AddRange([ForwardCharlmPath, BackwardCharlmPath]);
        return shared;
    }

    /// <summary>A comma-separated processor list as a set, checking every name.</summary>
    internal static HashSet<string> ParseProcessors(string processors, string paramName)
    {
        var set = processors.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(p => p.ToLowerInvariant()).ToHashSet();
        foreach (var p in set)
            if (!Requires.ContainsKey(p))
                throw new ArgumentException($"Unknown processor '{p}'. Available: {AllProcessors}", paramName);
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
    /// output is the same as it would be alone. Every call runs libtorch's operations on up to
    /// <see cref="PipelineOptions.Threads"/> threads of its own, so concurrent calls compete for the same cores:
    /// throughput is bounded by the CPU, not by the number of callers.
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
        var doc = Step("tokenize", () => _tokenizer.Process(text, _splitSentences, cancellationToken), cancellationToken);
        return Annotate(doc, cancellationToken);
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
        return Annotate(Tokenizer.Pretokenized(list), cancellationToken);
    }

    /// <summary>
    /// Runs the loaded processors on many texts at once, like Stanza's <c>Pipeline.bulk_process</c>: one
    /// <see cref="Document"/> per text, in order, each with offsets relative to its own text. Sentences from all texts
    /// are batched together, which is much faster than one <see cref="Process(string)"/> call per text when the texts
    /// are short.
    /// </summary>
    /// <remarks>
    /// The output equals Stanza's <c>bulk_process</c> for the same texts. It can differ slightly from processing each
    /// text alone, as it does in Stanza: the sentiment classifier sees the padding of its batches, so a sentence's
    /// label can depend on the other sentences batched with it (on the golden validation texts, 172 of 854 labels
    /// change). The dependency parser's scores depend on its batches too, though no parse changed there. Everything
    /// else is the same as processing each text alone, except that sentence ids (<see cref="Sentence.SentId"/>)
    /// continue across the documents, as in Stanza.
    /// </remarks>
    /// <param name="texts">The texts; an empty text gives an empty document.</param>
    /// <example>
    /// <code>
    /// List&lt;Document&gt; docs = nlp.Process(new[] { "First text.", "Second one." });
    /// </code>
    /// </example>
    public List<Document> Process(IEnumerable<string> texts) => Process(texts, CancellationToken.None);

    /// <summary>
    /// <see cref="Process(IEnumerable{string})"/> that can be canceled, like <see cref="Process(string, CancellationToken)"/>.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was canceled; no documents are returned.</exception>
    public List<Document> Process(IEnumerable<string> texts, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var list = texts.ToList();
        if (list.Contains(null!))
            throw new ArgumentNullException(nameof(texts), "A text is null");
        if (list.Count == 0)
            return [];
        var docs = Step("tokenize", () => _tokenizer.Process(list, _splitSentences, cancellationToken), cancellationToken);
        // The other processors run on all sentences as one document (UDProcessor.bulk_process).
        var combined = new Document();
        combined.Sentences.AddRange(docs.SelectMany(d => d.Sentences));
        Annotate(combined, cancellationToken);
        // NERProcessor.bulk_process: entities again, with each document's own text.
        if (_ner != null)
            foreach (var doc in docs)
                foreach (var sentence in doc.Sentences)
                    Entity.Build(sentence, doc.Text);
        return docs;
    }

    /// <summary>
    /// Everything after the tokenizer, in Stanza's order. All state is per call (the document and the charlm cache);
    /// the models are only read, which is what makes Process thread-safe.
    /// </summary>
    private Document Annotate(Document doc, CancellationToken ct)
    {
        try
        {
            return AnnotateWithModels(doc, ct);
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

    private Document AnnotateWithModels(Document doc, CancellationToken ct)
    {
        if (_mwt != null)
            Step("mwt", () => _mwt.Process(doc), ct); // one batch, and only for the words the dictionary lacks
        // NER, the parser and the sentiment classifier reuse the tagger's charlm outputs instead of computing them again.
        // A _nocharlm tagger (default_fast) has no charlm outputs to share.
        using var charlms = _pos is { UsesCharlm: true } && (_ner != null || _parser != null || _sentiment != null) && _cacheOptions.IsEnabled ? new CharlmCache(_cacheOptions.MaxWords, _device) : null;
        if (_pos != null)
            Step("pos", () => _pos.Process(doc, charlms, ct), ct);
        if (_lemma != null)
            Step("lemma", () => _lemma.Process(doc, ct), ct);
        // Stanza's order (PIPELINE_NAMES). Only the CoNLL-U comment order shows it, and Conllu.Write fixes that.
        if (_parser != null)
            Step("constituency", () => _parser.Process(doc, charlms, ct), ct);
        // No CharlmCache: depparse runs the charlms with a ROOT word in front, so the tagger's outputs don't apply.
        if (_depparse != null)
            Step("depparse", () => _depparse.Process(doc, ct), ct);
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
        _charlmForward?.Dispose();
        _charlmBackward?.Dispose();
        _pretrain?.Dispose();
    }
}
