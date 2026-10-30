using System.Diagnostics;
using System.Globalization;
using System.Runtime.ExceptionServices;
using Microsoft.Extensions.Logging;
using StanzaSharp.Constituency;
using StanzaSharp.Coref;
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
/// Runs Stanza's English pipeline in its order: tokenize → mwt → pos → lemma → constituency → coref → depparse → sentiment →
/// ner (by default all those of the selected package; coref only when named).
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

    // Every processor in Stanza's order, coref (optional, never in a package's default list) included.
    internal const string StanzaOrder = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos},{Processor.Lemma},{Processor.Constituency},{Processor.Coref},{Processor.Depparse},{Processor.Sentiment},{Processor.Ner}";

    // The multilingual coref model, English's and German's (docs/coref-plan.md).
    internal const string CorefModel = "udcoref_xlm-roberta-lora";

    /// <summary>The package <see cref="PipelineOptions.Package"/> selects by default: Stanza's English <see langword="default"/>.</summary>
    public const string DefaultPackage = "default";

    internal const string English = "en";

    // The packages StanzaSharp runs per language, by Stanza's names (ported and tested; any other package throws). Their
    // models, and each model's dependencies and MD5, come from Stanza's resources.json (StanzaResources). English:
    // default_fast has pos, depparse and ner with their own character LSTMs and no constituency; default_accurate the
    // _charlm tokenizer and lemmatizer and google/electra-large-discriminator (Transformers); the treebank packages
    // tokenize to depparse only (combined has default's models, eslspok no mwt and the identity lemmatizer, the biomedical
    // craft, genia and mimic no mwt and their own word vectors). German: default_fast keeps ner and sentiment's charlm
    // models; default_accurate is on german-nlp-group/electra-base-german-uncased.
    internal static readonly Dictionary<string, string[]> SupportedPackages = new()
    {
        [English] =
        [
            DefaultPackage, "default_fast", "default_accurate",
            "combined", "ewt", "gum", "lines", "partut", "atis", "childes", "eslspok", "craft", "genia", "mimic",
        ],
        ["de"] = [DefaultPackage, "default_fast", "default_accurate", "combined", "gsd", "hdt"],
    };

    // Each supported package's models (processor → model), from resources.json.
    internal static readonly Dictionary<string, Dictionary<string, Dictionary<string, string>>> Packages =
        SupportedPackages.ToDictionary(l => l.Key, l => l.Value.ToDictionary(p => p, p => StanzaResources.Package(l.Key, p)));

    // Stanza's optional processors (resources.json's packages' "optional"): a package's model for them, run only when
    // Processors or Models names the processor (owner's decision, 2026-10-20: coref is opt-in).
    internal static readonly Dictionary<string, Dictionary<string, Dictionary<string, string>>> Optional =
        SupportedPackages.ToDictionary(l => l.Key, l => l.Value
            .Select(p => (Package: p, Models: StanzaResources.Optional(l.Key, p)))
            .Where(p => p.Models.Count > 0)
            .ToDictionary(p => p.Package, p => p.Models));

    // Models beyond the packages' that PipelineOptions.Models can choose, per language and processor (docs/per-processor-models-plan.md).
    // Each must be in resources.json, which has its MD5 and its shared files.
    internal static readonly Dictionary<string, Dictionary<string, string[]>> Alternatives = new()
    {
        [English] = new()
        {
            ["constituency"] = ["ptb3_charlm", "ptb3-revised_electra-large", "ptb3_electra-large"],
            ["sentiment"] = ["sstplus_electra-large"],
            ["coref"] = [CorefModel],

            // the treebanks' _charlm tokenizers and lemmatizers (per-character charlm states); lemma combined_charlm also has
            // contextual lemma classifiers ('s, her), which read the pretrain too
            ["tokenize"] =
            [
                "combined_charlm", "ewt_charlm", "gum_charlm", "lines_charlm", "partut_charlm", "atis_charlm", "childes_charlm", "eslspok_charlm",
            ],
            ["lemma"] = ["combined_charlm", "ewt_charlm", "gum_charlm", "lines_charlm", "partut_charlm", "atis_charlm", "childes_charlm"],

            // the treebanks' own _nocharlm taggers and parsers (mimic has only a parser); combined's are default_fast's
            // and default_accurate's frozen-transformer tagger, parser and NER model (Transformers), with the treebanks' own
            ["pos"] =
            [
                "ewt_nocharlm", "gum_nocharlm", "lines_nocharlm", "partut_nocharlm", "atis_nocharlm", "childes_nocharlm", "eslspok_nocharlm",
                "combined_electra-large", .. TreebankElectra,
            ],
            ["depparse"] =
            [
                "ewt_nocharlm", "gum_nocharlm", "lines_nocharlm", "partut_nocharlm", "atis_nocharlm", "childes_nocharlm", "eslspok_nocharlm",
                "mimic_nocharlm", "combined_electra-large", .. TreebankElectra,
            ],
            ["ner"] =
            [
                "ontonotes_charlm", "ontonotes_nocharlm", "combined_charlm", "conll03_charlm", "ontonotes-ww-multi_electra-large",
                "ontonotes_electra-large", "conll03_electra-large",

                // clinical (mimic's vectors and charlms) and biomedical (biomed's vectors, pubmed's charlms)
                "i2b2", "radiology", "anatem", "bc4chemd", "bc5cdr", "bionlp13cg", "jnlpba", "linnaeus", "ncbi_disease", "s800",
            ],
        },
        ["de"] = new()
        {
            ["tokenize"] = ["combined_charlm", "gsd_charlm", "hdt_charlm"],
            ["lemma"] = ["combined_charlm", "gsd_charlm", "hdt_charlm"],
            ["pos"] = ["gsd_nocharlm", "hdt_nocharlm", "gsd_german-nlp-electra", "hdt_german-nlp-electra"],
            ["depparse"] = ["gsd_nocharlm", "hdt_nocharlm", "gsd_german-nlp-electra", "hdt_german-nlp-electra"],
            ["ner"] = ["conll03"],
            ["sentiment"] = ["sb10k_dbmdz-bert-german-cased"],
            ["coref"] = [CorefModel],
        },
    };

    // The treebanks' frozen-transformer taggers and parsers (#168): combined_electra-large's architecture on each treebank's
    // vocabularies and heads (legacy tag heads; lines and partut composite XPOS), as their _charlm twins.
    private static string[] TreebankElectra =>
        ["ewt_electra-large", "gum_electra-large", "lines_electra-large", "partut_electra-large", "atis_electra-large", "childes_electra-large", "eslspok_electra-large"];

    /// <summary>Every model <paramref name="processor"/> of <paramref name="language"/> can run: its packages' and <see cref="Alternatives"/>.</summary>
    internal static IEnumerable<string> ModelsOf(string language, string processor) =>
        Packages[language].Values.Select(p => p.GetValueOrDefault(processor)).OfType<string>()
            .Concat(Alternatives[language].GetValueOrDefault(processor, []))
            .Distinct();

    /// <summary>Stanza's lemma "model" that sets each word's lemma to its text (LemmaProcessor's use_identity); it has no file.</summary>
    internal const string IdentityLemma = "identity";

    // The shared files (pretrains, charlms) each supported model reads, by "<processor>/<model>": resources.json's
    // "dependencies", which the model's name doesn't tell (German ner conll03 reads charlms and its own pretrain). mwt, the
    // _nocharlm tokenizers and lemmatizers read none; the _charlm tokenizers read only the forward charlm.
    internal static readonly Dictionary<string, Dictionary<string, string[]>> Dependencies =
        SupportedPackages.Keys.ToDictionary(l => l, l => SupportedModels(l).ToDictionary(m => m, m => StanzaResources.Dependencies(l, m)));

    /// <summary>
    /// Every model of <paramref name="language"/> StanzaSharp runs, as <c>processor/model</c>: its supported packages'
    /// (optional ones too) and <see cref="Alternatives"/>, but the identity lemmatizer, which has no file.
    /// </summary>
    internal static IEnumerable<string> SupportedModels(string language) =>
        Packages[language].Values.Concat(Optional[language].Values).SelectMany(p => p)
            .Concat(Alternatives[language].SelectMany(a => a.Value.Select(m => KeyValuePair.Create(a.Key, m))))
            .Where(m => m.Value != IdentityLemma)
            .Select(m => $"{m.Key}/{m.Value}")
            .Distinct();

    // Dependencies resources.json lists but the model never reads (owner's decision, 2026-10-18, #173): downloaded like
    // Stanza (FilesFor), but Load neither needs nor loads them.
    internal static readonly Dictionary<string, Dictionary<string, string[]>> Unread = new()
    {
        [English] = new()
        {
            ["sentiment/sstplus_electra-large"] = ["forward_charlm/1billion", "backward_charlm/1billion"],
        },
        ["de"] = new()
        {
            ["sentiment/sb10k_dbmdz-bert-german-cased"] = ["forward_charlm/newswiki", "backward_charlm/newswiki"],
        },
    };

    /// <summary>The shared files the model <paramref name="name"/> (<c>processor/model</c>) reads: its <see cref="Dependencies"/> minus <see cref="Unread"/>.</summary>
    internal static string[] ReadDependencies(string name, string language) =>
        [.. Dependencies[language].GetValueOrDefault(name, []).Except(Unread.GetValueOrDefault(language)?.GetValueOrDefault(name) ?? [])];

    // The transformer each transformer model reads, by "<processor>/<model>": the model config's bert_model, a Hugging Face
    // repository (ModelDownloader.Transformers has its files), not a Stanza resource, so it is kept out of Dependencies,
    // ModelDownloader.Files and CI's model caches (docs/transformer-plan.md). The pipeline loads each one once.
    internal static readonly Dictionary<string, Dictionary<string, string>> Transformers = new()
    {
        [English] = new()
        {
            ["pos/combined_electra-large"] = Electra,
            ["depparse/combined_electra-large"] = Electra,
            ["ner/ontonotes-ww-multi_electra-large"] = Electra,
            ["constituency/ptb3-revised_electra-large"] = Electra,
            ["constituency/ptb3_electra-large"] = Electra, // TOP_DOWN, with its LoRA adapters: a pass of its own
            ["sentiment/sstplus_electra-large"] = Electra, // with its LoRA adapters: a pass of its own
            ["pos/ewt_electra-large"] = Electra,
            ["depparse/ewt_electra-large"] = Electra,
            ["pos/gum_electra-large"] = Electra,
            ["depparse/gum_electra-large"] = Electra,
            ["pos/lines_electra-large"] = Electra,
            ["depparse/lines_electra-large"] = Electra,
            ["pos/partut_electra-large"] = Electra,
            ["depparse/partut_electra-large"] = Electra,
            ["pos/atis_electra-large"] = Electra,
            ["depparse/atis_electra-large"] = Electra,
            ["pos/childes_electra-large"] = Electra,
            ["depparse/childes_electra-large"] = Electra,
            ["pos/eslspok_electra-large"] = Electra,
            ["depparse/eslspok_electra-large"] = Electra,
            ["ner/ontonotes_electra-large"] = Electra,
            ["ner/conll03_electra-large"] = Electra,
            ["coref/" + CorefModel] = XlmRoberta, // with its LoRA adapters
        },
        ["de"] = new()
        {
            ["pos/combined_german-nlp-electra"] = GermanElectra,
            ["depparse/combined_german-nlp-electra"] = GermanElectra,
            ["constituency/spmrl_german-nlp-electra"] = GermanElectra, // LoRA adapters: a pass of its own
            ["sentiment/sb10k_german-nlp-electra"] = GermanElectra, // LoRA adapters: a pass of its own
            ["pos/gsd_german-nlp-electra"] = GermanElectra,
            ["depparse/gsd_german-nlp-electra"] = GermanElectra,
            ["pos/hdt_german-nlp-electra"] = GermanElectra,
            ["depparse/hdt_german-nlp-electra"] = GermanElectra,
            ["ner/germeval2014_german-nlp-electra"] = GermanElectra,
            ["sentiment/sb10k_dbmdz-bert-german-cased"] = "dbmdz/bert-base-german-cased", // LoRA adapters
            ["coref/" + CorefModel] = XlmRoberta, // with its LoRA adapters
        },
    };

    private const string Electra = "google/electra-large-discriminator", GermanElectra = "german-nlp-group/electra-base-german-uncased",
        XlmRoberta = "FacebookAI/xlm-roberta-large";

    /// <summary>The transformers <paramref name="models"/> of <paramref name="language"/> read (<see cref="Transformers"/>), each once.</summary>
    internal static List<string> TransformersOf(IReadOnlyDictionary<string, string> models, string language) =>
        [.. models.Select(m => Transformers[language].GetValueOrDefault($"{m.Key}/{m.Value}")).OfType<string>().Distinct()];

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
        ["coref"] = ["tokenize"], // the words' text (and UPOS, when the tagger ran, to choose a representative mention)
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
    private readonly CorefResolver? _coref;
    private readonly TransformerEmbedder[] _transformers = []; // every transformer the models read (Transformers), once each
    private readonly Pretrain[] _pretrains = [];
    private readonly BackendModels _models;
    private readonly CharlmCacheOptions _cacheOptions;
    private readonly bool _splitSentences, _trimNativeHeap, _depparseReadsCache, _identityLemma;
    private readonly Func<IReadOnlyList<IReadOnlyList<PostprocessorToken>>, IEnumerable<IEnumerable<PostprocessorToken>>>? _postprocessor;

    // Each processor's forward charlm (Dependencies), or none: a call's CharlmCache serves the tagger's pair only.
    private readonly Dictionary<string, string?> _charlmOf = [];
    private readonly ILogger? _logger;
    private readonly SemaphoreSlim? _gate;

    private Pipeline(string modelDir, Dictionary<string, string> models, PipelineOptions options, BackendModels backend)
    {
        _models = backend;
        _cacheOptions = options.CharlmCache;
        _splitSentences = options.SplitSentences;
        _postprocessor = options.TokenizePostprocessor;
        _depparseReadsCache = backend.DepparseReadsCache;
        _trimNativeHeap = options.TrimNativeHeap && NativeHeap.CanTrim;
        _logger = options.Logger;
        _gate = options.MaxConcurrentCalls is int max ? new SemaphoreSlim(max, max) : null;
        var custom = CustomPaths(options.ModelPaths, nameof(options));
        string Model(string processor) => custom.TryGetValue(processor, out var path) ? path.Model : Path.Combine(modelDir, processor, models[processor]);
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
            // A model of the caller's own that doesn't fit what it was given (a _charlm model without charlms, ...) names its ModelPath.
            int slash = name.IndexOf('/');
            if (slash > 0 && custom.TryGetValue(name[..slash], out var path))
            {
                var inner = load;
                load = () =>
                {
                    try
                    {
                        return inner();
                    }
                    catch (ArgumentException e)
                    {
                        throw new ArgumentException($"ModelPaths[\"{name[..slash]}\"] ({path.Model}): {e.Message}. Name the shared files it reads in its ModelPath "
                            + "(Pretrain, ForwardCharlm, BackwardCharlm); it must be a model of an architecture StanzaSharp runs.", nameof(options), e);
                    }
                };
            }

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

        // The shared files each processor's model reads, as (kind, key, file): its Dependencies, each replaced by the file its
        // ModelPath names, if any. The key is the dependency's name (pretrain/conll17) or a custom file's full path, so a
        // custom charlm pair is loaded apart from the package's and gets no CharlmCache it would share with another pair.
        List<(string Kind, string Key, string File)> DependenciesOf(string processor)
        {
            var path = custom.GetValueOrDefault(processor);
            var list = ReadDependencies(Name(processor), options.Language).Where(d => path == null || Override(path, d) == null)
                .Select(d => (d[..d.IndexOf('/')], d, Path.Combine(modelDir, d)))
                .ToList();
            if (path != null)
            {
                foreach (var (kind, file) in new[] { ("pretrain", path.Pretrain), ("forward_charlm", path.ForwardCharlm), ("backward_charlm", path.BackwardCharlm) })
                {
                    if (file != null)
                        list.Add((kind, file, file));
                }
            }

            return list;
        }

        var dependencies = models.Keys.ToDictionary(p => p, DependenciesOf);

        // Every pretrain the models read (German NER has its own), and every charlm (each pair once: default's 1billion,
        // and mimic's for ner i2b2).
        var shared = dependencies.Values.SelectMany(d => d).DistinctBy(d => d.Key)
            .OrderBy(d => d.Kind switch { "pretrain" => 0, "forward_charlm" => 1, _ => 2 })
            .ThenBy(d => d.Key, StringComparer.Ordinal)
            .ToList();
        var pretrains = shared.Where(d => d.Kind == "pretrain").ToDictionary(d => d.Key, d => Start(d.Key, () => backend.Pretrain(d.File)));
        var charlms = shared.Where(d => d.Kind != "pretrain")
            .Select(d => Start(d.Key, () =>
            {
                backend.Charlm(d.Key, d.File);
                return d.Key;
            }))
            .ToList();

        // The transformers, with the shared models: their readers need them to load. Each one once, by name, like the pretrains
        // (German default_accurate's ELECTRA next to sentiment sb10k_dbmdz-bert-german-cased's BERT, #172).
        var transformers = TransformersOf(models, options.Language).ToDictionary(name => name,
            name => Start($"transformer/{name}", () => backend.Transformer(ModelDownloader.TransformerDirectory(modelDir, name), name)));
        TransformerEmbedder? TransformerOf(string processor) =>
            Transformers[options.Language].TryGetValue(Name(processor), out var name) ? transformers[name].Result : null;

        // The pretrain and the charlms a processor's model reads, once loaded.
        Pretrain? PretrainIfAny(string processor) => dependencies[processor].Where(d => d.Kind == "pretrain").Select(d => pretrains[d.Key].Result).SingleOrDefault();
        Pretrain PretrainOf(string processor) => PretrainIfAny(processor)
            ?? throw new ArgumentException($"The {processor} model {Model(processor)} reads word vectors: name them in its ModelPath (Pretrain)", nameof(options));
        Charlms CharlmsOf(string processor) => new(
            dependencies[processor].Where(d => d.Kind == "forward_charlm").Select(d => d.Key).SingleOrDefault(),
            dependencies[processor].Where(d => d.Kind == "backward_charlm").Select(d => d.Key).SingleOrDefault());
        foreach (var processor in models.Keys)
            _charlmOf[processor] = CharlmsOf(processor).Forward;

        // The tokenizer and the lemmatizer go first, unless they read charlms (the _charlm ones): then after them.
        Task<Tokenizer>? tokenizer = null;
        Task<Lemmatizer>? lemma = null;
        _identityLemma = models.GetValueOrDefault("lemma") == IdentityLemma;
        bool hasLemma = models.ContainsKey("lemma") && !_identityLemma;
        void StartReaders(bool ofCharlms)
        {
            if (tokenizer == null && CharlmsOf("tokenize").Forward != null == ofCharlms)
                tokenizer = Start(Name("tokenize"), () => backend.Tokenizer(Model("tokenize"), CharlmsOf("tokenize")));
            if (hasLemma && lemma == null && CharlmsOf("lemma").Forward != null == ofCharlms)
                lemma = Start(Name("lemma"), () => backend.Lemma(Model("lemma"), CharlmsOf("lemma"), PretrainIfAny("lemma"), options.LemmaDictionaryOnly));
        }

        StartReaders(ofCharlms: false);
        var mwt = models.ContainsKey("mwt") ? Start(Name("mwt"), () => backend.Mwt(Model("mwt"), options.MwtDictionaryOnly)) : null;
        try
        {
            foreach (var c in charlms)
                c.GetAwaiter().GetResult();
            foreach (var p in pretrains.Values)
                p.GetAwaiter().GetResult();
            foreach (var loading in transformers.Values)
                loading.GetAwaiter().GetResult();
            StartReaders(ofCharlms: true);
            var pos = models.ContainsKey("pos") ? Start(Name("pos"), () => backend.Pos(Model("pos"), PretrainOf("pos"), CharlmsOf("pos"), TransformerOf("pos"))) : null;
            var depparse = models.ContainsKey("depparse")
                ? Start(Name("depparse"), () => backend.Depparse(Model("depparse"), PretrainOf("depparse"), CharlmsOf("depparse"), TransformerOf("depparse")))
                : null;
            var ner = models.ContainsKey("ner") ? Start(Name("ner"), () => backend.Ner(Model("ner"), PretrainOf("ner"), CharlmsOf("ner"), TransformerOf("ner"), options.NerTagSet)) : null;
            var parser = models.ContainsKey("constituency")
                ? Start(Name("constituency"), () => backend.Constituency(Model("constituency"), PretrainOf("constituency"), CharlmsOf("constituency"), TransformerOf("constituency")))
                : null;
            var sentiment = models.ContainsKey("sentiment")
                ? Start(Name("sentiment"), () => backend.Sentiment(Model("sentiment"), PretrainOf("sentiment"), CharlmsOf("sentiment"), TransformerOf("sentiment")))
                : null;
            var coref = models.ContainsKey("coref")
                ? Start(Name("coref"), () => backend.Coref(Model("coref"), TransformerOf("coref")
                    ?? throw new ArgumentException($"The coref model {Model("coref")} reads a transformer StanzaSharp doesn't know", nameof(options))))
                : null;
            Task.WaitAll(started);

            // The batch-size options: each a copy of its processor sharing the nets, made before any call.
            _tokenizer = options.TokenizeBatchSize is int t ? tokenizer!.Result.WithBatchSize(t) : tokenizer!.Result;
            _mwt = options.MwtBatchSize is int m ? mwt?.Result.WithBatchSize(m) : mwt?.Result;
            _lemma = options.LemmaBatchSize is int l ? lemma?.Result.WithBatchSize(l) : lemma?.Result;
            _lemma = options.LemmaBeamSize is int beam ? _lemma?.WithBeamSize(beam) : _lemma;
            _pos = options.PosBatchSize != null || options.PosBatchMaximumTokens != null
                ? pos?.Result.WithBatchSize(options.PosBatchSize, options.PosBatchMaximumTokens)
                : pos?.Result;
            _depparse = options.DepparseBatchSize != null || options.DepparseMinLengthToBatchSeparately != null
                ? depparse?.Result.WithBatchSize(options.DepparseBatchSize, options.DepparseMinLengthToBatchSeparately)
                : depparse?.Result;
            _ner = options.NerBatchSize is int n ? ner?.Result.WithBatchSize(n) : ner?.Result;
            _parser = options.ConstituencyBatchSize is int cp ? parser?.Result.WithBatchSize(cp) : parser?.Result;
            _sentiment = options.SentimentBatchSize is int b ? sentiment?.Result.WithBatchSize(b) : sentiment?.Result;
            _coref = coref?.Result;
            _pretrains = [.. pretrains.Values.Select(p => p.Result)];
            _transformers = [.. transformers.Values.Select(t => t.Result)];
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
    /// (e.g. constituency in <c>default_fast</c>), a processor without the ones it requires, or an unknown or
    /// unsupported model in <see cref="PipelineOptions.Models"/>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="PipelineOptions.Threads"/>, <see cref="PipelineOptions.MaxConcurrentCalls"/>
    /// or a batch size (<see cref="PipelineOptions.PosBatchSize"/> and the others) below 1, or a non-positive
    /// <see cref="CharlmCacheOptions.MaxWords"/>.</exception>
    /// <exception cref="FileNotFoundException">A model file the processors read is missing from
    /// <paramref name="modelDir"/>; download it with <see cref="ModelDownloader.DownloadAsync(string, PipelineOptions, IProgress{string}, CancellationToken)"/>.</exception>
    /// <exception cref="InvalidDataException">With <see cref="PipelineOptions.VerifyChecksums"/>: a model file whose
    /// checksum differs from Stanza's.</exception>
    /// <exception cref="InvalidOperationException">With <see cref="PipelineOptions.VerifyChecksums"/>: a model that
    /// would be read from converted files, which have no published checksum.</exception>
    public static Pipeline Load(string modelDir, PipelineOptions? options = null)
    {
        options ??= new PipelineOptions();
        if (!Directory.Exists(modelDir))
            throw new DirectoryNotFoundException($"Model directory not found: {modelDir} (download them with ModelDownloader.DownloadAsync or \"stanzasharp download\")");
        var models = SelectModels(options.Package, options.Processors, addRequired: false, nameof(options), options.Language, options.Models, options.Pretagged,
            options.ModelPaths);
        if (options.CharlmCache.IsEnabled && options.CharlmCache.MaxWords <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "CharlmCache.MaxWords must be positive; set IsEnabled = false to turn the cache off");
        if (options.Threads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Threads must be at least 1, or null for the default");
        if (options.MaxConcurrentCalls < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentCalls must be at least 1, or null for no limit");
        foreach (var (name, size) in BatchSizes(options))
        {
            if (size < 1)
                throw new ArgumentOutOfRangeException(nameof(options), $"{name} must be at least 1, or null for the model's own");
        }

        ThrowIfMissing(modelDir, models, options);
        var logger = options.Logger;
        long start = Stopwatch.GetTimestamp();
        if (options.VerifyChecksums)
        {
            ModelDownloader.Verify(modelDir, models, options.Language, CustomPaths(options.ModelPaths, nameof(options)));
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
            string.Join(",", StanzaOrder.Split(',').Where(models.ContainsKey).Select(p => options.Models == null ? p : $"{p}={models[p]}")),
            Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return pipeline;
    }

    /// <summary>The batch-size options by name, for <see cref="Load"/>'s checks.</summary>
    private static IEnumerable<(string Name, int? Size)> BatchSizes(PipelineOptions options) =>
    [
        (nameof(options.TokenizeBatchSize), options.TokenizeBatchSize),
        (nameof(options.MwtBatchSize), options.MwtBatchSize),
        (nameof(options.PosBatchSize), options.PosBatchSize),
        (nameof(options.PosBatchMaximumTokens), options.PosBatchMaximumTokens),
        (nameof(options.LemmaBatchSize), options.LemmaBatchSize),
        (nameof(options.LemmaBeamSize), options.LemmaBeamSize),
        (nameof(options.ConstituencyBatchSize), options.ConstituencyBatchSize),
        (nameof(options.DepparseBatchSize), options.DepparseBatchSize),
        (nameof(options.DepparseMinLengthToBatchSeparately), options.DepparseMinLengthToBatchSeparately),
        (nameof(options.SentimentBatchSize), options.SentimentBatchSize),
        (nameof(options.NerBatchSize), options.NerBatchSize),
    ];

    // A folder downloaded for fewer processors (or before tokenize brought mwt along, 1.x) lacks some files: say which
    // and how to fetch them, instead of the FileNotFoundException of whichever model reads it first.
    private static void ThrowIfMissing(string modelDir, Dictionary<string, string> models, PipelineOptions options)
    {
        // The caller's own files first: no download brings them.
        foreach (var (processor, path) in CustomPaths(options.ModelPaths, nameof(options)))
        {
            foreach (var (what, own) in new[] { ("Model", path.Model), ("Pretrain", path.Pretrain), ("ForwardCharlm", path.ForwardCharlm), ("BackwardCharlm", path.BackwardCharlm) })
            {
                if (own != null && !File.Exists(own + ".pt") && !File.Exists(own + ".json"))
                    throw new FileNotFoundException($"Model file not found: {own}.pt (ModelPaths[\"{processor}\"].{what})", own + ".pt");
            }
        }

        var missing = ModelFiles(models, options.Language, readOnly: true, CustomPaths(options.ModelPaths, nameof(options)))
            .FirstOrDefault(p => !File.Exists(Path.Combine(modelDir, p + ".pt")) && !File.Exists(Path.Combine(modelDir, p + ".json")));
        string chosen = options.Models == null ? "" : $" --model {string.Join(',', options.Models.Select(m => $"{m.Key}={m.Value}"))}";
        if (missing == null)
        {
            foreach (var name in TransformersOf(models, options.Language))
            {
                var directory = ModelDownloader.TransformerDirectory(modelDir, name);
                var absent = ModelDownloader.Transformers[name].Files.Select(f => Path.Combine(directory, f.File)).FirstOrDefault(f => !File.Exists(f));
                if (absent != null)
                {
                    throw new FileNotFoundException($"Transformer file not found: {absent}. Run \"stanzasharp download {modelDir} --lang {options.Language} "
                        + $"--package {options.Package}{chosen}\" or ModelDownloader.DownloadAsync(dir, options) again; it fetches only the missing files.", absent);
                }
            }

            return;
        }

        var file = Path.Combine([modelDir, .. (missing + ".pt").Split('/')]);
        string why = missing.StartsWith("mwt/", StringComparison.Ordinal) && options.Processors != null && !ParseProcessors(options.Processors, nameof(options)).Contains("mwt")
            ? " (tokenize loads the package's mwt too, as Stanza does)"
            : "";
        string processors = options.Processors == null ? "" : $" --processors {options.Processors}";
        throw new FileNotFoundException($"Model file not found: {file}{why}. Run \"stanzasharp download {modelDir} --lang {options.Language} "
            + $"--package {options.Package}{processors}{chosen}\" or ModelDownloader.DownloadAsync(dir, options) again; it fetches only the missing files.", file);
    }

    // How many models a managed pipeline loads at a time (round 17).
    private const int LoadLanes = 2;

    // English's shared files, for the samples that load them by hand.
    internal const string PretrainPath = "pretrain/conll17", ForwardCharlmPath = "forward_charlm/1billion", BackwardCharlmPath = "backward_charlm/1billion";

    /// <summary>
    /// The processors to run and each one's model: <paramref name="processors"/> (null: all of the package's, plus the
    /// keys of <paramref name="chosen"/>), checked against <paramref name="package"/> of <paramref name="language"/>,
    /// with <paramref name="chosen"/> (<see cref="PipelineOptions.Models"/>) replacing the package's models, plus the
    /// package's mwt wherever tokenize is (as Stanza adds it). With <paramref name="addRequired"/> the processors they
    /// require are added (what to download); without, a missing one throws (what to load), unless
    /// <paramref name="processors"/> is null.
    /// </summary>
    internal static Dictionary<string, string> SelectModels(string package, string? processors, bool addRequired, string paramName, string language = English,
        IReadOnlyDictionary<string, string>? chosen = null, bool pretagged = false, IReadOnlyDictionary<string, ModelPath>? paths = null)
    {
        if (!Packages.TryGetValue(language ?? "", out var packages))
            throw new ArgumentException($"Unknown language '{language}'. Available: {string.Join(", ", Packages.Keys)}", paramName);
        if (!packages.TryGetValue(package ?? "", out var packageModels))
        {
            string why = package == "default_accurate" ? $" ({language}'s default_accurate needs a transformer StanzaSharp doesn't run)" : "";
            throw new ArgumentException($"Unknown package '{package}' for language '{language}'{why}. Available: {string.Join(", ", packages.Keys)}", paramName);
        }

        var overrides = ResolveModels(chosen, language!, paramName);
        var models = new Dictionary<string, string>(packageModels);
        foreach (var (p, model) in overrides)
            models[p] = model;

        // A path of the caller's own keeps the replaced model's name, which says what it reads; one for a processor the
        // package lacks (or for the identity lemmatizer) reads only what its ModelPath names (CustomModel).
        var custom = CustomPaths(paths, paramName);
        if (custom.Keys.FirstOrDefault(overrides.ContainsKey) is { } both)
            throw new ArgumentException($"Both Models and ModelPaths name {both}", paramName);

        // An optional processor's path replaces the package's optional model (coref: its transformer stays XLM-R), on a
        // package without one too, as Models may choose coref on any package.
        foreach (var p in custom.Keys)
        {
            if (!models.ContainsKey(p) || models[p] == IdentityLemma)
                models[p] = Optional.GetValueOrDefault(language!)?.GetValueOrDefault(package!)?.GetValueOrDefault(p) ?? (p == "coref" ? CorefModel : CustomModel);
        }

        // Stanza's add_mwt: a replaced tokenizer brings the mwt of its own treebank, if there is one.
        if (chosen != null && overrides.ContainsKey("tokenize") && !overrides.ContainsKey("mwt") && models.ContainsKey("mwt")
            && MwtFor(chosen.First(kv => kv.Key.Trim().Equals("tokenize", StringComparison.OrdinalIgnoreCase)).Value.Trim().ToLowerInvariant(), language!) is { } mwt)
        {
            models["mwt"] = mwt;
        }

        var set = processors == null ? models.Keys.ToHashSet() : ParseProcessors(processors, paramName);

        // An optional processor (coref) runs only when named, with the package's model unless Models chose another.
        foreach (var (p, model) in Optional.GetValueOrDefault(language!)?.GetValueOrDefault(package!) ?? [])
        {
            if (set.Contains(p))
                models.TryAdd(p, model);
        }

        if (processors != null && overrides.Keys.FirstOrDefault(p => !set.Contains(p)) is { } unlisted)
            throw new ArgumentException($"Models names {unlisted}, which Processors ({processors}) doesn't list", paramName);
        if (processors != null && custom.Keys.FirstOrDefault(p => !set.Contains(p)) is { } unlistedPath)
            throw new ArgumentException($"ModelPaths names {unlistedPath}, which Processors ({processors}) doesn't list", paramName);

        // Pretagged (Stanza's pretagged): lemma, depparse and constituency need nothing; the tokenizer always loads, so the
        // text overloads keep working (owner's decision, 2026-10-17).
        if (pretagged)
            set.Add("tokenize");

        // Stanza's add_mwt: tokenize brings the package's mwt along (tokenize,ner expands "don't" too).
        if (set.Contains("tokenize") && models.ContainsKey("mwt"))
            set.Add("mwt");
        foreach (var p in set.ToList())
        {
            // What the package lacks isn't required (eslspok has no mwt); the identity lemmatizer reads no UPOS.
            var needs = pretagged && p is "lemma" or "depparse" or "constituency" ? []
                : p == "lemma" && models.GetValueOrDefault(p) == IdentityLemma ? ["tokenize"]
                : Requires[p];
            var missing = needs.Where(n => !set.Contains(n) && models.ContainsKey(n)).ToList();
            if (missing.Count > 0 && !addRequired && processors != null)
                throw new ArgumentException($"Processor '{p}' requires {string.Join(", ", missing)}", paramName);
            set.UnionWith(missing);
        }

        foreach (var p in set)
        {
            if (!models.ContainsKey(p))
            {
                var others = packages.Where(kv => kv.Value.ContainsKey(p) || Optional.GetValueOrDefault(language!)?.GetValueOrDefault(kv.Key)?.ContainsKey(p) == true)
                    .Select(kv => $"\"{kv.Key}\"");
                throw new ArgumentException($"Package '{package}' has no {p} model; use Package = {string.Join(" or ", others)} for {p}, "
                    + $"or choose one with Models: {string.Join(", ", ModelsOf(language!, p))}", paramName);
            }
        }

        return set.ToDictionary(p => p, p => models[p]);
    }

    /// <summary>
    /// <see cref="PipelineOptions.Models"/> as processor → model: each key a processor, each value one of the models
    /// <see cref="ModelsOf"/> lists, or a package with that processor (its model), as Stanza resolves them. Names are
    /// trimmed and lowercased, like Stanza's.
    /// </summary>
    private static Dictionary<string, string> ResolveModels(IReadOnlyDictionary<string, string>? chosen, string language, string paramName)
    {
        var resolved = new Dictionary<string, string>();
        foreach (var (key, value) in chosen ?? new Dictionary<string, string>())
        {
            var p = key?.Trim().ToLowerInvariant() ?? "";
            if (!Requires.ContainsKey(p))
                throw new ArgumentException($"Unknown processor '{key}' in Models. Available: {StanzaOrder}", paramName);
            if (resolved.ContainsKey(p))
                throw new ArgumentException($"Models names {p} twice", paramName);
            var name = value?.Trim().ToLowerInvariant() ?? "";
            var supported = ModelsOf(language, p).ToList();
            if (supported.Contains(name))
            {
                resolved[p] = name;
            }
            else if (Packages[language].TryGetValue(name, out var package) && package.TryGetValue(p, out var model))
            {
                resolved[p] = model;
            }
            else if (Optional.GetValueOrDefault(language)?.GetValueOrDefault(name)?.TryGetValue(p, out var optional) == true)
            {
                resolved[p] = optional;
            }
            else
            {
                string why = name.Contains("electra", StringComparison.Ordinal) || name.Contains("bert", StringComparison.Ordinal)
                    ? " (a transformer model StanzaSharp doesn't run yet)"
                    : "";
                var packagesWith = Packages[language].Where(kv => kv.Value.ContainsKey(p)).Select(kv => kv.Key);
                throw new ArgumentException($"Unknown or unsupported {p} model '{value}' for language '{language}'{why}. Supported: "
                    + $"{string.Join(", ", supported)}; or a package's name ({string.Join(", ", packagesWith)})", paramName);
            }
        }

        return resolved;
    }

    /// <summary>The mwt model Stanza's add_mwt picks for tokenize <paramref name="tokenize"/> (as given: a package's or a model's name), or null.</summary>
    private static string? MwtFor(string tokenize, string language)
    {
        var mwt = ModelsOf(language, "mwt").ToHashSet();
        if (Packages[language].TryGetValue(tokenize, out var package))
            return package.GetValueOrDefault("mwt");
        if (mwt.Contains(tokenize))
            return tokenize;
        var stem = tokenize.EndsWith("_nocharlm", StringComparison.Ordinal) || tokenize.EndsWith("_charlm", StringComparison.Ordinal) ? tokenize[..tokenize.LastIndexOf('_')] : null;
        return stem != null && mwt.Contains(stem) ? stem : null;
    }

    /// <summary>
    /// The shared files <paramref name="models"/> of <paramref name="language"/> read (paths without extension), from
    /// <see cref="Dependencies"/>: pretrains first, then the forward and the backward charlm.
    /// </summary>
    internal static List<string> SharedModels(IReadOnlyDictionary<string, string> models, string language = English, bool readOnly = false,
        IReadOnlyDictionary<string, ModelPath>? custom = null) =>
        models.SelectMany(m => (readOnly ? ReadDependencies($"{m.Key}/{m.Value}", language) : Dependencies[language].GetValueOrDefault($"{m.Key}/{m.Value}", []))
                .Where(d => custom?.GetValueOrDefault(m.Key) is not { } c || Override(c, d) == null))
            .Distinct()
            .OrderBy(p => p.StartsWith("pretrain/", StringComparison.Ordinal) ? 0 : p.StartsWith("forward_charlm/", StringComparison.Ordinal) ? 1 : 2)
            .ThenBy(p => p, StringComparer.Ordinal)
            .ToList();

    /// <summary>
    /// Every package file <paramref name="models"/> read (paths without extension): their own, but none for the identity
    /// lemmatizer or a processor with a <paramref name="custom"/> path, then <see cref="SharedModels"/> (<paramref name="readOnly"/>:
    /// without the <see cref="Unread"/> ones, which Load skips; none a custom path replaces).
    /// </summary>
    internal static IEnumerable<string> ModelFiles(IReadOnlyDictionary<string, string> models, string language, bool readOnly = false,
        IReadOnlyDictionary<string, ModelPath>? custom = null) =>
        models.Where(m => m.Value != IdentityLemma && m.Value != CustomModel && custom?.ContainsKey(m.Key) != true)
            .Select(m => $"{m.Key}/{m.Value}")
            .Concat(SharedModels(models, language, readOnly, custom));

    /// <summary>
    /// The model name of a processor whose <see cref="PipelineOptions.ModelPaths"/> path the package has no model for: it
    /// reads only the shared files its <see cref="ModelPath"/> names.
    /// </summary>
    internal const string CustomModel = "(custom)";

    /// <summary>
    /// <see cref="PipelineOptions.ModelPaths"/> by processor (trimmed and lowercased, like <see cref="PipelineOptions.Models"/>),
    /// each path made a full base path (without <c>.pt</c>, <c>.json</c> or <c>.safetensors</c>).
    /// </summary>
    internal static Dictionary<string, ModelPath> CustomPaths(IReadOnlyDictionary<string, ModelPath>? paths, string paramName)
    {
        var resolved = new Dictionary<string, ModelPath>();
        foreach (var (key, value) in paths ?? new Dictionary<string, ModelPath>())
        {
            var p = key?.Trim().ToLowerInvariant() ?? "";
            if (!Requires.ContainsKey(p))
                throw new ArgumentException($"Unknown processor '{key}' in ModelPaths. Available: {StanzaOrder}", paramName);
            if (resolved.ContainsKey(p))
                throw new ArgumentException($"ModelPaths names {p} twice", paramName);
            if (string.IsNullOrWhiteSpace(value?.Model))
                throw new ArgumentException($"ModelPaths[\"{p}\"] has no Model", paramName);
            resolved[p] = new ModelPath(BasePath(value.Model)!, BasePath(value.Pretrain), BasePath(value.ForwardCharlm), BasePath(value.BackwardCharlm));
        }

        return resolved;

        static string? BasePath(string? path)
        {
            if (path == null)
                return null;
            var full = Path.GetFullPath(path);
            foreach (var extension in new[] { ".pt", ".json", ".safetensors" })
            {
                if (full.EndsWith(extension, StringComparison.Ordinal))
                    return full[..^extension.Length];
            }

            return full;
        }
    }

    /// <summary>The file <paramref name="custom"/> names instead of the shared file <paramref name="dependency"/> (<c>pretrain/conll17</c>), or null.</summary>
    internal static string? Override(ModelPath custom, string dependency) => dependency[..dependency.IndexOf('/')] switch
    {
        "pretrain" => custom.Pretrain,
        "forward_charlm" => custom.ForwardCharlm,
        "backward_charlm" => custom.BackwardCharlm,
        _ => null,
    };

    /// <summary>A comma-separated processor list as a set, checking every name.</summary>
    internal static HashSet<string> ParseProcessors(string processors, string paramName)
    {
        var set = processors.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(p => p.ToLowerInvariant())
            .ToHashSet();
        foreach (var p in set)
        {
            if (!Requires.ContainsKey(p))
                throw new ArgumentException($"Unknown processor '{p}'. Available: {StanzaOrder}", paramName);
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
            var doc = Step("tokenize", () => _tokenizer.Process(text, _splitSentences, cancellationToken, _postprocessor), cancellationToken);
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

    /// <summary>
    /// Processes <paramref name="texts"/> lazily, <paramref name="batchSize"/> at a time, like Stanza's <c>Pipeline.stream</c>:
    /// each batch is one <see cref="Process(IEnumerable{string})"/> call, and its documents are yielded once it is done, so a
    /// long or unbounded sequence never has to be in memory at once. Sentence ids (<see cref="Sentence.SentId"/>) count across
    /// the whole stream.
    /// </summary>
    /// <param name="texts">The texts, read as the stream is enumerated; a null text throws when its batch is reached.</param>
    /// <param name="batchSize">Texts per batch, Stanza's <c>batch_size</c> (50 by default).</param>
    /// <param name="cancellationToken">Checked like <see cref="Process(IEnumerable{string}, CancellationToken)"/>'s, in the batch running.</param>
    /// <exception cref="ArgumentNullException"><paramref name="texts"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="batchSize"/> is less than 1.</exception>
    /// <remarks>
    /// The output equals Stanza's <c>stream</c> with the same batch size. Within a batch the documents are processed as by
    /// <see cref="Process(IEnumerable{string})"/>, so the sentiment labels and the parser's scores can depend on the batch
    /// they share (see there); a larger batch is faster, a batch of 1 gives each text's own output.
    /// </remarks>
    /// <example>
    /// <code>
    /// foreach (var doc in nlp.Stream(File.ReadLines("reviews.txt")))
    ///     Console.WriteLine(doc.Sentences[0].Sentiment);
    /// </code>
    /// </example>
    public IEnumerable<Document> Stream(IEnumerable<string> texts, int batchSize = 50, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        return Batches();

        IEnumerable<Document> Batches()
        {
            int sentences = 0;
            foreach (var batch in texts.Chunk(batchSize))
            {
                foreach (var doc in Process(batch, cancellationToken))
                {
                    foreach (var sentence in doc.Sentences)
                        sentence.SentId = (sentences++).ToString(CultureInfo.InvariantCulture);
                    yield return doc;
                }
            }
        }
    }

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
            var docs = Step("tokenize", () => _tokenizer.Process(list, _splitSentences, cancellationToken, _postprocessor), cancellationToken);

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
        // A _nocharlm tagger (default_fast) has no charlm outputs to share. The cache holds the tagger's pair only: a
        // processor reading another pair (ner i2b2's mimic next to default's 1billion) gets none, so nothing crosses pairs.
        string? tagger = _pos is { UsesCharlm: true } ? _charlmOf["pos"] : null;
        bool Shares(string processor) => tagger != null && _charlmOf.GetValueOrDefault(processor) == tagger;
        using var charlms = tagger != null && (Shares("ner") || Shares("constituency") || Shares("sentiment") || (Shares("depparse") && _depparseReadsCache))
            && _cacheOptions.IsEnabled ? new CharlmCache(_cacheOptions.MaxWords) : null;
        CharlmCache? CacheFor(string processor) => Shares(processor) ? charlms : null;

        // The transformer states, for the processors after the first that reads them (Stanza runs it once per processor):
        // one cache per net (the frozen transformer, each LoRA reader's own). On a short call (managed) every reader's
        // sentences are encoded up front in one sweep over the layers (#169), else each net runs as its readers ask.
        var caches = new Dictionary<ITransformerNet, TransformerCache>();
        TransformerCache CacheOf(ITransformerNet net) => caches.TryGetValue(net, out var cache) ? cache : caches[net] = new TransformerCache(net);
        TransformerCache? TransformersFor(TransformerReader? reader) => reader == null ? null : caches.GetValueOrDefault(reader.Embedder.Net);
        try
        {
            var readers = new[] { _pos?.TransformerReader(doc), _parser?.TransformerReader(doc), _depparse?.TransformerReader(doc),
                _sentiment?.TransformerReader(doc), _ner?.TransformerReader(doc), };

            // Per transformer: its frozen readers and its LoRA ones share its tokenizer (a LoRA embedder copies the frozen one's).
            foreach (var transformer in _transformers)
            {
                var own = readers.OfType<TransformerReader>().Where(r => r.Embedder.Tokenizer == transformer.Tokenizer).ToList();
                bool joint = Step("transformer", () => TransformerEmbedder.Prefill(own, CacheOf, ct), ct);
                if (!joint && own.Count(r => r.Embedder.Net == transformer.Net) > 1)
                    CacheOf(transformer.Net);
            }

            return Annotate(doc, charlms, CacheFor, readers, TransformersFor, ct);
        }
        finally
        {
            foreach (var cache in caches.Values)
                cache.Dispose();
        }
    }

    // The processors after mwt, in Stanza's order; readers[i] is the transformer reader of pos, constituency, depparse,
    // sentiment and ner (null without one).
    private Document Annotate(Document doc, CharlmCache? charlms, Func<string, CharlmCache?> cacheFor, TransformerReader?[] readers,
        Func<TransformerReader?, TransformerCache?> transformersFor, CancellationToken ct)
    {
        if (_pos != null)
            Step("pos", () => _pos.Process(doc, charlms, ct, transformersFor(readers[0])), ct);
        if (_lemma != null)
            Step("lemma", () => _lemma.Process(doc, ct), ct);
        else if (_identityLemma)
            Step("lemma", () => Lemmatizer.Identity(doc), ct);

        // Stanza's order (PIPELINE_NAMES). Only the CoNLL-U comment order shows it, and Conllu.Write fixes that.
        if (_parser != null)
            Step("constituency", () => _parser.Process(doc, cacheFor("constituency"), ct, transformersFor(readers[1])), ct);
        if (_coref != null)
            Step("coref", () => _coref.Process(doc, ct), ct);

        // Depparse runs the charlms with a ROOT word in front, so of the tagger's outputs only the backward ones apply, with
        // ROOT run on from the tagger's final state (managed; DependencyParser.Process).
        if (_depparse != null)
            Step("depparse", () => _depparse.Process(doc, cacheFor("depparse"), ct, transformersFor(readers[2])), ct);
        if (_sentiment != null)
            Step("sentiment", () => _sentiment.Process(doc, cacheFor("sentiment"), ct, transformersFor(readers[3])), ct);
        if (_ner != null)
            Step("ner", () => _ner.Process(doc, cacheFor("ner"), ct, transformersFor(readers[4])), ct);
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
        _coref?.Dispose();
        _depparse?.Dispose();
        _ner?.Dispose();
        _pos?.Dispose();
        _lemma?.Dispose();
        _mwt?.Dispose();
        _tokenizer.Dispose();
        foreach (var transformer in _transformers)
            transformer.Net.Dispose();
        _models.Dispose();
        foreach (var pretrain in _pretrains)
            pretrain.Dispose();
        _gate?.Dispose();
    }
}
