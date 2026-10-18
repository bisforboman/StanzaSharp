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

            // The most accurate: the _charlm tokenizer and lemmatizer, and google/electra-large-discriminator (Transformers)
            // under pos, constituency, depparse and ner (frozen, one shared pass) and sentiment (its LoRA adapters).
            ["default_accurate"] = new()
            {
                ["tokenize"] = "combined_charlm",
                ["mwt"] = "combined",
                ["pos"] = "combined_electra-large",
                ["lemma"] = "combined_charlm",
                ["constituency"] = "ptb3-revised_electra-large",
                ["depparse"] = "combined_electra-large",
                ["sentiment"] = "sstplus_electra-large",
                ["ner"] = "ontonotes-ww-multi_electra-large",
            },

            // The treebank packages: tokenize to depparse only. combined has default's models; eslspok has no mwt and
            // Stanza's identity lemmatizer (no model); the biomedical craft, genia and mimic have no mwt and their own
            // word vectors (mimic also its own charlms).
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
            ["craft"] = Biomedical("craft", "nocharlm"),
            ["genia"] = Biomedical("genia", "nocharlm"),
            ["mimic"] = Biomedical("mimic", "charlm"),
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

    // Models beyond the packages' that PipelineOptions.Models can choose, per language and processor (docs/per-processor-models-plan.md).
    // Like the packages' models, each needs its MD5 in ModelDownloader.Files and its shared files in Dependencies.
    internal static readonly Dictionary<string, Dictionary<string, string[]>> Alternatives = new()
    {
        [English] = new()
        {
            ["constituency"] = ["ptb3_charlm", "ptb3-revised_electra-large"],
            ["sentiment"] = ["sstplus_electra-large"],

            // the treebanks' _charlm tokenizers and lemmatizers (per-character charlm states); lemma combined_charlm also has
            // contextual lemma classifiers ('s, her), which read the pretrain too
            ["tokenize"] =
            [
                "combined_charlm", "ewt_charlm", "gum_charlm", "lines_charlm", "partut_charlm", "atis_charlm", "childes_charlm", "eslspok_charlm",
            ],
            ["lemma"] = ["combined_charlm", "ewt_charlm", "gum_charlm", "lines_charlm", "partut_charlm", "atis_charlm", "childes_charlm"],

            // the treebanks' own _nocharlm taggers and parsers (mimic has only a parser); combined's are default_fast's
            // and default_accurate's frozen-transformer tagger, parser and NER model (Transformers)
            ["pos"] =
            [
                "ewt_nocharlm", "gum_nocharlm", "lines_nocharlm", "partut_nocharlm", "atis_nocharlm", "childes_nocharlm", "eslspok_nocharlm",
                "combined_electra-large",
            ],
            ["depparse"] =
            [
                "ewt_nocharlm", "gum_nocharlm", "lines_nocharlm", "partut_nocharlm", "atis_nocharlm", "childes_nocharlm", "eslspok_nocharlm",
                "mimic_nocharlm", "combined_electra-large",
            ],
            ["ner"] =
            [
                "ontonotes_charlm", "ontonotes_nocharlm", "combined_charlm", "conll03_charlm", "ontonotes-ww-multi_electra-large",

                // clinical (mimic's vectors and charlms) and biomedical (biomed's vectors, pubmed's charlms)
                "i2b2", "radiology", "anatem", "bc4chemd", "bc5cdr", "bionlp13cg", "jnlpba", "linnaeus", "ncbi_disease", "s800",
            ],
        },
        ["de"] = new()
        {
            ["tokenize"] = ["combined_charlm", "gsd_charlm", "hdt_charlm"],
            ["lemma"] = ["combined_charlm", "gsd_charlm", "hdt_charlm"],
            ["pos"] = ["gsd_nocharlm", "hdt_nocharlm"],
            ["depparse"] = ["gsd_nocharlm", "hdt_nocharlm"],
            ["ner"] = ["conll03"],
        },
    };

    /// <summary>Every model <paramref name="processor"/> of <paramref name="language"/> can run: its packages' and <see cref="Alternatives"/>.</summary>
    internal static IEnumerable<string> ModelsOf(string language, string processor) =>
        Packages[language].Values.Select(p => p.GetValueOrDefault(processor)).OfType<string>()
            .Concat(Alternatives[language].GetValueOrDefault(processor, []))
            .Distinct();

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

    // A biomedical package: no mwt, a _nocharlm tokenizer and lemmatizer, the tagger and parser as <kind>.
    private static Dictionary<string, string> Biomedical(string name, string kind) => new()
    {
        ["tokenize"] = $"{name}_nocharlm",
        ["pos"] = $"{name}_{kind}",
        ["lemma"] = $"{name}_nocharlm",
        ["depparse"] = $"{name}_{kind}",
    };

    private static readonly string[] Conll17 = ["pretrain/conll17"],
        EnglishCharlm = ["pretrain/conll17", "forward_charlm/1billion", "backward_charlm/1billion"],
        Pubmed = ["pretrain/biomed", "forward_charlm/pubmed", "backward_charlm/pubmed"],
        GermanCharlm = ["pretrain/conll17", "forward_charlm/newswiki", "backward_charlm/newswiki"],
        EnglishForward = ["forward_charlm/1billion"],
        EnglishPair = ["forward_charlm/1billion", "backward_charlm/1billion"],
        GermanForward = ["forward_charlm/newswiki"],
        GermanPair = ["forward_charlm/newswiki", "backward_charlm/newswiki"];

    // The shared files (pretrains, charlms) each model reads, by "<processor>/<model>": resources.json's "dependencies",
    // which the model's name doesn't tell (germeval2014 reads charlms and its own pretrain). Models without an entry
    // (mwt, the _nocharlm tokenizers and lemmatizers) read none; the _charlm tokenizers read only the forward charlm.
    internal static readonly Dictionary<string, Dictionary<string, string[]>> Dependencies = new()
    {
        [English] = new()
        {
            ["pos/combined_charlm"] = EnglishCharlm,
            ["constituency/ptb3-revised_charlm"] = EnglishCharlm,
            ["depparse/combined_charlm"] = EnglishCharlm,
            ["sentiment/sstplus_charlm"] = EnglishCharlm,
            ["sentiment/sstplus_electra-large"] = EnglishCharlm, // resources.json lists the charlms, which the model doesn't read
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
            ["pos/craft_nocharlm"] = ["pretrain/biomed"],
            ["depparse/craft_nocharlm"] = ["pretrain/biomed"],
            ["pos/genia_nocharlm"] = ["pretrain/biomed"],
            ["depparse/genia_nocharlm"] = ["pretrain/biomed"],
            ["pos/mimic_charlm"] = ["pretrain/mimic", "forward_charlm/mimic", "backward_charlm/mimic"],
            ["depparse/mimic_charlm"] = ["pretrain/mimic", "forward_charlm/mimic", "backward_charlm/mimic"],

            // Alternatives
            ["constituency/ptb3_charlm"] = EnglishCharlm,
            ["tokenize/combined_charlm"] = EnglishForward,
            ["tokenize/ewt_charlm"] = EnglishForward,
            ["tokenize/gum_charlm"] = EnglishForward,
            ["tokenize/lines_charlm"] = EnglishForward,
            ["tokenize/partut_charlm"] = EnglishForward,
            ["tokenize/atis_charlm"] = EnglishForward,
            ["tokenize/childes_charlm"] = EnglishForward,
            ["tokenize/eslspok_charlm"] = EnglishForward,
            ["lemma/combined_charlm"] = EnglishCharlm,
            ["lemma/ewt_charlm"] = EnglishPair,
            ["lemma/gum_charlm"] = EnglishPair,
            ["lemma/lines_charlm"] = EnglishPair,
            ["lemma/partut_charlm"] = EnglishPair,
            ["lemma/atis_charlm"] = EnglishPair,
            ["lemma/childes_charlm"] = EnglishPair,
            ["pos/ewt_nocharlm"] = Conll17,
            ["depparse/ewt_nocharlm"] = Conll17,
            ["pos/gum_nocharlm"] = Conll17,
            ["depparse/gum_nocharlm"] = Conll17,
            ["pos/lines_nocharlm"] = Conll17,
            ["depparse/lines_nocharlm"] = Conll17,
            ["pos/partut_nocharlm"] = Conll17,
            ["depparse/partut_nocharlm"] = Conll17,
            ["pos/atis_nocharlm"] = Conll17,
            ["depparse/atis_nocharlm"] = Conll17,
            ["pos/childes_nocharlm"] = Conll17,
            ["depparse/childes_nocharlm"] = Conll17,
            ["pos/eslspok_nocharlm"] = Conll17,
            ["depparse/eslspok_nocharlm"] = Conll17,
            ["depparse/mimic_nocharlm"] = ["pretrain/mimic"],
            ["ner/ontonotes_charlm"] = ["pretrain/fasttextcrawl", "forward_charlm/1billion", "backward_charlm/1billion"],
            ["ner/ontonotes_nocharlm"] = ["pretrain/fasttextcrawl"],
            ["ner/combined_charlm"] = EnglishCharlm,
            ["ner/i2b2"] = ["pretrain/mimic", "forward_charlm/mimic", "backward_charlm/mimic"],
            ["ner/radiology"] = ["pretrain/mimic", "forward_charlm/mimic", "backward_charlm/mimic"],
            ["ner/conll03_charlm"] = ["pretrain/glove", "forward_charlm/1billion", "backward_charlm/1billion"],
            ["ner/anatem"] = Pubmed,
            ["ner/bc4chemd"] = Pubmed,
            ["ner/bc5cdr"] = Pubmed,
            ["ner/bionlp13cg"] = Pubmed,
            ["ner/jnlpba"] = Pubmed,
            ["ner/linnaeus"] = Pubmed,
            ["ner/ncbi_disease"] = Pubmed,
            ["ner/s800"] = Pubmed,
            ["pos/combined_electra-large"] = EnglishCharlm,
            ["depparse/combined_electra-large"] = EnglishCharlm,
            ["ner/ontonotes-ww-multi_electra-large"] = EnglishCharlm,
            ["constituency/ptb3-revised_electra-large"] = EnglishCharlm,
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

            // Alternatives
            ["tokenize/combined_charlm"] = GermanForward,
            ["lemma/combined_charlm"] = GermanPair,
            ["tokenize/gsd_charlm"] = GermanForward,
            ["lemma/gsd_charlm"] = GermanPair,
            ["tokenize/hdt_charlm"] = GermanForward,
            ["lemma/hdt_charlm"] = GermanPair,
            ["pos/gsd_nocharlm"] = Conll17,
            ["depparse/gsd_nocharlm"] = Conll17,
            ["pos/hdt_nocharlm"] = Conll17,
            ["depparse/hdt_nocharlm"] = Conll17,
            ["ner/conll03"] = ["pretrain/fasttextwiki", "forward_charlm/newswiki", "backward_charlm/newswiki"],
        },
    };

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
            ["sentiment/sstplus_electra-large"] = Electra, // with its LoRA adapters: a pass of its own
        },
        ["de"] = [],
    };

    private const string Electra = "google/electra-large-discriminator";

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
    private readonly TransformerEmbedder? _transformer; // the one transformer the models read (Transformers), if any
    private readonly Pretrain[] _pretrains = [];
    private readonly BackendModels _models;
    private readonly CharlmCacheOptions _cacheOptions;
    private readonly bool _splitSentences, _trimNativeHeap, _depparseReadsCache, _identityLemma;

    // Each processor's forward charlm (Dependencies), or none: a call's CharlmCache serves the tagger's pair only.
    private readonly Dictionary<string, string?> _charlmOf = [];
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

        // Every pretrain the models read (German NER has its own), and every charlm (each pair once: default's 1billion,
        // and mimic's for ner i2b2).
        var shared = SharedModels(models, options.Language);
        var pretrains = shared.Where(p => p.StartsWith("pretrain/", StringComparison.Ordinal))
            .ToDictionary(p => p, p => Start(p, () => backend.Pretrain(Path.Combine(modelDir, p))));
        var charlms = shared.Where(p => !p.StartsWith("pretrain/", StringComparison.Ordinal))
            .Select(p => Start(p, () =>
            {
                backend.Charlm(p, Path.Combine(modelDir, p));
                return p;
            }))
            .ToList();

        // The transformer, with the shared models: its readers need it to load. One per pipeline (only ELECTRA-large so far).
        var transformerName = TransformersOf(models, options.Language).SingleOrDefault();
        var transformer = transformerName == null
            ? null
            : Start($"transformer/{transformerName}", () => backend.Transformer(ModelDownloader.TransformerDirectory(modelDir, transformerName), transformerName));
        TransformerEmbedder? TransformerOf(string processor) =>
            Transformers[options.Language].ContainsKey(Name(processor)) ? transformer!.Result : null;

        // The pretrain and the charlms a processor's model reads (Dependencies), once loaded.
        string[] DependenciesOf(string processor) => Dependencies[options.Language].GetValueOrDefault(Name(processor), []);
        Pretrain PretrainOf(string processor) => pretrains[DependenciesOf(processor).Single(p => p.StartsWith("pretrain/", StringComparison.Ordinal))].Result;
        Pretrain? PretrainIfAny(string processor) => DependenciesOf(processor).Any(p => p.StartsWith("pretrain/", StringComparison.Ordinal)) ? PretrainOf(processor) : null;
        Charlms CharlmsOf(string processor) => new(
            DependenciesOf(processor).SingleOrDefault(p => p.StartsWith("forward_charlm/", StringComparison.Ordinal)),
            DependenciesOf(processor).SingleOrDefault(p => p.StartsWith("backward_charlm/", StringComparison.Ordinal)));
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
            transformer?.GetAwaiter().GetResult();
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
            Task.WaitAll(started);
            _tokenizer = tokenizer!.Result;
            _mwt = mwt?.Result;
            _lemma = lemma?.Result;
            _pos = pos?.Result;
            _depparse = depparse?.Result;
            _ner = ner?.Result;
            _parser = parser?.Result;
            _sentiment = sentiment?.Result;
            _pretrains = [.. pretrains.Values.Select(p => p.Result)];
            _transformer = transformer?.Result;
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
    /// <exception cref="ArgumentOutOfRangeException"><see cref="PipelineOptions.Threads"/> below 1, or a
    /// non-positive <see cref="CharlmCacheOptions.MaxWords"/>.</exception>
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
        var models = SelectModels(options.Package, options.Processors, addRequired: false, nameof(options), options.Language, options.Models, options.Pretagged);
        if (options.CharlmCache.IsEnabled && options.CharlmCache.MaxWords <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "CharlmCache.MaxWords must be positive; set IsEnabled = false to turn the cache off");
        if (options.Threads < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Threads must be at least 1, or null for the default");
        if (options.MaxConcurrentCalls < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxConcurrentCalls must be at least 1, or null for no limit");
        ThrowIfMissing(modelDir, models, options);
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
            string.Join(",", AllProcessors.Split(',').Where(models.ContainsKey).Select(p => options.Models == null ? p : $"{p}={models[p]}")),
            Stopwatch.GetElapsedTime(start).TotalMilliseconds);
        return pipeline;
    }

    // A folder downloaded for fewer processors (or before tokenize brought mwt along, 1.x) lacks some files: say which
    // and how to fetch them, instead of the FileNotFoundException of whichever model reads it first.
    private static void ThrowIfMissing(string modelDir, Dictionary<string, string> models, PipelineOptions options)
    {
        var missing = ModelFiles(models, options.Language)
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
        IReadOnlyDictionary<string, string>? chosen = null, bool pretagged = false)
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

        // Stanza's add_mwt: a replaced tokenizer brings the mwt of its own treebank, if there is one.
        if (chosen != null && overrides.ContainsKey("tokenize") && !overrides.ContainsKey("mwt") && models.ContainsKey("mwt")
            && MwtFor(chosen.First(kv => kv.Key.Trim().Equals("tokenize", StringComparison.OrdinalIgnoreCase)).Value.Trim().ToLowerInvariant(), language!) is { } mwt)
        {
            models["mwt"] = mwt;
        }

        var set = processors == null ? models.Keys.ToHashSet() : ParseProcessors(processors, paramName);
        if (processors != null && overrides.Keys.FirstOrDefault(p => !set.Contains(p)) is { } unlisted)
            throw new ArgumentException($"Models names {unlisted}, which Processors ({processors}) doesn't list", paramName);

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
                var others = packages.Where(kv => kv.Value.ContainsKey(p)).Select(kv => $"\"{kv.Key}\"");
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
                throw new ArgumentException($"Unknown processor '{key}' in Models. Available: {AllProcessors}", paramName);
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
        // A _nocharlm tagger (default_fast) has no charlm outputs to share. The cache holds the tagger's pair only: a
        // processor reading another pair (ner i2b2's mimic next to default's 1billion) gets none, so nothing crosses pairs.
        string? tagger = _pos is { UsesCharlm: true } ? _charlmOf["pos"] : null;
        bool Shares(string processor) => tagger != null && _charlmOf.GetValueOrDefault(processor) == tagger;
        using var charlms = tagger != null && (Shares("ner") || Shares("constituency") || Shares("sentiment") || (Shares("depparse") && _depparseReadsCache))
            && _cacheOptions.IsEnabled ? new CharlmCache(_cacheOptions.MaxWords) : null;
        CharlmCache? CacheFor(string processor) => Shares(processor) ? charlms : null;

        // The transformer states, for the processors after the first that reads them (Stanza runs it once per processor).
        int transformerReaders = (_pos?.UsesTransformer == true ? 1 : 0) + (_parser?.UsesTransformer == true ? 1 : 0) + (_depparse?.UsesTransformer == true ? 1 : 0)
            + (_ner?.UsesTransformer == true ? 1 : 0);
        var transformers = transformerReaders > 1 ? new TransformerCache(_transformer!.Net) : null;
        if (_pos != null)
            Step("pos", () => _pos.Process(doc, charlms, ct, transformers), ct);
        if (_lemma != null)
            Step("lemma", () => _lemma.Process(doc, ct), ct);
        else if (_identityLemma)
            Step("lemma", () => Lemmatizer.Identity(doc), ct);

        // Stanza's order (PIPELINE_NAMES). Only the CoNLL-U comment order shows it, and Conllu.Write fixes that.
        if (_parser != null)
            Step("constituency", () => _parser.Process(doc, CacheFor("constituency"), ct, transformers), ct);

        // Depparse runs the charlms with a ROOT word in front, so of the tagger's outputs only the backward ones apply, with
        // ROOT run on from the tagger's final state (managed; DependencyParser.Process).
        if (_depparse != null)
            Step("depparse", () => _depparse.Process(doc, CacheFor("depparse"), ct, transformers), ct);
        if (_sentiment != null)
            Step("sentiment", () => _sentiment.Process(doc, CacheFor("sentiment"), ct), ct);
        if (_ner != null)
            Step("ner", () => _ner.Process(doc, CacheFor("ner"), ct, transformers), ct);
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
        _transformer?.Net.Dispose();
        _models.Dispose();
        foreach (var pretrain in _pretrains)
            pretrain.Dispose();
        _gate?.Dispose();
    }
}
