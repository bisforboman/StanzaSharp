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

namespace StanzaSharp;

/// <summary>
/// One pipeline's models on one backend (docs/backends.md, the StanzaSharp.Cuda split): the backend's settings, the
/// shared pretrain and charlms, and each processor. <see cref="PipelineBackend"/> makes one per <see cref="Pipeline.Load"/>;
/// the TorchSharp one lives in StanzaSharp.Cuda (the StanzaSharp.Cuda package), which this package never references.
/// </summary>
internal abstract class BackendModels : IDisposable
{
    /// <summary>Applies <see cref="PipelineOptions.Threads"/> (and any other process-wide setting) before the models load.</summary>
    public abstract void Configure(PipelineOptions options);

    /// <summary>Runs the whole load (TorchSharp: on the backend's device).</summary>
    public virtual T Load<T>(Func<T> load) => load();

    /// <summary>Whether to run a GC after each model: the managed models allocate far more while loading than they keep.</summary>
    public virtual bool CollectAfterEachModel => false;

    /// <summary>Whether the models may load concurrently, on the thread pool (managed: nothing is per thread).</summary>
    public virtual bool LoadInParallel => false;

    /// <summary>Whether the dependency parser reads the tagger's backward charlm outputs (CharlmCache.TryGetBackward).</summary>
    public virtual bool DepparseReadsCache => false;

    public abstract Pretrain Pretrain(string basePath);

    /// <summary>
    /// Loads the charlm at <paramref name="basePath"/> as <paramref name="path"/> (e.g. <c>forward_charlm/1billion</c>), the
    /// name a processor's <see cref="Charlms"/> give: a pipeline loads every charlm its models read (Pipeline.Dependencies).
    /// </summary>
    public abstract void Charlm(string path, string basePath);

    /// <param name="charlms">The forward charlm a <c>_charlm</c> tokenizer reads; both null for the others.</param>
    public abstract Tokenizer Tokenizer(string basePath, Charlms charlms);

    public abstract MwtExpander Mwt(string basePath);

    /// <param name="charlms">The charlms a <c>_charlm</c> lemmatizer reads; both null for the others.</param>
    /// <param name="pretrain">The pretrain its contextual classifiers read (English <c>combined_charlm</c>); else null.</param>
    public abstract Lemmatizer Lemma(string basePath, Charlms charlms, Pretrain? pretrain);

    /// <summary>
    /// Loads the transformer <paramref name="name"/> (a Hugging Face repository) from its <paramref name="directory"/>; the
    /// pipeline owns it and gives it to every model that reads it (<see cref="Pipeline.Transformers"/>).
    /// </summary>
    public abstract TransformerEmbedder Transformer(string directory, string name);

    /// <param name="transformer">The transformer a <c>_electra-large</c> model reads; null for the others.</param>
    public abstract PosTagger Pos(string basePath, Pretrain pretrain, Charlms charlms, TransformerEmbedder? transformer);

    /// <param name="transformer">The transformer a <c>_electra-large</c> model reads; null for the others.</param>
    public abstract DependencyParser Depparse(string basePath, Pretrain pretrain, Charlms charlms, TransformerEmbedder? transformer);

    /// <param name="transformer">The transformer a <c>_electra-large</c> model reads; null for the others.</param>
    public abstract NerTagger Ner(string basePath, Pretrain pretrain, Charlms charlms, TransformerEmbedder? transformer);

    public abstract ConstituencyParser Constituency(string basePath, Pretrain pretrain, Charlms charlms);

    public abstract SentimentClassifier Sentiment(string basePath, Pretrain pretrain, Charlms charlms);

    /// <summary>The charlm loaded as <paramref name="path"/> into <paramref name="loaded"/>, or null for none.</summary>
    protected static T? Find<T>(Dictionary<string, T> loaded, string? path) where T : class
    {
        if (path == null)
            return null;
        lock (loaded)
            return loaded.TryGetValue(path, out var charlm) ? charlm : throw new InvalidOperationException($"Charlm {path} was not loaded");
    }

    /// <summary>Adds a loaded charlm (the managed models load in parallel).</summary>
    protected static void Add<T>(Dictionary<string, T> loaded, string path, T charlm)
    {
        lock (loaded)
            loaded.Add(path, charlm);
    }

    /// <summary>Frees the charlms.</summary>
    public virtual void Dispose()
    {
    }
}

/// <summary>
/// <see cref="PipelineBackend.Managed"/>'s models. Nothing here may touch TorchSharp: its first call loads native
/// libtorch, which the managed backend doesn't need (ManagedCheckTests).
/// </summary>
internal sealed class ManagedModels : BackendModels
{
    private readonly Dictionary<string, ManagedCharLanguageModel> _charlms = [];

    public override void Configure(PipelineOptions options)
    {
        // Like libtorch's threads: null caps the pool's current count (the default unless set lower) at the default.
        ManagedThreads.Count = options.Threads ?? Math.Min(ManagedThreads.Count, ManagedThreads.Default);
        options.Logger?.LogInformation("Using {Threads} managed threads", ManagedThreads.Count);
    }

    // Each weight is read into an array, then packed into another: about 1.4 GB allocated for 630 MB kept (default
    // package). Collecting after each model lets the next one reuse that memory: load peak 1,035 → ~850 MB for about
    // 0.05 s (docs/backends.md).
    public override bool CollectAfterEachModel => true;

    public override bool DepparseReadsCache => true;

    // Each model's load is mostly one thread's work (parsing, packing, the JIT), so loading two side by side cuts the
    // load time (Pipeline, docs/performance/round-17-startup.md).
    public override bool LoadInParallel => true;

    public override Pretrain Pretrain(string basePath) => Nn.Pretrain.LoadManaged(basePath);

    public override void Charlm(string path, string basePath) => Add(_charlms, path, ManagedCharLanguageModel.Load(basePath));

    public override Tokenizer Tokenizer(string basePath, Charlms charlms) => Tokenize.Tokenizer.LoadManaged(basePath, Find(_charlms, charlms.Forward));

    public override MwtExpander Mwt(string basePath) => MwtExpander.LoadManaged(basePath);

    public override Lemmatizer Lemma(string basePath, Charlms charlms, Pretrain? pretrain) =>
        Lemmatizer.LoadManaged(basePath, Find(_charlms, charlms.Forward), Find(_charlms, charlms.Backward), pretrain);

    public override TransformerEmbedder Transformer(string directory, string name) =>
        TransformerEmbedder.Load(directory, name, (config, ckpt) => ManagedTransformer.Load(config, ckpt));

    public override PosTagger Pos(string basePath, Pretrain pretrain, Charlms charlms, TransformerEmbedder? transformer) =>
        PosTagger.LoadManaged(basePath, pretrain, Find(_charlms, charlms.Forward), Find(_charlms, charlms.Backward), transformer);

    public override DependencyParser Depparse(string basePath, Pretrain pretrain, Charlms charlms, TransformerEmbedder? transformer) =>
        DependencyParser.LoadManaged(basePath, pretrain, Find(_charlms, charlms.Forward), Find(_charlms, charlms.Backward), transformer);

    public override NerTagger Ner(string basePath, Pretrain pretrain, Charlms charlms, TransformerEmbedder? transformer) =>
        NerTagger.LoadManaged(basePath, pretrain, Find(_charlms, charlms.Forward), Find(_charlms, charlms.Backward), transformer);

    public override ConstituencyParser Constituency(string basePath, Pretrain pretrain, Charlms charlms) =>
        ConstituencyParser.LoadManaged(basePath, pretrain, Find(_charlms, charlms.Forward)!, Find(_charlms, charlms.Backward)!);

    public override SentimentClassifier Sentiment(string basePath, Pretrain pretrain, Charlms charlms) =>
        SentimentClassifier.LoadManaged(basePath, pretrain, Find(_charlms, charlms.Forward)!, Find(_charlms, charlms.Backward)!);
}

/// <summary>
/// The pair of charlms a processor's model reads, by their Pipeline.Dependencies names (e.g. <c>forward_charlm/1billion</c>);
/// both null for a model that reads none (<c>_nocharlm</c>).
/// </summary>
internal readonly record struct Charlms(string? Forward, string? Backward);
