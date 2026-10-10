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

    /// <summary>Whether the dependency parser reads the tagger's charlm outputs (CharlmCache.TryGetBackwardState).</summary>
    public virtual bool DepparseReadsCache => false;

    public abstract Pretrain Pretrain(string basePath);

    /// <summary>Loads a charlm the processors below then read; the forward one first.</summary>
    public abstract void Charlm(string basePath, bool forward);

    public abstract Tokenizer Tokenizer(string basePath);
    public abstract MwtExpander Mwt(string basePath);
    public abstract Lemmatizer Lemma(string basePath);
    public abstract PosTagger Pos(string basePath, Pretrain pretrain);
    public abstract DependencyParser Depparse(string basePath, Pretrain pretrain);
    public abstract NerTagger Ner(string basePath, Pretrain pretrain);
    public abstract ConstituencyParser Constituency(string basePath, Pretrain pretrain);
    public abstract SentimentClassifier Sentiment(string basePath, Pretrain pretrain);

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
    private ManagedCharLanguageModel? _forward, _backward;

    public override void Configure(PipelineOptions options)
    {
        // The same semantics as libtorch's threads: null caps the pool's current count (ProcessorCount unless set lower).
        ManagedThreads.Count = options.Threads ?? Math.Min(ManagedThreads.Count, Environment.ProcessorCount);
        options.Logger?.LogInformation("Using {Threads} managed threads", ManagedThreads.Count);
    }

    // Each weight is read into an array, then packed into another: about 1.4 GB allocated for 630 MB kept (default
    // package). Collecting after each model lets the next one reuse that memory: load peak 1,035 → ~850 MB for about
    // 0.05 s (docs/backends.md).
    public override bool CollectAfterEachModel => true;

    public override bool DepparseReadsCache => true;

    public override Pretrain Pretrain(string basePath) => Nn.Pretrain.LoadManaged(basePath);

    public override void Charlm(string basePath, bool forward)
    {
        var charlm = ManagedCharLanguageModel.Load(basePath);
        if (forward)
            _forward = charlm;
        else
            _backward = charlm;
    }

    public override Tokenizer Tokenizer(string basePath) => Tokenize.Tokenizer.LoadManaged(basePath);
    public override MwtExpander Mwt(string basePath) => MwtExpander.LoadManaged(basePath);
    public override Lemmatizer Lemma(string basePath) => Lemmatizer.LoadManaged(basePath);
    public override PosTagger Pos(string basePath, Pretrain pretrain) => PosTagger.LoadManaged(basePath, pretrain, _forward, _backward);
    public override DependencyParser Depparse(string basePath, Pretrain pretrain) => DependencyParser.LoadManaged(basePath, pretrain, _forward, _backward);
    public override NerTagger Ner(string basePath, Pretrain pretrain) => NerTagger.LoadManaged(basePath, pretrain, _forward, _backward);
    public override ConstituencyParser Constituency(string basePath, Pretrain pretrain) => ConstituencyParser.LoadManaged(basePath, pretrain, _forward!, _backward!);
    public override SentimentClassifier Sentiment(string basePath, Pretrain pretrain) => SentimentClassifier.LoadManaged(basePath, pretrain, _forward!, _backward!);
}
