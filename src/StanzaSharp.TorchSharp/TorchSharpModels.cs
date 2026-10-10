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
using TorchSharp;

namespace StanzaSharp;

/// <summary>The TorchSharp backend's models (<see cref="CudaBackend"/>), on <paramref name="device"/> (null: the CPU).</summary>
/// <param name="disableTf32">Sets both torch TF32 switches to false at load, process-wide, and doesn't restore them.</param>
internal sealed class TorchSharpModels(torch.Device? device, bool disableTf32) : BackendModels
{
    private CharLanguageModel? _forward, _backward;

    public override void Configure(PipelineOptions options)
    {
        // libtorch defaults to the host's physical cores; ProcessorCount respects a container's CPU quota.
        int threads = options.Threads ?? Math.Min(torch.get_num_threads(), Environment.ProcessorCount);
        if (threads != torch.get_num_threads())
            torch.set_num_threads(threads);
        options.Logger?.LogInformation("Using {Threads} torch intra-op threads (Environment.ProcessorCount is {ProcessorCount})", threads, Environment.ProcessorCount);
        if (disableTf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = false;
    }

    public override T Load<T>(Func<T> load) => Weights.On(device, load);

    public override Pretrain Pretrain(string basePath) => Nn.Pretrain.Load(basePath);

    public override void Charlm(string basePath, bool forward)
    {
        var charlm = CharLanguageModel.Load(basePath);
        if (forward)
            _forward = charlm;
        else
            _backward = charlm;
    }

    public override Tokenizer Tokenizer(string basePath) => Tokenize.Tokenizer.Load(basePath);
    public override MwtExpander Mwt(string basePath) => MwtExpander.Load(basePath);
    public override Lemmatizer Lemma(string basePath) => Lemmatizer.Load(basePath);
    public override PosTagger Pos(string basePath, Pretrain pretrain) => PosTagger.Load(basePath, pretrain, _forward, _backward);
    public override DependencyParser Depparse(string basePath, Pretrain pretrain) => DependencyParser.Load(basePath, pretrain, _forward, _backward);
    public override NerTagger Ner(string basePath, Pretrain pretrain) => NerTagger.Load(basePath, pretrain, _forward, _backward);
    public override ConstituencyParser Constituency(string basePath, Pretrain pretrain) => ConstituencyParser.Load(basePath, pretrain, _forward!, _backward!);
    public override SentimentClassifier Sentiment(string basePath, Pretrain pretrain) => SentimentClassifier.Load(basePath, pretrain, _forward!, _backward!);

    public override void Dispose()
    {
        _forward?.Dispose();
        _backward?.Dispose();
    }
}
