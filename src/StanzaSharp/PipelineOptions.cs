using Microsoft.Extensions.Logging;
using StanzaSharp.Nn;
using TorchSharp;

namespace StanzaSharp;

/// <summary>Settings for <see cref="Pipeline.Load(string, PipelineOptions?)"/>.</summary>
/// <example>
/// <code>
/// using var fast = Pipeline.Load("models/stanza/en", new PipelineOptions { Package = "default_fast" });
/// using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions
/// {
///     Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos}", // or "tokenize,mwt,pos"
///     Threads = 4,
///     Backend = PipelineBackend.TorchSharp, // needs TorchSharp-cpu; the default is PipelineBackend.Managed
///     CharlmCache = new() { MaxWords = 100_000 },
/// });
/// </code>
/// </example>
public sealed class PipelineOptions
{
    /// <summary>
    /// Stanza's name for the set of English models to use: <c>"default"</c> (<see cref="Pipeline.DefaultPackage"/>,
    /// all eight processors) or <c>"default_fast"</c> (no constituency; pos, depparse and ner use their own small
    /// character models instead of the large character language models, which makes them faster and smaller).
    /// Any other name throws.
    /// </summary>
    public string Package { get; init; } = Pipeline.DefaultPackage;

    /// <summary>
    /// Comma-separated subset of <see cref="Pipeline.AllProcessors"/>; each needs the ones before it. Null (the
    /// default) runs every processor of the <see cref="Package"/>. A processor the package lacks throws.
    /// </summary>
    public string? Processors { get; init; }

    /// <summary>
    /// The implementation that runs the models: <see cref="PipelineBackend.Managed"/> (the default; no native
    /// dependencies) or <see cref="PipelineBackend.TorchSharp"/> (needs a native libtorch package such as
    /// <c>TorchSharp-cpu</c>). Output is the same on both. Left unset while the obsolete <see cref="Device"/> or
    /// <see cref="DisableTf32"/> is set, it is <see cref="PipelineBackend.TorchSharp"/>.
    /// </summary>
    /// <exception cref="ArgumentNullException">Set to null.</exception>
    public PipelineBackend Backend
    {
        get => _backend ?? (UsesTorchSharpDevice ? PipelineBackend.TorchSharp : PipelineBackend.Managed);
        init => _backend = value ?? throw new ArgumentNullException(nameof(value));
    }

    private readonly PipelineBackend? _backend;

    /// <summary>Whether <see cref="Backend"/> was set rather than defaulted.</summary>
    internal bool BackendIsSet => _backend is not null;

#pragma warning disable CS0618 // the obsolete options are read here and in Pipeline.Load only
    /// <summary>Whether the obsolete <see cref="Device"/> or <see cref="DisableTf32"/> is set.</summary>
    internal bool UsesTorchSharpDevice => Device is not null || DisableTf32;
#pragma warning restore CS0618

    /// <summary>
    /// Runs the models on TorchSharp on this device, e.g. <c>torch.CUDA</c> with a <c>TorchSharp-cuda-*</c> package.
    /// Setting it selects <see cref="PipelineBackend.TorchSharp"/>; with <see cref="Backend"/> set to
    /// <see cref="PipelineBackend.Managed"/>, <see cref="Pipeline.Load"/> throws an <see cref="ArgumentException"/>.
    /// Output identical to Python Stanza needs the CPU, or a GPU with <see cref="DisableTf32"/>.
    /// </summary>
    [Obsolete(TorchSharpOnly)]
    public torch.Device? Device { get; init; }

    /// <summary>
    /// Turns off TF32 for matrix multiplications and cuDNN, so GPU output matches the CPU. Off by default. Setting it
    /// selects <see cref="PipelineBackend.TorchSharp"/>, like <see cref="Device"/>.
    /// <b>Process-wide:</b> it sets <c>torch.backends.cuda.matmul.allow_tf32</c> and
    /// <c>torch.backends.cudnn.allow_tf32</c> to false for all TorchSharp code in the process, and does
    /// not restore them.
    /// </summary>
    [Obsolete(TorchSharpOnly)]
    public bool DisableTf32 { get; init; }

    private const string TorchSharpOnly = "Device and DisableTf32 apply only to the TorchSharp backend, and setting either "
        + "selects it: use Backend = PipelineBackend.TorchSharp. GPU support moves to the StanzaSharp.Cuda package; "
        + "both options leave the main package in 1.0.";

    /// <summary>Sharing of character-model outputs between the tagger, the constituency parser and sentiment.</summary>
    public CharlmCacheOptions CharlmCache { get; init; } = new();

    /// <summary>
    /// The number of threads each operation runs on, set by <see cref="Pipeline.Load"/>. On
    /// <see cref="PipelineBackend.Managed"/> it is the size of StanzaSharp's own thread pool; on
    /// <see cref="PipelineBackend.TorchSharp"/> it is libtorch's intra-op thread count (<c>torch.set_num_threads</c>).
    /// Null (the default) caps the current count at <see cref="Environment.ProcessorCount"/>, which .NET limits to a
    /// container's CPU quota, and keeps a lower count set earlier.
    /// </summary>
    /// <remarks>
    /// <b>Process-wide</b> on both backends, and the last pipeline loaded wins.
    /// <list type="bullet">
    /// <item><b>Managed:</b> one pool for the whole process. Concurrent <see cref="Pipeline.Process(string)"/> calls share
    /// it: each works its own part and idle threads help, so N callers don't start N teams.</item>
    /// <item><b>TorchSharp:</b> libtorch's own default is the host's physical core count, so in a container limited to 2
    /// CPUs on a 32-core node it would start 32 threads per operation. libtorch applies a new count to the loading
    /// thread and to threads that haven't run a tensor operation yet; a thread that already has keeps its count, so
    /// load before processing starts. Each thread that calls Process runs its own operations on up to this many
    /// threads, so with N concurrent calls consider <c>ProcessorCount / N</c>. libtorch's inter-op thread pool is not
    /// used by StanzaSharp and is left alone.</item>
    /// </list>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown by <see cref="Pipeline.Load"/> for a value below 1.</exception>
    public int? Threads { get; init; }

    /// <summary>
    /// At most this many <see cref="Pipeline.Process(string)"/> calls run on this pipeline at once; more callers wait
    /// for a turn (cancellable through the call's token). Null, the default, means no limit.
    /// </summary>
    /// <remarks>
    /// Output never depends on it. On <see cref="PipelineBackend.Managed"/> concurrent calls evict each other's model
    /// weights from the CPU cache, so beyond about two calls in flight throughput falls (default package, Ryzen 7 5800X,
    /// one sentence per call: 1 caller 29.6 calls/s, 2 callers 36.0, 8 callers 23.6; 8 callers limited to one call at a
    /// time 36.4). A service with many concurrent requests on one pipeline can set 2. Bulk input
    /// (<see cref="Pipeline.Process(IEnumerable{string})"/>) is faster still when requests can be grouped.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown by <see cref="Pipeline.Load"/> for a value below 1.</exception>
    public int? MaxConcurrentCalls { get; init; }

    /// <summary>
    /// Whether the tokenizer splits paragraphs into sentences (true, the default, as Stanza does). False is Stanza's
    /// <c>tokenize_no_ssplit</c>: each paragraph, i.e. text between blank lines, becomes one sentence, for input that
    /// already has one sentence per paragraph. Tokens and multi-word tokens are the same either way. Pretokenized input
    /// (<see cref="Pipeline.Process(IEnumerable{IEnumerable{string}})"/>) keeps its given sentences regardless, as in Stanza.
    /// </summary>
    public bool SplitSentences { get; init; } = true;

    /// <summary>
    /// Checks every model file <see cref="Pipeline.Load"/> reads against the MD5 that Stanza publishes for it (the
    /// checksums <see cref="ModelDownloader"/> verifies downloads with) before loading, which catches corrupted files
    /// or files from another Stanza version. It reads every file once more, which takes a second or two for the
    /// <c>default</c> package. Off by default.
    /// </summary>
    /// <remarks>
    /// Only Stanza's <c>.pt</c> files have published checksums. If a model would be read from converted
    /// <c>.json</c> + <c>.safetensors</c> files instead, Load throws rather than skip the check.
    /// </remarks>
    public bool VerifyChecksums { get; init; }

    /// <summary>
    /// Receives timings: each model's load time and the thread count at <see cref="LogLevel.Information"/>, and each
    /// processor's time per <see cref="Pipeline.Process(string)"/> call at <see cref="LogLevel.Debug"/>. Null (the
    /// default) logs nothing and costs nothing.
    /// </summary>
    public ILogger? Logger { get; init; }

    /// <summary>
    /// Linux with glibc: return the native heap's free memory to the OS at the end of each Process call
    /// (<c>malloc_trim(0)</c>, see <see cref="NativeHeap"/>). Always on for users (decided: no public API, its cost is
    /// in the noise); internal so the benchmark can measure without it.
    /// </summary>
    internal bool TrimNativeHeap { get; init; } = true;
}

/// <summary>
/// The tagger keeps its character-model outputs (about 8 KB per word) for the constituency parser,
/// which needs the same ones. Output is identical either way; the cache trades memory for speed.
/// </summary>
public sealed class CharlmCacheOptions
{
    /// <summary>Whether the parser reuses the tagger's outputs (true) or computes its own.</summary>
    public bool IsEnabled { get; init; } = true;

    /// <summary>
    /// The most words kept per document. Sentences past it are recomputed by the parser. The default,
    /// 32,768 words, is about 256 MB.
    /// </summary>
    public int MaxWords { get; init; } = Nn.CharlmCache.DefaultMaxWords;
}
