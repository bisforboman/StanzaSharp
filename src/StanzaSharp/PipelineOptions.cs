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
///     Device = torch.CUDA,
///     DisableTf32 = true,
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
    /// Where the models run: the CPU by default, or e.g. <c>torch.CUDA</c> with a <c>TorchSharp-cuda-*</c>
    /// package. Output identical to Python Stanza on CPU needs the CPU, or a GPU with <see cref="DisableTf32"/>.
    /// </summary>
    public torch.Device? Device { get; init; }

    /// <summary>
    /// Turns off TF32 for matrix multiplications and cuDNN, so GPU output matches the CPU. Off by default.
    /// <b>Process-wide:</b> it sets <c>torch.backends.cuda.matmul.allow_tf32</c> and
    /// <c>torch.backends.cudnn.allow_tf32</c> to false for all TorchSharp code in the process, and does
    /// not restore them.
    /// </summary>
    public bool DisableTf32 { get; init; }

    /// <summary>Sharing of character-model outputs between the tagger, the constituency parser and sentiment.</summary>
    public CharlmCacheOptions CharlmCache { get; init; } = new();

    /// <summary>
    /// The number of threads libtorch uses inside each operation (intra-op threads), set by <see cref="Pipeline.Load"/>.
    /// Null (the default) caps libtorch's current setting at <see cref="Environment.ProcessorCount"/>, which .NET limits
    /// to a container's CPU quota. libtorch's own default is the host's physical core count, so in a container limited
    /// to 2 CPUs on a 32-core node it would start 32 threads per operation. Outside containers the default leaves
    /// libtorch's setting alone, and a lower count set earlier with <c>torch.set_num_threads</c> is kept.
    /// </summary>
    /// <remarks>
    /// <b>Process-wide:</b> this is <c>torch.set_num_threads</c>, which applies to all TorchSharp code in the process,
    /// and the last pipeline loaded wins. libtorch applies a new count to the loading thread and to threads that haven't
    /// run a tensor operation yet; a thread that already has keeps its count, so load before processing starts. Each thread that calls <see cref="Pipeline.Process(string)"/> runs its own
    /// operations on up to this many threads, so with N concurrent calls consider <c>ProcessorCount / N</c>.
    /// libtorch's inter-op thread pool is not used by StanzaSharp and is left alone.
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">Thrown by <see cref="Pipeline.Load"/> for a value below 1.</exception>
    public int? Threads { get; init; }

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
