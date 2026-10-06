using StanzaSharp.Nn;
using TorchSharp;

namespace StanzaSharp;

/// <summary>Settings for <see cref="Pipeline.Load(string, PipelineOptions?)"/>.</summary>
/// <example>
/// <code>
/// using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions
/// {
///     Processors = "tokenize,mwt,pos",
///     Device = torch.CUDA,
///     DisableTf32 = true,
///     CharlmCache = new() { MaxWords = 100_000 },
/// });
/// </code>
/// </example>
public sealed class PipelineOptions
{
    /// <summary>Comma-separated subset of <see cref="Pipeline.AllProcessors"/>; each needs the ones before it.</summary>
    public string Processors { get; init; } = Pipeline.AllProcessors;

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

    /// <summary>Sharing of character-model outputs between the tagger and the constituency parser.</summary>
    public CharlmCacheOptions CharlmCache { get; init; } = new();
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
