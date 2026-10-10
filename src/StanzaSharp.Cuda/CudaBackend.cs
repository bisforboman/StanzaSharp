using TorchSharp;

namespace StanzaSharp;

/// <summary>
/// The TorchSharp (libtorch) backends for <see cref="PipelineOptions.Backend"/>: on an NVIDIA GPU, or on the CPU. They
/// need the native libtorch next to this package: <c>TorchSharp-cuda-windows</c> or <c>TorchSharp-cuda-linux</c> for the
/// GPU, <c>TorchSharp-cpu</c> for the CPU, with the TorchSharp version this package was built with.
/// </summary>
/// <example>
/// <code>
/// using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions { Backend = CudaBackend.Create(disableTf32: true) });
/// </code>
/// </example>
public static class CudaBackend
{
    /// <summary>
    /// TorchSharp on CUDA device <paramref name="deviceIndex"/>. Output identical to Python Stanza on the CPU needs
    /// <paramref name="disableTf32"/>; with TF32 on (the default) matrix products round to 10 mantissa bits, so tags and
    /// parses can differ in rare near-ties.
    /// </summary>
    /// <param name="deviceIndex">The CUDA device's index; 0 is the first GPU.</param>
    /// <param name="disableTf32">Turns off TF32 for matrix multiplications and cuDNN. <b>Process-wide:</b>
    /// <see cref="Pipeline.Load"/> sets <c>torch.backends.cuda.matmul.allow_tf32</c> and
    /// <c>torch.backends.cudnn.allow_tf32</c> to false for all TorchSharp code in the process and doesn't restore them.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="deviceIndex"/> is negative.</exception>
    public static PipelineBackend Create(int deviceIndex = 0, bool disableTf32 = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        // The device is made at Load: torch.device would load libtorch here already.
        return new PipelineBackend($"Cuda:{deviceIndex}", () => new TorchSharpModels(torch.device(DeviceType.CUDA, deviceIndex), disableTf32));
    }

    /// <summary>
    /// TorchSharp on the CPU, with <c>TorchSharp-cpu</c>: the backend StanzaSharp used up to 0.4. The default
    /// <see cref="PipelineBackend.Managed"/> gives the same output without a native dependency and is usually faster.
    /// </summary>
    public static PipelineBackend Cpu { get; } = new("TorchSharp", () => new TorchSharpModels(device: null, disableTf32: false));
}
