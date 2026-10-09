using StanzaSharp.Nn;
using TorchSharp;

namespace StanzaSharp;

/// <summary>
/// The implementation that runs the models' networks, chosen with <see cref="PipelineOptions.Backend"/>. Both give
/// the same tags, lemmas, parses, entities and sentiment.
/// </summary>
/// <example>
/// <code>
/// using var nlp = Pipeline.Load("models/stanza/en"); // PipelineBackend.Managed
/// using var torch = Pipeline.Load("models/stanza/en", new PipelineOptions { Backend = PipelineBackend.TorchSharp });
/// </code>
/// </example>
public sealed class PipelineBackend
{
    // Device and TF32 travel inside the backend, so a GPU backend (the planned StanzaSharp.Cuda package, through
    // InternalsVisibleTo) needs no TorchSharp type in this package's public API.
    internal PipelineBackend(Backend kind, string name, torch.Device? device = null, bool disableTf32 = false)
    {
        Kind = kind;
        _name = name;
        Device = device;
        DisableTf32 = disableTf32;
    }

    private readonly string _name;

    /// <summary>
    /// The default: C# kernels on the CPU (SIMD where the CPU has it), with no native dependencies, so it runs wherever
    /// .NET runs. It uses <see cref="PipelineOptions.Threads"/> threads of one process-wide pool.
    /// </summary>
    public static PipelineBackend Managed { get; } = new(Backend.Managed, nameof(Managed));

    /// <summary>
    /// TorchSharp (libtorch) on the CPU. It needs a native libtorch package next to StanzaSharp, e.g. <c>TorchSharp-cpu</c>
    /// with the same versions StanzaSharp was built with. It will leave the main package in 1.0.
    /// </summary>
    public static PipelineBackend TorchSharp { get; } = new(Backend.TorchSharp, nameof(TorchSharp));

    internal Backend Kind { get; }

    /// <summary>The TorchSharp device; null is the CPU.</summary>
    internal torch.Device? Device { get; }

    internal bool DisableTf32 { get; }

    /// <summary>The backend's name: <c>Managed</c> or <c>TorchSharp</c>.</summary>
    public override string ToString() => _name;
}
