namespace StanzaSharp;

/// <summary>
/// The implementation that runs the models' networks, chosen with <see cref="PipelineOptions.Backend"/>. All of them
/// give the same tags, lemmas, parses, entities and sentiment.
/// </summary>
/// <remarks>
/// This package has <see cref="Managed"/>. The <c>StanzaSharp.Cuda</c> package adds TorchSharp (libtorch) on a GPU,
/// <c>CudaBackend.Create()</c>, or on the CPU, <c>CudaBackend.Cpu</c>.
/// </remarks>
/// <example>
/// <code>
/// using var nlp = Pipeline.Load("models/stanza/en"); // PipelineBackend.Managed
/// using var gpu = Pipeline.Load("models/stanza/en", new PipelineOptions { Backend = CudaBackend.Create() }); // StanzaSharp.Cuda
/// </code>
/// </example>
public sealed class PipelineBackend
{
    // Other packages (StanzaSharp.Cuda) make their backends through InternalsVisibleTo, so this package's public API has
    // no TorchSharp type and this package references none of theirs.
    internal PipelineBackend(string name, Func<BackendModels> models)
    {
        _name = name;
        Models = models;
    }

    private readonly string _name;

    /// <summary>
    /// The default: C# kernels on the CPU (SIMD where the CPU has it), with no native dependencies, so it runs wherever
    /// .NET runs. It uses <see cref="PipelineOptions.Threads"/> threads of one process-wide pool.
    /// </summary>
    public static PipelineBackend Managed { get; } = new(nameof(Managed), () => new ManagedModels());

    /// <summary>Makes the models' loader for one <see cref="Pipeline.Load"/>.</summary>
    internal Func<BackendModels> Models { get; }

    /// <summary>The backend's name, e.g. <c>Managed</c>.</summary>
    public override string ToString() => _name;
}
