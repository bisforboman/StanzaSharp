namespace StanzaSharp.Nn;

/// <summary>
/// Which implementation runs a processor's network (issue #29; docs/backends.md); the public form is
/// <c>PipelineBackend</c>. The seam is per processor: each processor has a small network interface (e.g.
/// <c>ITokenizerNet</c>) with array inputs and outputs, a TorchSharp implementation and a managed one built from
/// <c>Nn.Managed</c>. Batching, decoding and everything else stays shared, so both backends see exactly the same batches.
/// </summary>
internal enum Backend
{
    /// <summary>TorchSharp/libtorch (StanzaSharp.Cuda), on <c>Weights.Device</c>: the test reference. Processor <c>Load</c>s default to it.</summary>
    TorchSharp,

    /// <summary>The managed kernels (<c>Nn.Managed</c>), CPU only, on <c>ManagedThreads.Count</c> threads: the
    /// pipeline's default.</summary>
    Managed,
}
