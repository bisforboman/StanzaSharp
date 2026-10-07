namespace StanzaSharp.Nn;

/// <summary>
/// Which implementation runs a processor's network (issue #29; docs/backends.md). The seam is per processor: each
/// ported processor has a small network interface (e.g. <c>ITokenizerNet</c>) with array inputs and outputs, a
/// TorchSharp implementation (today's code) and a managed one built from <c>Nn.Managed</c>. Batching, decoding and
/// everything else stays shared, so both backends see exactly the same batches.
/// </summary>
internal enum Backend
{
    /// <summary>TorchSharp/libtorch, on <see cref="Weights.Device"/>: the default, and the test reference.</summary>
    TorchSharp,

    /// <summary>The managed kernels (<c>Nn.Managed</c>), CPU only, on <c>ManagedThreads.Count</c> threads. Processors
    /// not ported yet run on TorchSharp.</summary>
    Managed,
}
