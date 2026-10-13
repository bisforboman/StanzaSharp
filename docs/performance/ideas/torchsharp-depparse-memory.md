# Depparse memory (TorchSharp only)

An open idea ([list](../../performance.md#remaining-ideas)). Concerns only the `StanzaSharp.Cuda` (TorchSharp) backend.

The [batch, width, width, 53] label scores (275 MB on the warm-up batch) set
the peak there. The managed backend never had that tensor (round 9).
