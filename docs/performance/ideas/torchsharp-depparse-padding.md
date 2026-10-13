# Padding in depparse's input (TorchSharp only)

An open idea ([list](../../performance.md#remaining-ideas)). Concerns only the `StanzaSharp.Cuda` (TorchSharp) backend.

Depparse builds its input and charlm outputs padded; building
them in packed order like the tagger (round 4) would remove one padded copy of the 2423-wide input per batch.
