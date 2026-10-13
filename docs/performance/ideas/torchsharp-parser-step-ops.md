# Parser step loop: per-state tensor ops (TorchSharp only)

An open idea ([list](../../performance.md#remaining-ideas)). Concerns only the `StanzaSharp.Cuda` (TorchSharp) backend.

Each step takes one `WordHx` row view per state and
stacks per-state tensors; a shared "bank" tensor per stack (`index_select` / `index_copy_`) would make the op count
per step constant. Exact (data movement only); about 2% on the CPU, likely more on a GPU (docs/gpu.md).
