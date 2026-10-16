# The GC after each model at load

An open idea ([list](../../performance.md#remaining-ideas)).

The managed `Pipeline.Load` runs `GC.Collect()` after each model (`BackendModels.CollectAfterEachModel`). Each weight is
read into an array and then packed into another, so a model leaves its unpacked arrays behind, and collecting lets the
next model reuse that memory (load peak 1,035 → ~850 MB). [Round 34](../round-34-load-packed-cache.md) measured these
GCs at 130-270 ms of thread time, about 9% of the load. Skipping them on a packed-cache hit saved 15-80 ms of wall time
and raised the peak by 50-180 MB.

If the weights that only feed packing were read into rented buffers, the GCs could go without raising the peak. For
example, `Checkpoint.Tensor` could take an `ArrayPool` lease, or a `PackedMatrix` could be packed straight from the
file's bytes. Round 34's prototype found those tensors by weak reference: tensors of 256 KB or more that are dead
after the load.
