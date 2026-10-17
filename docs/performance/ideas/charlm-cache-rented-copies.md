# The tagger's CharlmCache copies, rented

An open idea ([list](../../performance.md#remaining-ideas)).

The largest garbage a long call still makes ([round 38](../round-38-long-text-garbage-left.md)): the tagger copies each
cached sentence's forward and backward charlm outputs into new arrays, 8 KB per word up to `CharlmCache.MaxWords`
(32,768), so about 256 MB per call once a text has that many words, most of it on the large-object heap. Renting the
copies from `ArrayPool<float>.Shared` and giving them back when the call's cache is dropped would let a second call
reuse them.

The catch is a memory decision for the owner: the pool would then hold those ~256 MB after the call until it trims
them (a gen2 GC a minute or more later), where today they are garbage the next gen2 GC frees. The pool also keeps at
most 513 arrays per size, fewer than a long text's sentences, so part of it would still be allocated.
