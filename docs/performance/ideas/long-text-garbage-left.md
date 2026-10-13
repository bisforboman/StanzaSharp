# Long texts: what is left of the garbage

An open idea ([list](../../performance.md#remaining-ideas)).

After round 23, a long call allocates about 8 KB per word once the `ArrayPool` is warm (11 KB on a fresh process's first
call; the difference is the pool's first fill, which stays live and isn't garbage, see
[round 23](../round-23-array-pool-misses.md)). What remains: the tagger's `CharlmCache` copies (by design), and sites
under 0.3 KB/word each (constituency's `Score` output and `Owned` lists, Chu-Liu/Edmonds' per-cycle arrays, the
tokenizer's per-batch net input).

On the peak rather than garbage: `ManagedConstituencyNet.EncodeWords` could fill the word LSTM's packed input directly,
without the padded `[n, width, 2268]` copy, which is exact by construction. From round 23's trace it would save one 64 MB
buffer on that text (estimated, not built), which is within the peak's noise, so it's worth doing only if a text with longer sentences shows more.
