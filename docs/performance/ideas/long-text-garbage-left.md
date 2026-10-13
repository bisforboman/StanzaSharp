# Long texts: what is left of the garbage

An open idea ([list](../../performance.md#remaining-ideas)).

After round 23, a long call allocates about 8 KB per word once the `ArrayPool` is warm (11 KB on a fresh process's first
call; the difference is the pool's first fill, which stays live and isn't garbage, see
[round 23](../round-23-array-pool-misses.md)). What remains: the tagger's `CharlmCache` copies (by design), and sites
under 0.3 KB/word each (constituency's `Score` output and `Owned` lists, Chu-Liu/Edmonds' per-cycle arrays, the
tokenizer's per-batch net input).

Filling constituency's word-LSTM input in packed rows (no padded copy) was tried in
[round 24](../round-24-constituency-packed-input.md): exact, but the pool's first fill only moved to depparse and
sentiment, so neither the peak nor the allocation dropped.
