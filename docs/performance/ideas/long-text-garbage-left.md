# Long texts: what is left of the garbage

An open idea ([list](../../performance.md#remaining-ideas)).

(round 22, 11 KB/word): `ArrayPool` misses (`PoolExhausted`) in
`ManagedLstm.ForwardPacked`/`ForwardPadded` for the tokenizer and the parser's word LSTM (~0.4 GB at 201k words), the
tagger's `CharlmCache` copies (by design), and sites under 0.3 KB/word each (constituency's `Score` output and
`Owned` lists, Chu-Liu/Edmonds' per-cycle arrays, the tokenizer's per-batch net input).
