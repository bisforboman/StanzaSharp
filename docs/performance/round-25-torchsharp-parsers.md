# Results, round 25: the TorchSharp parsers' step loop, input and label scores

Part of [performance.md](../performance.md)'s measurement history.

Three open ideas for the TorchSharp backend (`StanzaSharp.Cuda`, also `CudaBackend.Cpu`), all shipped. The managed
backend is unchanged.

1. **Constituency step loop** (`ConstituencyNet`): every vector and stack state the parser holds is now a row of a
   per-thread "bank" tensor (one for word and constituent vectors, one per stack holding h and c), and a handle is its
   row number. A step reads each bank with one `index_select` and writes the new states with one `index_copy_`, so the
   op count per step no longer grows with the 50 states in flight. Before, `Score` took a row view per state and
   stacked them three times, and each push stacked the parents' h and c and unbound the result into per-state views.
   Compose gathers every close's children in one `[closes, most children]` index (short lists repeat their first child,
   which doesn't change a max). A bank has a free list; it grows by doubling and is freed when its last row comes back
   (the end of a Parse call). Banks are per thread (`ThreadLocal`), since Parse is synchronous, so concurrent calls
   never share one.
2. **Depparse input** (`DepparseNet`): built in packed order like the tagger's (round 4): embeddings looked up in
   packed order, the charlm (or character model) outputs concatenated and reordered once, `Rnn.Pack`, the highway
   LSTM on packed rows, then one `pad_packed_sequence` for the scorers. The padded `[batch, width, 2423]` input and its
   pack copy are gone, and with them `HighwayLstm`'s padded `Forward` overload.
3. **Depparse label scores** (`DepparseNet.DeepBiaffine`): with `labelScores` false (the pipeline), each chunk of
   sentences adds the bias and takes its argmax at once, so the `[batch, width, width, 53]` label scores never exist
   whole; only the int64 argmax does. Tests asking for label scores still get the whole tensor, built the old way.

## Exactness

A temporary test (not committed) ran the pipeline on the TorchSharp backend (`CudaBackend.Cpu`, 8 threads) through
wrapped nets, hashing (SHA-256) every constituency `Score` output, every depparse batch's arc log-probs and labels as
the pipeline asks for them, the same batch's full label scores (asked for again with `labelScores` true; its labels
were also checked equal to the pipeline's), and the CoNLL-U. Texts: every `tests/golden/*.txt` and
`tests/golden/de/*.txt`, each alone and all of a language's as one document. Packages: English `default` and
`default_fast`, German `default`, `default_fast`, `gsd` and `hdt`, so both the charlm and the character-model
depparse input, and the English and German constituency parsers.

| | CoNLL-U | arcs | labels | label scores | constituency scores |
|---|---|---|---|---|---|
| main | `6BA9C55837837A3A` | `D8D6F0E8FFD84887` | `A3D75B3031D6251A` | `0BE14218BC0CE555` | `68C63809409F293E` |
| all three items | `6BA9C55837837A3A` | `D8D6F0E8FFD84887` | `A3D75B3031D6251A` | `0BE14218BC0CE555` | `68C63809409F293E` |

Every score is bit-identical, not only the discrete outputs. Items 1 and 3 only move data. Item 2 also changes the
shape of `trans_pretrained` and `trans_char` (rows `[words, n]` instead of `[batch, width, n]`), which moved last bits
in the tagger in round 4; here it did not (MKL gave each row the same sums). An earlier run of the same test over
English and German `default` only, after each item in turn, gave the same hashes at every step.

## Measured

Windows, Ryzen 7 5800X, converted models, `--backend torch --threads 8`, one process per run, builds alternated, two
runs each.

Stage times (`--copies 4 --runs 3`, 13,132 words, medians of 3 runs; "peak" is the depparse warm-up's):

| build | constituency | depparse | depparse peak | total | peak working set |
|---|---:|---:|---:|---:|---:|
| main | 3.93 / 3.75 s | 5.77 / 5.46 s | 1,839 / 1,842 MB | 22.79 / 21.47 s | 2,447 / 2,419 MB |
| label scores (3) | 3.70 / 3.64 s | 5.43 / 5.42 s | 1,731 / 1,730 MB | 21.57 / 21.34 s | 2,459 / 2,482 MB |
| + bank (1, 3) | 3.15 / 3.10 s | 5.53 / 5.43 s | 1,720 / 1,720 MB | 21.08 / 20.80 s | 2,511 / 2,505 MB |
| + packed input (all) | **3.05 / 2.99 s** | 5.48 / 5.43 s | **1,608 / 1,628 MB** | 21.00 / 20.82 s | 2,499 / 2,501 MB |

`--memory 20000 --verbose` (26,882 words, one Process call):

| build | constituency | its managed allocation | depparse peak (all eight) | peak (all eight) | peak, `tokenize,mwt,pos,lemma,depparse` |
|---|---:|---:|---:|---:|---:|
| main | 8.49 / 8.37 s | 183 MB | 2,479 / 2,464 MB | 2,810 / 2,804 MB | 1,948 / 1,954 MB |
| all three | **6.66 / 6.60 s** | **96 MB** | **2,199 / 2,190 MB** | 2,817 / 2,805 MB | **1,658 / 1,664 MB** |

- **Item 1:** constituency is about 20% faster on the CPU, and allocates half the managed memory (no per-state `Tensor`
  objects). TorchSharp's count of tensors outside dispose scopes no longer goes negative after constituency (the
  per-state views had made it −507,348). The working set after constituency is ~100 MB higher (the banks' freed
  memory stays with the allocator); its peak is unchanged.
- **Items 2 and 3:** depparse's own peak drops by ~110 MB each on the benchmark (~280 MB together on 26,882 words, and
  ~290 MB when depparse is the last processor). Its time doesn't change measurably.
- The all-eight peak is set by sentiment, after depparse, so it doesn't drop (within the ±50 MB noise).

## On a GPU (expectation, not measured)

No GPU was available. docs/gpu.md found the constituency parser gaining least from CUDA because kernel launches and
the per-step read-back dominate; a constant number of ops per step instead of several per state should help more
there than on the CPU. Items 2 and 3 should lower the GPU memory peak of depparse the same way.
