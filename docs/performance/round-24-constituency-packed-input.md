# Results, round 24: constituency's word encoder without the padded input copy (not shipped)

Part of [performance.md](../performance.md)'s measurement history.

Round 23 named the last large buffer a long call could avoid: `ManagedConstituencyNet.EncodeWords` built a padded
`[n, width, 2268]` input for the word LSTM, and `ManagedLstm.ForwardPadded` then copied it into packed rows. This round
built the input in packed rows directly. It is exact, but the peak and the allocation don't drop, so it isn't shipped.

## The change (built, measured, reverted)

`EncodeWords` computed `ForwardPadded`'s row order itself: lengths + 2, sentences sorted longest first (stable),
`PackedLstm.BatchSizes`, and row `offsets[t] + rank[i]` for sentence i's step t. It filled the packed rows with the
sentinels, embeddings and charlm columns, and called `ForwardPacked`. The word LSTM's output and the
`word_to_constituent` GEMM then stayed packed too, so the padded `x`, `y` and `z` buffers were all gone, along with
`ForwardPadded`'s own two copies. This adds about 15 lines and one subtlety:

- **The LSTM:** `ForwardPadded` only packs and unpacks around `ForwardPacked`, so the same packed rows give the same bits.
  Switching to packed rows doesn't regroup the sequences (round 9's issue): the batch and the order are the same.
- **`word_to_constituent`:** `Gemm.Run` gives every row the same arithmetic except row m − 1 when m % 6 == 1, which goes
  through the single-row kernel. Over the padded rows that row is sentence n − 1's last step. It is real only if that
  sentence is one of the longest, and then it is also the last packed row. So the packed GEMM ran `rowInvariant` on all
  rows but that one, and ran that row alone (non-row-invariant) when `n · width % 6 == 1` and sentence n − 1 has the full
  width. That condition never held in the golden texts (0 of 294 `EncodeWords` calls), so this branch is exact by
  argument only.

## Exactness

A temporary test (not committed) wrapped the parser's net and hashed (SHA-256) every `Score` output, which is every
constituency step's scores. It also hashed the all-eight CoNLL-U of every `tests/golden/*.txt`, each alone and all of them
as one document, on Vector256, Vector128 and Scalar, at 1 and 8 threads. Main and the change gave the same hashes:

| path | CoNLL-U | constituency scores |
|---|---|---|
| Vector256, 1 and 8 threads | `1F0DCD923D879A66` | `B735992ABF9FF41B` |
| Vector128, 1 and 8 threads | `1F0DCD923D879A66` | `C3FB35F6C0DD8ADF` |
| Scalar, 1 and 8 threads | `1F0DCD923D879A66` | `7F9221E458CCBE8C` |

The CoNLL-U hash is round 22's.

## Measured

Windows, Ryzen 7 5800X, converted models, `--threads 8 --calls 2`, one process per run, main and the change alternated,
two runs each. Other sessions loaded the machine, so times are not compared.

`--memory 150000 --verbose` (201,313 words), MB allocated per processor in call 1 (identical in both runs):

| | tokenize | mwt | pos | lemma | constituency | depparse | sentiment | ner | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main | 439 | 52 | 494 | 84 | 781 | 214 | 79 | 92 | 2,235 |
| packed input | 439 | 52 | 494 | 84 | **685** | **278** | **111** | 92 | 2,235 |

Constituency allocates 96 MB less, but depparse allocates 64 MB more and sentiment 32 MB more. Constituency's padded
buffers had been the pool's first fill of the 64 MB and 32 MB sizes. Those two processors then reused the buffers, and
now they rent them first. The total is the same to the megabyte. Call 2 (pool warm): 1,605 MB on main in both runs; the
change gave 1,671 and 1,606 (depparse took 278 MB again in one run, and 213 in the other). Constituency took 556 MB on
main and 557 MB with the change.

| | peak after call 1 | peak after call 2 | `--memory 20000` (26,882 words): call 1 allocated | peak after call 1 | peak after call 2 |
|---|---:|---:|---:|---:|---:|
| main | 1,851 / 1,847 MB | 1,953 / 1,965 MB | 786 / 786 MB | 1,316 / 1,346 MB | 1,549 / 1,559 MB |
| packed input | 1,845 / 1,842 MB | 2,014 / 1,951 MB | 778 / 779 MB | 1,308 / 1,314 MB | 1,538 / 1,480 MB |

The peaks are within the ±50 MB noise. On 20,000 words, call 1 allocates 8 MB less (1%).

## Why it isn't shipped

The packed-input buffers this change dropped were scratch that the `ArrayPool` shares between processors. Removing one
only moves its first allocation to the next processor that rents that size. The working set is set by the largest
batch of any processor, so it stays the same. Neither the peak nor the allocation drops measurably. The code would be
slightly longer, and its single-row branch would never run on the golden data. The idea is dropped. The patch was
about 30 lines in `ManagedConstituencyNet.EncodeWords` and is described above.
