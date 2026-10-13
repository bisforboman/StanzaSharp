# Results, round 2: memory (six processors)

Part of [performance.md](../performance.md)'s measurement history.

"Before" is `main` at 2627542 (all six processors, GPU support merged). "After" is `agent/perf2`.
Both run the six-processor default; the benchmarks gained lemma and depparse stages. Times are
seconds per run on 26,264 words. The machine was shared (15–50% CPU load from other jobs during
these runs), so differences under about 3% are noise. Every run below wrote the same CoNLL-U as
Python.

**8 threads** (C# columns: median of two benchmark runs)

| stage                 | C# before | C# after | Python |
|-----------------------|----------:|---------:|-------:|
| load                  |      1.19 |     0.86 |   4.17 |
| tokenize              |      1.63 |     1.66 |   2.24 |
| mwt                   |      0.01 |     0.02 |   0.37 |
| pos                   |     17.20 |    17.25 |  23.43 |
| lemma                 |      0.90 |     0.90 |   1.50 |
| depparse              |     14.13 |    14.33 |  23.40 |
| constituency          |     10.77 |    10.41 |  26.33 |
| **total**             | **44.64** | **44.55**| **77.26** |
| words/s               |       588 |      590 |    340 |
| load peak (MB)        |     1,322 |      634 |    906 |
| **peak memory (MB)**  | **6,081** | **2,690**| **4,253** |

**1 thread**

| stage                 | C# before | C# after | Python |
|-----------------------|----------:|---------:|-------:|
| load                  |      1.01 |     0.86 |   3.92 |
| tokenize              |      1.54 |     1.58 |   2.23 |
| mwt                   |      0.01 |     0.01 |   0.32 |
| pos                   |     44.52 |    44.76 |  65.86 |
| lemma                 |      1.20 |     1.23 |   1.84 |
| depparse              |     47.04 |    47.77 |  64.30 |
| constituency          |     23.23 |    23.22 |  61.34 |
| **total**             |**117.54** |**118.58**|**195.89** |
| words/s               |       223 |      221 |    134 |
| **peak memory (MB)**  | **6,182** | **3,054**| **4,196** |

Summary:
- Peak memory fell from 6.1 GB to 2.7 GB at 8 threads, now 1.6 GB below Python's. The load peak
  halved, to 0.63 GB (Python: 0.91 GB).
- Speed is unchanged: 1.7× faster than Python at 8 threads, 1.65× at 1 thread.
- The two new stages: lemma takes 2% of the time, depparse 32% at 8 threads (40% at 1 thread). Pos
  is 39%, constituency 23%.

## What changed in round 2

| commit | change | why it keeps output identical |
|---|---|---|
| Load models without whole-file copies | `SafeTensorFile` reads only the header, then each tensor from the file on demand. `Weights.ToTensor`/`LoadFrom` read straight into the TorchSharp tensor's memory (`Tensor.bytes`), instead of byte[] of the file → float[] → temporary tensor → copy. `Checkpoint` parses JSON from a stream, and `UnitToId` reads `_unit2id` as JSON text instead of building a `JsonNode` per entry (250k in the pretrain vocabulary). | the same bytes end up in the same tensors |
| Depparse: chunked biaffine scorers | the label scorer's einsum intermediate is [batch, width, 401, 53] floats: 1.5 GB for 250 sentences padded to 72 words, plus a permuted copy inside einsum. Stanza peaks the same way. `DeepBiaffine` now runs both einsums on a few sentences at a time (≈128 MB of intermediate), writes into a preallocated output, frees each intermediate at once and adds the bias in place. | each sentence's scores read only its own rows; golden depparse files and the benchmark stay byte-identical |
| POS: cut batches at 5000 words | like Stanza's `LengthLimitedBatchSampler` (`batch_maximum_tokens`). Never triggers on the benchmark text. On a synthetic text of 600 sentences of 10–60 words, which it does split, C# still equals Python. | matches Python's batching |
| CharlmCache: at most 32k words | the cache held 8 KB per word for the whole document. It now keeps at most `MaxWords` (default 32,768 ≈ 256 MB) and disposes the rest on `Add`; the parser recomputes those, as it already did for sentences the tagger didn't keep. Text under 32k words is unaffected. | as for any sentence missing from the cache (see the note in round 1) |
| Parser: unbind | `Push`, the open markers and the reduce outputs use one `unbind` per tensor instead of one view op per state per step. | the same views |

Freeing CharlmCache entries as the parser consumes them was not done: the cache is full when the
tagger finishes, which is before the parser starts, so that would not lower the peak. The word cap
bounds it instead.
