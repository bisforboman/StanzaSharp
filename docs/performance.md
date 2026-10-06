# Performance

How fast StanzaSharp runs compared with Python Stanza, what was optimized, and what is left.
Every change kept the output byte-identical: all golden tests pass, and on the benchmark text the
C# CoNLL-U equals Python's line for line, before and after.

## Method

- **Benchmark:** `samples/StanzaSharp.Benchmark` (C#) and `tools/benchmark.py` (Python) build the
  same text and print the same report:

  ```powershell
  dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --models models\converted\en --threads 8 --out cs.conllu
  tools\.venv\Scripts\python tools\benchmark.py --models models\stanza --threads 8 --out py.conllu
  ```

- **Input:** `tests/golden/validation.txt`, `corpus.txt` and `tokenize_stress.txt`, joined by blank
  lines and repeated 8 times. That is 116k characters, 1,992 sentences and 26,264 words after MWT.
  - The text repeats, but batches cut across the copies differently, so each copy is batched
    differently.
  - `tokenize_stress.txt` adds long paragraphs and long run-on sentences.
- **Runs:** one warm-up run on a single copy, then 3 timed runs. The tables show medians.
  - Each stage is timed separately: tokenize, mwt, pos, lemma, depparse, constituency (round 1:
    the first three and constituency). Words/s is computed over all
    26,264 words.
  - Load time covers all six models: C# reads the converted `.safetensors` + `.json` files, Python
    the `.pt` files.
  - Peak memory is the process's peak working set.
- **Correctness check:** both benchmarks write their CoNLL-U output with `--out`. Every C# run below
  matched the Python output exactly.
- **Machine:** AMD Ryzen 7 5800X (8 cores, 16 threads), 64 GB RAM, Windows 11. Software: .NET 10,
  TorchSharp 0.107 (CPU), Python 3.10, torch 2.14.1, stanza 1.15.0.
- **Threads:** results are given for 8 and for 1 torch intra-op threads, set with `--threads` on
  both sides. 8 is torch's default on this machine (physical cores).
- **Caveats:** other jobs shared the machine overnight.
  - Round 1's final runs were taken when it was mostly idle (5–10% CPU load before and after);
    round 2's at 15–50% load, with before and after alternated.
  - Earlier development runs, made under heavy load, were 1.5–3× slower. Compare numbers from the
    same table only.
  - Under load, the default thread count suffers most, because libtorch's OpenMP threads spin
    waiting for each other.

## Results, round 2: memory (six processors)

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

### What changed in round 2

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

## Results, round 1: speed (four processors)

"Before" is `main` at b4c1464, the merge base of this work. "After" is `agent/perf`. Times are
seconds per run on 26,264 words.

**8 threads**

| stage            | C# before | C# after | Python |
|------------------|----------:|---------:|-------:|
| load             |      0.80 |     0.80 |   3.71 |
| tokenize         |      1.52 |     1.49 |   2.21 |
| mwt              |      0.02 |     0.01 |   0.38 |
| pos              |     43.85 |    17.30 |  23.37 |
| constituency     |     37.49 |    10.45 |  26.04 |
| **total**        | **82.89** | **29.25**| **52.01** |
| words/s          |       317 |      898 |    505 |
| peak memory (MB) |     2,544 |    2,159 |  1,753 |

**1 thread**

| stage            | C# before | C# after | Python |
|------------------|----------:|---------:|-------:|
| load             |      0.72 |     0.69 |   3.56 |
| tokenize         |      1.42 |     1.47 |   2.13 |
| mwt              |      0.02 |     0.01 |   0.35 |
| pos              |    159.37 |    44.27 |  65.10 |
| constituency     |    120.15 |    22.93 |  59.77 |
| **total**        | **280.95**| **68.67**| **127.35** |
| words/s          |        93 |      382 |    206 |
| peak memory (MB) |     2,528 |    2,142 |  1,730 |

Summary:
- C# went from 1.6× slower than Python to 1.8× faster at 8 threads.
- At 1 thread, it went from 2.2× slower to 1.9× faster.
- Pos and constituency together account for over 90% of the time. Tokenize and mwt are minor.

## Where the time went (round 1)

Measured with stopwatch instrumentation on the stages and inside the parser.

1. **The charlm ran over padding.** Each charlm is a 1-layer LSTM with 1024 hidden units. It costs
   about 9 MFLOP per character, per direction.
   - `BuildCharRepresentation` ran it over the batch padded to its longest sentence, in characters.
   - The tagger's batches are 250 sentences in document order. One long sentence therefore made
     most of the work padding.
   - This was most of the pos time and a large part of the constituency time.
2. **The charlm ran twice per sentence.** The tagger and the parser feed the same words through
   the same two charlms. Python Stanza does this too.
3. **The parser ran in lockstep batches.** It took 50 sentences in document order and stepped them
   until the slowest one finished. Stanza instead keeps 50 states in flight, sorted longest first,
   and refills the batch as states finish.
   - The lockstep version stepped half-empty batches for a long time.
   - All tensors of a batch stayed in one `DisposeScope` until the batch ended.
4. **Peak memory.** Padding the packed charlm output back to [batch, longest sentence, 1024], only
   to read a few rows, created a tensor of about 1 GB for a 250-sentence batch with one very long
   sentence.

Not worth changing:
- JSON/safetensors loading: 0.7–0.8 s, against Python's 3.6 s.
- Tokenizer array building: the tokenizer is already faster than Python's.
- The parser's per-step overhead. After the changes it is about 2.5 s of the 10.5 s parser time
  at 8 threads, and much of that is the constituent-stack LSTM's own compute.

## What changed in round 1

Each change has its own commit, with numbers measured at the time.

| commit | change | why it keeps output identical |
|---|---|---|
| Pack the charlm input | `BuildCharRepresentation` packs sequences (`Rnn.RunPacked`), as Stanza does, so the LSTM never runs over padding | the LSTM is one-directional and padding follows each sentence, so the positions read are unchanged |
| Parser scheduling | `ConstituencyParser.Parse` schedules like Stanza's `parse_sentences`. Sentences are sorted longest first, 50 states are in flight, and finished states are refilled from the next 50 sentences. Each step gets its own `DisposeScope`. A state's kept tensors are owned by `ParserState` and freed when that state finishes. | each state's computation is independent. The batch makeup now matches Python's. |
| Charlm reuse | `Nn.CharlmCache`: `PosTagger.Process(doc, cache)` keeps each sentence's charlm outputs, and `ConstituencyParser.Process(doc, cache)` reuses them. The cache is keyed by the `Sentence` object, so it never matches across repeated text. The tagger keeps only sentences whose words `simplify_punct` left unchanged. The parser computes just the missing ones. | see the note below |
| `Rnn.RunPackedAt` | the charlm reads its word-end rows directly from the packed output instead of padding it first | pure indexing; values are unchanged |

Note on charlm reuse: the charlm is **not bitwise batch-invariant**.
- In a direct test, 149 of 208 sentences differed in their last float bits depending on which
  sentences shared the batch.
- So the parser's input from the cache can differ from a fresh computation, by about 1e-7.
- That is far below the existing C#-vs-Python drift (parser scores ≈ 2e-5).
- Outputs stayed identical on all golden files and on the 26k-word benchmark text, at 1 and at 8
  threads.
- If an output ever differed because of this, the fix is to drop the cache:
  `Pipeline.Process` would pass `null`.

## Remaining ideas

Rough payoff estimates at 8 threads, against the current 44.5 s six-processor total.

- **Parser step loop: per-state tensor ops.** Round 2 replaced the per-state `select`s with
  `unbind`, but each step still takes one `WordHx` row view per state and stacks per-state tensors.
  - A shared "bank" tensor per stack (state rows addressed by index, `index_select` /
    `index_copy_`) would make the op count per step constant.
  - Payoff: about 0.5–1 s (≈2%) on the CPU. On a GPU, where each op is a kernel launch, likely
    more; that is where the parser gains least today (docs/gpu.md). Moderate refactor; exact, since
    it is data movement only.
- **Remaining depparse memory.** The peak is now set in depparse on the warm-up batch, about 1.1 GB
  above pos: the [batch, width, width, 53] label scores (275 MB for that batch) plus the charlm and
  LSTM activations of a 5000-word batch. Scoring labels only for the chosen heads would remove the
  first, but changes the summation order, so it needs checking against near-ties. Payoff: a few
  hundred MB.
- **`.pt` loading.** Loading Stanza's `.pt` files directly still reads each file whole and copies
  the tensors into one buffer (`TorchCheckpoint.Materialize`). Reading storages from the zip
  entries on demand would bring it to the converted path's memory. Only matters for users who skip
  the conversion.
- **Very large documents.** `CharlmCache` is capped, but the document itself, and depparse's and
  the parser's per-document lists, still grow with the input. Callers with huge inputs should split
  them, for example by paragraph, and call `Process` per part.
- **Thread count.** On a loaded machine, fewer threads were often faster than the default 8,
  because spinning OpenMP threads compete. The README now says so; `torch.set_num_threads` is the
  knob.