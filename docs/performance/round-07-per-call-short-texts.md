# Results, round 7: one `Process` call per short text

Part of [performance.md](../performance.md)'s measurement history.

Services often call `Process` once per short text (a chat message, one sentence). Bulk processing showed C# slower than
Python there on the TorchSharp backend (see "Bulk processing"). Round 7 measures that case on both backends, finds where
the time goes on the managed one (the default since 0.5) and cuts what can be cut without changing a bit of output.

**Method.** `--per-call N` (new, both benchmarks) makes N calls of `Process(text)` (Python: `nlp(text)`) on one
sentence each, after 20 warm-up calls: the 499 sentences of 5–25 words among the `# text` lines of
`tests/golden/validation*.conllu`, cycled. It reports the median and p90 per call and per stage (C#: the pipeline's
Debug log; Python: each processor timed in turn) and the GC allocation per call. `--concurrent T` runs T threads making
N calls each on one shared pipeline and reports calls/s; `--text T` times one fixed text. Converted models (C#), `.pt`
(Python); machine idle (0–10% CPU before each run); "before" is main at 72cc50a with only the benchmark added, run
alternately with "after".

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --per-call 200 --threads 8 [--package default_fast] [--backend torch] [--concurrent 8] [--text "Hi."]
tools\.venv\Scripts\python tools\benchmark.py --models models\stanza --per-call 200 --threads 8 [--package default_fast] [--concurrent 8]
```

Per call, ms, median / p90 (200 calls):

| | managed before | managed after | TorchSharp | Python |
|---|---:|---:|---:|---:|
| default, 8 threads | 30.0 / 50.7 | **27.9 / 44.6** | 192.5 / 339.0 | 188.8 / 258.6 |
| default, 1 thread | 93.9 / 170.3 | **83.2 / 141.3** | 185.6 / 313.4 | 319.3 / 464.9 |
| default_fast, 8 threads | 15.4 / 24.1 | 14.1 / 21.8 | 98.3 / 143.0 | 73.2 / 95.0 |
| default_fast, 1 thread | 42.4 / 68.7 | 42.0 / 69.6 | 85.2 / 125.1 | 88.6 / 121.7 |

- One call on a short sentence takes 28 ms on the managed backend, **6.8× less than Python** (3.8× on 1 thread), and
  `default_fast` 14 ms (5.2×). The TorchSharp backend is where the old observation came from: as slow as Python on
  `default` with 8 threads, 1.3× slower on `default_fast` (sentiment 49 against 24 ms, tokenize 7.6 against 2.6 ms:
  libtorch's per-operation cost on tiny tensors). The changes below leave it unchanged (before/after within noise).
- `default_fast` changed only within noise: the change applies to the `_charlm` models' shared charlms.

Per stage, 8 threads, managed, default (median / p90, ms):

| | tokenize | mwt | pos | lemma | constituency | depparse | sentiment | ner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 0.32 / 0.57 | 0.01 / 0.10 | 6.8 / 11.3 | 0.29 / 0.74 | 6.5 / 10.8 | 10.5 / 16.1 | 2.4 / 9.0 | 1.32 / 7.57 |
| after | 0.30 / 0.55 | 0.01 / 0.10 | 7.3 / 11.5 | 0.28 / 0.77 | 6.5 / 10.7 | **8.4 / 12.0** | 2.5 / 9.4 | **1.25 / 1.60** |
| Python | 2.66 / 4.95 | 0.09 / 2.46 | 27.0 / 33.9 | 4.15 / 8.44 | 71.4 / 110.1 | 37.9 / 50.8 | 22.9 / 30.5 | 20.9 / 28.4 |

On 1 thread depparse went from 35.7 / 56.8 to 26.1 / 39.9 ms and ner from 3.6 / 23.9 to 3.2 / 4.9.

## Where the time goes

There is no large fixed cost per call left to remove on the managed backend: the time goes into the networks, and they
are limited by memory bandwidth, not by setup.

- **A minimal call** (`--text "Hi."`, two tokens) takes 14.0 ms on 8 threads (TorchSharp: 52 ms). Instrumented (counters
  in `Gemm.Run` and `PackedLstm.Step`, since removed): per call 71 GEMMs read 424 MB of packed weights in 9 ms, about
  47 GB/s, which is this machine's DRAM bandwidth, and 69 LSTM steps read 622 MB in 5 ms. With one or two rows a GEMM
  reads each weight once and does one multiply-add with it, so it runs at the speed of memory, whatever the kernel.
  The models hold about 680 MB of weights; a call reads most of the big matrices once (pos, depparse and constituency
  input layers, depparse's label scorer, sentiment's convolutions, NER's input transform).
- **A 5–25-word sentence**: 274 GEMMs (994 MB) take 12 ms and 564 LSTM steps (5.7 GB, mostly the charlms' W_hh, 16 MB
  per direction and character) 19 ms of the 32 ms. The charlm steps run from the L3 cache at about 300 GB/s: one
  direction's W_hh fits the Ryzen's 32 MB L3, both don't, so the two directions run one after the other.
- **Everything else is small.** A thread-time profile (`dotnet-trace`, `dotnet-sampled-thread-time`, 1 thread) puts 91%
  of the call inside the kernels' parallel regions and under 6% outside them: tokenizing, document building,
  batching, sorting, decoding. GC allocation is about 1 MB per call (0.2 MB for "Hi."); gen0 collections cost about 2%.
- **Thread-pool dispatch** isn't a cost worth removing: a call opens about 850 regions, and the tiny ones (tokenizer,
  MWT) already run their few tasks in microseconds. Running small LSTM steps inline was measured before (docs/backends.md,
  "Small batches") and was slower: a 1–8 MB W_hh read by one core takes longer than the region's synchronization.

## What changed

The charlms are most of the work (about half of a call), so the change removes charlm passes. It rests on one fact:
**a charlm call on a single sentence depends only on that sentence's text**. Every step then runs one row through the
same 1-row kernel, so any other single-sentence call on the same text computes the same bits. (In a batch, a row's
kernel, 1-row or 3/6-row, depends on how many rows are left at each step, which is why `CharlmCache` entries in general
are not exact, see "Batched sentences" below.)

- `CharlmCache` keeps the outputs of single-sentence charlm calls by text (`AddAlone`/`TryGetAlone`), with the backward
  pass's final cell state. Pos, sentiment and NER compute both charlms through `ManagedCharLanguageModel.BuildBoth`,
  which copies a single sentence from an earlier call on the same text, or computes and keeps it.
- **Depparse** (commit 1, generalized in 2): the parser runs the charlms over `"\n"` (ROOT) + the words. Backward, that
  is the tagger's sequence followed by ROOT, so for a one-sentence document the parser continues from the tagger's
  final (h, c) over ROOT only (`ManagedCharLanguageModel.Continue`): two characters instead of the whole sentence, one
  of its two charlm passes. Forward, ROOT comes first, so that pass is unchanged.
- **NER** on a sentence with multi-word tokens (commit 2): it reads tokens, the tagger words, so the tagger's charlm
  outputs don't apply and NER ran both charlms again. Sentiment, which runs before it, reads the same tokens; NER now
  takes them from there. That removes the p90 tail (7.6 → 1.6 ms).
- Exactness: `DepparseTests.OneSentenceDocument_ContinuesTheTaggersBackwardCharlm_BitForBit` compares the parser's arc
  and label scores with and without the cache bit for bit on 130 golden sentences, and
  `PipelineTests.OneSentenceDocuments_EqualTheUncachedPipeline` compares the whole pipeline's CoNLL-U with the cache on
  and off on 96 validation sentences (including multi-word tokens). All golden tests pass unchanged.
- Batched sentences (multi-sentence documents, bulk) are unchanged: the tagger's batch and the parser's differ, so
  reusing the backward pass there would change last bits, and the parser's log-softmax can flip on near-ties.

## Concurrent callers

Throughput with T threads each calling `Process` on one shared pipeline (60 calls each, 8 pool threads; TorchSharp and
Python 40 calls each), calls/s:

| callers | managed before | managed after | TorchSharp | Python |
|---:|---:|---:|---:|---:|
| default: 1 | 27.2 | 29.6 | | |
| 2 | 31.5 | **36.0** | | |
| 8 | 20.8 | 23.6 | 7.0 | 3.9 |
| default_fast: 1 | 66.0 | 59.6 | | |
| 2 | 76.9 | 72.4 | | |
| 8 | 61.0 | 61.3 | 21.4 | 17.1 |

(`default_fast` didn't change; its before/after differences are noise of 60 calls.)

- On the managed backend more than about two concurrent callers **lower** the throughput of `default`: 8 callers get
  23.6 calls/s, one caller alone 29.6. Each call streams the charlms' 16 MB matrices; one caller keeps one of them in the
  L3 cache for a whole pass, but several callers at different points evict each other's and fall back to DRAM. On
  1 pool thread, 2 callers got only 1.24× the throughput of one (9.2 → 11.4 calls/s).
- An experiment (a lock around `Process` in the benchmark, not committed): 8 callers serialized got 36.4 calls/s
  instead of 22.8, at a lower median latency; 2 serialized callers 30.7 instead of 38.7. So at most about two calls in
  flight is best. `PipelineOptions.MaxConcurrentCalls` limits it.
- Still 3.4× TorchSharp's and 6× Python's throughput with 8 callers.

## Guidance

- **Many short texts at once: use bulk** `Process(IEnumerable<string>)`. It batches all sentences, so each weight is
  read once per batch instead of once per text: 2,000 one-sentence texts ran 7.7–9× faster than one call each on the
  TorchSharp backend ("Bulk processing"). Its output can differ slightly from one call per text (sentiment's batch
  padding, as in Stanza), so use it where that is acceptable.
- **One text per request** (a service): about 28 ms per short sentence on 8 threads (`default`), 14 ms with
  `default_fast`. Keep about two requests in flight per pipeline (`PipelineOptions.MaxConcurrentCalls = 2`), or batch
  requests that arrive together into one bulk call; many more concurrent calls lower the throughput.
