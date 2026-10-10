# Performance

How fast StanzaSharp runs compared with Python Stanza, what was optimized, and what is left.
The measurements here are of the TorchSharp backend (add `--backend torch` to the benchmark commands; since 0.5 the
default is the managed backend, measured in [backends.md](backends.md)), except rounds 6 and 7, which compare both.
Every change kept the output byte-identical: all golden tests pass, and on the benchmark text the
C# CoNLL-U equals Python's line for line, before and after.

## Run it yourself

The numbers below are from one machine. To measure yours, install the tool and run its benchmark:

```
dotnet tool install -g StanzaSharp.Tool
stanzasharp download models/stanza/en
pip install stanza==1.15.0            # optional: without it only StanzaSharp is measured
stanzasharp benchmark                 # --package default_fast, --threads N, --words N, --quick, --python PATH
```

It follows the method below on a smaller text: the golden texts (validation.txt, corpus.txt, tokenize_stress.txt,
shipped with the tool) cut at about 1,500 words (`--words`); the managed backend; a warm-up run on each side, then 3
timed runs with StanzaSharp and Stanza alternating (Stanza runs in one Python process that stays loaded); medians per
processor; then 50 `Process` calls on one sentence each (the text's sentences of 5-25 words; median and p90, after 5
warm-up calls); the peak working set of each process (macOS: `ru_maxrss`). Both sides read the same `.pt` files, and
the block says whether their CoNLL-U was identical. Threads: each side's default unless `--threads` sets both
(StanzaSharp: all logical processors; torch: the physical cores). It prints a Markdown block with the CPU model, cores
and logical processors, RAM, OS, .NET version, SIMD path (Vector256, Vector128 or Scalar) and StanzaSharp version,
ready to paste into an issue. About two minutes with Python on 4 cores; `--quick` (300 words, one run, 10 calls) is a
smoke test.

On GitHub's hosted runners: **Actions > Benchmark > Run workflow** (or `gh workflow run benchmark.yml -f package=default`)
runs it with Python Stanza on ubuntu-24.04, ubuntu-24.04-arm, windows-2025, windows-11-arm and macos-15 (free standard
runners only). Each job puts its block in its summary and the artifact `benchmark-<runner>`; the last job combines them
into one table (summary and artifact `benchmarks`, `benchmarks.md`, made by `tools/benchmark_table.py`). Hosted runners
are shared virtual machines (the macOS one is virtualized Apple Silicon), so compare numbers within one run.

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

## Results, round 7: one `Process` call per short text

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

### Where the time goes

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

### What changed

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

### Concurrent callers

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
  flight is best. `Process` itself doesn't limit this (see "Remaining ideas").
- Still 3.4× TorchSharp's and 6× Python's throughput with 8 callers.

### Guidance

- **Many short texts at once: use bulk** `Process(IEnumerable<string>)`. It batches all sentences, so each weight is
  read once per batch instead of once per text: 2,000 one-sentence texts ran 7.7–9× faster than one call each on the
  TorchSharp backend ("Bulk processing"). Its output can differ slightly from one call per text (sentiment's batch
  padding, as in Stanza), so use it where that is acceptable.
- **One text per request** (a service): about 28 ms per short sentence on 8 threads (`default`), 14 ms with
  `default_fast`. Keep about two requests in flight per pipeline (`PipelineOptions.MaxConcurrentCalls = 2`), or batch
  requests that arrive together into one bulk call; many more concurrent calls lower the throughput.

## Results, round 6: memory after `Process` on the managed backend

Round 5 measured the TorchSharp backend, whose tensors live in libtorch's native heap. The managed backend (the default
since 0.5) allocates nothing natively: the models and every buffer are .NET arrays. Is the memory kept after a call a
problem there too? Same benchmark mode as round 5 (`--memory N --calls 2 --verbose`, `.pt` models, 8 threads,
workstation GC), with `--backend managed` and `--backend torch`, `tokenize,mwt,pos,constituency` and all eight
processors; `--idle-gc` (new) then runs an aggressive GC at once and another after 61 s idle. 500, 5,000 and 15,000 are
680, 6,761 and 20,513 words after tokenizing. Another job kept the host near 100% CPU, so times are not compared.

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --memory 5000 --calls 2 --backend managed --threads 8 --models models\stanza\en [--processors tokenize,mwt,pos,constituency] [--verbose] [--idle-gc]
```

Windows, MB: load peak / after load, then peak during call 1 (call 2's in parentheses where higher), then the working
set after call 1 / call 2:

| | managed, 4 processors | TorchSharp, 4 processors | managed, all eight | TorchSharp, all eight |
|---|---:|---:|---:|---:|
| load | 561 / 518 | 536 / 536 | 863 / 844 | 870 / 870 |
| 680 words: peak | 561 | 663 (705) | 863 | 1,338 |
| after call 1 / 2 | 551 / 558 | 657 / 667 | 852 / 856 | 1,074 / 1,020 |
| 6,761 words: peak | 711 (755) | 1,053 (1,165) | 2,366 (3,056) | 4,253 (5,319) |
| after call 1 / 2 | 704 / 720 | 1,046 / 1,148 | 2,360 / 3,056 | 4,034 / 4,749 |
| 20,513 words: peak | 933 (1,106) | 1,455 (1,473) | 2,027 (2,197) | 2,844 (3,015) |
| after call 1 / 2 | 927 / 994 | 1,125 / 1,124 | 2,027 / 2,136 | 2,792 / 2,982 |

Linux (Docker Desktop, `aspnet:10.0`, as in round 5), managed, RSS after call 1 / 2 in MB, with the trim and with
`--no-trim`:

| | 4 processors | 4, `--no-trim` | all eight | all eight, `--no-trim` |
|---|---:|---:|---:|---:|
| after load | 544 | 529 | 856 | 856 |
| 680 words | 573 / 584 | 565 / 573 | 863 / 868 | 863 / 868 |
| 6,761 words | 710 / 760 | 726 / 761 | 2,366 / 2,428 | 2,376 / 3,089 |
| 20,513 words | 939 / 1,061 | 947 / 1,038 | 2,018 / 2,345 | 2,025 / 2,070 |

So yes, the managed backend keeps memory too, at about its peak, on both systems. Who holds it:

- **Not the native heap.** On Linux glibc's arenas stay at 13–22 MB in every run, so `malloc_trim` has nothing to give
  back (with and without it the numbers differ only by run-to-run noise). `NativeHeap.Trim` still runs after large
  calls; it returns nothing on this backend and still matters on the TorchSharp backend.
  `MIMALLOC_PURGE_DELAY` is libtorch's, so it does nothing here either.
- **The GC heap: the models, then the call's scratch buffers in `ArrayPool<T>.Shared`.** After loading the GC heap is
  370 MB (4 processors) or 660 MB (all eight): the weights. After a 6,761-word call with all eight it is 2.8 GB, and a
  full GC leaves 2.75 GB: the pool holds the per-batch buffers of the managed kernels (the largest, rounded up to a
  power of two: a charlm LSTM's gate buffer of 386 MB, sentiment's 345 MB input, ...), on purpose, so the next call
  reuses them instead of making the GC find hundreds of MB of garbage. Without the pool (large arrays left to the GC,
  tried) the peak rose from 2.4 to 3.1 GB in call 1 and 3.8 GB in call 2.
- .NET's shared pool gives an array back only from a gen2 GC at least a minute after the array was last used, and the
  GC keeps memory it freed committed until an aggressive GC (or decommits it slowly, over later GCs). So in an idle
  service the memory stays, and no GC right after the call can free it:

| managed, Windows, MB | working set after call 2 | aggressive GC at once | 61 s idle, then a full GC | then an aggressive GC |
|---|---:|---:|---:|---:|
| 4 processors, 6,761 words | 720 | 608 (53 ms) | 604 | **364** (56 ms) |
| 4 processors, 20,513 words | 949 | 686 (134 ms) | 687 | **425** (142 ms) |
| all eight, 6,761 words | 2,417 | 2,229 (50 ms) | 2,226 | **821** (230 ms) |
| all eight, 20,513 words | 2,786 | 1,940 (745 ms) | 1,941 | **663** (196 ms) |

  An aggressive GC a minute after the last call brings the process back to about its size after loading (below it,
  as it also compacts what loading left). That is the guidance in the README: a service that wants the memory back when
  idle calls `GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true)` itself, a minute or more
  after its last call. The library does not: a GC is process-wide, and its cost grows with the application's own heap.
  On the TorchSharp backend it does nothing (all eight, 6,761 words: 4,824 MB after call 2, 4,547 MB after both GCs;
  the GC heap is 40 MB): there the memory is mimalloc's, as round 5 found.
- Splitting the text into ~1,000-word calls does not shrink what stays (all eight, 20,513 words in 15 parts: 1.4–1.8 GB
  after, 0.67 GB after the idle aggressive GC): the parts' batches have different sizes, and the pool keeps arrays of
  every size it was asked for.
- Call 2 sometimes ends higher than call 1 (all eight, 6,761 words: 2.4 → 3.1 GB, GC heap 2.8 → 3.8 GB, in 2 of 4
  runs on Windows and Linux): the shared pool keeps arrays per thread and per core, so it can hold a second set.

Tried and not kept (branch `exp/managed-buffer-pool`, for a decision):
- A private pool (`ArrayPool<T>.Create`) instead of the shared one: one set of buffers, no growth on call 2 or 3
  (2.4–2.5 GB), but it never trims, so the idle aggressive GC no longer helps (2.2 GB after it).
- The private pool dropped after each call of 1,000+ words: an aggressive GC right after the call then brings the
  process to 730 MB (330 ms), but without that GC the next call's buffers are allocated before the old ones are
  collected, and call 2 peaked at 3.2 GB instead of 2.5 GB.

## Results, round 5: memory after `Process` returns, Linux and Windows

Issue #19's reporter runs a long-running service on Linux x64 (glibc, `mcr.microsoft.com/dotnet/aspnet:10.0`), where
what matters is the memory the process keeps between calls, not only its peak. Round 3 measured Windows only and found
the working set stays near the peak after `Process`. `--memory N --calls 2` (both benchmarks) now reports the memory
after each of two calls on the same text; `--verbose` adds the GC heap and, on Linux, glibc's `mallinfo2` and
`/proc/self/smaps_rollup`.

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --memory 5000 --calls 2 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza\en [--verbose] [--no-trim]
tools\.venv\Scripts\python tools\benchmark.py --memory 5000 --calls 2 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza
```

- **Linux:** Docker Desktop (WSL 2) on the same machine. The benchmark is published for `linux-x64` and run on
  `mcr.microsoft.com/dotnet/aspnet:10.0` (Ubuntu 24.04, glibc 2.39); Python on `python:3.12-slim` with
  `tools/requirements.txt` (CPU torch 2.14.1). The `.pt` models are copied into a Docker volume: read through a
  Windows bind mount, loading took minutes (issue #48: one `fstat` per pickle opcode, fixed since). Memory is the
  RSS (`VmRSS`), peaks are `VmHWM`.
- **Windows:** as in round 3 (`.pt` models, workstation GC).
- **0.4.0** is the tag `v0.4.0` with this benchmark; "now" is this branch. `--memory 500`, `5000` and `15000` are
  680, 6,761 and 20,513 words after tokenizing.
- **Times are not comparable:** other jobs kept the host at 100% CPU throughout, and in the container libgomp's
  spinning threads made calls 2–10× slower (some Linux runs use `OMP_WAIT_POLICY=PASSIVE`, which changes no
  memory). Only memory is compared in the tables; the trim's timing was measured later, on a quieter host.

Linux, MB. "Peak" is the peak during call 1 (call 2's in parentheses where it is higher):

| | 0.4.0 | main before | **now** (malloc_trim) | Python |
|---|---:|---:|---:|---:|
| load peak / after load | 1,018 / 1,011 | 568 / 561 | 595 / 588 | 800 / 733 |
| 680 words: peak | 1,124 | 674 (704) | 705 (762) | 841 (857) |
| after call 1 / after call 2 | 952 / 958 | 660 / 651 | 705 / 711 (not trimmed) | 842 / 811 |
| 6,761 words: peak | 2,656 | 1,410 (1,562) | 1,403 (1,515) | 2,225 (2,290) |
| after call 1 / after call 2 | 1,071 / 1,269 | 929 / 947 | **655 / 648** | 889 / 855 |
| 20,513 words: peak | 2,668 | 1,510 (1,597) | 1,490 (1,595) | 2,282 (2,333) |
| after call 1 / after call 2 | 1,163 / 1,193 | 960 / 977 | **642 / 641** | 918 / 961 |

Windows, MB (StanzaSharp's Windows behaviour is unchanged by this round; the last C# column is an environment
setting, see below):

| | 0.4.0 | now | now, `MIMALLOC_PURGE_DELAY=0` | Python |
|---|---:|---:|---:|---:|
| load peak / after load | 968 / 968 | 538 / 537 | 535 / 530 | 733 / 660 |
| 680 words: peak | 1,085 | 702 (712) | 639 (688) | 761 (797) |
| after call 1 / after call 2 | 930 / 922 | 702 / 712 | 594 / 603 | 699 / 703 |
| 6,761 words: peak | 2,628 (2,792) | 1,650 | 1,283 | 2,343 (2,404) |
| after call 1 / after call 2 | 2,082 / 2,773 | 1,620 / 1,601 | **583 / 579** | 713 / 716 |
| 20,513 words: peak | 2,863 | 1,744 (1,781) | 1,358 (1,384) | 2,399 |
| after call 1 / after call 2 | 2,415 / 2,303 | 1,419 / 1,440 | **581 / 596** | 741 / 811 |

Who holds the memory after a call:

- **Not StanzaSharp, and not the GC.** No tensor survives a call (round 3), the second call ends where the first did,
  and the GC heap is 80–125 MB (committed 100–180 MB) before and after. Once the native heap is trimmed, the process
  is back within 60–100 MB of its size after loading.
- **Linux: glibc's malloc.** libtorch allocates CPU tensors with `malloc` (its `c10` allocator does not cache on
  the CPU). glibc serves blocks over its mmap threshold with `mmap` and unmaps them on `free`, but the threshold is
  dynamic: each freed mmapped block raises it to that block's size, up to 32 MB. After the first large tensors are
  freed, nearly every later tensor comes from the arenas, and memory freed there stays in the process unless it is at
  the top of a heap. After a 6,761-word call `mallinfo2` shows 394 MB of arenas of which 299 MB are free, plus
  265 MB mmapped (the large weights). Per-thread arenas are not the problem: `MALLOC_ARENA_MAX=2` changes nothing.
- **Windows: mimalloc inside libtorch.** libtorch 2.10's `c10.dll` has mimalloc 2.2.4 built in for CPU tensors
  (`MIMALLOC_VERBOSE=1` prints its options). It keeps freed pages committed and schedules their purge for later
  (`purge_delay`, 10 ms), but after a call the purge never came: neither waiting up to a second nor allocating and
  freeing a 256 MB tensor or a small one afterwards lowered the working set. Python's torch 2.14.1 has mimalloc
  2.4.1 with the same options and does drop back, so the newer mimalloc (or torch around it) behaves differently.
  `MIMALLOC_PURGE_DELAY=0` purges on every free: the working set after a call drops by 0.8–1 GB and the peak by
  370 MB, as freed pages are no longer counted while new ones are committed.

What changed:

- **Linux: `malloc_trim(0)` after each call of at least 1,000 words** (`NativeHeap`, internal switch
  `PipelineOptions.TrimNativeHeap`, default on; `--no-trim` in the benchmark). It gives the free pages of every
  arena back: 270–320 MB after 6,761 and 20,513 words, leaving the RSS 55–70 MB above its size after loading and
  about 250 MB below Python's. The trim itself takes 23–42 ms after such a call (125–145 ms while the host was at
  full load and the calls took 55–145 s), and the next call faults the pages in again. In three alternated pairs of
  runs (6,761 words, two calls each, calls of about 22 s) the median call took 22.8 s with the trim and 22.4 s
  without, which is within the noise.
  Below 1,000 words a call frees little (15 MB at 680 words) while trimming still walks every arena (2–12 ms under
  load), so it is skipped; one-sentence calls in a loop pay nothing.
- It is found with `NativeLibrary` in `libc.so.6`, so it never runs on Windows, macOS or musl.

Environment settings, Linux, 6,761 words, without the trim:

| setting | peak | after call 1 / 2 |
|---|---:|---:|
| none | 1,410 (1,562) | 929 / 947 |
| `MALLOC_ARENA_MAX=2` | 1,411 (1,571) | 938 / 987 |
| `MALLOC_MMAP_THRESHOLD_=131072` | 1,277 (1,360) | 726 / 785 |
| `MALLOC_TRIM_THRESHOLD_=131072` | 1,341 (1,401) | 764 / 809 |
| `MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=2` | 1,342 (1,397) | 764 / 812 |
| the trim (no setting) | 1,414 (1,513) | 646 / 646 |

Setting any `MALLOC_*_THRESHOLD_` turns off the dynamic threshold, so large tensors are mmapped and unmapped again
and the peak is 70–130 MB lower, but tensors under the threshold still fragment the arenas. With the trim no setting
is needed; a fixed mmap threshold could be combined with it to lower the peak, at the cost of an `mmap` per large
tensor (not measured for time).

Windows: `MIMALLOC_PURGE_DELAY=0` costs time. Three alternated runs of three calls each on 6,761 words (host at
97–100% CPU): median 18.5 s per call against 15.2 s without it, about 20% slower. It must be in the environment before
libtorch loads: set it outside the process, or with `Environment.SetEnvironmentVariable` before the first TorchSharp
call (checked: mimalloc then reports `purge_delay: 0`).

## Results, round 4: the highway LSTMs on packed rows

Round 3 left the tagger's padded batch as the peak: batches are 250 sentences in document order, so one long sentence
padded the whole [250, longest, 2248] input, both charlm outputs and every highway layer's intermediates. Now nothing
in the tagger is padded:
- `Rnn.PackedOrder` gives the row order pack_padded_sequence (enforce_sorted: false) would produce, and the tagger
  builds its input straight in that order: embeddings looked up in packed order, each charlm's per-sentence outputs
  concatenated and reordered once.
- `Rnn.Pack` makes the `PackedSequence` without a padded input (TorchSharp can't build one from data: it packs a
  one-wide zero input and points the result's data at the packed rows with `set_`, no copy).
- `HighwayLstm.Forward(PackedSequence)` runs the gate and highway layers on the packed rows and adds them in place to
  the LSTM's packed output, which is the next layer's input. The heads read the last layer's rows in sentence order.
- Depparse keeps its padded `Forward(Tensor, lengths)`, now a wrapper (pack, the packed path, pad). Its batches are
  sorted by length, so it had little padding; its code is unchanged.

The LSTMs see exactly the same packed data, batch sizes and order as before. Only the linear layers (`trans_pretrained`,
`trans_char`, gate, highway) now multiply [words, n] instead of [batch, longest, n] matrices, which moves the last
float bits. Measured against the previous implementation over `corpus.txt` and every `validation*.txt`
(tokenize,mwt,pos,lemma,depparse, both packages), max abs difference:

| output | default | default_fast |
|---|---:|---:|
| UPOS logits | 2.3e-5 | 2.3e-5 |
| XPOS scores | 4.6e-5 | 4.6e-5 |
| UFeats scores (21 heads) | 4.6e-5 | 4.6e-5 |
| depparse arc log-probs | 1.5e-5 | 2.3e-5 |
| depparse label log-probs | 1.9e-5 | 2.3e-5 |

That is the size of the existing drift from Python (UPOS logits ≈ 2e-5); the tests' tolerances are unchanged and pass.
Every discrete output is identical: the 26 CoNLL-U files of that comparison byte for byte, the full test suite (all
golden files: pipeline, validation, lemma, depparse, ner, fast, sentiment, bulk, pretokenized) with converted and
with `.pt` models, nothing skipped, and the 26k-word benchmark output.

`--memory N --processors tokenize,mwt,pos,constituency --threads 8`, peak working set in MB, workstation GC; before is
`main` at aaf6e02. Before and after were alternated, two runs each (the peaks agreed within 45 MB). The machine was
shared with other jobs, so the times are noisy: the pos columns give the range over the four runs of each size (both
model formats), leaving out two runs that took 2–3× as long (one after at 5k words, one before at 15k).

| words | converted, before | converted, after | `.pt`, before | `.pt`, after | pos before | pos after |
|---|---:|---:|---:|---:|---:|---:|
| 499 | 596–611 | 611 | 656 | 656 | 0.6–1.0 s | 0.6–0.8 s |
| 5,196 | 1,567 | **1,016** | 1,615 | **1,054** | 6.8–9.1 s | 4.7–6.0 s |
| 14,767 | 1,655 | **1,248** | 1,678–1,722 | **1,270** | 15.7–16.6 s | 12.0–14.1 s |

- At 5k words the peak falls by 550 MB (35%), at 15k by 410 MB. Python Stanza peaked at 2,356 MB on the same
  5,196 words (round 3).
- The tagger no longer sets the peak at 15k words: after pos the process is at about 1,110 MB, and the constituency
  parser takes it to 1,250 MB.
- 499 words has no long sentence in a batch, so nothing changes there.
- pos is about 20–25% faster where a batch has a long sentence: the gate and highway layers no longer multiply padding.

All eight processors, the usual benchmark (26,264 words, converted models, 8 threads, 3 timed runs after a warm-up),
before and after alternated twice; the shared machine made the other stages vary by up to 2.5× between runs:

| | before | after |
|---|---:|---:|
| pos | 19.72 s, 15.71 s | 13.02 s, 11.03 s |
| total | 96.72 s, 71.53 s | 68.86 s, 68.31 s |
| peak memory | 3,007 MB, 3,002 MB | 3,015 MB, 3,002 MB |

pos is 25–35% faster. The 8-processor peak is set by later stages, so it does not change. Both outputs were
byte-identical.

## Results, round 3: memory of a short-lived process

Issue #19: a process that loads `tokenize,mwt,pos,constituency` (`default` package) from Stanza's `.pt` files, makes one
`Process` call and exits peaked 230–310 MB above Python Stanza. `--memory N` measures exactly that, one fresh process
per run: the text is `corpus.txt` and `validation*.txt`, repeated and cut after the paragraph that reaches N
whitespace-separated words.

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --memory 3750 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza\en [--verbose]
tools\.venv\Scripts\python tools\benchmark.py --memory 3750 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza
```

Peak working set in MB, 8 threads, Windows. "Load" is the peak after `Pipeline.Load`, "after load" the working set
then. Server GC is `DOTNET_gcServer=1`. "Before" is `main` at eb87c77. Python is 1.15.0 with torch 2.14.1.

| | .pt, workstation GC | .pt, Server GC | converted, workstation | converted, Server | Python |
|---|---:|---:|---:|---:|---:|
| load peak, before | 968 | 751–1,090 | 484 | 482 | 733 |
| load peak, after | 536 | 498–537 | 484 | 482 | |
| after load, before | 967 | 750–1,090 | 483 | 482 | 659 |
| after load, after | 536 | 498–537 | 483 | 482 | |
| 499 words, before | 1,066 | 1,209 | 607 | 602 | 746 |
| 499 words, after | **652** | **652** | 604 | 600 | |
| 5,196 words, before | 2,604 | 2,756 | 2,357 | 2,292 | 2,356 |
| 5,196 words, after | **1,613** | **1,586** | 1,567 | 1,580 | |
| 15,151 words, before | 2,826 | 2,821 | 2,519 | 2,565 | 2,460 |
| 15,151 words, after | **1,684** | **1,715** | 1,660 | 1,706 | |

Process times did not change (0.8 s, 6.2 s and 15.8 s; Python 1.3 s, 13.1 s and 33.3 s). The 5k-word output equals
Python's byte for byte.

Where the memory went:
- **Loading `.pt` files** (430–590 MB). `TorchCheckpoint` read each file into a `byte[]` and copied its tensors into a
  second one; the 290 MB of checkpoints this processor set reads became about 580 MB of garbage. The load ended before
  a GC returned it, so it stayed in the working set into `Process`. Server GC collected at different points, hence
  its spread. Now the unpickler reads from a stream, only the storages' positions are noted, and each tensor is read
  from the file straight into its memory, as for safetensors. The `.pt` path now costs 50 MB over the converted one
  (the unpickled object graph, garbage until the next GC).
- **The tagger's padded batches** (800 MB at 5k words, in C# and Python alike). Batches are 250 sentences in
  document order, padded to the longest: with a 139-word sentence, the concatenated input alone is
  [250, 139, 2248] floats, 312 MB, and the padded charlm outputs another 285 MB. Everything stayed in the batch's
  dispose scope: the inputs, their concatenation, and each highway layer's six [250, 139, 400] intermediates. Now
  the inputs are freed once concatenated, the concatenation after the first layer, and each layer's intermediates when
  it ends, with the gate's elementwise steps in place (the same kernels, so the same values). That is why C# now
  peaks 0.8 GB below Python on the longer texts. The reporter's transcript has shorter sentences, so this gains less
  there.
- **Not the cause:** tensors left to the finalizer (a full GC with finalizers frees nothing after `Process`), the
  `CharlmCache` (8 KB per word: 40 MB at 5k words; off, the peak differs by under 50 MB, while the run takes 30–40%
  longer), and the constituency parser, whose 50 states in flight never set the peak.
- After `Process` the working set stays near the peak although every tensor is freed: the native allocator keeps
  the pages for reuse. That raises the steady state of a long-running process, not its peak. Round 5 finds who keeps
  them, and returns them on Linux.

Splitting the input (15,151 words, converted models, workstation GC). The output of the parts equals one call's
except for sentence ids, offsets (each part's own) and the whitespace at each cut (`SpaceAfter` of a part's last
token, `SpacesBefore` of its first):

| calls | peak | time |
|---|---:|---:|
| one `Process` call | 1,664 MB | 15.5 s |
| 12 calls of about 1,000 words | 1,156 MB | 18.0 s |
| 6 calls of about 2,000 words (`.pt`) | 1,836 MB | 16.7 s |
| 629 calls, one paragraph each | 574 MB | 81.0 s |
| one bulk `Process(texts)` call on the 12 parts | 1,705 MB | 15.6 s |

The peak follows the largest padded tagger batch, so parts help only once they hold well under 250 sentences: the
2,000-word parts happened to put the longest sentence in a fuller batch than one call did. Bulk processing batches all
texts together, like one call.

## Bulk processing: many short texts

`agent/input-modes`: 2,000 one-sentence texts (the `# text` lines of `tests/golden/validation*.conllu`, cycled),
all processors of each package, 8 torch threads. "One by one" calls `Process(text)` per text (Python `nlp(text)`);
"bulk" makes one `Process(texts)` call (Python `nlp.bulk_process(texts)`). One timed run each, after a warm-up
on 50 texts:

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --models models\converted\en --threads 8 --documents 2000 [--package default_fast]
tools\.venv\Scripts\python tools\benchmark.py --models models\stanza --threads 8 --documents 2000 [--package default_fast]
```

| | C# default | Python default | C# default_fast | Python default_fast |
|---|---:|---:|---:|---:|
| one by one | 421.96 s (5 docs/s) | 366.33 s (5 docs/s) | 217.10 s (9 docs/s) | 130.06 s (15 docs/s) |
| bulk | 54.54 s (37 docs/s) | 91.87 s (22 docs/s) | 24.23 s (83 docs/s) | 32.20 s (62 docs/s) |
| speed-up | 7.7× | 4.0× | 9.0× | 4.0× |

- Bulk is 7.7–9× faster in C#, and 1.3–1.7× faster than Python's bulk.
- One by one, C# is *slower* than Python (by 15% and 67%): a single short sentence pays a fixed cost per call
  that bulk spreads over many sentences. Per call on one sentence (default_fast, 8 threads), C# vs Python:
  sentiment 53 vs 26 ms, tokenize 6.4 vs 2.2 ms, depparse 22.5 vs 20.3 ms, pos 11.7 vs 12.0 ms. That was the TorchSharp
  backend; on the managed backend (the default since 0.5) one call takes 5–7× less time than Python's (round 7).
- The output of bulk differs from one by one only in sentiment labels (and sentence ids), exactly as in Stanza:
  see `Pipeline.Process(IEnumerable<string>)`.

## Results, packages: default_fast vs default

`agent/default-fast`: `--package default_fast` on both benchmarks (`tools/benchmark.py --package`), and
the default package for comparison, on the usual text (26,264 words, 8 torch threads, medians of 3 runs
after a warm-up). For each package the C# CoNLL-U is byte-identical to Python's. The two packages give
different output: they are different models.

| stage | C# fast | Python fast | C# default | Python default |
|---|---:|---:|---:|---:|
| load | 0.95 s | 5.28 s | 1.34 s | 5.61 s |
| tokenize | 1.36 s | 1.77 s | 1.36 s | 1.90 s |
| mwt | 0.01 s | 0.32 s | 0.01 s | 0.31 s |
| pos | 3.55 s | 4.34 s | 12.11 s | 20.61 s |
| lemma | 0.82 s | 1.34 s | 0.79 s | 1.29 s |
| constituency | — | — | 9.10 s | 22.93 s |
| depparse | 6.61 s | 11.41 s | 13.30 s | 20.45 s |
| sentiment | 9.60 s | 12.43 s | 5.87 s | 11.91 s |
| ner | 1.44 s | 4.15 s | 11.99 s | 27.62 s |
| **total** | **23.40 s** | **35.76 s** | **54.54 s** | **107.03 s** |
| words/s | 1,122 | 734 | 482 | 245 |
| peak memory | 2,580 MB | 4,311 MB | 3,366 MB | 4,591 MB |

- `default_fast` is 2.3× faster than `default` in C# (3.0× in Python) and 1.5× faster than Python's
  `default_fast`.
- Its pos, depparse and ner run small character LSTMs (400 or 2×100 wide) instead of the two 1024-wide
  charlms; ner drops from 12.0 s to 1.4 s.
- Sentiment still runs the charlms, and is slower than in the default package (9.6 vs 5.9 s): with a
  nocharlm tagger there are no charlm outputs to reuse (`CharlmCache`), so it computes all of them. It is
  41% of the fast total.
- This branch also stopped the tagger from scoring batch padding: its 23 output heads ran on the padded
  `[batch, longest, hidden]` LSTM output, mostly padding when a batch of 250 sentences holds one long
  sentence. They now see the real words only, as Stanza's do (packed data). pos went from 7.0 to 3.6 s
  (fast) and from 16.1 to 12.1 s (default), with identical output.
- The machine was otherwise idle (no other benchmarks or agents' test runs at the same time).

## Results, all eight processors (0.1.0)

`main` with NER and sentiment, which are part of the default since 0.1.0-alpha.2. The input is the
usual text: 26,264 words, 8 torch threads, medians of 3 runs after a warm-up run. Both sides run the same
stages in Stanza's order, and the C# CoNLL-U output is byte-identical to Python's (`--out`, compared
with `cmp`).

| stage | C# | Python 1.15.0 | C# speed-up |
|---|---:|---:|---:|
| load | 1.28 s | 7.07 s | 5.5× |
| tokenize | 1.43 s | 1.82 s | 1.3× |
| mwt | 0.01 s | 0.31 s | |
| pos | 17.28 s | 21.33 s | 1.2× |
| lemma | 0.80 s | 1.33 s | 1.7× |
| constituency | 9.54 s | 24.51 s | 2.6× |
| depparse | 14.11 s | 21.24 s | 1.5× |
| sentiment | 6.20 s | 12.51 s | 2.0× |
| ner | 14.10 s | 29.25 s | 2.1× |
| **total** | **63.48 s** | **112.29 s** | **1.77×** |
| peak memory | 3,374 MB | 4,603 MB | |

- NER and sentiment add 20.3 s, about 32% of the C# total.
- Both reuse the tagger's charlm outputs through `CharlmCache` wherever the input is the same. That is
  most of why they are about 2× faster than Python, which runs the charlms again for each.
- No other agents or benchmarks were running. Load time is measured inside the process (after .NET
  start-up) and reads the converted models.

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
- **Very large documents.** `CharlmCache` is capped, but the document itself, and depparse's and
  the parser's per-document lists, still grow with the input. Callers with huge inputs should split
  them, for example by paragraph, and call `Process` per part (round 3 measures it).
- **Padding in depparse's input.** Done for the tagger in round 4. Depparse still builds its input and charlm outputs
  padded and pads `HighwayLstm`'s output for its scorers, which need [batch, width, width] anyway; its batches are
  sorted by length, so there is little padding to save. Building its input in packed order like the tagger would
  remove one padded copy of the 2423-wide input per batch.
- **Per-call cost on short texts.** Done in round 7 for the managed backend, which is now 6.8× faster than Python per
  call; what is left is memory bandwidth. Open:
  - *Concurrent callers* (owner's decision): more than about two concurrent `Process` calls lower the managed backend's
    throughput (8 callers: 23.6 calls/s, serialized: 36.4). `Pipeline` could let at most two calls run their networks
    at once and queue the rest, exactly (the output doesn't depend on it); today it is documented guidance.
  - *TorchSharp per call*: 192 ms (Python 189), libtorch's per-operation cost on tiny tensors; sentiment in
    `default_fast` is 49 ms against Python's 24. Not pursued, as the managed backend is the default.
  - *Depparse's forward charlm*: every sentence starts `"

 "` (start, ROOT, space), so those three steps could start
    from a stored state for single sentences (3 of ~90 steps).
- **Sentiment in default_fast.** With nothing to reuse it runs both charlms on every token. Only a faster
  charlm (or Stanza changing the package) would help.
- **Thread count.** On a loaded machine, fewer threads were often faster than the default 8,
  because spinning OpenMP threads compete. The README now says so; `torch.set_num_threads` is the
  knob.