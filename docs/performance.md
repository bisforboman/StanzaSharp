# Performance

How fast StanzaSharp runs compared with Python Stanza, what was optimized, and what is left.
The measurements here are of the TorchSharp backend (add `--backend torch` to the benchmark commands; since 0.5 the
default is the managed backend, measured in [backends.md](backends.md)), except rounds 6 and 7, which compare both,
and rounds 8 on, which measure the managed backend only.
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
shipped with the tool) cut after the paragraph that reaches 1,500 whitespace-separated words (`--words`), which is
2,009 words after tokenizing, the count the block reports; the managed backend; warm-up runs on the whole text until
each side settles (no stage more than 10% and 20 ms faster than in the run before; 2 to 5 runs, the block says how
many; round 16), then 3 timed runs with StanzaSharp and Stanza alternating (Stanza runs in one Python process that
stays loaded); medians per
processor; then 50 `Process` calls on one sentence each (the text's sentences of 5-25 words; median and p90, after 5
warm-up calls); the peak working set of each process (macOS: `ru_maxrss`). Both sides read the same `.pt` files, and
the block says whether their CoNLL-U was identical. Threads: each side's default unless `--threads` sets both
(StanzaSharp: the physical cores from 4 up, else all logical processors; torch: the physical cores). It prints a Markdown block with the CPU model, cores
and logical processors, RAM, OS, .NET version, SIMD path (Vector512, Vector256, Vector128 or Scalar) and StanzaSharp version,
ready to paste into an issue. About four minutes with Python on 4 cores; `--quick` (300 words, one warm-up run, one
timed run, 10 calls) is a smoke test.

On GitHub's hosted runners: **Actions > Benchmark > Run workflow** (or `gh workflow run benchmark.yml -f package=default`)
runs it with Python Stanza on ubuntu-26.04, ubuntu-26.04-arm, windows-2025, windows-11-arm and macos-26 (free standard
runners only). Each job puts its block in its summary and the artifact `benchmark-<runner>`; the last job combines them
into one table (summary and artifact `benchmarks`, `benchmarks.md`, made by `tools/benchmark_table.py`). Hosted runners
are shared virtual machines (the macOS one is virtualized Apple Silicon), so compare numbers within one run. The block
names the StanzaSharp version as `git describe` gives it (`1.0.0-67-g97089bb`: 67 commits after v1.0.0), since the
workflow builds from source.

Latest run ([38293654596](https://github.com/bisforboman/StanzaSharp/actions/runs/38293654596), 2026-10-12, commit
093e7c2, after round 16: the new warm-up, so not directly comparable with the runs below, which warmed up on a tenth
of the text). The CoNLL-U was identical on every runner, Python Stanza included on windows-11-arm for the first time:

| runner | CPU, cores/logical | threads | words/s | speed-up | per call, median ms | peak memory MB |
|---|---|---|---:|---:|---:|---:|
| ubuntu-26.04 | AMD EPYC 7763, 2/4 | 4 / 2 | 250 / 66 | 3.81x | 83.5 / 299.3 | 1,051 / 3,539 |
| ubuntu-26.04-arm | Neoverse-N2, 4/4 | 4 / 4 | 240 / 80 | 2.99x | 83.2 / 642.1 | 1,088 / 3,712 |
| windows-2025 | AMD EPYC 7763, 2/4 | 4 / 2 | 249 / 59 | 4.21x | 85.3 / 348.8 | 1,037 / 3,597 |
| windows-11-arm | Cobalt 100, 4/4 | 4 / 4 | 228 / 29 | 7.95x | 86.4 / 560.1 | 908 / 3,434 |
| macos-26 | Apple M1 (virtual), 3/3 | 3 / 3 | 218 / 64 | 3.42x | 107.5 / 511.4 | 832 / 2,531 |

- StanzaSharp now leads in every stage on every runner: tokenize 4.2–20x, ner 4.0–11x, constituency 3.3–10.7x,
  depparse 2.6–8.0x, pos 1.4–7.0x. Load took 1.7–4.3 s against 5.2–8.7 s.
- windows-11-arm's Python Stanza (Arm64 CPython, torch 2.14.1+cpu) is the slowest anywhere: 29 words/s, a third of
  ubuntu-26.04-arm's on the same kind of core.
- StanzaSharp took 4–5 warm-up runs, Python Stanza 2–3.

Previous run ([38203172080](https://github.com/bisforboman/StanzaSharp/actions/runs/38203172080), 2026-10-12, commit d2e93b4, after round 12 and the move to ubuntu-26.04 and
macos-26; same text and settings). The CoNLL-U was identical on every runner with Python:

| runner | CPU, cores/logical | threads | words/s | speed-up | per call, median ms | peak memory MB |
|---|---|---|---:|---:|---:|---:|
| ubuntu-26.04 | Intel Xeon Platinum 8573C, 2/4 | 4 / 2 | 217 / 105 | 2.06x | 131.1 / 284.5 | 1,140 / 3,557 |
| ubuntu-26.04-arm | Neoverse-N2, 4/4 | 4 / 4 | 219 / 80 | 2.73x | 91.0 / 639.9 | 1,145 / 3,708 |
| windows-2025 | AMD EPYC 7763, 2/4 | 4 / 2 | 238 / 63 | 3.77x | 73.2 / 323.1 | 1,065 / 3,594 |
| windows-11-arm | Cobalt 100, 4/4 | 4 | 219 / - | - | 91.3 / - | 1,023 / - |
| macos-26 | Apple M1 (virtual), 3/3 | 3 / 3 | 203 / 62 | 3.25x | 117.8 / 397.3 | 865 / 2,226 |

- ubuntu-26.04 got an Intel Xeon (Emerald Rapids) this time instead of the AMD EPYC, and Python Stanza ran about 40%
  faster there than in the first run (likely torch's AVX-512 kernels; StanzaSharp's stop at AVX2, Vector256), so the
  gap narrows to 2.1x. Not measured on the same machine.
- Per stage StanzaSharp led everywhere except tokenize on macos-26 (0.70x) and ubuntu-26.04 (0.97x); ner 3.2–7.5x,
  constituency 1.6–4.7x, depparse 2.0–6.1x. Load took 1.8–3.3 s against 4.2–7.6 s.
- The workflow's Python Stanza install still failed on windows-11-arm: numpy 2.2.6 has no win_arm64 wheel (numpy's first
  is 2.3.0), and pip's source build fails with the runner's x64 MinGW gcc. The workflow now installs numpy 2.3.5 on that
  runner only (Stanza 1.15.0 and torch 2.14.1+cpu as everywhere, Arm64 CPython); a quick run there gave identical
  CoNLL-U on both sides (run 38274497249).
- Round 15 (below) made tokenize 1.7–2.7x faster on these runners; it now leads Python Stanza on every one.

First run ([38150941695](https://github.com/bisforboman/StanzaSharp/actions/runs/38150941695), 2026-10-11, commit
6a08cf6, before round 12's pool change and CI's move to ubuntu-26.04 and macos-26; `default` package, 2,009 words, each
side's default threads; pairs are StanzaSharp / Python Stanza). The CoNLL-U was identical on every runner with Python:

| runner | CPU, cores/logical | threads | words/s | speed-up | per call, median ms | peak memory MB |
|---|---|---|---:|---:|---:|---:|
| ubuntu-24.04 | AMD EPYC 7763, 2/4 | 4 / 2 | 246 / 75 | 3.29x | 68.8 / 248.8 | 1,143 / 3,471 |
| ubuntu-24.04-arm | Neoverse-N2, 4/4 | 4 / 4 | 228 / 81 | 2.83x | 81.2 / 678.4 | 1,166 / 3,643 |
| windows-2025 | AMD EPYC 7763, 2/4 | 4 / 2 | 238 / 63 | 3.77x | 73.6 / 327.0 | 1,120 / 3,600 |
| windows-11-arm | Cobalt 100, 4/4 | 4 | 221 / - | - | 85.9 / - | 1,016 / - |
| macos-15 | Apple M1 (virtual), 3/3 | 3 / 3 | 183 / 60 | 3.04x | 765.9 / 338.5 | 877 / 2,454 |

- Python Stanza's install step failed on windows-11-arm in that run, so that row is StanzaSharp only.
- The macOS per-call time was Spotlight indexing (below and round 12); the workflow turns it off since.
- Per stage, StanzaSharp was ahead in every processor on every runner; the largest gains were in ner (3.9–7.0x),
  constituency (3.4–4.7x) and depparse (2.3–3.9x). Load took 1.7–3.1 s against 5.0–6.9 s.

On a fresh macOS runner Spotlight indexes the disk for its first minutes, and again after pip installs torch:
`mds`, `mds_stores` and several `mdworker_shared` take about half of the 3 vCPUs. One-sentence calls then took 717-845 ms
instead of 94 (Python Stanza was hardly affected: 496-586 ms in those runs, 383-716 in the others; see round 12), so the workflow turns
indexing off first (`sudo mdutil -a -i off`). On your own machine, measure while it is otherwise idle: StanzaSharp's
per-call times are more sensitive to other busy processes than its throughput is.

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

## Results, round 19: Vector512 on Intel, and two panels per LSTM step

Round 14 measured Vector512 only on AMD. The Benchmark workflow's `avx512` jobs (`avx512_only`) now also try three
windows-2025 runners (Git Bash's `/proc/cpuinfo` has the flags too) and print the kernels' GEMM rates per path
(`managed-spike gemm`). In 8 dispatches (69 jobs) the CPUs drawn were mostly the EPYC 7763 (no AVX-512) and the EPYC 9V74
(about a third of those with AVX-512 masked), then the EPYC 9V45. Intel came up 8 times: a Xeon Platinum 8573C (Emerald
Rapids) 5 times (3 on ubuntu-26.04, 2 on windows-2025), a Xeon 6973P-C (Granite Rapids) twice, both on ubuntu-26.04, and a
Xeon Platinum 8370C (Ice Lake) once. Same method as round 14 (2,009 words, `default` package, Vector256 and Vector512
alternated on one machine, two rounds each; totals of both rounds).

**Before the change** (main, 1.3.0):

| CPU (2 cores / 4 logical) | threads | Vector256 total | Vector512 total | gain | per call, median |
|---|---:|---:|---:|---:|---:|
| Xeon 6973P-C (Granite Rapids) | 4 | 8.21–8.39 s | 6.82–6.97 s | +18–23% | 133–136 → 119–124 ms |
| Xeon 6973P-C (Granite Rapids) | 1 | 18.27–18.91 s | 15.15–15.97 s | +14–25% | 278–293 → 237–253 ms |
| Xeon Platinum 8573C (Emerald Rapids), Windows | 4 | 10.87–11.01 s | 8.50–8.65 s | +26–30% | 159–169 → 141–150 ms |
| Xeon Platinum 8573C (Emerald Rapids), Windows | 1 | 24.49–24.79 s | 17.99–18.14 s | +35–38% | 323–328 → 268–271 ms |

Runs [38308654625](https://github.com/bisforboman/StanzaSharp/actions/runs/38308654625) and
[38311028336](https://github.com/bisforboman/StanzaSharp/actions/runs/38311028336), 2026-10-12. A second Granite Rapids
draw ([38311308180](https://github.com/bisforboman/StanzaSharp/actions/runs/38311308180)) was a noisy machine (4 threads
barely faster than 1) but agreed: +14% at 4 threads, +20–22% at 1.

So Vector512 was faster on Intel too, but by less than on Zen 5, and `Speed_IsReported` showed why: the charlm, which is
the LSTM recurrence, gained only 8% on Granite Rapids (263 → 243 ms), while the highway gained 32%. `PackedLstm.Step`
called the blocked kernel one panel at a time. On Vector512 that is 6 accumulators, and per k it does 1 panel load and 6
broadcasts for 6 FMAs, while the GEMM's `KernelPair512` does 2 panel loads and 6 broadcasts for 12 FMAs.

**The change.** `PackedLstm.Step` hands Vector512 two adjacent panels of its task per call
(`Gemm.KernelBlockedPair512`: 12 accumulators for 6 rows, 6 for 3), then runs the LSTM cell on each panel. Each column
gets the single-panel kernel's FMAs in the same order and the same flushes, so the results are still Vector256's bit for
bit (`Vector512_GivesVector256sBits`: its charlm and highway cases go through paired and single panels). Measured with
main's tool (Vector512) built and alternated in the same job (a temporary step on the branch); totals are each round's, gains are each job's mean of
its two rounds (the range over jobs). The JIT's code for the pair is what you would write: 2 `vmovups`, 6
`vbroadcastss` from memory and 12 `vfmadd231ps` per k.

| CPU | threads | Vector256 | Vector512 before | Vector512 after | after vs before | after vs Vector256 |
|---|---:|---:|---:|---:|---:|---:|
| Xeon Platinum 8573C (Emerald Rapids), 3 jobs | 4 | 9.10–10.17 s | 6.92–7.78 s | 6.69–7.25 s | +5–7% | +37–39% |
| Xeon Platinum 8573C (Emerald Rapids), 3 jobs | 1 | 20.96–22.94 s | 14.81–16.28 s | 13.27–15.13 s | +7–9% | +51–54% |
| Xeon Platinum 8370C (Ice Lake) | 4 | 9.94–9.95 s | 10.54–10.65 s | 9.51–9.59 s | +11% | +4% |
| Xeon Platinum 8370C (Ice Lake) | 1 | 21.86–22.06 s | 23.12–23.20 s | 20.75 s | +12% | +6% |
| AMD EPYC 9V45 (Zen 5), 6 jobs | 4 | 5.10–5.66 s | 3.36–3.78 s | 2.96–3.27 s | +11–15% | +65–76% |
| AMD EPYC 9V45 (Zen 5), 6 jobs | 1 | 10.26–11.81 s | 6.56–7.71 s | 5.71–7.07 s | +11–15% | +73–82% |
| AMD EPYC 9V74 (Zen 4), 4 jobs | 4 | 6.95–7.60 s | 6.15–6.45 s | 6.21–6.63 s | −3 to +1% | +12–16% |
| AMD EPYC 9V74 (Zen 4), 4 jobs | 1 | 13.53–15.11 s | 12.83–13.96 s | 12.60–13.31 s | +3–5% | +8–16% |

Runs [38312471435](https://github.com/bisforboman/StanzaSharp/actions/runs/38312471435),
[38314372520](https://github.com/bisforboman/StanzaSharp/actions/runs/38314372520),
[38317679896](https://github.com/bisforboman/StanzaSharp/actions/runs/38317679896) and
[38318677685](https://github.com/bisforboman/StanzaSharp/actions/runs/38318677685), 2026-10-13. A fourth 8573C job, on
windows-2025, agrees at 1 thread (+9% over before, +50% over Vector256); its 4-thread rounds were disturbed (one
Vector256 round had a 592 ms median call). Per stage on the Emerald Rapids at 4 threads (run 38312471435), Vector256 → before → after: pos 1.24 → 1.10 → 0.96 s, depparse 2.20 → 1.83 → 1.65 s, sentiment
2.49 → 1.92 → 1.83 s, constituency 2.63 → 1.74 → 1.73 s, ner 0.71 → 0.63 → 0.60 s. `Speed_IsReported`'s charlm, Vector256
→ Vector512: 250 → 184 ms there (0.74x; Granite Rapids before the change: 0.92x), 147 → 87 ms on the 9V45.

- **Ice Lake** was the only CPU where Vector512 lost before the change: 5–7% slower than Vector256 in total (pos, depparse
  and ner slower, constituency faster). After it, it is 4–6% faster in total; pos and depparse at 1 thread are still about
  3–6% slower than on Vector256, constituency 20% faster. One draw only. Its GEMM rates show the AVX-512 frequency drop
  of that generation: at 1 thread Vector512 beats Vector256 by 7–16% (80.9 vs 75.7 GFLOP/s on the charlm's shape), at 4
  threads it loses by 4% (150 vs 157). Not enough to make a CPU rule worth its risk: Vector512 is ahead overall there now.
- **No throttling on the newer Intel parts**: on the Emerald Rapids at 1 thread Vector512 reached 134–156 GFLOP/s against
  Vector256's 74–81 (torch's AVX-512 kernels 170–196), so the 512-bit FMAs run at nearly twice the 256-bit rate.
- **No vendor rule.** Vector512 is faster than Vector256 on every Intel and AMD CPU drawn, so it stays the default
  wherever .NET accelerates it. (.NET itself leaves `Vector512` unaccelerated on Skylake-X and Cascade Lake, which
  throttle hardest.)
- **Where Intel loses to Zen 5 is not the kernels.** The 9V45 runs the pipeline 2.3–3.5x faster than these Xeons with the
  same code, most of all constituency (0.41 s against 1.65–2.1 s at 4 threads). The single-row GEMM, which streams the 16 MB
  charlm-sized weight matrix, ran at 13–16 GFLOP/s on the Xeons against 43–48 on the 9V45 (and so did torch's): about a
  third of the memory bandwidth per core. The constituency parser's many small steps (a few rows against large weights)
  are bound by that, so wider FMAs can't help them; only reading the weights fewer times would.
- Not tried: more accumulators. The pair already has 12 independent chains, more than the 8 that
  two FMA ports with a 4-cycle latency need, and its broadcasts come from memory (load ports, not the shuffle port that
  Intel's second FMA unit shares).

## Results, round 18: depparse's forward charlm start; garbage on long texts

Two small items from "Remaining ideas". One shipped (the charlm's character ids), one didn't (the stored start state).

### The `ArrayPool`'s churn wasn't the garbage; the charlm's character strings were

Round 11 guessed that what is left of the garbage on a long call is mostly `ArrayPool<T>.Shared` trimming idle buffers at a
gen2 GC and renting fresh ones later. Measured with `--memory 150000 --verbose` (201,313 words, all eight processors,
8 threads, Windows, one process each; other sessions' jobs loaded the machine, so times are not compared):

| | allocated by the call | peak | gen2 GCs |
|---|---:|---:|---:|
| main | 3,673 MB (18.2 KB/word) | 1,946 MB | 20 |
| main, `DOTNET_SYSTEM_BUFFERS_ARRAYPOOL_TRIMSHARED=false` (the pool never trims) | 3,674 MB | 1,948 MB | 20 |
| this round | **3,064 MB (15.2 KB/word)** | 1,958 MB | 19 |

With trimming off, every processor allocated the same to the megabyte: the pool's churn costs nothing here, so a per-call
pool or holding the call's largest buffers can't help (not tried). An in-process `EventListener` (temporary) on the
GC's allocation ticks and `ArrayPoolEventSource` showed where it goes instead: pool misses (a rent that finds no array)
add ~60M elements between 20,513 and 201,313 words (219 buffers trimmed, none needed again), while per-character
objects grow with the text: `String` +290 MB, `Int32[]` +380 MB, boxed `StringRuneEnumerator` +120 MB and a LINQ
`Select` iterator over it +100 MB. They come from `ManagedCharLanguageModel`'s input: every character became a
one-rune string (`EnumerateRunes().Select(r => r.ToString())`) for the vocabulary lookup, and the ids grew in `List<int>`s.
The charlms run over every word several times (pos, depparse, and sentiment, NER and constituency where the cache
misses), so this was ~3 KB per word.

**What changed:** the ids are looked up from each rune's UTF-16 in a stack buffer through the vocabulary's
`GetAlternateLookup<ReadOnlySpan<char>>()`, into an array of the exact length (code points counted first), and the
backward charlm fills it from the end instead of reversing enumerables. The same ids, so the same bits. Per processor
at 201,313 words, allocated MB before → after: pos 649 → 516, constituency 1,024 → 914, depparse 646 → 505, sentiment
199 → 86, ner 439 → 326 (tokenize, mwt and lemma unchanged): −610 MB, −17%. The peak is unchanged within its
±50 MB run-to-run noise: it is set by live data (the models, the document, the cache and the largest batch's buffers),
not by these short-lived objects.

What is left (the same trace): `Int32[]` and `String` outside the charlm (tokenizer inputs, the document), constituency's
`StackState`s (~125 MB), `bool[]`, `object[]` and `string[]` of ~110 MB each; each under 1 KB per word.

### Depparse's forward charlm start: exact, but under the noise

The parser feeds the forward charlm `"\n"` (start), `"\n"` (ROOT) and `" "` before every sentence's words. A row's
arithmetic in an LSTM step doesn't depend on the other rows: since every step sums h·W_hhᵀ through `Gemm.KernelBlocked`
(2026-10-10, docs/backends.md), which has a 3-row and a 6-row kernel with the same per-row FMA order and no single-row
one, so the state after those three steps is the same in any batch. A version that computed it once per kernel path (lazily, read-only after) and ran every
sentence on from it gave the same bits: the depparse arc log-probs, label scores and labels of all 845 golden sentences
alone and in batches of 40, the parser's CoNLL-U on every golden depparse file, the `tokenize,mwt,pos,lemma,depparse`
CoNLL-U of every golden text and of all of them as one document, and 60 one-sentence calls, SHA-256 on Vector256,
Vector128 and Scalar at 1 and 8 threads: identical to main (Vector256 alone `B1272F36…`, batched `E09AECE1…`; CoNLL-U
`84599FFE…` and pipeline `EF8F01BD…` on every path).

It saves 3 of the forward charlm's ~90 steps per sentence. Per-call A/B (same binary, an environment switch, alternated,
`--per-call 300 --processors tokenize,mwt,pos,lemma,depparse`, 8 threads, four undisturbed rounds): depparse median
11.79 ms before, 11.71 after; the rounds themselves spread ±0.6 ms. Timed alone (200 golden sentences of 5–25 words, one at a time, three
processes): the forward charlm took 2.82 / 2.88 / 3.09 ms per sentence (48 steps on average) and 2.62 / 2.75 / 2.96
ms from the stored state, 0.13–0.2 ms saved, under 1% of a 23 ms five-processor call and about 0.5% of a 40 ms
all-eight one. Under the noise of a call, so not shipped.

## Results, round 17: startup (load and the first calls)

A short-lived process (a CLI run, a serverless function, a one-off job) pays for `Pipeline.Load` and for its first
calls, which run while the JIT is still at tier 0. Output unchanged, bit for bit (below).

**Method.** `StanzaSharp.Benchmark startup` loads the default package (all eight processors, default threads) in a
fresh process and makes a few `Process` calls, each with the methods the JIT compiled meanwhile (`JitInfo`) and the
process's CPU time; `tools/startup.py` runs it once per process under each setting, the settings alternating within
each round, and tabulates medians and the CoNLL-U hash. The Benchmark workflow's `startup` jobs (`gh workflow run
benchmark.yml -f startup_only=true [-f startup_base=origin/main] [-f startup_settings=...]`) do that on the five free
runners, with Stanza's `.pt` files, for two texts: one 13-word sentence per call (5 processes, 6 calls) and the
benchmark text, 2,009 words per call (3 processes, 4 calls). Settings: a plain build (JIT), a ReadyToRun publish of
the same app (`-r <rid> -p:PublishReadyToRun=true`), `DOTNET_*` knobs, and with `startup_base` the library at another
ref under the same harness. Load peaks are the working set's peak after `Load` (not measured on macOS).

### Where the time went

Locally (Ryzen 7 5800X, Windows; a machine busy with other work, so only shares are reliable), `Pipeline.Load` took
2.2–3.0 s with converted models and 3.5–4.2 s with `.pt` files:
- **Checkpoint reads:** parsing (`Checkpoint.Load`: the JSON, or the pickle mirrored into JSON) 0.6–0.9 s converted,
  1.6–1.9 s `.pt`; tensor bytes 0.2–0.5 s (630 MB). Almost all of the parsing is the two 250k-word vocabularies:
  the pretrain's and NER's `word` vocab (conll17's again). Warm, in a loop, the same reads take a fraction of that
  (pretrain `.pt` 0.29 s + 0.04 s for its vocabulary, converted 0.04 + 0.07 s; NER 0.33 + 0.06 / 0.05 + 0.12 s): the
  rest of the cold cost is the parsers' code at tier 0.
- **JIT:** 700–1,200 methods, 0.1–0.35 s of JIT time on the runners (`JitInfo`, all threads).
- **Packing** (repacking weights into GEMM panels, `BiaffinePerUpos`, gate reordering) and a GC after each model: the
  rest, about half. Each model's load is one thread's work, and the models loaded one after another.

The first one-sentence call compiled another ~770 methods. A JIT summary (`DOTNET_JitDisasmSummary=1`) of the first
three calls showed what ran unoptimized:
- **`VectorMath.ExpSingle`**, inside every sigmoid and tanh: the runtime's `Vector128/256/512.Exp` is too large to
  inline, and its ReadyToRun code is rejected (it depends on the machine's vector width), so it started at tier 0 even
  inside our fully optimized kernels (the LSTM step, the GEMM epilogues), with its `Widen`/`Narrow` helpers as calls.
- **The thread pool's region machinery** (`ManagedThreads.For`, `Pool.Run`, `Region.Drain`, `PackedLstm.Step`), which
  an LSTM enters once per time step, thousands of times per call, stayed at instrumented tier 0 for the first calls.
- `Act.LstmCell` and `Act.AddGatedTanh` (inlined into optimized callers) were compiled at tier 0 for the callers still
  at tier 0 (the constituency parser's stack LSTM, the lemmatizer's decoder, the highway LSTM).

So on the runners the 4th to 6th one-sentence call of a fresh process took 88–218 ms, while a warmed process
(`stanzasharp benchmark`) takes 63–136 ms.

### What changed

- **`VectorExp`** (`Nn/Managed/VectorExp.cs`): .NET 10's `ExpSingle`, copied operation for operation (the same double
  vector width, the same conversions, `MultiplyAddEstimate` in the same order), `AggressiveInlining |
  AggressiveOptimization`. `Act.Sigmoid` uses it. It gives the runtime's bits for all 2^32 float inputs (checked on
  Vector256, with AVX off and with AVX2 off, and on AVX-512 on a runner); `ManagedBackendTests.VectorExp_GivesTheRuntimesBits`
  checks every 61st pattern on each machine CI runs.
- **`AggressiveOptimization`** on `ManagedThreads.For`, the pool's `Run`, `Help`, `Work` and `Region.Drain`,
  `PackedLstm.Step`, `Act.LstmCell` and `Act.AddGatedTanh`: optimized from the first call. Optimized callers still
  inline the last two (checked with `DOTNET_JitDisasm`), and none of them gained from PGO: the warmed calls 4+ on the
  benchmark text were unchanged on every runner (table below).
- **Two models load at a time** (managed backend; `BackendModels.LoadInParallel`, `Pipeline.LoadLanes`): the pretrain,
  the charlms, tokenize, mwt and lemma first, the pretrain's readers once it is loaded, at most two at a time on the
  thread pool, each still followed by a GC. TorchSharp loads as before (its device and dispose scopes are per thread).
  A load failure still throws the first failing model's own exception, after the other loads finish.

Every model at once, each followed by a GC, took 28–46% less time on the runners and peaked 95–152 MB higher (without
the GCs 250–310 MB); two at a time keeps most of the gain for a quarter to a half of the extra memory: 23–40% less
time, 22–72 MB more
(run [38315161541](https://github.com/bisforboman/StanzaSharp/actions/runs/38315161541)):

| runner | load, one at a time | two at a time | all at once (GC after each) | load peak MB: one / two / all |
|---|---:|---:|---:|---|
| macos-26 | 1,558 ms | 1,177 | 1,001 | - |
| ubuntu-26.04 | 2,613 | 1,565 | 1,409 | 847 / 869 / 969 |
| ubuntu-26.04-arm | 1,787 | 1,260 | 1,215 | 937 / 971 / 1,089 |
| windows-11-arm | 1,601 | 1,223 | 1,017 | 904 / 956 / 1,054 |
| windows-2025 | 1,253 | 969 | 906 | 839 / 911 / 934 |

**Before and after** (run [38317542272](https://github.com/bisforboman/StanzaSharp/actions/runs/38317542272): main
and this branch under the same harness, alternating; ms; one-sentence text, then the benchmark text):

| runner | load | load peak MB | call 1 | calls 4–6 | load + 6 calls | benchmark text: load + 4 calls | calls 4+ (warm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| macos-26 | 1,124 → **890** | - | 192 → 183 | 115 → **95** | 1,889 → **1,551** (-18%) | 28.0 → 27.2 s | 6,518 → 6,098 |
| ubuntu-26.04 | 1,773 → **1,251** | 849 → 938 | 216 → 192 | 88 → **71** | 2,547 → **1,839** (-28%) | 28.4 → 26.9 s | 6,269 → 6,105 |
| ubuntu-26.04-arm | 1,845 → **1,340** | 936 → 1,000 | 262 → 239 | 135 → **95** | 2,806 → **2,219** (-21%) | 37.2 → 36.2 s | 8,466 → 8,467 |
| windows-11-arm | 1,673 → **1,291** | 904 → 965 | 293 → 264 | 218 → **104** | 2,934 → **2,365** (-19%) | 36.8 → 36.0 s | 8,400 → 8,326 |
| windows-2025 | 2,301 → **1,439** | 839 → 864 | 329 → 304 | 190 → **118** | 3,587 → **2,540** (-29%) | 36.5 → 35.2 s | 7,912 → 7,896 |

The load takes 21–37% less, its peak is 25–89 MB higher (+3–11%; a call on a longer text peaks higher anyway, about
1.0–1.1 GB on the benchmark text), the first call 5–11% less and the next ones 17–52% less. `stanzasharp benchmark`
on the branch (run [38317573839](https://github.com/bisforboman/StanzaSharp/actions/runs/38317573839)) loaded in
1.2–1.8 s on four runners (3.2 s on macos-26) against 1.7–4.3 s in the latest run above, with identical CoNLL-U to
Python Stanza on every runner; its words/s were within the run-to-run spread.

Tried and not kept: the NER checkpoint read started before the pretrain was loaded (the two 250k-word vocabularies
are the longest chain: pretrain, then NER): with every model at once another 5–10% off the load locally, but 100 MB
more peak with converted models.

### ReadyToRun

`PublishReadyToRun` doesn't apply to a library's own package; a library ships ReadyToRun code by compiling its
assemblies with Crossgen per RID and packing them as `runtimes/<rid>/lib/net10.0/` assets. Measured with a ReadyToRun
publish of the app instead (`-r <rid> -p:PublishReadyToRun=true`), which compiles every assembly the app ships, ours
included, so it is what such a package would give:
- **Size:** the package's assemblies are 470 KB of IL and 1.2 MB with win-x64 ReadyToRun code (600 KB zipped). One
  set per RID (linux-x64, linux-musl-x64, linux-arm64, win-x64, win-arm64, osx-arm64) would take the package from
  277 KB to about 3.9 MB.
- **What it covers:** with ReadyToRun the first three one-sentence calls compiled about 185 of our methods instead of 876,
  but not the hot ones: the GEMM kernels are `AggressiveOptimization` (never precompiled), and every method touching
  `Vector256`/`Vector512` (`Act.*`, the runtime's own `ExpSingle`) is rejected at run time and JIT-compiled anyway,
  also when compiled with `--instruction-set:x86-64-v3` (they reference `Vector512`, whose support isn't known at
  compile time). Precompiled code is replaced at tier 1 when hot, as usual.
- **Effect** (same run, this branch): call 1 23–38% faster (183–304 → 126–188 ms), the load the same (±5%), load + 6
  one-sentence calls −1 to −5% on four runners (macos-26: +32%, noisy), the benchmark text the same.

So it saves about 50–120 ms on the first call for 14 times the package size; not shipped. An app that wants it gets
it from its own `PublishReadyToRun`.

### Settings a user can choose

Same run, this branch, process totals (load + 6 one-sentence calls; load + 4 benchmark-text calls):

| setting | one sentence | benchmark text | notes |
|---|---:|---:|---|
| default | 1,551–2,540 ms | 26.9–36.2 s | |
| `DOTNET_TieredPGO=0` | -3 to +5% | -2 to -4% | |
| ReadyToRun publish | -5 to +32% | -1 to +1% | call 1 -23 to -38% |
| ReadyToRun + `TieredPGO=0` | -12 to +2% | -4 to +1% | |
| `DOTNET_TieredCompilation=0` | **0 to -16%** | -3 to -6% | call 1 +40 to +52%, later calls the fastest |
| ReadyToRun + `TieredCompilation=0` | +1 to -24% | **+19 to +58% on Arm64**, -3 to -5% on x64 | precompiled code is never re-optimized |

`DOTNET_TieredCompilation=0` (or `<TieredCompilation>false</TieredCompilation>` in the app's project) suits a
short-lived process: everything is compiled optimized once, so the first call is slower and every later one is as
fast as in a warm service. Before this round it made the benchmark text 28–43% slower on the Arm64 runners (the runtime's
`ExpSingle` stayed a call without PGO's inlining); with `VectorExp` it is as fast as the default there. Don't combine
it with ReadyToRun on Arm64 for long texts. `DOTNET_TC_QuickJitForLoops=0` (run
[38312449894](https://github.com/bisforboman/StanzaSharp/actions/runs/38312449894), before this round's changes) was
mixed (-17% to +23% on the one-sentence process).

### Exactness

The CoNLL-U of every `tests/golden/*.txt` (SHA-256, each text processed alone), main against this branch, at 1 and 4
threads: identical (`4FD0F4EE…`), also with `DOTNET_EnableAVX2=0` (the Vector128 path), with tiering off and from
the `.pt` files (the Scalar path is covered by ManagedBackendTests). Every startup process on every runner, every
setting (ReadyToRun, the knobs, main), gave the same CoNLL-U for both texts; `stanzasharp benchmark` matched Python
Stanza on all five runners. `VectorExp` equals the runtime's `Exp` on all 2^32 inputs (above).
PipelineTests, OptionsTests and ManagedBackendTests pass (93 tests, converted models).

## Results, round 16: tokenizer follow-ups (LSTM step split, warm-up, URL prefilter)

Round 15's three leftovers. Output unchanged, bit for bit (below).

### Four tasks per thread in an LSTM step

`PackedLstm.Step` split each time step into one task of whole 16-column panels per thread; a step is a parallel region
that waits for its slowest task. The tokenizer's LSTMs (hidden 64: 32 panels over both directions, a few µs per step)
run thousands of such steps on the benchmark text, and on macos-26 (3 vCPUs) the default 3 threads were slower than 1.

What was tried, on the free runners (experiment runs [38288718064](https://github.com/bisforboman/StanzaSharp/actions/runs/38288718064),
[38291143656](https://github.com/bisforboman/StanzaSharp/actions/runs/38291143656) and [38292254170](https://github.com/bisforboman/StanzaSharp/actions/runs/38292254170);
each setting in its own process, 3–5 rounds, with this round's warm-up):
- **A work threshold** (fewer tasks when rows × panels × 16 × K multiply-adds are few): no threshold helped everywhere.
  16k multiply-adds per task (one row of the tokenizer → 2 tasks) made macos-26 ~40% faster but ubuntu-26.04-arm
  10–35% slower; there even one-row steps (32k multiply-adds, ~8k per task) are fastest on all 4 threads.
- **Measured per step size**: time each step, per power-of-two size class, under each task count (4, 2, 1), then run the
  fastest. Better on macos-26 (90 → 66 ms) but 25–50% worse on both Arm runners: a parallel candidate timed while
  workers were asleep (a class had just run serially) looked slow and was then rarely retried. Dropped.
- **The caller's wait**: spinning instead of `SpinWait.SpinOnce` (which yields) when waiting for the last task: mixed,
  within noise except on macOS (~20% faster).
- **More tasks than threads** (2, 4, 8 per thread, or one per panel): 2 or 4 per thread was the fastest tokenizer on
  every runner, and 4 also sped up the big LSTMs (below). A thread that runs
  late (a vCPU the host is busy with, a slower core, an SMT sibling) then holds a quarter of its share, while the
  others claim the rest; with one task per thread the step waits for its whole share.

**What changed:** `Step` runs min(4 × threads, panels) tasks (`TasksPerThread`). A panel's arithmetic (its rows in
blocks of `Gemm.MR`, `KernelBlocked`, the cell update) doesn't depend on which task runs it, so the split can't change a
result; the pool already hands out a region's items one at a time.

Same-VM A/B (run 38292254170: each runner ran both settings alternately; tokenize alone is the median of 5 processes,
the full pipeline of 2; ms):

| runner | tokenize alone, 1 → 4 tasks/thread | all eight: total | constituency | depparse | per call |
|---|---:|---:|---:|---:|---:|
| macos-26 | 84 → **60** | noisy (7,878 → 9,057, one run 10.2 s) | 1,071 → 1,104 | 2,021 → 2,615 | 117.1 → 107.5 |
| ubuntu-26.04 (EPYC 9V74) | 89 → 86 | 6,383 → **6,248** (-2%) | 853 → 822 | 1,759 → 1,717 | 60.1 → 60.9 |
| ubuntu-26.04-arm | 73 → **63** | 8,776 → **8,441** (-4%) | 1,112 → **997** | 2,540 → 2,383 | 94.1 → 90.7 |
| windows-11-arm | 72 → 75 | 8,659 → **8,396** (-3%) | 1,107 → **1,014** | 2,498 → 2,350 | 85.4 → 84.9 |
| windows-2025 (EPYC 7763) | 104 → **94** | 8,174 → **7,966** (-2.5%) | 1,165 → 1,113 | 2,239 → 2,161 | 76.6 → 75.4 |

Locally (Ryzen 7 5800X, 8 threads, alternating, 5 and 3 processes): tokenize alone 52.9 → 45.1 ms; all eight: pos -3%,
constituency -4%, depparse -7%, ner -3%, total -3%, one-sentence calls 39.9 → 39.0 ms. The one cost: a one-sentence
`tokenize,mwt` call, all tiny steps, takes ~0.07 ms longer (0.38 → 0.45 ms here; +0.1 ms on the Arm runners, less
elsewhere), more claims for the same work; with all eight processors it is lost in the noise of a ~40–100 ms call.

### Benchmark warm-up

`stanzasharp benchmark` warmed up on a tenth of the text, which has no long paragraph, so the tokenizer's windowed path
was first compiled in the timed runs, and its per-step methods stayed at JIT tier 0 through all three. A warmed service
is faster than the numbers showed. Now each side runs the whole text until it settles: no stage got more than 10% and
20 ms faster than in its previous warm-up run (at least 2 runs, at most 5; `--quick`: 1). The first criterion tried,
10% alone, never settled on the runners (some 50 ms stage always moved 10%), so the noise floor of 20 ms was added.
Python Stanza goes through the same loop; its stages settle after 2–3 runs. The block names both counts ("after 4
warm-up runs (Stanza 2)").

**Numbers from before this round aren't directly comparable.** How much they move, on the same runner type (runs
[38288777481](https://github.com/bisforboman/StanzaSharp/actions/runs/38288777481) main and
[38288755731](https://github.com/bisforboman/StanzaSharp/actions/runs/38288755731) the warm-up change alone; only the
two Arm runners drew the same CPU in both):

| | StanzaSharp total | tokenize | lemma | other stages | Python Stanza total |
|---|---:|---:|---:|---:|---:|
| ubuntu-26.04-arm | 9.03 → 8.60 s (-5%) | 110 → 62 ms | 86 → 53 ms | -3 to -5% | -1% |
| windows-11-arm | 9.12 → 8.65 s (-5%) | 110 → 65 ms | 84 → 60 ms | -4 to -7% | -1% |
| Ryzen 7 5800X, 4 threads | 5.2 → 5.2 s | ~70 → ~47 ms | ~62 → ~44 ms | within noise | - |

So StanzaSharp's words/s rise by up to ~5% and its advantage over Python Stanza by as much; tokenize and lemma, the
stages with the most small per-step calls, get about 30–45% faster. The x64 runners drew different CPUs in the two runs, so
they say nothing here. The one-sentence calls are unchanged (they had 5 warm-up calls before and still do). A run now
takes 4–8 minutes per runner instead of 2–5.

### E-mail/URL regex prefilter

The tokenizer runs Stanza's e-mail/URL pattern over every paragraph to keep matches whole (`FixLabels`). Every match
contains one of `@`, `://`, `www.`, `.gov`, `.org`, `.edu`, `.net`, `.co`, read off the pattern's alternatives (the
pattern is case-sensitive): the e-mail alternative has a literal `@`; the four URL forms start `https?:\/\/` or
`www\.`; the last one has `\.(?:gov|org|edu|net|com|co)`, and `.com` contains `.co`. `AddParagraph` checks for them
once with `SearchValues<string>` (vectorized) while it builds the features, and `FixLabels` skips the regex, and the
joined string it would need, when none occurs. 44 of the benchmark text's 48 paragraphs have none.

**Exactness:** `MaskLiterals_EveryAlternativeNeedsOne`: a match for each alternative of the pattern that contains only
the one literal it needs (so the list misses none of these, and each literal is needed); and
`MaskLiterals_PresentWhereverTheRegexMatches`: a literal is present in every paragraph of every golden text where the
regex matches (14) and in every one of 200,000 random strings built from the pattern's own pieces (`http`, `://`,
`www`, `@`, `.`, the TLDs, quotes, brackets, digits, `WWW.` ...) that it matches (3,559).

**Measured:** less than round 15's profile suggested. On the benchmark text the regex costs 0.23 ms per tokenize call
once warm and 0.07 ms with the prefilter (about 0.3% of tokenize); the "~5%" in round 15's profile was probably
the source-generated regex still at JIT tier 0 (its first call here: 2.2 ms). The prefilter keeps that cold cost off the
paragraphs without an address too.

### Exactness

As in rounds 9–15, a temporary test hashed (SHA-256) every tokenizer logit the processor reads (per network call), the
`tokenize` CoNLL-U of every `tests/golden/*.txt` and all of them as one document, both charlms' outputs over the
corpus, validation and stress texts' sentences (batches of 40 and one alone), the all-eight CoNLL-U of those texts as
one document, and 20 one-sentence calls, on Vector256, Vector128 and Scalar at 1, 4 and 16 threads (no local
AVX-512). Before (main) and after this round: identical everywhere (tokenizer logits Vector256 `66E9A9BC…`, Vector128
`A9C8C202…`, Scalar `258F2053…`; charlm `33A8BFDA…` on Vector256/128, `26F21566…` Scalar; pipeline `C904102E…` and
one-sentence `68AB6199…` on all three). The measured-step variant above gave the same hashes too. `ManagedBackendTests.LstmStep_EveryThreadCountGivesTheSameBits`
keeps the property: 4 to 32 tasks over 32 panels, bit for bit.

## Results, round 15: the tokenizer's long paragraphs

In the latest benchmark run tokenize was the one stage where StanzaSharp did not lead everywhere (macos-26 0.70x,
ubuntu-26.04 0.97x). The benchmark text has one paragraph of 1,883 characters (from tokenize_stress.txt) among 48.

**Profile** (`dotnet-trace`, `stanzasharp benchmark --processors tokenize,mwt`, Ryzen 7 5800X; share of the tokenize
stage, before this round on 1 thread and after it on 4):

| | before, 1 thread | after, 4 threads |
|---|---:|---:|
| the biLSTMs' recurrence (`PackedLstm.Recur`) | 55% | 56% (31% of it spinning at each step's region) |
| the biLSTMs' input GEMMs | 24% | 9% |
| heads, gating, log-sigmoids (rest of `ManagedTokenizerNet`) | 6% | 8% |
| units, ids, features and their regexes (`AddParagraph`) | 4% | 8% |
| `FixLabels` (the e-mail/URL regex) | < 2% | 6% |
| decoding into tokens and sentences, `MarkWhitespace` | 2% | 2% |
| of all the above, the long paragraph's batch (`PredictWindowed`) | **60%** | 38% |

Allocation is small (round 11: 3 KB per word). The time was in `PredictWindowed`: a paragraph over the window (1,000
units) runs in windows, and every window after the first ran **all** rows of its batch at the full window width, as
Stanza's `advance_old_batch` pads them. The other 31 paragraphs of the batch are finished by then, so the network ran
31 rows of pure padding for each of them: about 30,000 row-steps for 900 useful ones.

**What changed:**
- Windows after the first run only the unfinished rows (`Tokenizer.LiveRows`). Rows are independent in every layer, and
  each row keeps its full-width run, so a row's arithmetic is unchanged, with one exception: `Gemm.Run` gives row m − 1
  of a call the single-row kernel when m % 6 is 1 (m = rows × width, in both LSTMs' input GEMMs and the two head
  GEMMs). That row must use it exactly when it did with all rows, so finished rows (all padding, outputs unused) go
  before the last live row until m % 6 matches. Both directions of that rule occur in the test texts below.
- The six regexes are source-generated (`[GeneratedRegex]`) instead of interpreted: the same patterns (checked
  string-equal to the old ones) and options. About 2 ms of a 38 ms tokenize once warm; no difference while the JIT is
  still tiering up, since the interpreter is precompiled and the generated code starts at tier 0.

**Exactness** (a temporary test, as in rounds 9–11): SHA-256 of every tokenizer logit the processor reads (each row
with any non-padding unit, per network call) and of the `tokenize,mwt` CoNLL-U, over every `tests/golden/*.txt`
(corpus, validation*, tokenize_stress, long_token) and all of them as one document, per kernel path, on 1 and 4
threads; plus the logits of 300 synthetic texts (random cuts of the golden texts, 1-40 paragraphs, many over 1,000
characters, and 100 batches of four long paragraphs whose shortest row outlives the others), which take the
row-count adjustment in both directions (33 and 3 windows). Before and after: identical on every path (golden
logits Vector256 `A3F25BD4…`, Vector128 `CACA5738…`, Scalar `AD1E3B8F…`; CoNLL-U `F1EACF94…` on all three; synthetic
`4ECEF22B…`, `C1EDECCA…`, `75AAC477…`). Without the adjustment the synthetic hashes change on Vector256 and Vector128,
so the test sees the single-row kernel.

**Measured locally** (Ryzen 7 5800X, Windows; `stanzasharp benchmark --processors tokenize,mwt`, 2,009 words; medians
of 5 runs, before and after alternating):

| tokenize | before | after |
|---|---:|---:|
| 8 threads (the default) | 144 ms | **60 ms** (2.4x) |
| 4 threads | 189 ms | **85 ms** (2.2x) |
| 1 thread | 257 ms | **145 ms** (1.8x) |
| 4 threads, `DOTNET_TieredCompilation=0` | 101 ms | **38 ms** (2.7x) |

Python Stanza on this machine: 330–400 ms (8 threads). The one-sentence calls are unchanged (with tiering off,
0.29 → 0.27 ms for `tokenize,mwt`).

**Free runners**, the `default` package, sequential runs of the same commit pair ([38284092033](https://github.com/bisforboman/StanzaSharp/actions/runs/38284092033)
main, [38283443584](https://github.com/bisforboman/StanzaSharp/actions/runs/38283443584) this branch; CoNLL-U identical to
Python's in both). Tokenize, StanzaSharp before → after, and Python Stanza in the after run:

| runner | before | after | Python Stanza | after vs Python |
|---|---:|---:|---:|---:|
| macos-26 | 0.55 s | **0.33 s** | 0.72 s | 2.19x |
| ubuntu-26.04 (AMD EPYC 7763) | 0.23 s | **0.13 s** | 0.32 s | 2.54x |
| ubuntu-26.04-arm | 0.20 s | **0.10 s** | 0.47 s | 4.62x |
| windows-2025 (AMD EPYC 7763) | 0.30 s | **0.18 s** | 0.46 s | 2.54x |
| windows-11-arm | 0.24 s | **0.09 s** | - | - |

What is left, and why the times above still vary so much between runs:
- **JIT warm-up.** The benchmark warms up on a tenth of the text, which has no long paragraph, so the windowed path is
  first compiled in the timed runs, and the per-step methods (`PackedLstm.Step`, `ManagedThreads.For`, `Pool.Run`,
  `Region.Drain`) stay at instrumented tier 0 through all three of them (`DOTNET_JitDisasmSummary`). Locally,
  tokenizing the benchmark text 40 times in one process takes 147, 62, 67, 70, 89 ... ms and settles at 27–33 ms after
  about ten calls; with tiering off it is 26 ms from the second call. So a warmed service tokenizes about twice as
  fast as the benchmark shows.
- **One parallel region per LSTM step.** The tokenizer's LSTMs are tiny (hidden 64) and long (the long paragraph runs
  ~2,000 sequential steps per level), so a step's work is a few µs, about what waking the pool's workers costs. On
  macos-26 (3 vCPUs, virtual) an experiment ([38283174666](https://github.com/bisforboman/StanzaSharp/actions/runs/38283174666),
  `tokenize,mwt` only, three rounds) measured the default 3 threads at 190–1,260 ms, `--threads 1` at 108–126 ms, and every
  step on the calling thread (GEMMs still parallel) at 79–109 ms; on ubuntu-26.04-arm the same serial steps were
  slower than the default (119–131 vs 73–79 ms). Serializing only steps of at most 4 or 8 rows was inconsistent on
  macOS. A fix belongs in `PackedLstm.Step` (fewer tasks when M × panels is small) or the pool, both outside this round.
- The e-mail/URL regex is still ~5% (its first character class matches almost any ASCII character); a literal
  prefilter (`@`, `http`, `www`, `.gov` ...) could skip it for most paragraphs.

## Results, round 14: AVX-512 kernels (Vector512)

The managed kernels stopped at AVX2 (`Vector256`), while on the AVX-512 runner Python Stanza ran about 40% faster than on
an EPYC 7763 (above). A fourth kernel path, `Vector512`, is now the detected one wherever .NET accelerates `Vector512`
(AVX-512 that doesn't throttle; `DOTNET_PreferredVectorBitWidth=256` turns it off):

- The panels stay 16 floats wide, so one Vector512 register holds a panel row: the 6-, 3- and 1-row kernels and the
  LSTM's blocked kernel have one accumulator per row instead of two. `Gemm.Run` gives each kernel call two adjacent
  panels (12 accumulators for 6 rows, enough to cover the FMA latency). The LSTM cell takes one exp over its 16 gate
  values; the highway's gated tanh runs 16 lanes at a time.
- Every output gets the same fused multiply-adds in the same order as on Vector256, and the activations are the same
  lane by lane, so the results are Vector256's **bit for bit** (`ManagedBackendTests.Vector512_GivesVector256sBits`:
  GEMMs of 1–7 and 97 rows with paired and single panels, the blocked kernel, the charlm, the highway). Every
  tolerance and golden file holds unchanged, and which CPU a CI run draws can't change a result.

**Method.** The Benchmark workflow's new `avx512` jobs (`avx512_only` runs only them): six ubuntu-26.04 jobs; each prints
its CPU and stops if `/proc/cpuinfo` has no `avx512f`. One that has it runs `stanzasharp benchmark` without Python
(`default` package, 2,009 words, 3 timed runs, 50 one-sentence calls) eight times, alternating Vector256
(`DOTNET_PreferredVectorBitWidth=256`) and Vector512, two rounds each at the default threads (4) and at 1, then the
managed kernel and pipeline tests. Medians of the two rounds (each itself a median of 3 runs):

| CPU (2 cores / 4 logical) | threads | Vector256 total | Vector512 total | words/s | per call, median |
|---|---:|---:|---:|---:|---:|
| AMD EPYC 9V45 (Zen 5) | 4 | 5.48–5.60 s | 3.69–3.77 s | 359–366 → 533–545 (+48%) | 53.5 → 43.8 ms |
| AMD EPYC 9V45 (Zen 5) | 1 | 10.75–11.06 s | 7.08–7.15 s | 182–187 → 281–284 (+53%) | 117 → 92 ms |
| AMD EPYC 9V74 (Zen 4), run 1 | 4 | 7.15 s, 7.15 s | 6.50–6.63 s | 281 → 303–309 (+9%) | 65 → 59 ms |
| AMD EPYC 9V74 (Zen 4), run 1 | 1 | 13.65–13.66 s | 13.13–13.23 s | 147 → 152–153 (+4%) | 138 → 117 ms |
| AMD EPYC 9V74 (Zen 4), run 2 | 4 | 7.16–7.28 s | 6.48–6.66 s | 276–281 → 302–310 (+10%) | 67 → 59 ms |
| AMD EPYC 9V74 (Zen 4), run 2 | 1 | 13.80–14.45 s | 13.19–13.77 s | 139–146 → 146–152 (+3%) | 142 → 120 ms |

Runs [38276588610](https://github.com/bisforboman/StanzaSharp/actions/runs/38276588610) (EPYC 9V74) and
[38277973171](https://github.com/bisforboman/StanzaSharp/actions/runs/38277973171) (EPYC 9V45 and 9V74), 2026-10-12. Per
stage on the 9V45 at 4 threads: pos 0.92 → 0.68 s, constituency 0.72 → 0.51 s, depparse 1.44 → 1.00 s, sentiment 1.69 →
1.04 s, ner 0.45 → 0.32 s. `Speed_IsReported` (320 words, 2 threads) on the 9V45: charlm 144 → 109 ms, highway 18.7 →
10.3 ms (TorchSharp 321 / 23.2 ms).

- Zen 5 executes 512-bit FMAs at full width, so the kernels gain about half; Zen 4 splits them into two 256-bit halves,
  and the gain is fewer instructions and loads (+3–10% throughput, 10–15% shorter one-sentence calls). Nothing got slower.
- Of 6 + 6 jobs, the ubuntu-26.04 runners were an EPYC 7763 (no AVX-512) 9 times, an EPYC 9V74 with AVX-512 twice and
  once without it (masked), and an EPYC 9V45 once; none drew the Intel Xeon Platinum 8573C of the round 12 benchmark
  run. On that Xeon (two 512-bit FMA ports) the gain should be nearer the 9V45's; not measured.
- So Vector512 ships as the default where .NET accelerates it. 105 tests (ManagedBackendTests on every path, PipelineTests
  on Vector512) passed on both AVX-512 CPUs.

## Results, round 13: physical cores by default from 4 up

The owner's rule after round 12 (2026-10-12): with `Threads` null the managed pool now starts at the physical cores when
there are at least 4 of them, otherwise at all logical processors, and never above `Environment.ProcessorCount` (a
container's CPU quota). `CpuCores.Physical()` finds the cores without starting a process (Linux: distinct sysfs
`thread_siblings_list`; Windows: `GetLogicalProcessorInformationEx`; macOS: `sysctlbyname("hw.physicalcpu")`) and falls
back to the logical processors if that fails. The benchmark's core count uses it too. TorchSharp is unchanged (libtorch
already defaults to the physical cores).

**Ryzen 7 5800X** (8 cores / 16 logical): 8 threads (the new default) against 16 (the old one), `samples/StanzaSharp.Benchmark`,
three rounds, 8 and 16 back to back with the order flipped each round. Other sessions shared the machine, so each run
waited for two idle readings (< 5% CPU) and was repeated if the machine was busy (≥ 10%) right after it (two runs were):

| | 8 threads | 16 threads | 8 vs 16 (medians) |
|---|---|---|---:|
| default, 26,264 words, words/s | 1,197, 1,230, 1,196 | 1,315, 1,247, 1,313 | -9% |
| default_fast, 26,264 words, words/s | 2,682, 2,721, 2,633 | 2,921, 2,858, 2,758 | -6% |
| default, one sentence, median ms | 42.1, 32.2, 37.5 | 34.4, 33.3, 40.2 | +9% |
| default, one sentence, p90 ms | 60.1, 49.8, 59.7 | 59.1, 71.4, 72.9 | -16% |
| default_fast, one sentence, median ms | 17.3, 15.9, 16.1 | 17.3, 17.4, 22.9 | -7% |
| default_fast, one sentence, p90 ms | 26.0, 24.8, 23.9 | 38.7, 54.7, 38.7 | -36% |
| 2 callers (default), calls/s | 30.3, 33.6, 36.1 | 26.4, 30.3, 30.2 | +11% |
| 2 callers, slowest call ms | 135, 119, 107 | 386, 246, 279 | -57% |

(Big text: `--runs 1`, 8 copies; one sentence: `--per-call 50`; 2 callers: `--per-call 60 --concurrent 2`.) The
one-sentence `default` median moved back and forth, so six more back-to-back rounds of it alone followed: over all
nine, 8 threads had the shorter median in 5 (overall medians 36.8 vs 35.8 ms) and a shorter p90 in 8 (57.0 vs 72.5 ms).
So on a busy desktop 8 threads mostly remove the slow calls rather than shorten the typical one, which round 12 (idle)
found 18-27% shorter. The big texts lose 1-9% (medians: 9% and 6%), as round 12 found; the owner accepted that with the rule. CoNLL-U of the
big runs is byte-identical on 8 and 16 threads (MD5 db91abcc… default, a3a3916d… default_fast).

**Free runners** ([38232917125](https://github.com/bisforboman/StanzaSharp/actions/runs/38232917125), this branch): the
detection found 2 cores on ubuntu-26.04 (AMD EPYC 7763) and windows-2025 (Intel Xeon Platinum 8573C), 4 on both Arm
runners and 3 on macos-26, so the default stays 4, 4, 4, 4 and 3 threads there, as before; CoNLL-U identical to Python
Stanza's on every runner that ran it.

## Results, round 12: concurrent calls and the managed thread pool

How `ManagedThreads`, the managed backend's pool (used by every GEMM and LSTM step), behaves when several `Process` calls
run at once on one pipeline, and why one-sentence calls were slow on the macOS runner.

**Method.** `--per-call 60 --concurrent T` (T threads, 60 one-sentence calls each, on one shared pipeline; default
package); new: the slowest and fastest caller's median, and `--with-long`, where one more thread processes one copy of
the golden texts (14,534 characters, 3,283 words; alone 4.5 s on 8 threads, 19.2 s on 1) in a loop meanwhile. Ryzen 7
5800X (8 cores, 16 logical processors), Windows, converted models. Other sessions' test runs loaded the machine
throughout (20-100% CPU), so single numbers moved by up to 2× between runs: compare within a table, where before and
after were alternated.

**Callers and threads, before this round** (two sweeps, calls/s, then median / max ms per call):

| callers | 8 threads | 1 thread |
|---:|---|---|
| 1 | 22.9, 23.8; 42 / 101 | 8.2, 6.9; 115 / 298 |
| 2 | 26.3, 27.6; 71 / 182 | 12.8, 10.4; 142 / 415 |
| 4 | 21.8, 21.3; 155 / 562 | 18.9, 15.4; 176 / 649 |
| 8 | 18.9, 18.9; 319 / 2,442 | 18.7, 19.2; 350 / 1,355 |

- **Serialize or oversubscribe?** Neither, while `Threads` is at most the number of logical processors minus the
  callers: each caller works its own regions, the pool's 7 workers help whichever region has work, so 8 callers use
  15 threads. Latency grows with the callers because throughput stays flat, not because calls queue.
- **Is N callers' throughput at least one caller's?** Up to 2 callers yes (+15%), beyond that no (8 callers −20%).
  The pool isn't the cause: on 1 thread, with no pool at all, 8 callers reach the same ~19 calls/s. That is the
  memory bandwidth and L3 limit of round 7, and `MaxConcurrentCalls = 2` remains the advice.
- **Does a long call starve short ones?** No, they share the pool. One short caller next to the long call (8 threads)
  took 103-117 ms per call instead of ~40, and the long call 6.9-11.7 s instead of 4.5. Among 8 short callers the
  long call slowed most (6-25 s); the short callers' medians spread 1.8-2.2× (1.3-1.4× without it), and no caller
  stopped making progress.
- **But `Threads` = all logical processors (the default) oversubscribed.** 15 workers + N callers exceed 16 logical
  processors, and every waiting thread spins. A region ends with its slowest item, so a worker descheduled while it
  holds one stalls its caller for a scheduler quantum: with 8 callers on 16 threads throughput fell to 3.4-10 calls/s,
  single calls took up to 19.6 s, and the slowest caller's median was 2-10× the fastest's.

**What changed** (`ManagedThreads.Pool`): workers no longer wake or keep spinning while the busy threads (callers
inside a region plus awake workers) already number the logical processors (or `Threads`, if larger). One caller still
gets its full team; N callers still use at most `Threads` − 1 + N threads, but now also no more than the logical
processors (or N, when there are more callers). Items are still claimed one at a time from the same split, so results can't change. Before / after, alternated,
three runs each (calls/s; p90 / max ms):

| threads, callers | before | after |
|---|---|---|
| 16, 2 | 8.4, 7.7, 8.4; 455-570 / 1,041-1,707 | 11.2, 6.7, 13.0; 271-599 / 483-1,182 |
| 16, 8 | 10.1, 9.2, 10.2; 1,660-1,917 / 5,221-6,111 | **16.6, 13.1, 14.7**; 808-875 / 1,927-4,940 |
| 16, 4 + a long call | 10.2, 6.3, 8.3; 704-1,399 / 2,731-3,804 | **14.4, 8.9, 13.4**; 439-659 / 973-1,502 |
| 8, 1 | 21.2, 13.0, 18.0 | 23.5, 20.9, 18.1 |
| 8, 8 | 16.9, 15.3, 13.7 | 17.9, 17.0, 14.7 |

(16 threads, 1 caller: unchanged within noise.) With 8 callers on 16 threads the slowest and fastest callers' medians
are now 1.2-1.4× apart instead of 2-3×; the median call rose (164-204 → 349-449 ms) because no caller runs ahead of
the others any more. The long call took 7-16 s instead of 13-21 s.

Tried and not kept:
- **Threads as one budget across calls** (workers stay out while callers + workers exceed `Threads`): on 16 threads
  as good as the change above, but with 8 threads and 4-8 callers 10-15% less throughput, since callers then leave
  the other 8 logical processors idle.
- **Blocking instead of spinning** while the caller waits for a region's last items (an event after 50 spins): no
  better on the macOS runner (below), within noise.

Exactness: `Pipeline` CoNLL-U for one copy of the golden texts, both packages, on 1, 3, 8 and 16 threads after the
change and on 16 threads before, SHA-256 identical (default `8B626630…`, default_fast `F3311B8E…`); ConcurrencyTests,
ManagedBackendTests, OptionsTests and PipelineTests pass.

**macOS runner, one-sentence calls** (run 38150941695: 766 ms median, p90 1,197, against 69-86 ms on the other runners
and 338 for Python Stanza on the same Mac). Reproduced on macos-15 (Apple M1 virtual, 3 vCPUs), `--per-call`, per-call
median ms:

| | total | pos | constituency | depparse |
|---|---:|---:|---:|---:|
| benchmark command right after `pip install` (Spotlight indexing torch's files) | 845 | 264 | 215 | 187 |
| the same, Spotlight on, at the indexing peak | 717 | | | |
| the same, Spotlight on, between peaks | 117-152 | 55 | 27 | 40 |
| the same, Spotlight off (`mdutil -a -i off`), two runs | 94, 95 | | | |
| no Python process, after the first minutes | 104 | 38 | 19 | 25 |
| `--threads 1` (with Python; without) | 266; 207 | 92; 69 | 58; 48 | 81; 57 |

- The cause is the runner, not the pool's dispatch: `top` showed `mds`, `mds_stores` and six `mdworker_shared` at
  15-30% each (load average 16-33 on 3 vCPUs). Every processor slowed alike; with Spotlight off the median was 94 ms.
- Why StanzaSharp suffers more than Python: a call opens about 850 parallel regions (one per LSTM step), each ending
  with its slowest item, so a helper the host preempts holds up the whole call; torch runs far fewer, larger
  operations. Pool variants, alternated four times in one job (the caller blocking instead of spinning, workers
  spinning 10× shorter before they sleep; in another job 100× longer or not at all, which was slower), moved the median by less than the run-to-run noise
  (136-234 ms on 3 threads; 192-217 on 1, 181-273 on 2).
- `benchmark.yml` now turns Spotlight off on macOS first; "Run it yourself" says so.
- On **macos-26** (the runner CI has used since; macOS 26.6.2, the same Apple M1 virtual with 3 vCPUs) the slow calls don't
  show: the same benchmark command right after `pip install`, two runs each, 149 and 104 ms with Spotlight on (the
  first while `mds_stores` used 57% of a vCPU), 132 and 133 ms with it off; Python Stanza 303-704 ms. The `mdutil` step
  stays, as it costs nothing and removes a source of noise.

**Default thread count: physical cores or logical processors?** Loaded, one caller on 16 threads (the default, all
logical processors) took 98-180 ms per call against 42-65 on 8 (the physical cores). Measured again on an idle machine
(under 5% CPU before each run) and on the free runners, alternated, three rounds each (macos-26 and the Arm runners
have no SMT, so physical = logical there):

| | Ryzen 7 5800X, idle: 8 / 16 threads | ubuntu-26.04 (EPYC, 2 / 4) | windows-2025 (EPYC, 2 / 4) |
|---|---|---|---|
| default, words/s | 1,085, 1,068, 1,078 / 1,076, 1,137, 1,124 | 227, 231, 226 / 240, 233, 241 | 174, 214, 213 / 119, 120, 147 |
| default, per call median ms | 42.6, 45.9, 45.3 / 52.5, 57.7, 58.7 | 80.6, 86.5, 84.8 / 73.9, 76.0, 72.2 | 85.6, 92.5, 91.4 / 77.9, 72.7, 78.3 |
| default_fast, words/s | 2,379, 2,395, 2,403 / 2,452, 2,574, 2,603 | 378, 375, 366 / 402, 394, 395 | 352, 349, 355 / 382, 389, 391 |
| default_fast, per call median ms | 22.8, 23.4, 22.6 / 27.9, 29.9, 31.0 | 40.1, 41.0, 40.8 / 40.0, 50.3, 39.2 | 43.0, 44.0, 43.6 / 39.5, 41.2, 41.8 |
| 2 callers, calls/s | 22.6, 22.4, 21.2 / 19.1, 19.5, 15.3 | 9.7, 9.3, 9.4 / 10.3, 10.5, 10.4 | 10.6, 10.4, 9.9 / 12.1, 11.5, 11.5 |

(Ryzen: the big call is the benchmark text, 26,264 words, one run; runners: `stanzasharp benchmark --python none`, about
2,000 words and 50 one-sentence calls; 2 callers: `--per-call 60 --concurrent 2`.) Neither wins everywhere. On the
idle Ryzen the physical cores give 18-27% shorter calls and 15-39% more calls/s with 2 callers, but up to 8% fewer
words/s on the big text. On the 2-core runners the logical processors give 3-8% more words/s (except below), 7-17% more
calls/s with 2 callers and 8-21% shorter `default` calls (`default_fast` calls: even on ubuntu, 3-10% shorter on
windows), and lose only windows-2025's big `default` text (31-44% fewer words/s). So the default stayed all logical processors; round 13 changes it.

## Results, round 11: garbage on long texts (managed backend)

Round 8 found that the peak on a long text is 0.6–1.1 GB above the live heap: arrays the `ArrayPool` doesn't hold, which the
GC lets accumulate. This round finds where they come from and rents or reuses them.

**Where it goes** (an allocation trace, `dotnet-trace --profile gc-verbose`, grouped by the allocating method; and the
benchmark's per-processor allocation at 201,313 words, all eight processors): constituency 30.9 KB per word, depparse 9.4,
tokenize 3.5, pos 3.2 (1.3 of it the charlm cache, kept by design), ner 3.0, sentiment 2.9, lemma 0.4, mwt 0.3: 54 KB per
word, 10.8 GB in all. Constituency's is its parser states: every transition made a stack LSTM state (two [2, 512] arrays,
8 KB, for the constituent stack) and every close a 2 KB vector, each live until its sentence was done, long enough to reach
gen2. Depparse's is Chu-Liu/Edmonds (an n × n matrix per sentence plus one per contracted cycle, and LINQ in Tarjan) and
the [batch, width, width] arc and label outputs (large-object heap).

**What changed** (all exact; see below):
- Constituency: the stack states, the word vectors and the Compose outputs come from a `Recycler`, per-thread free lists
  of float arrays (thread-static, so per `Process` call: the parse is synchronous on the caller's thread). The parser
  already disposes what the net made for a sentence when the sentence is done, and that now gives the arrays back; when
  every array is back (the parse is over) the lists are dropped, so nothing outlives the call. A call allocates the states
  of the sentences in flight and reuses them. The transition inputs are embedding slices instead of copies, and the
  per-step score rows are no longer copied.
- Depparse: Chu-Liu/Edmonds works on flat n × n buffers, two rented ones taking turns as cycles are contracted (no
  [c, n] head-score matrix: each score is recomputed, the same double), and Tarjan's dependents come from one counting
  sort instead of a LINQ query per node. `DependencyParser.Parse` rents the arc log-probs and labels (`DepparseBatch.RentScores`)
  and returns them after decoding; `Scores` (tests) still gets exact arrays.
- Tokenizer: the two head GEMMs write into rented buffers. Sentiment: the pooled features and the fully connected
  layers' outputs are rented. NER: Viterbi's trellis and backpointers are rented, and `Process` no longer copies each
  sentence's emissions (only tests ask for them).
- Not changed: the outputs a net returns through its interface (pos scores and tags, NER logits, the tokenizer's
  predictions), since the TorchSharp nets return plain arrays and the caller can't tell which to give back; the
  tokenizer's per-paragraph inputs; the charlm cache; the `Document`. Each is under 1 KB per word.
- Proof: every managed output on all golden texts (corpus.txt, every validation*.txt, tokenize_stress.txt and all of them
  as one document): the CoNLL-U of both packages, every constituency step's transition scores, every sentiment logit
  (without and with the tagger's charlm cache) and the tokenizer logits of every paragraph (1,885,714 floats), SHA-256 per
  kernel path, on both builds (a temporary test): Vector256 `89AAC382…`, Vector128 `2E6181FF…`, Scalar `3A018614…`
  (CoNLL-U `AC2D88C0…` on all three), identical before and after. Chu-Liu/Edmonds also passes its golden tie-heavy
  matrices (`mst.json`).

**Measured** (`--memory N --verbose`, converted models, 8 threads, Windows; before = 8a849eb (PR #69's branch), after =
this round, alternated; other sessions' test suites shared the machine, so times are only indicative):

| | 20,513 words | 201,313 words |
|---|---:|---:|
| peak, before | 1,299 / 1,302 / 1,297 MB | 2,310 / 2,209 MB |
| peak, after | **1,271 / 1,270 / 1,269 MB** | **1,916 / 1,944 MB** |
| allocated, before | 1,486 MB (72 KB/word) | 10.8 / 10.9 GB (54 KB/word) |
| allocated, after | 831 MB (41 KB/word) | 3.7 GB (18 KB/word) |
| process time, before → after (same load) | 25.8 / 25.2 → 26.0 / 24.5 s | 333 → 307 s |

Per processor at 201,313 words (second pair), allocation per word and the peak after each processor, before → after:

| | allocated, KB/word | peak, MB |
|---|---:|---:|
| tokenize | 3.5 → 3.0 | 1,181 → 1,091 |
| mwt | 0.26 → 0.26 | 1,211 → 1,123 |
| pos | 3.2 → 3.2 | 1,442 → 1,437 |
| lemma | 0.42 → 0.42 | 1,443 → 1,452 |
| constituency | 30.9 → 5.1 | 1,862 → 1,791 |
| depparse | 10.0 → 3.2 | 1,998 → 1,805 |
| sentiment | 2.9 → 1.0 | 2,092 → 1,841 |
| ner | 3.0 → 2.2 | 2,209 → 1,944 |

- The peak on 201,313 words drops ~0.3–0.4 GB (−13 to −17%) with no heap limit. On 20,513 words it drops ~30 MB: there
  the garbage was ~0.2 GB and the pool's buffers (~0.4 GB) set most of the peak.
- What is left: the per-word outputs listed above (under 1 KB each) and, likely the larger part on a call this long, the
  `ArrayPool`'s churn: at a gen2 GC it trims buffers left idle for a minute (an earlier processor's), which become
  garbage, and later rents allocate fresh ones.
- Constituency also got faster (104 → 75 s, and 4.9 → 4.8 s at 20k): fewer gen0/gen1 collections promoting live states.
## Results, round 10: sentiment memory on the managed backend

With all eight processors, sentiment set the managed pipeline's peak: on 6,761 words it added ~1.15 GB (1,150 → 2,300 MB).

**Where it goes** (a temporary probe of the batch shapes, `--memory`, 8 threads): Stanza's batches are 5000 *real* tokens,
but each is padded to its longest sentence, and the classifier's LSTM and convolutions run over the padding. The first
batch of the 6,761-word text is 257 sentences × 139 tokens = **35,723 rows** for 5,000 tokens (20,513 words: 129 × 139 =
17,931 rows). Every per-row buffer was multiplied by that: the input [rows, 2148], the LSTM's GEMM output (both
directions' gates, [rows, 2400]), two layer outputs and the copied output ([rows, 600] each), its batch-major copy, and
each full-width convolution's output ([rows, 1008]): ~7,950 floats = 32 KB per row, 1.14 GB on 35,723 rows.

**What changed** (exact; see below):
- `ManagedLstm.ForwardUnpacked` (new; sentiment's unpacked biLSTM, which ran `ForwardPacked` with every batch size = n):
  per layer and direction, the input projection is computed for a chunk of ~2048 rows (a few time steps) just before
  those steps run, through a panel view of that direction's gate columns, and the steps are driven with
  `PackedLstm.Step` one direction at a time. The layer-0 input rows are built per chunk by a callback (embeddings +
  charlm rows, which stay per real token: from the cache, or one [tokens, 1024] buffer per direction), so the [rows, 2148]
  input never exists. The last layer writes batch-major, which is what the convolutions read, so the copy is gone too.
  What is left per row: layer 0's output and the final output, 2 × 600 floats.
- The full-width convolutions run 2048 rows at a time into one [2048, 1008] buffer, each chunk max-pooled as it comes.
- `Gemm.RunRows(a, r0, r1, m, ...)` computes rows [r0, r1) of an m-row GEMM with the bits the whole call gives them:
  every kernel but the single-row one gives a row the same arithmetic (the 3-row kernel's per-row sum is the 6-row
  one's), and `Run` uses the single-row kernel only for row m − 1 when m % 6 == 1 (MC = 96 is a multiple of 6). So a
  chunk runs `rowInvariant`, and that one row alone. Round 9's "splitting by rows is not exact" was about splitting a
  batch's sequences (fewer rows per LSTM step); splitting a GEMM's rows this way is.
- Proof: `ManagedBackendTests.Gemm_RowRangesGiveTheWholeGemmsBits` (chunks of 1, 5, 7, 13, 96 rows; m % 6 == 1 and not)
  and `ManagedLstm_UnpackedGivesThePackedBits` (sentiment's checkpoint; one-step chunks, multi-step chunks, n·width % 6 == 1)
  on every path. And every managed sentiment logit and label on all golden sentiment data (the 14 files and all of them
  as one document, each without and with the tagger's charlm cache: 10,632 logits), SHA-256 per kernel path, on both
  builds (a temporary test): Vector256 `01356120…`, Vector128 `9763DA4D…`, Scalar `8473EDF3…`, identical before and after.

**Measured** (`--memory N`, converted models, Windows; before = 475211d (PR #65's branch), after = this round,
alternated, two rounds; peaks first, sentiment stage time in brackets. Up to three other agents' test suites kept the CPU
at 100% throughout, so these times are noise and only the peaks count; idle-machine times are below the table):

| | 8 threads | 1 thread |
|---|---:|---:|
| all eight, 6,761 words | 2,306 / 2,304 → **1,314 / 1,311 MB** (71 / 68 → 115 / 86 s) | 2,308 / 2,291 → **1,318 / 1,336 MB** (95 / 77 → 85 / 75 s) |
| all eight, 20,513 words | 1,859 / 1,873 → **1,379 / 1,372 MB** (147 / 76 → 63 / 175 s) | 1,864 / 1,880 → **1,378 / 1,412 MB** (135 / 100 → 24 / 22 s) |
| `default_fast`, 6,761 words | 2,002 / 2,001 → **1,058 / 1,056 MB** (203 / 97 → 241 / 110 s) | 2,015 / 2,011 → **1,060 / 1,062 MB** (135 / 135 → 135 / 115 s) |
| `default_fast`, 20,513 words | 1,476 / 1,478 → **1,071 / 1,073 MB** (272 / 180 → 113 / 109 s) | 1,463 / 1,483 → **1,075 / 1,076 MB** (261 / 96 → 260 / 80 s) |

- The default pipeline's peak drops ~1.0 GB at 6,761 words (−43%) and ~0.48 GB at 20,513 words (−26%; its widest batch
  has half the rows); sentiment now adds ~0.18 GB on 6,761 words (1,126 → 1,303 MB in the per-stage report), which is the
  two [rows, 600] layer outputs (171 MB on 35,723 rows). `default_fast` drops ~0.95 GB (−47%) and ~0.4 GB (−27%).
- Speed: the same multiply-adds; the input rows are built twice (once per direction of layer 0), a copy of 2148 floats
  per row against 2148 · 1200 multiply-adds. On the idle machine afterwards (one run each, all eight): the sentiment
  stage took 3.51 → 3.50 s (6,761 words, 8 threads), 14.6 → 13.7 s (1 thread), 4.26 → 4.10 s (20,513 words, 8 threads),
  20.4 → 20.1 s (1 thread), with the same peaks as in the table (2,286 → 1,329 MB; 1,908 → 1,409 MB).

**What is left:** layer 0's time-major output and the final batch-major output (1,200 floats per padded row). Layer 1's
backward direction frees layer 0's rows as it goes while the output fills, so a shared layout could hold ~900 floats
per row (−43 MB at 6,761 words); not worth the bookkeeping. The padding itself is Stanza's batching and must stay.

## Results, round 9: depparse memory on the managed backend

Without sentiment, depparse sets the managed pipeline's peak: `tokenize,mwt,pos,lemma,depparse` on 6,761 words rises
from ~700 MB after lemma to ~900 MB. Round 8 measured where that goes and cut what can be cut exactly.

**Where it goes** (working set at points inside `ManagedDepparseNet.Forward`, first 5000-word batch, 8 threads; a
temporary probe, not committed): the charlm pass +57 MB (the 2423-wide input, 51 MB, and the charlm outputs), the
highway LSTM +107 MB (its GEMM output, 12·400 floats per row = 100 MB, plus two layer outputs), the scorers +34 MB (all
four scorers' W1/W2 in one [rows, 3200] GEMM, 67 MB, plus the arc scorers' T, 26 MB), the label scorer +1 MB.

**What changed** (both exact: a GEMM column depends only on its 16-column panel, so splitting a GEMM by whole panels
gives the same bits; `ManagedBackendTests.Gemm_ColumnPanelsAreIndependent` on every path):
- `ManagedHighwayLstm` (pos and depparse): per layer the GEMM wrote both directions' gates and the gate and highway
  layers (12·hidden per row). Now the 8·hidden gate columns go first, the recurrence runs, and then the gate/highway
  columns are computed into the same buffer. Both GEMMs are panel views of the one packed matrix (`new
  PackedMatrix(source, firstPanel, panels)`, no copy). Depparse's buffer for 5,250 rows: 100 → 67 MB; pos's: 48 → 32 MB.
- `ManagedDepparseNet`: each scorer computes its own W1/W2 [rows, 800] just before it is used, and the arc scorers run
  one after another (each pair's three terms are accumulated in the same order as before, in a [rows, width] double
  buffer), so only one scorer's in1/in2 and T are live.
- Proof beyond the panel test: every managed depparse arc log-prob and label score of every pair (`labelScores: true`)
  and every pos UPOS logit on all golden data (depparse/ and fast/, 8,401 words, both checkpoints), hashed per kernel
  path (Vector256, Vector128, Scalar): identical before and after (a temporary test, run on both builds).

**Measured** (`--memory N`, all `.json`/`.safetensors` models, Windows; before = main at 629634f, after = this round,
alternated, two rounds; another job kept the CPU at 100% throughout, so the times are noise and only the peaks count):

| | peak, 8 threads | peak, 1 thread | depparse, 8 threads | depparse, 1 thread |
|---|---:|---:|---:|---:|
| 5 processors, 6,761 words | 896 / 901 → 826 / 831 MB | 900 / 900 → 830 / 830 MB | 10.4 / 8.3 → 12.1 / 8.1 s | 21.2 / 20.2 → 21.3 / 19.7 s |
| 5 processors, 20,513 words | 994 / 1,018 → 933 / 949 MB | 1,000 / 1,015 → 945 / 947 MB | 21.2 / 16.8 → 18.9 / 16.9 s | 57.8 / 48.5 → 50.2 / 46.1 s |
| all eight, 6,761 words | 2,353 / 2,354 → 2,402 / 2,409 MB | 2,358 / 2,356 → 2,395 / 2,409 MB | 8.4 / 9.3 → 8.9 / 7.5 s | 22.5 / 20.3 → 21.7 / 18.7 s |
| all eight, 20,513 words | 1,976 / 2,021 → 2,012 / 2,005 MB | 2,004 / 2,014 → 2,019 / 2,006 MB | 18.2 / 13.7 → 10.2 / 17.3 s | 51.4 / 53.6 → 57.1 / 53.5 s |

- Without sentiment the peak drops 60–70 MB (7%). Each change alone (6,761 words, 8 threads): the LSTM split −60 MB, the
  per-scorer W1/W2 −10 MB.
- With all eight, the peak is sentiment's, not depparse's: sentiment adds ~1.2 GB on 6,761 words (1,156 → 2,344 MB).
  Its buffers come from the same `ArrayPool`, and with the LSTM split pos's buffer moved from the 16M-float bucket to
  the 8M one, so sentiment no longer finds a touched 16M array to reuse and touches a new 64 MB one: +50–60 MB at 6,761
  words (the per-scorer change alone: −3 MB), nothing at 20,513. `default_fast` (all seven): 2,107 / 2,109 → 2,106 /
  2,107 MB at 6,761 words, 1,632 → 1,647 MB at 20,513.
- Speed (`--processors tokenize,mwt,pos,lemma,depparse`, 26,264 words, 8 threads, one run each, same load): pos 25.4 →
  26.3 s, depparse 20.2 → 21.0 s, tokenize (unchanged code) 5.9 → 9.1 s: within the noise. The work is the same; the
  LSTM reads its input twice per layer.

**Tried and dropped:**
- *Labels only for the chosen heads* (the parser decodes the trees first, then the label scorer runs one head per word,
  bit for bit the full matrix's scores). The managed net only ever kept an int per pair (~5 MB) and T dominates the label
  scorer (the 400 × 401·49 GEMM every word needs), so it saved ~5 MB and no measurable time, for an extra decode
  callback in the network interface.
- *Exact-size arrays instead of `ArrayPool`* for the three biggest buffers (input, LSTM GEMM output, scorer W1/W2): peak
  820 → 970 MB. The pool's power-of-two rounding costs no working set (the tail of a rented array is never touched);
  fresh arrays pile up on the large-object heap between collections.
- *Splitting the batch* for the encoder is not exact: the charlm and the packed LSTM step all of a batch's rows through
  one GEMM, whose single-row kernel sums differently, so a different split moves the last bits (the arc log-softmax is
  per row, but its inputs would change). After the per-scorer change the scorers add ~20 MB, so splitting them alone
  wasn't worth the bookkeeping that keeps their GEMMs' row blocks as they are.

**What is left:** sentiment's ~1.2 GB on a 5000-token batch (its padded [rows, 2148] input, the unpacked biLSTM's GEMM
output, the convolutions' outputs) is now the default pipeline's peak; depparse's input (51 MB) and LSTM buffers stay.

## Results, round 8: one call on a very long text

What happens to memory when a whole book goes into one `Process(text)` call? Round 3 measured up to 15,000 words; this
round goes to 670,000, on the managed backend with all eight processors, and asks what grows with the input apart from
the output itself.

**Method.** `--memory N --verbose` as in round 3 (converted models, 8 threads, workstation GC, Windows), one fresh
process per run. New in the benchmark: `--verbose` ends with the documents' own size (the GC heap with the returned
documents minus without them, after full GCs), each processor's log line shows what it allocated, the `process` line
the call's total allocation, and `--live` adds a full GC and the live heap after each processor. 15,000, 50,000,
150,000 and 500,000 whitespace-separated words are 20,513, 67,120, 201,313 and 670,972 words after tokenizing.
"Before" is main at 629634f. Other jobs (three test suites of parallel sessions) kept the machine at 100% CPU for every
"after" run and the 670,000-word "before" run, so their times are 3–15× the idle ones and are not compared;
alternated under the same load, three before and three after runs on 20,513 words took a median 233 s and 224 s
(peaks 1,995–2,092 MB and 2,008–2,077 MB): the time is unchanged within noise.

Peak working set, MB, and the returned documents on the GC heap:

| words after tokenizing | 20,513 | 67,120 | 201,313 | 670,972 |
|---|---:|---:|---:|---:|
| peak, before | 2,006 | 2,010 | 2,389 | 3,636 |
| peak, after | 2,029 | 1,935 | 2,259 | 3,024 |
| documents, before | 12 MB | 38 MB | 114 MB | 380 MB (595 B/word) |
| documents, after | 10 MB | 31 MB | 92 MB | 308 MB (482 B/word) |
| allocated by the call (after) | 2.5 GB | 4.2 GB | 11.2 GB | 34.5 GB |
| time, before (idle machine) | 26 s | 86 s | 288 s | |

Loading peaks at 840 MB and leaves 827 MB (a 638 MB GC heap: the models) in every run. Single runs: the peak moves by
±50 MB between identical runs at 20k words, and more on the long ones, where it depends on when the GC runs.

**What the peak is made of.** The live heap after each processor (`--live`, 670,972 words, after) separates what is
kept from garbage: 983 MB after tokenize, 1,406 after pos, 1,203 after constituency, 1,520 after depparse, 1,784 after
sentiment, and 961 at the end, against 627 MB after loading. At the end that is exactly the models plus the documents
(627 + 308 MB): nothing else survives the call. During it:

- **Independent of the length:** the models (640 MB on the GC heap), the scratch buffers of the largest batches, which
  `ArrayPool<T>.Shared` keeps (0.3–0.8 GB; every processor's batches are capped: pos, depparse and sentiment at 5,000
  words, the tokenizer at 32 paragraphs of at most 1,000 characters, the parser at 50 sentences, NER at 32), and the
  `CharlmCache` (8 KB per word up to its 32,768-word cap, so 256 MB from about 30,000 words on).
- **The output**, the `Document`: 482 bytes per word after this round. It is the only thing that grows. For all eight
  processors (measured per processor set at 6,500 words): tokens and words with their texts, whitespace and offsets
  310 B/word, the constituency tree 150, entities and NER tags 21; tags, features, lemmas, heads and relations about 3
  (fields of `Word` pointing at shared strings; the feats strings took 46 B/word before they were shared).
  `Document.Text` and each `Sentence.Text` add 2 bytes per character each.
- **Per-processor lists over the whole document** (the round-2 worry in "Remaining ideas"): each processor's lists of
  sentences, words or (word, tag) pairs and its sort order are references and indices, 10–40 bytes per word, and are
  dropped when the processor returns. The largest transient one is the tokenizer's: every paragraph's code points,
  offsets, vocabulary ids and features are built before the first batch, about 85 bytes per character (about 0.5 KB per
  word), during tokenizing only, which never sets the peak (1.7 GB at 670,972 words against 3.0 GB later).
- **Garbage the GC hasn't collected yet**, the rest: the peak was 0.6–1.1 GB above the live heap. The call allocates
  56–125 KB per word (per-batch arrays the pool doesn't hold: GEMM results, LSTM outputs, logits), and the GC lets more
  of it accumulate as its heap grows (its gen2 and large-object budgets scale with what survived). It is not needed
  memory: with the GC heap capped by `DOTNET_GCHeapHardLimit` (a container's memory limit sets 75% of it by default),
  201,313 words ran in a 1,792 MB heap (`0x70000000`) and peaked at 1,782 MB instead of 2,259 MB, with 28 gen2 GCs
  instead of 23 and no measurable slowdown under the same load; 670,972 words ran in a 2,304 MB heap
  (`0x90000000`), peaking at 2,230 MB instead of 3,024 MB (41 gen2 GCs instead of 37; 4,593 s against 4,207 s, within this load's noise).

**What changed** (no output changes; all golden tests pass):

- The `Document` shrank from 594 to 482 bytes per word (−19%; 72 MB at 670,972 words): `Token.Words` starts with room
  for one word (an empty `List` grows to four on its first `Add`), `SpaceAfter` shares one `" "` instead of a new
  substring per token, the tagger shares equal feats strings within a call, and constituency trees keep their children
  in arrays instead of lists.
- NER, which runs last, releases each batch's `CharlmCache` entries once it has read them, so the cache (256 MB on long
  texts) is gone by the end of NER instead of held to the end of the call.
- The peak fell by about 0.6 GB at 670,972 words (3.6 → 3.0 GB) and within noise below that; the documents account for
  72 MB of it, the cache release and GC timing for the rest.

**Splitting the input.** Not done inside the library: it would change output. Stanza (and this port) batch the
sentiment classifier and the dependency parser over the sentences of the whole call, sorted by length; a sentence's
sentiment label depends on its batch's padding, and depparse's arc log-softmax runs over its batch's padded width.
Calling `Process` per part of about 1,000 words on 20,513 words (all eight processors) changed 508 of 2,091 sentiment
labels (24%) and nothing else: every token, sentence boundary, lemma, tag, feature, head, relation, tree and NER tag
was the same. Sentence ids restart in each part, offsets are each part's own, and the whitespace at each cut is split
between the parts (round 3). For the memory: 201,313 words in 148 parts of about 1,000 words, keeping every part's
document, peaked at 2,056 MB against 2,259 MB for one call (smaller batches, so smaller scratch buffers and less garbage) and
allocated 15.1 GB instead of 11.2 GB. A caller that drops each part's `Document` after use also saves the
documents' 0.5 KB per word.

Guidance (README, "Very long texts"): one call is fine for a book; the memory beyond the models and batches is the
`Document` (about 0.5 KB per word) plus collectable garbage. In a memory-limited container the GC adapts to the limit.
To bound memory further, split at blank lines and drop each part's `Document` after use, accepting the sentiment and
sentence-id differences.

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
  flight is best. `PipelineOptions.MaxConcurrentCalls` limits it.
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

A short backlog. Items marked *TorchSharp only* concern the `StanzaSharp.Cuda` backend; the managed backend (the
default) doesn't have them.

**Open:**
- **Depparse's backward charlm in batches.** Today only a one-sentence document continues from the tagger's final
  backward state (`CharlmCache.TryGetAlone`), because a batched row's kernel was thought to depend on its batch. Since
  `Gemm.KernelBlocked` a row's step is batch-invariant (round 18 checked it bit for bit for the forward charlm), so the
  parser could continue every cached sentence from the tagger's state over ROOT alone, if the cache also kept each
  sentence's final cell state (4 KB per sentence). That is the whole backward pass of depparse's charlm work, not 3 steps.
  Check it on the backward charlm the same way first.
- **Long texts: what is left of the garbage** (round 18): per-word objects outside the charlm (tokenizer inputs,
  constituency's `StackState`s, `bool[]`/`object[]`/`string[]`), each under 1 KB per word. The peak is set by live data.
- **Loading `.pt` files.** The pretrain's and NER's 250k-word vocabularies are pickled, mirrored into JSON
  (`TorchCheckpoint.Splitter`), parsed back into nodes and then into a dictionary: about 0.3 s each when warm, more cold,
  and the pretrain then NER are the load's longest chain (round 17). Building the vocabulary straight from the pickle
  would cut most of it.
- **Sentiment in default_fast.** With nothing to reuse it runs both charlms on every token. Only a faster charlm (or
  Stanza changing the package) would help.
- **Parser step loop: per-state tensor ops** (*TorchSharp only*). Each step takes one `WordHx` row view per state and
  stacks per-state tensors; a shared "bank" tensor per stack (`index_select` / `index_copy_`) would make the op count
  per step constant. Exact (data movement only); about 2% on the CPU, likely more on a GPU (docs/gpu.md).
- **Padding in depparse's input** (*TorchSharp only*). Depparse builds its input and charlm outputs padded; building
  them in packed order like the tagger (round 4) would remove one padded copy of the 2423-wide input per batch.
- **Depparse memory** (*TorchSharp only*): the [batch, width, width, 53] label scores (275 MB on the warm-up batch) set
  the peak there. The managed backend never had that tensor (round 9).
- **Per-call cost** (*TorchSharp only*): 192 ms (Python 189), libtorch's per-operation cost on tiny tensors. Not pursued.

**Done** (see the round):
- Depparse's forward charlm from a stored start state (exact, 3 of ~90 steps: under the noise, not shipped); the
  `ArrayPool`'s churn on long texts (measured: none; the charlm's per-character strings were cut instead): round 18.
- Startup: the runtime's vector `Exp` and the thread pool at JIT tier 0, two models loading at a time; ReadyToRun and
  the `DOTNET_*` knobs measured: round 17.
- Small LSTMs on few cores (four tasks per thread in a step), the benchmark's warm-up, the e-mail/URL regex prefilter:
  round 16.
- AVX-512 kernels (`Vector512`): round 14.
- The tokenizer's long paragraphs (later windows run only unfinished rows): round 15.
- Thread count: the default is the physical cores when there are at least 4 (rounds 12–13).
- Per-call cost on short texts (managed): round 7; concurrent callers: `PipelineOptions.MaxConcurrentCalls` (round 7)
  and the pool's oversubscription fix (round 12).
- Very large documents: measured in round 8 (only the `Document` grows); allocations cut in round 11. The tokenizer's
  up-front inputs (about 85 bytes per character) never set the peak (round 8).
- Managed depparse and sentiment memory: rounds 9 and 10.
