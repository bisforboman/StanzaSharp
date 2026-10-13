# Performance

How fast StanzaSharp runs compared with Python Stanza, what was optimized, and what is left.
The measurements here are of the TorchSharp backend (add `--backend torch` to the benchmark commands; since 0.5 the
default is the managed backend, measured in [backends.md](backends.md)), except rounds 6 and 7, which compare both,
and rounds 8 on, which measure the managed backend only.
Every change kept the output byte-identical: all golden tests pass, and on the benchmark text the
C# CoNLL-U equals Python's line for line, before and after.

Measurement rounds: one file each in [docs/performance/](performance/) (`round-NN-*.md`, newest = highest number);
a PR adding a round adds a file and doesn't touch this page.

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
- [Round 15](performance/round-15-tokenizer-long-paragraphs.md) made tokenize 1.7–2.7x faster on these runners; it now leads Python Stanza on every one.

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
