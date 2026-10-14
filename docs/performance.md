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
stanzasharp download --lang de && stanzasharp benchmark --lang de    # German
```

It follows the method below on a smaller text: the golden texts (validation.txt, corpus.txt, tokenize_stress.txt,
shipped with the tool; `--lang de`: tests/golden/de's files of the same names) cut after the paragraph that reaches 1,500 whitespace-separated words (`--words`), which is
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

On GitHub's hosted runners: **Actions > Benchmark > Run workflow** (or `gh workflow run benchmark.yml -f package=default`; `-f lang=de` for German)
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

Open ideas are one file each in [performance/ideas/](performance/ideas/), so parallel pull requests don't collide
here. A new idea adds a file; when one is done (or dropped after measuring), its pull request deletes the file, and
its round file in [performance/](performance/) is the record. Files marked *TorchSharp only* concern the
`StanzaSharp.Cuda` backend.
