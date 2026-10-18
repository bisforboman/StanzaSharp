# Results, round 46: default_accurate on the runners

Part of [performance.md](../performance.md)'s measurement history.

The first `default_accurate` run of [benchmark.yml](../../.github/workflows/benchmark.yml) against Python Stanza
(run [39108325436](https://github.com/bisforboman/StanzaSharp/actions/runs/39108325436), 2026-10-18). It includes round
45's encoder speed-up. Python Stanza ran with tools/requirements.txt plus tools/requirements-transformer.txt (transformers
5.20.1, peft 0.21.2), which benchmark.yml installs for this package since #177. Each side used its default threads. The
text was the default one: 1,994 words after tokenizing and MWT.

These are medians of the three timed runs, read off the job logs. The full blocks (per stage, one-sentence calls, peak
memory, the CoNLL-U comparison) are in the run's summary and its `benchmark-<os>` artifacts.

| runner | StanzaSharp | Python Stanza | StanzaSharp's advantage |
|---|---:|---:|---:|
| ubuntu-26.04 (x64) | 27.8 s | 184.5 s | 6.6x |
| ubuntu-26.04-arm | 31.7 s | 245.1 s | 7.7x |
| windows-2025 (x64) | 16.4 s | 271.6 s | 16.6x |
| windows-11-arm | 34.5 s | 940.6 s | 27.3x |
| macos-26 (Apple Silicon) | 54.2 s | 241.7 s | 4.5x |

- **Linux is the fair comparison.** Both Linux runners were steady (every timed run within 1% on both sides): 6.6x on
  x64 and 7.7x on Arm64. Round 44's 5.51x on a 4-core Xeon predates round 45's encoder speed-up (29.1 → 19.3 s on that machine).
- **Windows mostly measures PyTorch on Windows.** Python Stanza took 272 s on windows-2025 and 941 s on windows-11-arm,
  against 185 s and 245 s on the Linux runners of the same architectures. StanzaSharp's own times there (16.4 s and 34.5 s)
  are in line with Linux. windows-2025's 16.4 s is the fastest of all; that runner's CPU wasn't recorded in the log.
- **macOS was noisy.** The warm-up runs ranged from 58 to 74 s for StanzaSharp and from 211 to 334 s for Stanza, so its 4.5x
  is the least reliable figure. Its 3 cores and 7 GB also hold both pipelines (about 4 GB and 6 GB at peak, round 44).
- **Run times.** The jobs took 22 min (ubuntu-26.04) to 90 min (windows-11-arm), almost all of it Python Stanza.
  benchmark.yml's job limit is 150 min for this package (60 otherwise).

Next: benchmark.yml should also print each block into its job log, so the per-stage tables can be read without the
artifacts.
