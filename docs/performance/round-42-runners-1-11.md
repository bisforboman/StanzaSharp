# Results, round 42: 1.11.0 on the free runners

Part of [performance.md](../performance.md)'s measurement history.

The benchmark (`gh workflow run benchmark.yml`, `default` package, English and German) on main after 1.11.0, so after
rounds 35–41 (faster German lemma dictionary, converted vocabularies, less decoding garbage). No code changed in this
round. Runs 39029617599 (English) and 39029619310 (German), 2026-10-17; StanzaSharp 1.11.0, managed backend; Stanza
1.15.0, torch 2.14.1. Every runner gave identical CoNLL-U on both sides.

## English, `default` (all eight processors), 2,009 words

| runner | CPU | cores/logical | SIMD | threads | StanzaSharp words/s | Stanza words/s | speed-up | per call, median ms | peak memory MB |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| macos-26 | Apple M1 (Virtual) | 3/3 | Vector128 | 3 / 3 | 284 | 93 | 3.04x | 106.2 / 362.8 | 794 / 2,715 |
| ubuntu-26.04 | AMD EPYC 7763 64-Core Processor | 2/4 | Vector256 | 4 / 2 | 297 | 67 | 4.42x | 87.7 / 281.3 | 1,001 / 3,543 |
| ubuntu-26.04-arm | Neoverse-N2 | 4/4 | Vector128 | 4 / 4 | 281 | 81 | 3.49x | 86.5 / 642.8 | 1,076 / 3,694 |
| windows-11-arm | Cobalt 100 | 4/4 | Vector128 | 4 / 4 | 286 | 29 | 9.87x | 85.5 / 539.1 | 1,046 / 3,507 |
| windows-2025 | AMD EPYC 7763 64-Core Processor | 2/4 | Vector256 | 4 / 2 | 307 | 63 | 4.91x | 78.2 / 337.8 | 989 / 3,577 |
## German, `default` (all eight processors), 1,933 words

| runner | CPU | cores/logical | SIMD | threads | StanzaSharp words/s | Stanza words/s | speed-up | per call, median ms | peak memory MB |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| macos-26 | Apple M1 (Virtual) | 3/3 | Vector128 | 3 / 3 | 231 | 90 | 2.57x | 163.9 / 503.8 | 807 / 1,981 |
| ubuntu-26.04 | AMD EPYC 7763 64-Core Processor | 2/4 | Vector256 | 4 / 2 | 326 | 90 | 3.64x | 74.1 / 254.4 | 1,322 / 2,966 |
| ubuntu-26.04-arm | Neoverse-N2 | 4/4 | Vector128 | 4 / 4 | 299 | 116 | 2.58x | 79.1 / 600.8 | 1,362 / 3,050 |
| windows-11-arm | Cobalt 100 | 4/4 | Vector128 | 4 / 4 | 290 | 40 | 7.26x | 84.1 / 529.4 | 1,289 / 2,743 |
| windows-2025 | AMD EPYC 9V45 96-Core Processor | 2/4 | Vector512 | 4 / 2 | 813 | 121 | 6.70x | 48.4 / 216.3 | 1,316 / 2,871 |
## Load

| runner | English, StanzaSharp / Stanza | German, StanzaSharp / Stanza |
|---|---:|---:|
| macos-26 | 3.28 / 9.95 s | 10.82 / 21.32 s |
| ubuntu-26.04 | 0.95 / 5.81 s | 1.25 / 7.84 s |
| ubuntu-26.04-arm | 0.66 / 5.07 s | 0.93 / 6.58 s |
| windows-11-arm | 0.75 / 6.78 s | 0.92 / 9.52 s |
| windows-2025 | 0.81 / 7.86 s | 0.73 / 4.67 s |

## Reading it

- Against the last English runner run (38293654596, 2026-10-12, performance.md), on the same CPUs: StanzaSharp
  250 → 297 words/s (ubuntu EPYC 7763), 240 → 281 (Neoverse-N2), 249 → 307 (windows EPYC 7763), 228 → 286 (Cobalt 100),
  218 → 284 (macOS): 17–30% faster, while Stanza stayed at 29–81 words/s. Load was 1.7–4.3 s, now 0.66–3.28 s. That is
  rounds 17–32's work (startup, kernels, depparse, sentiment, constituency, the pool), not 1.11 alone.
- German against round 33 (2026-10-14), same CPUs: 316 → 326 (EPYC 7763), 294 → 299 (Neoverse-N2), 295 → 290
  (Cobalt 100), 805 → 813 (EPYC 9V45): unchanged within runner noise.
- StanzaSharp is 3.0–9.9x faster than Python Stanza on English and 2.6–7.3x on German, with 2.1–3.6x less peak memory.
- The German windows-2025 job drew an AMD EPYC 9V45 (Zen 5, Vector512): 813 words/s, 2.5x the EPYC 7763 jobs.
  Runners are shared virtual machines, so compare jobs of the same CPU and day only.
- macOS loads are slow on both sides (German 10.8 s vs 0.7–1.3 s elsewhere; Stanza 21.3 s): the virtualized runner reads
  the model files cold. Processing there is in line with the other runners.
