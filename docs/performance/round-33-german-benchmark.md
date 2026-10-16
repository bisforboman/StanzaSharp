# Results, round 33: German on the free runners

Part of [performance.md](../performance.md)'s measurement history.

The first full German benchmark (`gh workflow run benchmark.yml -f lang=de`, `default` and `default_fast`), after #129
(German MWT decodes with Stanza's beam search of 5). It also answers whether the macOS German smoke run (0.90x, run
38495088127, 2026-10-14) was real. No code changed in this round.

## Result

StanzaSharp is 2.4–7.4x faster than Python Stanza on German `default`, and 1.8–4.4x on `default_fast`, with identical
CoNLL-U on every runner. Peak memory is 2.0–3.8x lower. The macOS smoke run was quick-mode noise, not Spotlight and not
a German problem: the two full runs on macOS give 2.43x and 2.81x.

## The runs

Text: the benchmark's German texts (tests/golden/de: validation, corpus, tokenize_stress), cut after 1,500 whitespace
words: 1,933 words after MWT, 142 sentences. Median of 3 timed runs after warm-up, the sides alternating; 50 one-sentence
calls. StanzaSharp 1.8.0-2-gec4f30f (main after #129), managed backend; Stanza 1.15.0, torch 2.14.1. Words/s are
StanzaSharp / Python Stanza.

`default` (all eight processors), two runs of the same commit (38784140005, 38793255979):

| runner | CPU (run 1 / run 2) | SIMD | words/s, run 1 | speed-up | words/s, run 2 | speed-up | per call ms (run 2) | peak MB (run 2) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| macos-26 | Apple M1 (Virtual) | Vector128 | 205 / 85 | 2.43x | 323 / 115 | 2.81x | 107 / 397 | 895 / 2,478 |
| ubuntu-26.04 | EPYC 7763 / EPYC 9V45 | Vector256 / Vector512 | 316 / 88 | 3.58x | 805 / 124 | 6.48x | 49 / 171 | 1,388 / 2,967 |
| ubuntu-26.04-arm | Neoverse-N2 | Vector128 | 294 / 114 | 2.58x | 284 / 110 | 2.58x | 91 / 697 | 1,442 / 3,000 |
| windows-11-arm | Cobalt 100 | Vector128 | 294 / 40 | 7.39x | 295 / 40 | 7.42x | 86 / 532 | 1,217 / 2,557 |
| windows-2025 | EPYC 9V45 / EPYC 9V74 | Vector512 / Vector256 | 778 / 121 | 6.40x | 291 / 75 | 3.90x | 82 / 336 | 1,405 / 2,873 |

`default_fast` (seven processors, no constituency; run 38784144270):

| runner | CPU | SIMD | words/s | speed-up | per call, median ms | peak MB |
|---|---|---|---:|---:|---:|---:|
| macos-26 | Apple M1 (Virtual) | Vector128 | 342 / 190 | 1.80x | 104 / 169 | 673 / 2,568 |
| ubuntu-26.04 | EPYC 9V74 | Vector512 | 519 / 169 | 3.06x | 50 / 94 | 1,114 / 2,783 |
| ubuntu-26.04-arm | Neoverse-N2 | Vector128 | 360 / 200 | 1.80x | 81 / 258 | 1,186 / 2,806 |
| windows-11-arm | Cobalt 100 | Vector128 | 322 / 73 | 4.41x | 79 / 254 | 1,041 / 2,453 |
| windows-2025 | EPYC 9V74 | Vector512 | 520 / 162 | 3.21x | 50 / 104 | 1,086 / 2,668 |

The x64 runners drew different CPUs between the runs, so their columns differ by the CPU (EPYC 7763 with Vector256
against 9V45 with Vector512: 316 vs 805 words/s). One `windows-2025` 9V74 reported no AVX-512 (Vector256, 291 words/s),
the other ones Vector512. The Arm runners repeat within 4%.

## Per stage (`default`, seconds, StanzaSharp / Stanza)

| stage | ubuntu-26.04 (7763, run 1) | ubuntu-26.04-arm (run 1) | windows-11-arm (run 1) | macos-26 (run 1) | macos-26 (run 2) |
|---|---:|---:|---:|---:|---:|
| tokenize | 0.06 / 0.35 | 0.06 / 0.53 | 0.06 / 1.27 | 0.17 / 0.82 | 0.14 / 0.48 |
| mwt | 0.01 / 0.06 | 0.01 / 0.08 | 0.01 / 0.14 | 0.02 / 0.13 | 0.01 / 0.08 |
| pos | 1.42 / 3.05 | 1.69 / 2.25 | 1.66 / 7.33 | 2.38 / 3.16 | 1.60 / 2.07 |
| lemma | 0.08 / 0.17 | 0.06 / 0.24 | 0.07 / 0.37 | 0.09 / 0.39 | 0.06 / 0.19 |
| constituency | 0.81 / 3.93 | 0.75 / 3.25 | 0.76 / 9.54 | 1.21 / 4.03 | 0.74 / 3.80 |
| depparse | 1.53 / 5.47 | 1.61 / 3.67 | 1.60 / 12.40 | 2.21 / 5.46 | 1.46 / 4.76 |
| sentiment | 1.38 / 4.60 | 1.47 / 4.28 | 1.48 / 9.07 | 2.00 / 4.29 | 1.15 / 2.17 |
| ner | 0.83 / 4.23 | 0.92 / 2.66 | 0.94 / 8.52 | 1.32 / 4.56 | 0.84 / 3.29 |
| **total** | 6.11 / 21.87 | 6.58 / 16.94 | 6.58 / 48.63 | 9.41 / 22.84 | 5.99 / 16.84 |
| load | 1.56 / 8.12 | 1.05 / 6.84 | 1.27 / 9.61 | 4.02 / 11.38 | 2.88 / 9.23 |

- **pos** is the stage with the smallest lead everywhere (1.3–2.2x outside windows-11-arm): it runs both charlms over
  the whole text, and on macOS Python's torch probably gets Apple's matrix units through Accelerate for those LSTMs
  (not measured) while the managed backend has NEON (Vector128). English `default` shows the same order (pos 2.02x on macOS, 3.20x on ubuntu; run 38735518300).
- **sentiment and ner** cost more in `default_fast` (sentiment 1.63 s, ner 1.20 s on the 9V74) than in `default` (1.38 s
  and 0.83 s on the slower 7763), because `default_fast` has no charlm tagger whose outputs `CharlmCache` could hand
  them: they run both charlms themselves. Stanza recomputes them in both packages.
- German `default` and English `default` run at similar rates on the same CPU (EPYC 7763: 316 and 274 words/s; English
  run 38735518300, a different day).

## The macOS smoke run

The smoke run (quick: 372 words, one timed run after one warm-up, 10 calls) gave macOS 134 / 149 words/s, 0.90x, while
every other runner showed 2.3–6.1x. Checked:

- **Spotlight** was off: the job's "Turn off Spotlight indexing" step succeeded, as in every run since round 12.
- **Quick mode on macOS repeats badly.** The same quick German run today (38788912170) gave 127 / 93 words/s, 1.37x:
  Python went from 149 to 93 words/s, StanzaSharp stayed at 127–134. The smoke's Python number was the outlier.
- **The full runs** give macOS 2.43x and 2.81x (205 and 323 words/s; same commit, same image), so the macOS runner's own
  run-to-run spread is about 1.5x, more than any other runner's.
- **Quick mode underrates StanzaSharp on macOS.** With one warm-up run the managed tagger took 2.3 ms per word there
  (0.85 s for 372 words) against 0.83–1.23 ms in the full runs; on ubuntu-26.04 quick and full differ little (0.91 against
  0.73 ms). English shows the same pattern: macOS 1.87x quick (38788921487) against 4.04x full (38735518300), while
  ubuntu-26.04's quick run is as fast as its full one.

So German on macOS is not slow; compare only full runs, and macOS ones only within a run.

## Reproduce

```
gh workflow run benchmark.yml -f lang=de                         # default
gh workflow run benchmark.yml -f lang=de -f package=default_fast
gh workflow run benchmark.yml -f lang=de -f quick=true           # the smoke run
```
