# Results, all eight processors (0.1.0)

Part of [performance.md](../performance.md)'s measurement history.

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
