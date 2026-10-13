# Results, packages: default_fast vs default

Part of [performance.md](../performance.md)'s measurement history.

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
