# Results, round 44: default_accurate against Python Stanza

Part of [performance.md](../performance.md)'s measurement history.

Round 43 had StanzaSharp's column only: the tool's Python side couldn't run Stanza's transformer models. Since `compare.py`'s
`load_stanza` reads the transformer from `<models>/transformer/` (this round's change), `stanzasharp benchmark --package
default_accurate --python tools/.venv/bin/python` runs both sides: a cloud session's 4-core Intel Xeon @ 2.10 GHz (Vector512),
10 GB, Ubuntu 24.04; managed backend, 4 threads on both sides; Stanza 1.15.0, torch 2.14.1, transformers 5.20.1, peft 0.21.2;
2026-10-18. The CoNLL-U was identical on both sides.

| | StanzaSharp | Python Stanza | StanzaSharp's advantage |
|---|---:|---:|---:|
| load | 18.18 s | 21.18 s | 1.17x |
| tokenize | 0.73 s | 2.74 s | 3.74x |
| pos | 10.18 s | 31.86 s | 3.13x |
| lemma | 0.38 s | 0.86 s | 2.24x |
| constituency | 2.52 s | 17.00 s | 6.76x |
| depparse | 1.00 s | 34.50 s | 34.60x |
| sentiment | 10.44 s | 41.92 s | 4.01x |
| ner | 1.56 s | 18.88 s | 12.10x |
| **total** (1,994 words) | 26.82 s | 147.78 s | 5.51x |
| per call, median / p90 | 330.7 / 490.6 ms | 987.5 / 1,117.6 ms | 2.99x / 2.28x |
| peak memory | 4,079 MB | 5,768 MB | 1.41x |

- **Where the time goes:** Stanza runs the transformer once per processor (pos, constituency, depparse and ner each pay
  a frozen pass, sentiment its LoRA pass); StanzaSharp runs the frozen pass once, in pos, and depparse, constituency and
  ner read its states from the call's `TransformerCache`. That is depparse's 34.6x and ner's 12.1x. pos and sentiment,
  each one pass on both sides, are 3–4x: the encoder itself.
- **Load:** 18.2 s here against round 43's 9.4 s: the two sides load in parallel, so they shared the 4 cores and the
  memory bandwidth. Stanza loads its transformer once per processor too.
- The per-call ratio (2.99x) is lower than the default package's, since a one-sentence call is mostly the two encoder
  passes, which both sides compute at a similar rate per piece.
