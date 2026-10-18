# Results, round 43: the default_accurate package

Part of [performance.md](../performance.md)'s measurement history.

The first measurement of `default_accurate` (docs/transformer-plan.md, PR 4): `stanzasharp benchmark --package
default_accurate` and, for comparison, `--package default`, one after the other on the same machine (a cloud session's
4-core Intel Xeon @ 2.10 GHz, Vector512, 10 GB, Ubuntu 24.04; managed backend, 4 threads; 2026-10-17). No code changed
in this round.

Python Stanza did not run: the tool's `benchmark.py` has no `transformers`/`peft` and doesn't make Stanza read the
transformer from the model folder (only `make_golden.py` does), so the block has StanzaSharp's column only. The output
itself is checked against Stanza by the golden row `models/default_accurate/` (`TransformerModelTests`).

| | default | default_accurate |
|---|---:|---:|
| words (after tokenize and mwt) | 2,009 in 163 sentences | 1,994 in 167 sentences |
| load | 1.26 s | 9.43 s |
| tokenize | 0.08 s | 0.81 s |
| pos | 0.87 s | 11.02 s |
| lemma | 0.07 s | 0.43 s |
| constituency | 1.68 s | 2.67 s |
| depparse | 0.95 s | 1.00 s |
| sentiment | 1.13 s | 11.52 s |
| ner | 0.51 s | 1.62 s |
| **total** | 5.29 s | 29.08 s |
| words/s | 380 | 69 |
| per call, median / p90 | 101.8 / 170.4 ms | 342.0 / 532.5 ms |
| peak memory | 1,083 MB | 4,158 MB |

- **5.5x slower, 3.8x the memory.** Almost all of the difference is the two encoder passes: the frozen one runs in pos
  (11 s; depparse, constituency and ner then read its states from the call's `TransformerCache` and cost about what their
  `_charlm` twins do), and sentiment's LoRA pass costs the same again (11.5 s). The tokenizer and lemmatizer are the
  `_charlm` models (per-character charlm states), about 10x and 6x the default package's.
- **Memory:** not broken down in this round. The transformer's float32 weights alone are 1.3 GB; the encoder's per-batch
  buffers (up to 8,192 pieces, 4,096 floats a piece in the intermediate layer) and the cache (at most 16,384 pieces, about
  256 MB) come on top.
- **Load:** 9.4 s, against 1.3 s for `default`; the transformer (`pytorch_model.bin`) is the difference.

Open ideas, not tried: the LoRA pass differs from the frozen one from the first layer on (query and value are adapted
in layer 0), so the two passes share only the embeddings; what could be shared is when they run (one sweep over the
layers, while each layer's weights are in cache). A smaller `BatchPieces` would lower the peak.
