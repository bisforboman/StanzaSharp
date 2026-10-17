# Results, round 39: sentiment in default_fast (dropped)

Part of [performance.md](../performance.md)'s measurement history.

In `default_fast`, sentiment (`sstplus_charlm`) is the only model that reads the charlms. The nocharlm tagger keeps no
charlm outputs, so nothing can be reused, and the classifier runs both charlms over every token. The idea file said only
a faster charlm (or Stanza changing the package) would help. This round measured how much of the package that is.

## Measured

Local Ryzen 7 5800X, Windows 11, managed backend, 8 threads, converted models; the machine had other load.

| `default_fast` | tokenize | mwt | pos | lemma | depparse | sentiment | ner | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6,566 words (`--copies 2`), s | 0.09 | 0.00 | 0.23 | 0.12 | 0.93 | **2.15** | 0.11 | 3.62 |
| one sentence (`--per-call 100`), median ms | 0.44 | 0.02 | 1.02 | 0.40 | 4.03 | **8.72** | 0.50 | 15.36 |

Sentiment is 59% of the package's time on a long text and 57% of a short call.

## Result

**Dropped; nothing changed.** The time is the charlm itself: two 1024-unit LSTMs stepped once per character. Their
kernels are the ones every `_charlm` model runs (`PackedLstm.Step`, `Gemm.KernelBlocked`), tuned over several rounds
and kept bit for bit equal across paths and batch sizes. A cheaper charlm isn't an exact change. A user who doesn't
need sentiment leaves it out (`Processors`), which makes `default_fast` about 2.4x faster.
