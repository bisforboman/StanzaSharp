# Results, round 40: the TorchSharp backend's per-call cost (dropped)

Part of [performance.md](../performance.md)'s measurement history. Concerns only the `StanzaSharp.Cuda` (TorchSharp)
backend.

[Round 7](round-07-per-call-short-texts.md) measured one short sentence at 192 ms on TorchSharp (Python 189 ms), and
put that down to libtorch's fixed cost per operation on tiny tensors. This round measured it again next to the managed
backend.

## Measured

`--per-call 50`, `default` package, all eight processors, 8 threads, CPU, local Ryzen 7 5800X (with other load), median
ms per call:

| backend | tokenize | pos | constituency | depparse | sentiment | ner | total | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TorchSharp (`--backend torch`) | 6.7 | 50.7 | 42.5 | 60.1 | 9.0 | 4.6 | **203.9** | 294.9 |
| managed (the default) | 0.4 | 7.4 | 6.2 | 8.7 | 2.5 | 1.3 | **28.3** | 38.8 |

## Result

**Dropped; nothing changed.** The cost is still spread over the processors in proportion to how many torch operations
each runs per call. No single site dominates that StanzaSharp could batch away: the nets already run one forward pass
per batch, and the constituency parser's handles were already turned into bank rows (round 25). The rest is libtorch's
dispatch overhead per operation. Short-text services have the managed backend, the default since 0.5, which is about
7x faster per call. The TorchSharp backend's purpose is the GPU, where long texts are what matters.
