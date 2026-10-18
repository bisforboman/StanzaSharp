# Results, round 47: rented transformer states

Part of [performance.md](../performance.md)'s measurement history.

#173's question was whether the transformer's per-call garbage is worth renting. It is. The managed encoder's outputs are
the last 4 layers of every piece: 16 KB a piece with ELECTRA-large. Each reader's layer mix (`TransformerEmbedder.Mix`)
adds another 4 KB a piece. Both were plain allocations, so a `default_accurate` call allocated about 150 KB a word, against
the default package's 7, and every array went to the large object heap.

Now:
- `ManagedTransformer.Encode` rents its outputs from `ArrayPool<float>.Shared` (`ITransformerNet.RentsOutputs`; the
  TorchSharp encoder keeps plain arrays).
- The call's `TransformerCache` owns the states it keeps and returns them when the call ends (it is `IDisposable` now,
  `using` in `AnnotateWithModels`). `Embed` returns every other state array once it has mixed it.
- The mix buffer, and `Windowed`'s long-sentence windows and states, are rented too.

A rented array can be longer than its states. Nothing reads past them, and the encoder tests compare the used part only.

Measured on a cloud session's 4-core Intel Xeon @ 2.10 GHz (Vector512), managed backend, 4 threads:
`StanzaSharp.Benchmark --package default_accurate --memory 2000 --calls 2 --verbose`, 2,905 words, the second call (the
pool already filled):

| processor | before | after |
|---|---:|---:|
| pos | 126 MB | 43 MB |
| constituency | 58 MB | 41 MB |
| depparse | 49 MB | 32 MB |
| sentiment | 99 MB | 55 MB |
| ner | 41 MB | 15 MB |
| **all eight** | 437 MB (150 KB/word) | 250 MB (86 KB/word) |
| process peak | 3,054 MB | 2,837 MB |
| time | 34.5 s | 34.7 s |

- **What's left:** each reader's per-word rows (4 KB a word, handed to its net), the processors' own allocations (lemma's
  59 MB is the contextual classifiers), and sentiment's LoRA pass, whose states are never cached.
- **Time:** unchanged within noise. The encoder's compute dominates; the gain is memory (−217 MB peak) and GC pressure.
- Outputs are byte-identical (the `Transformer*` tests on both backends).
