# Results, round 48: one encoder sweep for the frozen and the LoRA transformer

Part of [performance.md](../performance.md)'s measurement history.

#169's question: `default_accurate` runs ELECTRA-large twice per call. The frozen pass serves pos, constituency, depparse
and ner; sentiment's LoRA pass adds the adapters. Each pass reads all 1.3 GB of weights. Is a short call bound by memory
bandwidth, so that one sweep over the layers for both would save a read?

Measured on a cloud session's 4-core Intel Xeon @ 2.80 GHz (33 MB L3, Vector256), managed backend, 4 threads.

## Is a short call bound by memory bandwidth?

`StanzaSharp.Benchmark encoder-scaling models/stanza/en/transformer/google/electra-large-discriminator --threads 4 --lora
models/stanza/en/sentiment/sstplus_electra-large`: synthetic sentences of n pieces, last 4 layers kept, median of 25 calls.
The last two columns are one frozen and one LoRA sentence, as a pass of each net and as one `EncodeJoint` sweep.

| pieces | one sentence | two in one call | two calls | frozen + LoRA, a pass each | one sweep |
|---:|---:|---:|---:|---:|---:|
| 3 | 51.1 ms | 54.0 | 102.2 | 114.5 | **74.2** |
| 8 | 65.4 | 98.4 | 130.8 | 149.1 | **105.0** |
| 16 | 87.1 | 153.8 | 174.2 | 239.1 | **189.2** |
| 32 | 159.9 | 284.9 | 319.9 | 355.1 | **331.6** |
| 64 | 274.7 | 519.0 | 549.5 | 676.9 | **569.3** |
| 128 | 560.3 | 1,077.9 | 1,120.6 | 1,143.3 | 1,145.7 |
| 256 | 1,445.4 | 2,878.3 | 2,890.7 | 3,081.4 | **2,902.3** |

- **Yes, for short sentences.** 3 pieces take 51 ms, which is 1.34 GB of weights at about 26 GB/s, and a second sentence
  in the same call adds 3 ms.
- **What one sweep saves.** Frozen + LoRA is 35% faster at 3 pieces, 21–30% up to 16, 16% at 64, and even from 128 on,
  where the arithmetic outweighs the weight reads. On 1 thread (measured before the change, `Encode` with two sentences)
  long sentences came out the same within noise, so there is no length cutoff.

## The change

- `ITransformerNet.EncodeJoint` runs groups of (net, sentences) in one sweep over the layers. Each group is the frozen net
  or one of its `WithLora` nets. The base GEMMs run over every row; they are row-invariant, so a row's bits don't depend on
  the others. Each group's adapters run over its own rows only, in `WithLora`'s order (base and bias, then the adapter). So
  every sentence gets exactly the bits its own net gives it alone (`TransformerKernelTests.JointPass_GivesEachGroupItsOwnNetsBits`,
  every native path, 1 and 4 threads, either group order). Managed only. The TorchSharp net pads each call like Stanza, so
  regrouping could move its bits: it keeps a pass per net (`EncodesJointly` false).
- After mwt, the pipeline asks every transformer reader for its sentences (`TransformerReader(doc)`):
  - pos and depparse: the simplify_punct'ed words;
  - constituency: the words;
  - ner and sentiment: the token texts.

  `TransformerEmbedder.Prefill` then encodes the distinct ones into one cache per net, which each reader's `Embed` finds.
  It runs only when two or more readers read a 4-layer mix and every distinct sentence fits one encoder call
  (`BatchPieces`, 8,192 pieces), so it holds at most that much. Otherwise nothing changes. German's `default_accurate` is
  three groups in one sweep: frozen pos and depparse, constituency's adapters and sentiment's.
- The benchmark's per-stage table gets a `transformer` row: the sweep, which pos and sentiment no longer contain. Python's
  table has no such row (its encoder time sits in pos and sentiment).

## Results

`StanzaSharp.Benchmark --models models/stanza/en --package default_accurate --per-call 60 --threads 4`: 60 calls of one
5–25-word sentence each, `main` (12ff936) and this change alternated, two runs each. Median / p90, ms per call:

| stage | main, run 1 | main, run 2 | joint, run 1 | joint, run 2 |
|---|---:|---:|---:|---:|
| transformer | – | – | 193.4 / 306.4 | 177.1 / 270.5 |
| pos | 142.6 / 192.7 | 136.2 / 199.5 | 31.6 / 53.0 | 30.6 / 46.9 |
| constituency | 57.4 / 87.1 | 53.5 / 88.5 | 51.6 / 87.9 | 49.1 / 85.1 |
| depparse | 29.3 / 43.4 | 31.0 / 47.9 | 28.5 / 50.9 | 27.8 / 44.6 |
| sentiment | 124.1 / 182.0 | 131.6 / 198.7 | 4.7 / 6.9 | 4.5 / 6.7 |
| ner | 6.3 / 44.4 | 6.4 / 45.5 | 6.3 / 31.5 | 5.4 / 26.7 |
| **total** | **404.0 / 553.1** | **389.6 / 578.3** | **348.3 / 543.2** | **322.0 / 515.2** |
| allocated per call | 1.53 MB | 1.53 MB | 1.58 MB | 1.58 MB |

- A one-sentence call is about 15% faster (median 390–404 → 322–348 ms). The sweep (177–193 ms) replaces about 230 ms of
  encoder time in pos and sentiment.
- ner's p90 drops too (44 → 27–32 ms): sentences with a multi-word token, whose token texts differ from the words, used to
  need a pass of their own in ner. They are now in the same sweep.
- Memory (`--memory 150 --calls 3`, 232 words, both twice): the process peak is the load's (2,415–2,510 MB either way), and
  the working set after repeated calls stays the same. The first call on a fresh process allocates 166 MB instead of
  106 MB, the pool's first fill of the sweep's larger buffers (round 23); the steady state is the 1.58 MB per call above.
- Long documents are unchanged: past 8,192 pieces nothing is pre-encoded, and each reader runs its own passes as before.

Every golden row is byte-identical on both backends: `TransformerModelTests`, English and German `default_accurate`,
`electra-large` and every single-model row.
