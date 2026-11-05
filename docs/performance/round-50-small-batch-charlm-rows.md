# Results, round 50: extra rows in the tagger's charlm steps (small batches)

Part of [performance.md](../performance.md)'s measurement history.

The small-batch item ([backends.md](../backends.md), "Small batches"): what does one short `Process` call cost on the
managed backend, and what of it can go without changing a bit?

**Machine:** a cloud session's 4-core Intel Xeon @ 2.10 GHz (1 thread per core, 2 MB L2 per core, 260 MB L3 shared;
AVX-512, so `Gemm.Path` is Vector512), Linux, .NET 10, 4 managed threads (the default here). Models: `tools/ci-models.sh en`
(`default` package, `.pt`). Before = main at 02e5a90, after = this change, run alternately.

## Where a short call goes

`StanzaSharp.Benchmark --per-call 200` (one 5–25-word sentence per call, round 7's method), before:

| threads | tokenize | pos | constituency | depparse | sentiment | ner | total (median / p90) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7 | 71.2 | 89.7 | 64.8 | 10.9 | 6.0 | 256.6 / 431.1 |
| 2 | 0.7 | 37.8 | 47.0 | 33.9 | 6.1 | 3.3 | 133.9 / 232.0 |
| 4 | 0.7 | 22.7 | 30.9 | 19.3 | 4.0 | 2.0 | 84.5 / 151.9 |

- The call scales almost linearly with threads, so there is no fixed dispatch cost to remove: the kernels are busy.
- A thread-time profile (`dotnet-trace`, `dotnet-sampled-thread-time`, 1 thread) puts 46% of the call in the charlm
  recurrence (`ManagedCharLanguageModel.Recur`): the tagger's two passes 26%, the parser's forward pass 14% (plus 1% for
  its ROOT continuation), sentiment 6% (sentences with multi-word tokens, whose tokens aren't the tagger's words).
  Constituency already takes the tagger's outputs; its time is the parser's own steps.
- **A one-row charlm step runs at memory bandwidth.** A scratch benchmark (not committed) ran `PackedLstm.Step` on a
  random 4096 × 1024 W_hh (the English charlm's, 16 MB) against a plain Vector512 read of the same bytes, split the same
  way, µs per step:

  | rows | 1 thread | 4 threads |
  |---:|---:|---:|
  | plain read | 510–630 | 125–162 |
  | 1 | 514–557 | 144–149 |
  | 2 | 586 | 137 |
  | 3 | 551 | 150 |
  | 4 | 731 | 196 |
  | 6 | 623 | 188 |
  | 8 | 913 | 267 |
  | 12 | 1,106 | 309 |

  One row reads W_hh at 30 GB/s per core (115 GB/s on 4), the speed of the plain read, so no kernel can make it faster
  without reading fewer bytes (fewer bits per weight would change outputs). But 2 and 3 rows cost what 1 does (the
  3-row kernel computes 3 rows anyway), 6 rows 1.2–1.3×, 12 about 2×.

## The change

So the cheap fix is fewer passes: run the sequences the later processors need as extra rows of the tagger's steps. A
step gives a row the same arithmetic in any batch (`Gemm.KernelBlocked`; the input projection is a table lookup), so a
row's outputs depend on its own characters only, and an extra row is bit for bit what its processor would compute.

- **The parser's forward pass.** The parser runs the charlms over `"\n"` (ROOT) + the words. Its backward pass already
  continues from the tagger's (round 7, round 20), but forward ROOT comes first, so every state differs. On a batch of
  at most `ManagedCharLanguageModel.SmallBatch` (4) sentences, the tagger's forward pass now also runs ROOT + each
  sentence and keeps the rows by texts (`CharlmCache.RootedForward`, `AddRootedForward`); the managed parser takes them
  (`TryGetRootedForward`) and computes only the sentences it doesn't find.
- **Tokens of multi-word-token sentences.** Sentiment and NER read tokens, so on a sentence with multi-word tokens they
  ran both charlms again (sentiment computed them, NER took sentiment's). On the same small batches the tagger's two
  passes now also run those sentences' tokens and keep them in the single-sentence table (`CharlmCache.ReadsTokens`,
  `AddAlone`). `BuildBoth` now copies any sentence of a batch the table holds, not only a single sentence, so sentiment
  and NER find them in multi-sentence documents too.
- Both only when the pipeline has such a reader with the tagger's charlm pair (`Pipeline.AnnotateWithModels`; the
  TorchSharp parser doesn't read the cache). The kept rows count toward `CharlmCacheOptions.MaxWords` like the others.
- Larger batches (bulk, long texts: the tagger batches up to 250 sentences) are unchanged: their steps are
  compute-bound, where extra rows would cost as much as a separate pass and only add memory.
- Tests: `DepparseTests.SmallDocuments_TakeTheTaggersRootedForwardCharlm_BitForBit` (1 to 4 sentences of every golden
  file: the parser's arc and label scores with and without the tagger's rows, bit for bit, and that most sentences took
  them) and `PipelineTests.SmallDocuments_EqualTheUncachedPipeline` (documents of 2 to 4 validation sentences, some with
  multi-word tokens: the CoNLL-U equals the pipeline's with the cache off). Every row's bits on every path were already
  pinned by `KernelBlocked_SumsInBlocksThenAddsInit` (each mr against a scalar reference) and
  `Vector512_GivesVector256sBits`. The golden tests pass unchanged on both backends.

## Result

`--per-call 200`, 4 threads, three alternating runs each, ms (median / p90):

| | pos | depparse | sentiment | ner | total |
|---|---:|---:|---:|---:|---:|
| before | 22.4–23.6 / 37–41 | 19.0–20.0 / 30–31 | 3.9–4.0 / 24–26 | 2.0 / 2.7–2.8 | **81.7–85.5 / 138–147** |
| after | 23.5–24.6 / 38–44 | 8.8–9.1 / 13.2–13.6 | 3.6–3.7 / 5.1–5.4 | 2.0–2.1 / 2.7–3.1 | **68.8–71.4 / 113–120** |

- **About 15% off the median and 19% off the p90.** Depparse per call is more than halved (its forward pass is gone),
  and sentiment's p90, the multi-word-token sentences, drops from 24 to 5 ms.
- Pos costs up to 1 ms more: the extra rows, the copies of the kept rows and the 3-row kernel for 2 rows no longer
  hiding a third one for free, where the token rows make 3 or more.

Bulk is unchanged: `--copies 2 --runs 3` (6,566 words), two alternating runs each, gave 11.73 / 11.63 s before and
11.48 / 11.91 s after (pos 1.81–1.99 s on both sides, depparse 2.03–2.07 → 2.00 s), peak working set 1,339–1,372 MB
against 1,355–1,379 MB, and the CoNLL-U was identical byte for byte.

## Tried and dropped

- **A faster one-row kernel.** The step already reads W_hh at the plain read's speed (table above), so there is
  nothing to win in the kernel.
- **Extra rows on any batch.** Above about 8 rows a step costs nearly what two separate steps do (12 rows: 2.1× of one),
  so for bulk calls the extra rows would only hold more memory until the parser runs. `SmallBatch` = 4 keeps every
  step with extra rows at 12 rows or fewer.
