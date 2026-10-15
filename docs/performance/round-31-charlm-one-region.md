# Results, round 31: both charlms in one region per character step (dropped)

Part of [performance.md](../performance.md)'s measurement history.

[Round 27](round-27-whole-pipeline-profile.md) found that at 8 threads the caller waits 122 ms per call for the last task
of a charlm region, once per character step and model. The idea: step the forward and the backward charlm together, so
character step t of both is one `PackedLstm.Step` (one parallel region) instead of two, halving the regions while
keeping each task its size. It is exact, but on the free runners it was no faster for whole texts and slower for
one-sentence calls, so it is not shipped.

## What was tried

- `ManagedCharLanguageModel.Run` took several passes (model, sentences, optional start states, output, final c) and
  stepped them together: at step t every pass still running added its own `StepRows` (its W_hh, table rows, h and c)
  to one `PackedLstm.Step` call. `BuildBoth` (pos, ner, sentiment, and constituency, which called the two models one
  after the other) ran its two passes that way. Depparse ran three: the forward pass, the backward pass over the
  sentences the tagger's cache lacks, and the backward ROOT continuation of the others (round 20's `Continue`).
- `PackedLstm.Step` got an optional `tasksPerWork`: `TasksPerThread` × `Count` tasks per model rather than per region,
  so a task stayed 8 panels at 8 threads. `ManagedThreads` was not touched.

## Exactness

A temporary test (not committed) hashed with SHA-256, per path and thread count: both charlms batched over each of 5
golden files (`pipeline`, `validation_long`, `_social`, `_nonbmp`, `_contractions`; outputs and backward final c), 45
sentences alone through the single-sentence cache, the backward ROOT continuation of every sentence, the tagger's UPOS
logits (with a cache), the constituency parser's transition scores and trees, depparse's arc log-probs and label scores
per batch with the tagger's cache and without, sentiment's logits and NER's emissions with the cache and without, the
all-eight CoNLL-U of each file's text and of 20 one-sentence calls. Ryzen 7 5800X, Windows; main `1fe32df` and the branch:

| path, threads | charlm | alone | continue | pos | constituency | depparse (cached = fresh) | sentiment | ner | CoNLL-U | one-sentence |
|---|---|---|---|---|---|---|---|---|---|---|
| Vector256, 1 and 8 | `F65C038FE59C` | `1947C465690E` | `47436D15E9D5` | `93B736D0AB2C` | `B1591D7D92D9` | `2C8B58EE87E2` | `0DEA1F130F14` | `E9E703972CA6` | `DFA20957CEC9` | `FBAA4E57B6A4` |
| Vector128, 1 and 8 | `F65C038FE59C` | `1947C465690E` | `47436D15E9D5` | `515681D8627B` | `5EDC594AE448` | `D7D1315FE988` | `817977C23D2D` | `DD71EA0EA13C` | `DFA20957CEC9` | `FBAA4E57B6A4` |
| Scalar, 1 and 8 | `BCE8C4EB2978` | `AF9CDD1B85CF` | `4E9CB829AC04` | `A378E825B1E0` | `E46F35E5077A` | `038EEF171DFF` | `09E1CBF21616` | `A540950C431F` | `DFA20957CEC9` | `FBAA4E57B6A4` |

Every hash was the same on main and on the branch (Vector512 gives Vector256's bits by test). The targeted tests
(ManagedBackendTests, NnTests, `Batches_ContinueTheTaggersBackwardCharlm_BitForBit`,
`OneSentenceDocument_ContinuesTheTaggers...`, `OneSentenceDocuments_EqualTheUncachedPipeline`) passed.

## Speed

`benchmark.yml` on the branch and on main, two pairs, the same day (`stanzasharp benchmark`, 2,009 words, 4 threads; 3 on
macOS). Seconds; per call is the one-sentence median in ms. Pair 2's ubuntu-26.04 main job drew an EPYC 9V74
(Vector512), so it is left out.

| runner | | pos | ner | depparse | total | per call |
|---|---|---:|---:|---:|---:|---:|
| ubuntu-26.04 (EPYC 7763) | main | 1.31 | 0.69 | 1.61 | 7.35 | 80.5 |
| | branch | 1.32 / 1.38 | 0.72 / 0.76 | 1.62 / 1.64 | 7.37 / 7.59 | 100.0 / 103.3 |
| windows-2025 (EPYC 7763) | main | 1.32 / 1.43 | 0.72 / 0.89 | 1.66 / 1.67 | 7.54 / 8.32 | 88.7 / 73.3 |
| | branch | 1.32 / 1.35 | 0.75 / 0.74 | 1.63 / 1.65 | 7.49 / 7.53 | 93.6 / 96.6 |
| ubuntu-26.04-arm (Neoverse-N2) | main | 1.59 / 1.62 | 0.82 / 0.83 | 1.69 / 1.74 | 7.77 / 7.94 | 91.2 / 91.7 |
| | branch | 1.64 / 1.63 | 0.83 / 0.82 | 1.72 / 1.71 | 7.90 / 7.86 | 94.3 / 90.2 |
| windows-11-arm (Cobalt 100) | main | 1.57 / 1.64 | 0.78 / 0.83 | 1.70 / 1.73 | 7.76 / 7.92 | 86.9 / 94.1 |
| | branch | 1.75 / 1.63 | 0.83 / 0.82 | 1.74 / 1.71 | 8.07 / 7.87 | 92.5 / 89.8 |

macOS (M1, virtual) varied by 2.4 s between main's own runs and is left out. Peak memory was unchanged.

- **Whole text:** no stage got faster on any runner; the charlm stages (pos, ner) were the same or up to 5% slower.
  At 4 threads round 27 expected less than its 60 ms at 8, and nothing showed.
- **One sentence:** the x64 runners' median per call rose from 73–89 ms to 94–103 ms on all four branch runs. A
  one-sentence step is a product of one row with W_hh (1024 × 4096 floats, 16 MB): it streams the weights. One model at
  a time keeps its 16 MB in the 32 MB L3 for all of its steps; both models in a step stream 32 MB per step, which no
  longer stays in L3. The same holds, less visibly, for the batched steps.

## Verdict

Exact, but slower for short calls and no faster for long ones on the runners, so it is dropped (the branch's code is not
kept). A step that combined the models only when its rows make it compute-bound (many sentences) could keep the short
calls' cache behaviour, but its expected gain is the 8-thread join wait only (about 3% of a call on the desktop), which
the 4-thread runners can't show; it isn't worth its code now.
