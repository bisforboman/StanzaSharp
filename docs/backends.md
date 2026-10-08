# Backends (issue #29)

StanzaSharp runs each processor's network on one of two backends:

- **TorchSharp** (libtorch): today's code. It is the default, it runs on the GPU, and it is the reference the
  managed code is tested against.
- **Managed**: C# SIMD kernels with no native dependencies (`src/StanzaSharp.Nn/Managed/`, see
  [managed-backend-spike.md](managed-backend-spike.md)). It is CPU only.

The owner's plan: 0.5 makes the managed backend the default, with TorchSharp still selectable. At 1.0 TorchSharp leaves
the main package for an opt-in `StanzaSharp.Cuda` package. Both implementations stay.

Phase 1 added the seam and ported **tokenize** and **mwt**. Phase 2 ports the other processors one at a time; so far
**ner** and **pos**, plus a backend-neutral `CharlmCache` (see [Phase 2](#phase-2-progress)). Everything is `internal`;
nothing public changed.

## The seam

The seam sits at each processor's **network**: one small interface per processor, with arrays in and arrays out.

| processor | interface | TorchSharp | managed |
|---|---|---|---|
| tokenize | `ITokenizerNet.Forward(ids, feats, rows, width, lengths, ct)` → [rows, width, 5] log-probs | `TokenizerNet` | `ManagedTokenizerNet` |
| mwt | `IMwtNet.Forward(ids, rows, width, lengths)` → [rows, width, 2] logits | `MwtNet` | `ManagedMwtNet` |
| ner | `INerNet.Forward(sentences, wordIds, deltaIds, width, charlms, cacheKeys, ct)` → [batch, width, tags] emissions | `NerNet` | `ManagedNerNet` |
| pos | `IPosNet.Forward(sentences, wordIds, pretrainIds, charlms, cacheKeys, ct)` → `PosOutput` (UPOS scores, UPOS/XPOS/feats ids) | `PosNet` | `ManagedPosNet` |

- **Shared:** everything around the network stays in the processor and serves both backends. That covers paragraph
  splitting, features, sorting, batching, the 1000-character windows, padding, the argmax, `FixLabels`, decoding,
  the MWT dictionary and the cut decision. Both backends therefore see exactly the same batches. That matters for
  the processors whose output depends on the batch: sentiment, and depparse's padded log-softmax.
- **TorchSharp side:** today's module code, unchanged. Its array method makes the input tensors on its device, runs
  the old `Forward`, and copies the result back. Before, `Tokenizer`/`MwtExpander` did exactly those steps
  themselves, so the default path is byte-identical by construction.
- **Managed side:** composed from shared building blocks in `Nn.Managed`:
  - `PackedMatrix` + `Gemm.Run`: every Linear (a layer's heads packed into one matrix, e.g. the tokenizer's tok/sent/mwt).
  - `PackedLstm`: the recurrence of a packed (bi)LSTM, cell fused into the GEMM.
  - `ManagedLstm` (new): a PyTorch `nn.LSTM(batch_first)` with any layers and directions, from its state-dict
    prefix. It takes a padded batch with lengths and returns the padded output, zero past each length, exactly
    like `Rnn.RunPacked`. Hidden sizes that aren't a multiple of 4 (MWT's 50) are padded with zero units, which stay
    exactly 0, so the real units are unchanged.
  - `ManagedHighwayLstm` and `ManagedCharLanguageModel` (Phase 0) are ready for pos and depparse.
- **Why not a finer seam** (per layer or per op)? A per-layer TorchSharp implementation would copy between host
  and device at every layer, which defeats the GPU. A per-op seam is a tensor library. At network level each
  implementation stays simple: the TorchSharp one is today's code, and the managed one is plain loops over arrays and
  a few blocks.

### Choosing the backend

- `Nn.Backend` (`TorchSharp`, `Managed`) and an internal `PipelineOptions.Backend` (default `TorchSharp`).
- `Pipeline` passes the backend to each ported processor's `Load(basePath, device, backend)`. Processors not
  ported yet ignore it and run on TorchSharp. So `Backend.Managed` today means "managed where ported".
- **Threads:** with `Backend.Managed`, `Load` also sets `ManagedThreads.Count`, with the documented semantics: an
  explicit `Threads`, or null for `min(current count, Environment.ProcessorCount)`. The count is ProcessorCount unless
  something set it lower, and `ProcessorCount` respects a container's CPU quota. It is process-wide, like
  `torch.set_num_threads`. TorchSharp's threads are still set as before, for the processors that aren't ported.
- Benchmark: `StanzaSharp.Benchmark --backend managed [--processors tokenize,mwt]`.

### Plan for the `StanzaSharp.Cuda` split (proposal, not done)

The split needs no change to the processors' logic; it only moves where the TorchSharp networks are built.

1. **Assemblies:**
   - Each processor assembly keeps its network interface (`ITokenizerNet`, …) and its managed network.
   - The TorchSharp networks move into a new `StanzaSharp.TorchSharp` assembly: `TokenizerNet`, `MwtNet`, and the
     torch layers in `Nn` (`CharLanguageModel`, `HighwayLstm`, `Biaffine`, `Rnn`, `Weights`, `Scalars`, the tensor
     half of `Pretrain`/`CharlmCache`). That assembly is the only one referencing TorchSharp.
   - Directory.Build.props already gives every StanzaSharp assembly `InternalsVisibleTo` the others, so the new
     assembly can implement the internal interfaces.
2. **Plugging in:**
   - Processors receive their network instead of choosing it: `Tokenizer.Load(basePath, ITokenizerNet net)`.
   - The facade (which references every processor) defines an internal factory, roughly
     `INetFactory { ITokenizerNet Tokenizer(Checkpoint); IMwtNet Mwt(Checkpoint); … ICharlm Charlm(Checkpoint); }`.
     It also covers the shared pretrain and charlms, which several processors take.
   - `ManagedNets : INetFactory` lives in the main package; `TorchSharpNets : INetFactory` (device, TF32) lives in
     `StanzaSharp.TorchSharp`.
   - The public option carries a factory, so the main assembly never references the Cuda one (see the API proposal).
3. **Packages:**
   - `StanzaSharp` (managed): Core, Nn without torch, the processors, and the facade, with no native dependency.
   - `StanzaSharp.Cuda`: `StanzaSharp.TorchSharp.dll`, depending on `StanzaSharp` and managed `TorchSharp`. Users
     add `TorchSharp-cuda-*` themselves, as today.
   - The `StanzaSharp.Cpu.*` platform packages and the `buildTransitive` version check move with it, or are dropped,
     since CPU users no longer need libtorch.
4. **Order:**
   - 0.5: both backends in the main package (TorchSharp still a dependency), managed the default.
   - 1.0: the move above, once every processor is ported.

### Public API (decided 2026-10-08: A; ships with 0.5, not implemented yet)

- **A (chosen):** `PipelineOptions.Backend` of a sealed public `PipelineBackend` class with only internal members:
  - `PipelineBackend.Managed` (default from 0.5) and `PipelineBackend.TorchSharp` in the main package.
  - From 1.0, `CudaBackend.Create(int deviceIndex = 0, bool disableTf32 = false)` in `StanzaSharp.Cuda`.
  - `PipelineOptions.Device`/`DisableTf32` become `[Obsolete]` forwards in 0.5 and are removed in 1.0.
  - It survives the package split without a breaking change to the option itself.
  - A pipeline on the managed backend must load with no native libtorch package: from 0.5, `Load` calls no torch
    function (not even `torch.set_num_threads`) unless TorchSharp is used. `PipelineBackend.TorchSharp` leaves the main
    package at 1.0.
- **B (not chosen):** an enum `PipelineBackend { Managed, TorchSharp }` next to today's `Device`/`DisableTf32`. It is simpler
  for 0.5. But at 1.0 the main package would have to find the Cuda implementation by reflection, and `Device` (a
  TorchSharp type) would still be on the main package's options, so it breaks then.
- Either way `Threads` keeps its meaning: threads per operation for the backend in use.

## Exactness

All golden tests that exercise tokenize and mwt run on both backends (xUnit theories with `managed: true/false`).
On the managed backend, every one of them is **byte-identical** to the golden data, with no tolerance changed:

- `TokenizerTests`: corpus tokens, sentences, offsets and MWT flags; `tokenize_stress` (a paragraph over 1000
  characters and more paragraphs than one batch); empty and whitespace-only input.
- `MwtTests`: dictionary and classifier-only expansions (`mwt.json`); tokenize → mwt words and offsets.
- `PipelineTests`: `pipeline.conllu` and all 12 `validation*.conllu` through the full 8-processor pipeline.
- `InputModeTests.Bulk_MatchesStanzasBulkProcess`, and `NoSsplitTests` (no_ssplit golden, plain and bulk).
- `ConcurrencyTests.ConcurrentCalls_EqualSequentialOutput`: 8 threads, mixed input modes.

Logit drift (max |diff|):

| | TorchSharp | managed |
|---|---:|---:|
| tokenizer log-probs vs golden (Python), 3 sentences | 3.8e-6 | 1.1e-5 (5.7e-6, 7.6e-6, 1.1e-5) |
| tokenizer net, managed vs TorchSharp (random batch, all 3 kernel paths) | – | 1.5e-5 (Vector256, Vector128), 1.1e-5 (Scalar) |
| MWT logits, managed vs TorchSharp (random batch, all 3 paths) | – | 1.4e-6 |
| `ManagedLstm` vs `Rnn.RunPacked` (tokenizer's rnn; MWT's 2-layer encoder) | – | < 1e-5 (tested at 1e-5) |

The 1e-4 tolerance of the tokenizer's golden logits test holds for the managed backend too.

Seam tests (`ManagedBackendTests`, per kernel path) compare the two implementations block by block and network by
network: `ManagedLstm_MatchesRunPacked`, `TokenizerNet_ManagedMatchesTorchSharp` and `MwtNet_ManagedMatchesTorchSharp`.
They join the Phase 0 tests of GEMM, the LSTM, the charlm and the highway layer.

That class sets the process-wide `Gemm.Path` and `ManagedThreads.Count`. It now runs in its own collection with
parallelization off (`ManagedKernelsCollection`), so the managed golden tests in other classes never run on a path
it switched to.

## Speed

`StanzaSharp.Benchmark --processors tokenize,mwt --backend torch|managed --threads N --runs 5`, 8 copies (116,286
characters, 1,992 sentences, 26,264 words). Ryzen 7 5800X (8 cores, AVX2, Vector256 path), Windows 11, machine
nearly idle (8–16% load). Medians in seconds, two rounds each:

| threads | tokenize TorchSharp | tokenize managed | ratio | mwt TorchSharp | mwt managed |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.62 / 1.65 | 0.33 / 0.33 | **0.20** | 0.01 | 0.01 |
| 1 | 1.73 / 1.58 | 1.01 / 0.98 | **0.60** | 0.01 | 0.01 |

- The tokenizer's LSTMs are small (hidden 64, up to 32 rows, up to 1000 steps per window). libtorch spends most of
  each step on per-op overhead and gets nothing from 8 threads. The managed recurrence is one fused region per step.
- The stage times include the non-neural work: features, regexes and decoding.
- mwt is mostly dictionary lookups; the classifier sees a handful of tokens per document, so the stage is too short to
  compare.
- **Load:** all models load in 1.1–1.3 s on either backend; the tokenizer and MWT weights are small, and packing them
  doesn't show.

### Small batches (the kernels report's open item)

The kernels report had the highway biLSTM at 1.9× TorchSharp on 320 words (16 sentences × 20 words), from
`Speed_IsReported`. Remeasured in isolation, with medians of 7 after a warm-up:

| 320 words | TorchSharp | managed | of which GEMMs | recurrence (both layers) |
|---|---:|---:|---:|---:|
| 8 threads | 17.2–18.0 ms | 12.6–14.5 ms | 6.9–7.6 ms | 2.9–3.1 ms |
| 1 thread | 41.1–41.3 ms | 39.9–40.0 ms | 33 ms | 13–15 ms |

`Speed_IsReported` itself gave 1.02× on a quiet machine. The 1.9× was most likely load from other processes: that
test takes medians of only 3 runs, and it ran while other work was using the CPU.

Fewer threads for small steps was also tried: a minimum amount of work per task in `PackedLstm.Step`. On the
tokenizer, the smallest steps in the pipeline (32 rows × K 64 × 32 panels), it was slower:

| minimum work per task (panels × rows × K) | tokenize, 8 threads |
|---|---:|
| none (every step on all threads) | 0.40–0.42 s |
| 4,096 | 0.41–0.43 s |
| 16,384 | 0.46–0.57 s |
| 65,536 (one task) | 0.75–0.83 s |

So no change was made. Splitting by rows as well isn't needed yet: every recurrent step ported so far has at least
26 panels to share out (MWT: 2 directions × 13), more than the 8 threads. A unidirectional layer with a small hidden
size would need it; none is ported.

## Phase 2 progress

| processor | ported | exactness on the managed backend | ner stage, 8 threads (TorchSharp → managed) | 1 thread |
|---|---|---|---|---|
| tokenize | Phase 1 | byte-identical | 1.62 → 0.44 s | 1.82 → 1.06 s |
| mwt | Phase 1 | byte-identical | too short to compare | |
| **ner** (`_charlm`) | yes | byte-identical tags and entities; emissions 1.1e-5 from Python (TorchSharp 3.7e-6) | **24.01 → 6.46 s (0.27)** | **55.40 → 26.39 s (0.48)** |
| **ner** (`_nocharlm`, default_fast) | yes | byte-identical; emissions 1.05e-5 from Python | **1.70 → 0.56 s (0.33)** | **2.42 → 1.43 s (0.59)** |
| **pos** (`_charlm`) | yes | byte-identical; UPOS logits 6.1e-5 from Python (TorchSharp 5.3e-5; Scalar path 6.9e-5) | pos stage **11.51 → 5.72 s (0.50)** | **32.07 → 23.83 s (0.74)** |
| **pos** (`_nocharlm`, default_fast) | yes | byte-identical; UPOS logits 3.1e-5 from Python | | |
| depparse, sentiment, lemma, constituency | no | | | |

Speed: `StanzaSharp.Benchmark --processors tokenize,ner --backend torch|managed --threads N --runs 3` (so NER computes every
charlm itself; no tagger, no cache), 8 copies (24,840 words), medians, Ryzen 7 5800X, idle machine. In the full
8-processor pipeline (8 threads) the ner stage goes from 14.3 to 3.7 s and the run from 58.7 to 46.7 s; there NER
reads the tagger's cached charlm outputs for sentences without MWTs.

### pos

- **Seam:** `IPosNet.Forward(sentences, wordIds, pretrainIds, charlms, cacheKeys, ct)` → `PosOutput`: per word (sentences in
  order) the UPOS scores and the UPOS, XPOS and each feat's ids. The heads' argmaxes are in the net (XPOS and feats
  read the argmax UPOS's embedding); `PosTagger` keeps simplify_punct, the vocab lookups, batching (250 sentences /
  5000 words), the cache keys (only sentences simplify_punct left unchanged) and the id → string decoding.
  `PosTagger.Load` builds `PosNet` (today's code); `PosTagger.LoadManaged` builds `ManagedPosNet`.
- **Managed net:** input rows built at their packed positions (word embedding, `trans_pretrained` GEMM over the pretrain
  vectors, the charlm columns or `ManagedCharacterModel` + `trans_char`), `ManagedHighwayLstm`, then one GEMM for
  `upos_hid`, `tag_hid.xpos` and `tag_hid.feats` (they all read the LSTM output).
  - **Biaffine heads per UPOS:** `Bilinear(x, upos_emb[u])` is linear in x for each of the 21 UPOS values, so at load
    each scorer is contracted (in double) with every UPOS embedding: XPOS becomes one [21·53, 400] linear layer and the
    21 feats one [21·149, 100] layer. A row reads the block of its argmax UPOS. That is 0.45M + 0.31M multiply-adds per
    word against TorchSharp's 1.08M + 0.77M bilinear ones.
  - **`upos_clf` in double:** the UPOS logits reach |150|, and the GEMM's 400-term float sums put them up to 1.07e-4
    from Stanza's on the Scalar path (9.2e-5 on the SIMD paths), over the 1e-4 tolerance. Measured: computing the whole
    UPOS MLP in double gives the same result as only `upos_clf` in double (21 outputs per word), so only that runs in
    double: 3.1e-5 to 6.9e-5 from Stanza's. Its cost is below the run-to-run noise (pos stage 24.86 / 5.63 s in float
    vs 23.83 / 5.72 s at 1 / 8 threads).
- **Cache producer:** with the charlms, the managed tagger hands the cache `[words, 1024]` float arrays
  (`TryAdd(sentence, float[], float[], words)`), copied out of its pooled buffers only if the cache has room
  (`CharlmCache.HasRoom`). Managed NER reads them as they are; constituency and sentiment (TorchSharp) read them as
  tensors, converted once per entry. Measured against the TorchSharp tagger's entries: 6.0e-6 (charlm drift).
- **Mixed pipelines are byte-identical:** the default pipeline on `Backend.Managed` (tokenize, mwt, pos, ner managed;
  lemma, constituency, depparse, sentiment TorchSharp) reproduces `pipeline.conllu` and every validation file, also with
  the cache off and capped at 10 words (`CharlmCacheSettings_DoNotChangeOutput`), and the all-eight check against each
  golden folder (`Process_AllEightProcessorsMatchEachGoldenFolder`).
- **Tests on both backends:** `PosTests` (UPOS logits at 1e-4, golden tags), `LemmaTests` and `DepparseTests` pipelines
  (their inputs are the tags), `InputModeTests.Pretokenized_MatchesGolden` (both packages), the cache settings and
  all-eight tests above, and a managed case of `Canceled_InsideEachProcessor_ThrowsPromptly…`.
  `FastPackageTests.ManagedIntermediates_MatchGolden` covers the nocharlm logits. `ManagedBackendTests`:
  `PosNet_ManagedMatchesTorchSharp` (both checkpoints, a simplify_punct sentence, the cache entries) and
  `PosUposLogits_MatchGoldenOnEveryPath` (both checkpoints at 1e-4 on every kernel path).
- **Cancellation:** as before: after each charlm pass (or the character model), inside the highway LSTM (per GEMM
  block and time step) and before the heads (and between XPOS and feats).
- **Speed** (`--processors tokenize,mwt,pos`, 8 copies, 26,264 words, medians of 3, idle Ryzen 7 5800X): pos stage
  11.51 → 5.72 s at 8 threads, 32.07 → 23.83 s at 1 thread. Full 8-processor run at 8 threads: 59.88 → 40.88 s (pos
  11.79 → 5.71, ner 14.43 → 3.35; the rest unchanged); peak working set 3,021 → 3,144 MB (the managed charlms' input
  tables, and cache entries that hold both forms once converted). `--memory 6000` (one Process call): peak 4,327 →
  4,082 MB, 25.1 → 21.5 s.

### ner

- **Seam:** `INerNet.Forward(sentences, wordIds, deltaIds, width, charlms, cacheKeys, ct)` → [batch, width, tags]
  emissions. `NerTagger` keeps the vocab lookups (lowercasing, the delta PAD rule), batching (32 sentences in document
  order), Viterbi, `fix_singleton_tags` and the entities. `NerTagger.Load` builds `NerNet` (today's TorchSharp code);
  `NerTagger.LoadManaged` builds `ManagedNerNet` with the managed charlms (or none for `_nocharlm`).
- **Managed net:** each token's input row (pretrain + delta embedding, then the charlm or character-model columns) is
  built at its packed position (sentences longest first, as pack_padded_sequence orders them), so input_transform,
  the biLSTM (`ManagedLstm.ForwardPacked`, with `taggerlstm_h_init`/`c_init`) and the OntoNotes head never compute
  padding. TorchSharp runs input_transform on the padded batch.
- **`ManagedCharacterModel`** (in `Nn.Managed`) has both variants, NER's (bidirectional, final states) and the
  tagger/parser's (unidirectional, attention pooling), so pos and depparse's `_nocharlm` models can use it as is.
  Against `CharacterModel`: 3.4e-7 (NER), 3.4e-7 (pos), on every kernel path.
- **Shared models:** `Pipeline.ManagedProcessors` (tokenize, mwt, ner) says what `Backend.Managed` runs managed. Each
  backend's charlms load only if a `_charlm` processor on that backend reads them, so a managed pipeline with pos holds
  both (the managed ones add 31 MB of input tables). Managed NER reads the shared `Pretrain` through `CpuVectors()`:
  the CPU tensor's own memory, no copy.
- **Tests:** `NerTests` (emissions vs golden, the 13 golden files with and without the cache, `tokenize,ner` alone) and
  `FastPackageTests` (all 13 default_fast files, the nocharlm emissions) run on both backends; `PipelineTests`,
  `InputModeTests`, `NoSsplitTests` and `ConcurrencyTests` (now also default_fast) cover managed NER through the
  pipeline. `ManagedBackendTests.NerNet_ManagedMatchesTorchSharp` (both checkpoints, 9.5e-6, identical tags) and
  `CharacterModel_ManagedMatchesTorchSharp` run on every kernel path. Every discrete output is byte-identical; no
  tolerance changed (emissions stay at 1e-4).

### Backend-neutral `CharlmCache`

The tagger produces the cache; constituency, sentiment and ner read it. Once processors move to the managed backend one
by one, producer and readers can sit on different backends.

- **Design:** one cache, entries in their producer's form. `TryAdd(sentence, Tensor, Tensor)` (TorchSharp producer,
  unchanged) or `TryAdd(sentence, float[], float[], words)` (managed producer, for pos next). Readers ask for their
  form: `TryGet` → tensors, `TryGetArrays` → `[words, dim]` arrays.
  - A tensor entry read as arrays is copied per read (8 KB per word; the caller owns the copy).
  - An array entry read as tensors is converted once, on the cache's device (the pipeline's), and kept with the entry,
    so the TorchSharp readers' contract (tensors owned by the cache, never disposed by them) holds.
  - `MaxWords` counts words the same for both forms; `IsEnabled`, `MaxWords` and the dispose rules are unchanged.
- **Cost:** a pipeline whose producer and readers share a backend never converts, so the default path runs exactly the
  old code. Measured (8-processor benchmark, 8 threads, two rounds, origin/main → this branch): 58.41 / 58.64 →
  58.74 / 58.70 s; peak working set 2,981 / 2,985 → 3,015 / 3,004 MB; `--memory 6000` peak 4,328 / 3,742 →
  4,243 / 3,738 MB. The differences are within run-to-run noise (load peak +8 MB from more code). The golden files
  are byte-identical on the TorchSharp path.

## Phase 2: what next

1. **depparse**: the same blocks plus `DeepBiaffine` and the distance/linearization terms. Chu-Liu/Edmonds is already
   plain C#. Risk: the padded log-softmax makes heads depend on the batch, so batches must stay identical (they do,
   since batching is shared). The arc/label log-probs are tested at 1e-4 on both backends; like `upos_clf`, the
   last scorers may need double sums to stay inside it.
2. **sentiment**: an unpacked padded biLSTM (a `ManagedLstm` mode with lengths = width), Conv2d as GEMMs over
   overlapping rows, max-pool, MLP. Risk: near-ties (173 of 886 labels already flip with batching in Stanza).
3. **lemma**: LSTMCell decoder, dot attention, copy gate, greedy argmax over tiny batches. Risk: near-ties per
   character.
4. **constituency**: the most code; per-step stack LSTMs (50 states in flight) and a transition argmax over thousands
   of steps. The highest near-tie risk.

Cross-cutting:

- **Tolerances** (owner, 2026-10-08): the managed backend uses TorchSharp's score tolerances (1e-4 where TorchSharp
  has 1e-4), not 1e-3. Where it misses, the drift's source gets a cheap fix (e.g. double sums, as in `upos_clf`) or
  goes to the owner with measured options.

- **Shared models:** the factory of the Cuda split has to build `Pretrain` and the charlms per backend. `Pretrain` is
  still a TorchSharp tensor that managed processors read in place; a libtorch-free managed pipeline (0.5) needs it
  as a plain array.
- **NER batches:** NER runs the charlms 32 sentences at a time (Stanza's batches), where both backends are
  memory-bound on the 16 MB recurrent weights. Running the charlms over bigger groups would be faster but changes the
  last float bits; left as is.
- **CI time:** the both-backend theories rerun the full pipeline on the golden files; the full local suite takes
  29–34 minutes on 8 cores (238 tests with the converted models or the `.pt` files, none skipped; Release build).
  `ConcurrencyTests` now runs alone after the parallel collections: beside the heavier both-backend theories its
  cancellation test failed every full run (the timed run was slower than the canceled ones).