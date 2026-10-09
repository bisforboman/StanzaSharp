# Backends (issue #29)

StanzaSharp runs each processor's network on one of two backends:

- **TorchSharp** (libtorch): today's code. It is the default, it runs on the GPU, and it is the reference the
  managed code is tested against.
- **Managed**: C# SIMD kernels with no native dependencies (`src/StanzaSharp.Nn/Managed/`, see
  [managed-backend-spike.md](managed-backend-spike.md)). It is CPU only.

The owner's plan: 0.5 makes the managed backend the default, with TorchSharp still selectable. At 1.0 TorchSharp leaves
the main package for an opt-in `StanzaSharp.Cuda` package. Both implementations stay.

Phase 1 added the seam and ported **tokenize** and **mwt**. Phase 2 ported the other six one at a time: **ner**, **pos**,
**depparse**, **sentiment**, **lemma** and **constituency**, plus a backend-neutral `CharlmCache` (see
[Phase 2](#phase-2-progress)). Every processor of both packages now runs on either backend, and a managed pipeline loads no
TorchSharp charlms. Everything is `internal`; nothing public changed.

## The seam

The seam sits at each processor's **network**: one small interface per processor, with arrays in and arrays out.

| processor | interface | TorchSharp | managed |
|---|---|---|---|
| tokenize | `ITokenizerNet.Forward(ids, feats, rows, width, lengths, ct)` → [rows, width, 5] log-probs | `TokenizerNet` | `ManagedTokenizerNet` |
| mwt | `IMwtNet.Forward(ids, rows, width, lengths)` → [rows, width, 2] logits | `MwtNet` | `ManagedMwtNet` |
| ner | `INerNet.Forward(sentences, wordIds, deltaIds, width, charlms, cacheKeys, ct)` → [batch, width, tags] emissions | `NerNet` | `ManagedNerNet` |
| pos | `IPosNet.Forward(sentences, wordIds, pretrainIds, charlms, cacheKeys, ct)` → `PosOutput` (UPOS scores, UPOS/XPOS/feats ids) | `PosNet` | `ManagedPosNet` |
| depparse | `IDepparseNet.Forward(DepparseBatch, labelScores, ct)` → `DepparseScores` (arc log-probs, label argmax, optional label scores) | `DepparseNet` | `ManagedDepparseNet` |
| sentiment | `ISentimentNet.Forward(batch, ids, extraIds, width, charlms, cacheKeys, ct)` → [batch, classes] logits | `SentimentNet` | `ManagedSentimentNet` |
| lemma | `ILemmaNet.Encode(ids, batch, width, posIds, lengths, ct)` → `ILemmaDecoder` (edit logits; `Step(previous, ct)` → [batch, columns] log-probs) | `LemmaNet` | `ManagedLemmaNet` |
| constituency | `IConstituencyNet`: `EncodeWords`, `Word`, `Score` → [states, transitions] scores, `Open`, `Compose`, `PushTransitions`, `PushConstituents` over opaque handles | `ConstituencyNet` | `ManagedConstituencyNet` |

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
  `torch.set_num_threads`. With `Backend.Managed`, libtorch's threads are left alone (see below); on TorchSharp they are
  set as before.
- Benchmark: `StanzaSharp.Benchmark --backend managed [--processors tokenize,mwt]`.

### No libtorch on the managed backend (0.5 requirement)

A pipeline on `Backend.Managed` loads and runs with no native libtorch anywhere, and makes no TorchSharp object. The
managed `TorchSharp.dll` assembly is still referenced and loaded (types in signatures and fields), but nothing calls
into it. TorchSharp's `torch` class loads libtorch in its static constructor, so *any* `torch.*` member (even
`torch.CPU`) would load it.

Touchpoints found and how they were removed:

| Touchpoint | Fix |
|---|---|
| `Pipeline.Load`: `torch.get_num_threads`/`set_num_threads` (the first one ran `torch`'s static constructor, which loaded `LibTorchSharp`, `torch_cpu` and `c10`) | Managed `Load` sets only `ManagedThreads.Count`; the torch threads, TF32 and `Weights.On(Device)` moved into `Pipeline.LoadTorchSharp`, the TorchSharp path, unchanged |
| `Pretrain`: the embedding matrix was a tensor (`ckpt.ToTensor`), read by managed nets through `CpuVectors()` | `Pretrain.LoadManaged`: a plain `float[]` read straight from the safetensors / `.pt` storage (`Checkpoint.Tensor<float>`); `Pipeline` uses it when every processor reading the pretrain is managed. `Count`/`Dim` no longer read the tensor's shape. `Embeddings` throws on a managed pretrain |
| `TorchSharp.torch.Device`/`DisableTf32` | only read on the TorchSharp path |

Checked and already clean: every processor's `LoadManaged`/managed net reads its weights through `Checkpoint.Tensor<float>`
(Core: `SafeTensorFile`, `TorchCheckpoint`) into arrays; the shared processor code (batching, decoding) has no torch
call; `CharlmCache` makes tensors only in `TryGet` (TorchSharp readers) and disposes only tensors it holds; `Weights.On`
(tokenizer, mwt, lemma `Load`) only sets an `AsyncLocal` when no device is given; `Scalars`' static fields are never
touched on the managed path; `ModelDownloader`/`VerifyChecksums` (MD5), `Conllu`, the `Logger`, cancellation and
`NativeHeap` (glibc `malloc_trim`) don't use TorchSharp.

**Proof:**

- `tests/StanzaSharp.ManagedCheck`: a console app referencing the library projects but no `TorchSharp-cpu` /
  `libtorch-cpu-*` (`StanzaSharpNoLibTorch` in its csproj keeps Directory.Build.props from adding the Windows Arm64
  libtorch), so libtorch is not on disk next to it (only TorchSharp's own `LibTorchSharp`, which the managed package
  carries). For `default` and `default_fast` it loads a managed pipeline (with `Threads`, a `Logger`, and
  `VerifyChecksums` for `.pt` models), checks that a canceled `Process` throws, compares `corpus.txt` with
  `pipeline.conllu` / `fast/corpus.conllu` byte for byte, runs bulk, pretokenized and no-ssplit input, and fails if a
  native torch module is loaded (`Process.Modules`: `LibTorchSharp`, `torch_cpu`, `c10`; on Linux .NET reads
  `/proc/self/maps`). Public API only, plus reflection for the internal `PipelineOptions.Backend`.
- `ManagedCheckTests` runs it in its own process with `NUGET_PACKAGES` set to an empty folder: TorchSharp's fallback
  otherwise copies libtorch from the NuGet cache into a `cpu/` folder next to the app and loads it from there (that is
  how the first run, before the fixes, loaded `torch_cpu`/`c10` although none was in the output). With the fallback
  blocked, any torch call throws (`TypeInitializationException` from `torch..cctor`, which is how the touchpoints above
  were found). It also asserts no libtorch file is in the app's output.
- `tools/verify-package.ps1 -Managed` (CI: golden job, Linux, `.pt` models): packs StanzaSharp, builds a fresh app that
  references **only** `StanzaSharp` (no `TorchSharp-cpu`, no platform package), compiles the same `Program.cs` against
  it, checks no libtorch reached the output, and runs it with an empty `NUGET_PACKAGES`.

**Load memory** (`--memory`, fresh process, default package, converted models, Ryzen 7 5800X; the benchmark used to
load libtorch at startup through `torch.CPU`, which inflated earlier managed numbers):

| | load | load peak | after load |
|---|---:|---:|---:|
| TorchSharp | 1.25 s | 818–820 MB | 818 MB |
| managed, before (libtorch loaded, tensor pretrain) | 1.21 s | 1,127 MB | 1,127 MB |
| managed, no libtorch, array pretrain, no GC between models | 1.08 s | 1,035 MB | 1,024 MB |
| managed, now | 1.10–1.15 s | **837–870 MB** | 826 MB |

`default_fast`: TorchSharp 623 MB, managed 749 → **658 MB**. The managed load allocated 1,639 MB on the GC heap for
679 MB kept: each weight is read into an array, then packed into another. `PackedLstm.PackInput` copied every input
row, then concatenated them through a growing buffer; it now copies once into the final array (1,390 MB allocated,
627 MB kept, load ~0.1 s faster). `Pipeline` then runs `GC.Collect()` after each managed model, so the next one reuses
that memory (−200 MB peak, about +0.05 s; the `--memory` run's gen2 count includes those 11 collections). What is left
above TorchSharp is the packed form (padding to 16-column panels, charlm input tables, contracted biaffines: 627 MB of
arrays for 551 MB of checkpoint) and the runtime. Processing peaks are unchanged (`--memory 6000`: managed 2,299–2,304 MB
in 12.5 s, TorchSharp 3,771–4,302 MB in 23–27 s).

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
| **depparse** (`_charlm`) | yes | byte-identical heads and deprels; arc / label log-probs 2.3e-5 / 3.4e-5 from Python (TorchSharp 1.1e-5 / 1.5e-5; Scalar path 2.3e-5 / 3.8e-5) | depparse stage **13.27 → 7.94 s (0.60)** | **42.52 → 36.99 s (0.87)** |
| **depparse** (`_nocharlm`, default_fast) | yes | byte-identical; arc / label log-probs 1.1e-5 / 2.7e-5 from Python (every path ≤ 1.5e-5 / 3.1e-5) | **6.47 → 3.68 s (0.57)** | **19.85 → 17.33 s (0.87)** |
| **sentiment** (`sstplus_charlm`, both packages) | yes | byte-identical labels; logits within 1e-4 of Python's float32 or float64 logits (8.2e-5; 1.38e-4 from float32 on one ill-conditioned sentence, see [sentiment](#sentiment)) (TorchSharp 2.1e-5) | sentiment stage **9.82 → 5.98 s (0.61)** | **33.65 → 26.29 s (0.78)** |
| **lemma** (`combined_nocharlm`, both packages) | yes | byte-identical lemmas, decoding and edits; smallest top-2 margin 5.6e-3; log-probs 4.8e-4 from TorchSharp, reported, not asserted (owner's decision, 2026-10-08; see [lemma](#lemma)) | lemma stage **0.89 → 0.32 s (0.36)** | **1.18 → 0.72 s (0.61)** |
| **constituency** (`ptb3-revised_charlm`, default package) | yes | byte-identical trees; transition scores 3.4e-5 from Python (TorchSharp 3.1e-5; every path ≤ 4.6e-5; tolerance 1e-3 as before); smallest decision margin 3.6e-4 over 22,569 steps (see [constituency](#constituency)) | constituency stage **10.02 → 4.77 s (0.48)** | **21.73 → 17.82 s (0.82)** |

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

### depparse

- **Seam:** `IDepparseNet.Forward(DepparseBatch, labelScores, ct)` → `DepparseScores`. `DependencyParser` keeps
  simplify_punct, the lowercased vocab lookups, the ROOT word (id 3), the batches (longest first, 5000 words with ROOT,
  over 150 alone), Chu-Liu/Edmonds in float64 and the deprel strings; the net returns the arc log-probs [batch, width,
  width] (log-softmax over the padded width, as in Stanza), each pair's label argmax and, for tests, the label scores.
  `DependencyParser.Load` builds `DepparseNet` (today's code, moved unchanged); `LoadManaged` builds `ManagedDepparseNet`.
- **Managed net:** the input rows (trans_pretrained GEMM, word, lemma, UPOS+XPOS twice, then the charlms over "\n" +
  the words, or `ManagedCharacterModel` with ROOT = char id 3 + `trans_char`) are built at their packed positions and go
  through `ManagedHighwayLstm`; nothing is padded. The W1/W2 layers of all four deep biaffine scorers are one GEMM.
  - **Biaffine scorers in two steps:** T = in1·W_bilin per dependent (a GEMM, with in1's appended 1 folded into the
    bias), then T·in2 per word pair. Padding columns all see the same in2, ReLU(W2's bias), because the LSTM output there
    is 0; so the arc log-softmax runs over exactly TorchSharp's padded width without scoring padding rows, which nothing reads.
  - **Arcs** (unlabeled, linearization, distance; one output each): the pair sums, logsigmoid, softplus (torch's beta 1,
    threshold 20), the distance penalty and the log-softmax are in double. Cheap: 3 × 400 multiply-adds per pair.
  - **Labels** (49 relations): only real word pairs. Each dependent's T row is repacked as a [400, 49 → 64] weight panel
    and every head's in2 goes through the GEMM micro-kernel (`Gemm.Kernel`, every path), then the argmax. T is computed
    a chunk of whole sentences at a time, at most 32 MB (TorchSharp: 128 MB chunks plus einsum's permuted copy and the
    full [batch, width, width, 49] output).
  - In float, the label scores stay within 3.8e-5 of Stanza's on every path (two 400-term sums), so no double was needed.
- **Tests:** `DepparseTests` (golden intermediates, the 13 golden files from Stanza's tags and lemmas, the
  tokenize..depparse pipeline) on both backends; `FastPackageTests.ManagedIntermediates_MatchGolden` (nocharlm scores);
  `ManagedBackendTests.DepparseNet_ManagedMatchesTorchSharp` (both checkpoints, a padded batch of 6 sentences, every
  path: arcs 1.9e-5 to 3.4e-5, label scores 2.7e-5 to 3.4e-5, identical parses) and `DepparseScores_MatchGoldenOnEveryPath`
  (both checkpoints at 1e-4). The ner, fast, pretokenized, bulk, all-eight, cancellation and concurrency tests cover it
  through the pipeline. No tolerance changed.
- **Speed** (`--processors tokenize,mwt,pos,lemma,depparse`, 8 copies, 26,264 words, medians of 3, Ryzen 7 5800X, load
  ~15%): depparse stage 13.27 → 7.94 s at 8 threads, 42.52 → 36.99 s at 1 thread. Of the managed 7.9 s, about 4.5 s are
  the two charlm passes, 2.1 s the input and highway LSTM, 0.3 s the W1/W2 GEMM, 0.25 s the arcs and 0.9 s the label
  scorer. default_fast: 6.47 → 3.68 s and 19.85 → 17.33 s. Full 8-processor run at 8 threads: 59.76 → 34.06 s (depparse
  13.69 → 8.26).
- **Memory:** depparse stage peak (warm-up, 5-processor run) 1,825 → 1,291 MB. Full 8-processor run: peak working set
  2,533 MB (TorchSharp) → 2,978 MB (managed), against 3,132 MB with only tokenize, mwt, pos and ner managed (the extra
  over TorchSharp is the pos/ner one described there). `--memory 6000` (one Process call, two rounds): TorchSharp
  4,367 / 3,748 MB, managed 3,833 / 3,837 MB (pos and ner managed only: 4,131 / 4,136 MB), 24.0 → 14.4 s.

### sentiment

- **Seam:** `ISentimentNet.Forward(batch, ids, extraIds, width, charlms, cacheKeys, ct)` → [batch, classes] logits.
  `SentimentClassifier` keeps map_word, the delta vocab, `label_sentences` (sorted longest first, stable), the
  5000-token batches, the padded width (the longest sentence, at least the widest filter), the cache keys (sentences
  without multi-word tokens) and the argmax, so both backends see the same batches. `Load` builds `SentimentNet` (today's
  code, moved unchanged); `LoadManaged` builds `ManagedSentimentNet` with the managed charlms.
- **Managed net:**
  - Input rows time-major ([width, batch, 2148]): pretrain (or the learned `unk`; padding reads the PAD rows) + delta,
    then the charlm columns (from the cache, else computed; zeros on padding).
  - The **unpacked** 2-layer biLSTM is `ManagedLstm.ForwardPacked` with every batch size = n: every row runs the full
    padded width, as `nn.LSTM` does without packing. No new LSTM code.
  - The output is copied batch-major once; each full-width convolution (3, 4, 5 tokens × 600) is then **one GEMM whose A
    rows overlap**: row r starts at token r with lda = 600 and K = Height·600, so window t of sentence i is row
    i·width + t and nothing is copied (windows crossing into the next sentence are computed and ignored, (Height − 1)/width
    of the work). ReLU + max over each sentence's windows = max(0, max).
  - The (5, 5) filter with stride (1, 5) (8 channels × 120 columns, 25 products each) runs as scalar loops, parallel per
    sentence; it is ~2% of the convolutions' multiply-adds.
  - FC 3960 → 400 → 100 → 3 through `Gemm` with ReLU between.
- **Exactness:** labels byte-identical everywhere (14 golden files, `all.json`'s two batches, pipeline, validation,
  fast, bulk, pretokenized, concurrency). Logits vs Python's float32 logits (`SentimentTests`, tolerance 1e-4; how the
  test accepts the one miss: below):

  | | TorchSharp | managed |
  |---|---:|---:|
  | 14 files, per-file batches | 2.1e-5 | **1.38e-4** (one sentence); every other ≤ 8.2e-5 |
  | `all.json` (one document, two batches) | 3.8e-6 | 8.2e-5 |
  | with the tagger's cache (tolerance 1e-3) | 9.3e-5 | 1.4e-4 |
  | managed vs TorchSharp net, 6 sentences, every path | – | 7.6e-6 to 9.5e-6 |

  - **The one miss is Stanza's rounding, not the port's.** validation.txt sentence 41 ("Neil Armstrong's words — "That's
    one small step for man" — were heard by millions.", 22 tokens in a 71-wide batch). Running Stanza itself in float64
    (`model.double()`, same batches) gives the exact logits: Stanza's float32 result is **1.49e-4** from them, the managed
    one **1.2e-5**. Over all 14 files: |Stanza f32 − f64| ≤ 1.49e-4 (TorchSharp the same), |managed − f64| ≤ 9.1e-5.
    The network is ill-conditioned there: the TorchSharp classifier itself moves 9.3e-5 when only the charlm's last
    bits change (the cache test).
  - Cheap fixes tried, none brings it under 1e-4 of Stanza (they move the result toward the exact value, not toward
    Stanza's float32 error): FC layers in double 1.38e-4; convolutions in double 1.41e-4; the LSTM input projection in
    double 2.0e-4; Scalar / Vector128 paths 1.66e-4 / 1.38e-4. Matching Stanza closer would mean reproducing MKL's
    summation order.
  - **Rule (owner, 2026-10-08):** `make_golden.py --sentiment-only` also stores each sentence's float64 logits
    (`logits64`, Stanza's model in float64 on the same batches) next to the float32 ones, which stayed byte-identical.
    In `SentimentTests` a logit passes within the unchanged tolerance (1e-4; 1e-3 with the cache) of the float32 **or**
    the float64 value, on both backends; labels stay exact. Accepted drift, nearer reference per logit: TorchSharp
    9.6e-6, managed 8.2e-5 (with the cache 5.6e-5 / 8.2e-5).
- **Cache:** managed sentiment reads array entries as they are and TorchSharp tensor entries as copies
  (`TryGetArrays`); `SentimentNet_ManagedMatchesTorchSharp` checks a cache holding both forms (2.9e-6 against no cache).
- **Speed** (`--processors tokenize,mwt,sentiment`, no cache, 8 copies, 26,264 words, medians of 3, Ryzen 7 5800X):
  sentiment stage 9.82 → 5.98 s at 8 threads, 33.65 → 26.29 s at 1 thread. Full 8-processor run at 8 threads (sentiment
  reads the tagger's cache): 53.27 → 30.18 s (sentiment 5.99 → 3.42 s; pos 10.65 → 5.37, depparse 13.10 → 7.94, ner
  12.29 → 3.02; tokenize 1.37 → 0.32; lemma and constituency unchanged).
- **Memory:** sentiment stage peak (warm-up, `tokenize,mwt,sentiment`) 1,988 → 1,621 MB. Full 8-processor run: peak
  working set 2,538 → 2,759 MB (both charlm forms, since constituency still reads TorchSharp's). `--memory 6000` (one
  Process call): 4,299 → 2,839 MB, 22.7 → 11.8 s.

### constituency

- **Seam:** `IConstituencyNet` works on opaque handles the net makes and only it reads: a sentence's word vectors, a
  constituent's vector, a stack node's LSTM state. `ConstituencyParser` keeps the transition system (IN_ORDER, legality,
  `unary_limit` 4, the 20 × (length + 2) transition cap), the vocab lookups (pretrain with the lowercase fallback, delta and
  tag ids), the schedule (sentences longest first, word queues built 50 at a time, 50 states in flight, each finished state
  replaced), the stacks themselves, the trees and their `-LRB-`/`-RRB-` printing. Per step it calls `Score`, then `Open`
  (dummy embedding rows), `Compose` (MAX + `reduce_linear` + ReLU, per close), `PushTransitions` and `PushConstituents`, each
  over the batch. `ParserState` owns the handles that are `IDisposable` (TorchSharp's tensors, freed when its sentence ends);
  the managed ones are plain float arrays. `ConstituencyNet` is the TorchSharp code, moved unchanged (no_grad and a dispose
  scope per call instead of per step); `LoadManaged` builds `ManagedConstituencyNet`. `Parse` takes an optional `onStep`
  hook (row and legality per state and step) for the near-tie report.
- **Managed net:**
  - Word encoder: rows `[word_start | pretrain 100, delta 100, tag 20, charlms 2048 | word_end]` (charlm columns from the
    cache as arrays, the rest computed in one batch), `ManagedLstm.ForwardPadded` (2 layers, 2 × 512), then
    `word_to_constituent` as one GEMM and ReLU; each sentence keeps its rows as [hidden] arrays.
  - Stack LSTMs: each layer is one GEMM over `[x | h]` (K = 40 for the transition stack, 1024 for the constituent stack;
    gate rows in `PackedLstm.GateOrder`, b_ih + b_hh folded) and `Act.LstmCell`, with every row read from and written to its
    own state; a node holds its [layers, hidden] h and c (8 KB for the constituent stack). The start states are pushed from
    zeros at load, like `nn.LSTM` without hx.
  - Scores: `[word | transition top | constituent top]` (1044), ReLU before each of the two output layers, through `Gemm`.
  - **Per-state arithmetic:** every per-step GEMM uses `Gemm.Run(..., rowInvariant: true)`. `Gemm` serves a lone row of a
    6-row block with a 1-row kernel that splits the k sum into four accumulators; the flag sends it through the 3-row
    kernel, whose per-row arithmetic is the 6-row one's. So a state's scores, compositions and pushes are bitwise the same
    whatever else is in the batch (tested on 7 states vs each alone, on every path; without the flag the test fails on the
    SIMD paths). The word encoder stays batched per 50 sentences, as on TorchSharp and in Stanza.
- **Exactness** (all float, no double sums needed):

  | | TorchSharp | managed |
  |---|---:|---:|
  | transition scores vs Python, 3 golden sentences (100 steps; tolerance 1e-3, unchanged) | 3.1e-5 | 3.4e-5 (Vector128 4.2e-5, Scalar 4.6e-5) |
  | step scores managed vs TorchSharp, pipeline.conllu (10 sentences, 410 steps), every path | – | 5.0e-5 to 6.1e-5 |
  | step scores managed vs TorchSharp, all 845 golden sentences (22,569 steps) | – | 7.3e-5 |

  Trees are byte-identical: `pipeline.conllu` and every `validation*.conllu` through the full pipeline, the all-eight,
  cache-settings, bulk, pretokenized, no_ssplit and concurrency theories, `ConstituencyTests` (both backends) and the
  cancellation theory (the parser stops within one step).
- **Near-ties** (`ConstituencyTests.NearTies_AreReported`: both backends parse all 845 golden sentences from Stanza's words
  and XPOS, charlms computed; every step's decision is the same on both). The decision margin is the best legal
  transition's score minus the next legal one's; 20,833 of the 22,569 steps have at least two legal transitions. The raw
  top-2 margin of the whole row gives the same counts (an illegal top never came close).

  | decision margin | TorchSharp | managed |
  |---|---:|---:|
  | < 1e-2 | 16 | 16 |
  | < 1e-3 | 2 | 2 |
  | < 1e-4 | 0 | 0 |
  | smallest | 3.63e-4 | 3.61e-4 |

  The closest calls (margin TorchSharp / managed):

  | sentence | step | chosen over | margins |
  |---|---:|---|---|
  | validation_social 43 | 6 | Shift over Open(ADJP) | 3.63e-4 / 3.61e-4 |
  | validation_instructions 53 | 12 | Shift over Open(PP) | 8.63e-4 / 8.65e-4 |
  | validation_academic 15 | 9 | Shift over Open(NP) | 2.92e-3 / 2.92e-3 |
  | validation_tech 64 | 12 | Shift over Open(PP) | 3.21e-3 / 3.21e-3 |
  | validation_dialogue 14 | 2 | Shift over Close | 4.41e-3 / 4.41e-3 |

  The smallest margin is 5× the largest score difference between the backends (7.3e-5) and 10× the drift from Python
  (3.4e-5); the two backends' margins differ by at most 2e-6 there.
- **Speed** (`--processors tokenize,mwt,pos,constituency`, 8 copies, 26,264 words, medians of 3, Ryzen 7 5800X, load ~14%;
  the parser reads the tagger's cache): constituency stage **10.02 → 4.77 s** at 8 threads, **21.73 → 17.82 s** at 1 thread.
  Per step the managed net reads ~19 MB of weights (two constituent LSTM layers 16 MB, output layer 2 MB, reduce 1 MB) for up
  to 50 rows.
- **Fully managed pipeline** (all eight processors, 8 threads, same text): **56.44 → 25.93 s** (tokenize 1.43 → 0.34, pos
  11.19 → 5.50, lemma 0.89 → 0.36, constituency 9.97 → 4.70, depparse 13.68 → 8.28, sentiment 6.12 → 3.61, ner 13.15 →
  3.14).
- **Memory:** no processor reads the TorchSharp charlms any more, so a managed pipeline doesn't load them
  (`PipelineTests.ManagedBackend_LoadsNoTorchSharpCharlms`, both packages), and cache entries are never converted to
  tensors. Full 8-processor benchmark: peak working set **2,530 MB (TorchSharp) → 2,306 MB (managed)**, against 2,692 MB with
  constituency still on TorchSharp (both charlm forms). `--memory 6000` (load, one Process call of 8,071 words, two fresh
  processes each): TorchSharp 4,294 / 3,750 MB in 23.8 / 24.0 s, managed **2,533 / 2,531 MB** in **11.5 / 11.6 s** (with
  constituency still on TorchSharp, as measured for sentiment: 2,839 MB). The managed load peak is higher (1,166 vs 819 MB: the packed weights live on the GC
  heap, 800 MB, next to the TorchSharp pretrain) and the run does 9 gen2 GCs (TorchSharp 4).

### lemma

- **Seam:** `ILemmaNet.Encode(ids, batch, width, posIds, lengths, ct)` → an `ILemmaDecoder` (the batch's state, disposed
  after the batch) with `EditLogits` and `Step(previous, ct)` → [batch, Columns] log-probs, Columns = the vocabulary
  widened to the batch's largest DeltaVocab id + 1, as torch sizes the copy scatter. `Lemmatizer` keeps the dictionary
  skip, DeltaVocab, the `batch_size` 50 batches sorted like `sort_all`, the greedy loop (argmax per row: the first maximum,
  as torch's `max`; `max_dec_len`; done rows), the edits and the `<UNK>`/empty fallback; it now also checks cancellation
  per decoder step. `LemmaNet` is today's TorchSharp code, moved unchanged (the TorchSharp decoder keeps the batch's tensors
  in one dispose scope); `Lemmatizer.Load(..., backend)` picks the net.
- **Managed net:**
  - Encoder: the packed rows (UPOS embedding, then the characters; ids past the vocabulary embed as `<UNK>`) through
    `ManagedLstm.ForwardPacked`, which now also returns the last layer's final cell states (`PackedLstm.Recur`'s
    `finalC`), for the decoder's `(h0, c0) = (cat(hn[-1], hn[-2]), …)`. Edit classifier in double.
  - Each decoder step, over the batch: one GEMM for the LSTMCell (`[x | h]` against `[W_ih | W_hh]`, K = 250, gate rows in
    `PackedLstm.GateOrder`, then `Act.LstmCell`), one for `linear_in`, one for `linear_out` (+ tanh), one for `dec2vocab`
    with the copy gate as an extra output column. Per row, in double: the attention scores over the row's real positions,
    softmax and weighted context; log_softmax of the vocabulary; the copy mix. Masked positions and columns nothing is
    copied to are skipped: torch's −1e12 entries contribute exactly 0 there, so the result is the same.
- **Exactness:** byte-identical everywhere: the 13 `lemma/` files, `words.json` (pipeline lemma, raw seq2seq output, edit
  class; unknown and non-BMP characters, a word past `max_dec_len`), the depparse, ner, pipeline/validation (all eight),
  fast, bulk, pretokenized and concurrency theories. `LemmaNet_ManagedMatchesTorchSharp` runs both nets on the 868 golden
  seq2seq words (each file's misses as one call) plus `words.json`: 892 words, 504 decoder steps, identical steps,
  decoding and edits on every path.
  - **Near-ties:** the smallest top-2 margin of a row still decoding is **5.6e-3** (either backend), about 12× the largest
    log-prob difference between the backends.
  - **Log-probs (owner's decision, 2026-10-08: exact lemmas, decoding and edits plus the reported margin; no score tolerance):** managed vs TorchSharp max |diff| 4.8e-4 (Vector256/Vector128; Scalar 4.3e-4),
    2.7e-4 among entries within 10 of the row's maximum. The test reports them without asserting a tolerance: none
    exists (TorchSharp's lemmatizer is tested on discrete output only), and 1e-4 is out of reach for TorchSharp too.
    Measured on `words.json` (one batch, 50 steps) against Stanza run in float64 (`model.double()`, same batch):

    | per-step log-probs vs Stanza float64 | all entries | within 10 of the top |
    |---|---:|---:|
    | Stanza float32 | 2.9e-4 | 1.6e-4 |
    | TorchSharp | 2.7e-4 | 1.4e-4 |
    | managed | 3.8e-4 | 1.3e-4 |
    | managed, LSTMCell gates in double | 1.8e-4 | 9.3e-5 |

    TorchSharp is 1.7e-4 from Stanza's float32; even the sentiment rule (within 1e-4 of float32 **or** float64) fails for
    TorchSharp (1.2e-4). The source is the decoder recurrence amplifying float noise in low-probability entries; in the
    managed net the LSTMCell gate sums dominate (double attention, `linear_out` or `dec2vocab` change nothing). Gates in
    double (scalar) cost 0.32 → 1.23 s for the lemma stage, so they are not used.
  - **Measured on UD English EWT** (2026-10-08; `tools/lemma_divergence.py` + `StanzaSharp.Benchmark lemma-divergence`;
    the corpus is CC BY-SA and not in the repository). EWT's train/dev/test `# text` lines, 1,174 documents (one Process
    call each), 254,589 words, `tokenize,mwt,pos,lemma`, 8 threads, four runs: Stanza 1.15.0 (each batch also in float64),
    TorchSharp, managed, managed with the LSTMCell gate sums in double (`ManagedLemmaNet.DoubleGates`, study only).
    - Tokens and tags are identical in all four except 2 documents (127 words, excluded): Stanza replaces tokens longer
      than the tokenizer's `max_seqlen` (200 in its config) with `<UNK>`, our tokenizer only those over 1000, so two
      long URLs stayed whole here and their tags differed. A tokenizer difference, not a lemma one; fixed since in #41.
    - The dictionary covers EWT well (it is in the lemmatizer's training data): only 3,659 words (32,198 decoder steps) go
      through the seq2seq model. So a second pass sends **every** word through it (`--no-dict`): 254,462 words, 1,248,359
      decoder steps.
    - **No lemma differs** between any two of the five (the four runs and Stanza float64), in either pass, nor in
      `default_fast` (whose tags differ, so its lemmas are another test). Decoded strings and edit classes are identical too.
    - Top-2 margins over all 1.25M steps: 21 below 1e-2, 13 below 5e-3, 1 below 1e-3 (`gant`, 9.06e-4), none below 5e-4;
      the same counts on every backend. Edit classes: 2 below 1e-3 (min 7.6e-4, `Scientific`). A decision's margin moves
      by at most (vs Stanza float64): Stanza float32 1.5e-4, TorchSharp 4.9e-4, managed 1.2e-4, double gates 9.2e-5;
      the closest call is about 7× the managed net's largest error.
    - Cost of double gates, parallel over gate rows (the scalar loop above costs 16×): lemma stage 1.13 → 2.78 s on all
      of EWT (77 s for `tokenize,mwt,pos,lemma`), 36 → 114 s with every word through the model. Not worth it: it moves
      the managed margins from 1.2e-4 to 9.2e-5 of float64 and changes no output.
- **Speed** (`--processors tokenize,mwt,pos,lemma`, 8 copies, 26,264 words, medians of 3, idle Ryzen 7 5800X): lemma stage
  **0.89 → 0.32 s** at 8 threads, **1.18 → 0.72 s** at 1 thread. TorchSharp runs ~40 small ops per decoder step; the managed
  step is four small GEMMs and per-row loops. Full 8-processor run at 8 threads: 53.89 → **29.71 s** (lemma 0.84 → 0.34;
  pos 10.64 → 5.37, depparse 13.41 → 7.96, sentiment 6.09 → 3.41, ner 12.32 → 2.99, tokenize 1.44 → 0.37; constituency
  9.13 → 9.26, still TorchSharp).
- **Memory:** per-batch buffers are pooled (about 1 MB for a 50-word batch); the managed lemmatizer holds the
  weights once as packed matrices (~1.5 MB). Lemma stage peak (warm-up, `tokenize,mwt,pos,lemma`) 1,339 → 1,128 MB;
  peak working set 1,675 → 1,445 MB. Full 8-processor run: 2,535 → 2,692 MB (both charlm forms, since constituency still
  reads TorchSharp's).

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

1. ~~lemma~~: done ([lemma](#lemma)).
2. ~~constituency~~: done ([constituency](#constituency)). Phase 2 is complete.

For 0.5:

- ~~`Pretrain` as a plain array, and a managed `Load` that touches no libtorch~~: done ([No libtorch on the managed
  backend](#no-libtorch-on-the-managed-backend-05-requirement)).
- The public `PipelineBackend` option (decided: A), and managed as the default; `Device`/`DisableTf32` become `[Obsolete]`.
- Later (1.0): the `StanzaSharp.Cuda` split.

Cross-cutting:

- **Tolerances** (owner, 2026-10-08): the managed backend uses TorchSharp's score tolerances (1e-4 where TorchSharp
  has 1e-4), not 1e-3. Where it misses, the drift's source gets a cheap fix (e.g. double sums, as in `upos_clf`) or
  goes to the owner with measured options.

- **Shared models:** the factory of the Cuda split has to build `Pretrain` and the charlms per backend. `Pretrain`
  already has both forms (`Load`: tensor, `LoadManaged`: array); the tensor half moves with the split.
- **NER batches:** NER runs the charlms 32 sentences at a time (Stanza's batches), where both backends are
  memory-bound on the 16 MB recurrent weights. Running the charlms over bigger groups would be faster but changes the
  last float bits; left as is.
- **CI time:** the both-backend theories rerun the full pipeline on the golden files; the full local suite takes
  29–34 minutes on 8 cores (246 tests, none skipped, Release build; with managed depparse 32.2 min on the converted models, 28.9 min on the `.pt` files; with managed sentiment, 253 tests, 27.2 / 31.2 min; with managed lemma, 258 tests, 31.4 / 26.5 min; with managed constituency, 265 tests, 40 min on the converted models while another suite loaded the machine, 25.2 min on the `.pt` files).
  `ConcurrencyTests` now runs alone after the parallel collections: beside the heavier both-backend theories its
  cancellation test failed every full run (the timed run was slower than the canceled ones).
