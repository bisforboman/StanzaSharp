# Managed backend (#29)

The managed (C# SIMD) backend: the owner's decisions, the backend seam per processor, exactness and the kernel rules' background. Moved from CLAUDE.md on 2026-10-31; the binding rules are summarized in CLAUDE.md, and docs/backends.md has the design and measurements.

- **Goal:** the main `StanzaSharp` package runs on a pure managed backend (C# SIMD kernels, no native dependencies).
- **Owner's decisions** (2026-10-07; GPU revised 2026-10-08):
  - GPU stays, as a separate opt-in package `StanzaSharp.Cuda` carrying the TorchSharp backend. So the backend seam
    (Phase 1) is permanent with two implementations, managed and TorchSharp; the TorchSharp one is also the CPU test
    reference.
  - Tolerances (revised 2026-10-08): the managed backend uses the **same score tolerances as TorchSharp** (1e-4 where
    tests use 1e-4; 1e-3 only where TorchSharp already has it, e.g. sentiment with the cache), and discrete outputs are
    byte-identical. A managed comparison that misses 1e-4: find the drift's source, try cheap fixes (higher-precision
    accumulation, reduction order) and measure their cost; if none is cheap, report the drift, source and options with
    costs to the owner. Never a managed-only tolerance, never a skipped test.
  - Accepted: a slower Arm64 path at first; up to ~15% slower at 1 thread.
  - 0.5: managed backend the default, TorchSharp still selectable. 1.0: TorchSharp leaves the main package (not the
    project).
  - Public backend option (2026-10-08): **A**. `PipelineOptions.Backend` takes a sealed public `PipelineBackend` class whose
    constructor and members are internal: `PipelineBackend.Managed`, `PipelineBackend.TorchSharp`; later
    `CudaBackend.Create(device, disableTf32)` in `StanzaSharp.Cuda`, created through internals (the main package never
    references it; device and TF32 travel inside the backend). No enum.
    - 0.5: Managed is the default; `Device`/`DisableTf32` become `[Obsolete]` and mean "TorchSharp on that device". A
      managed pipeline must load with no native libtorch package: `Load` must not call `torch.set_num_threads` (or any
      other torch function) unless TorchSharp is used.
    - 1.0 (done): `Device`, `DisableTf32` and `PipelineBackend.TorchSharp` are gone. `PipelineBackend` (internal
      constructor: a name and a `Func<BackendModels>`; `ToString` = the name) has only `Managed` in the main package;
      `CudaBackend.Create(deviceIndex, disableTf32)` ("Cuda:N") and `CudaBackend.Cpu` ("TorchSharp") are in
      StanzaSharp.Cuda (owner, 2026-10-09: the CPU option stays public). Tests: `Repo.PipelineBackendFor`.
- **Status:** Phase 0 (kernels), Phase 1 (seam; tokenize and mwt ported) and Phase 2 (ner, pos, depparse, sentiment,
  lemma, constituency; `CharlmCache` backend-neutral) are done: every processor of both packages runs managed, and a
  managed pipeline loads no TorchSharp charlms. `docs/backends.md` has the design, the Cuda split plan, exactness, speed and
  the Phase 2 progress table. A managed pipeline needs no native libtorch (below). 0.5: the public `PipelineBackend`
  option, managed the default, platform packages discontinued, Docker sample managed, `alpine` and `linux-arm64` CI jobs.
  The `StanzaSharp.Cuda` split (owner's decisions, 2026-10-09: two PRs) is done. Step 1 (#55): all torch code is in
  `StanzaSharp.Cuda`. Processors take a net factory in an internal constructor and have `LoadManaged`; their
  TorchSharp `Load` is a C# 14 static extension in `StanzaSharp.Cuda` (`PosTaggerLoad` etc., in the processor's
  namespace), so `PosTagger.Load(...)` call sites are unchanged. `Pretrain.Native` (`IPretrainVectors`) and
  `CharlmCache` entries' `ICharlmReps` hold the tensor forms; `Embeddings`, `Pretrain.Load` and the tensor
  `TryAdd`/`TryGet` are extensions. Step 2 (1.0): the facade references no TorchSharp; `Pipeline` loads through an
  internal `BackendModels` (`Configure` threads/TF32, `Load` wrapper for the device, pretrain, charlms, each processor;
  `ManagedModels` in the facade, `TorchSharpModels` in StanzaSharp.Cuda), made by `PipelineBackend.Models`.
- **No libtorch on the managed backend** (0.5 requirement, done): TorchSharp's `torch` static constructor loads libtorch, so
  the managed path calls no `torch.*` member at all (not even `torch.CPU`) and makes no TorchSharp object. Since 1.0
  the main package has no TorchSharp at all, and ManagedCheck fails if a TorchSharp assembly loads. Managed
  `Pipeline.Load` sets only `ManagedThreads.Count` (`ManagedModels.Configure`); libtorch's threads, TF32 and
  `Weights.On(Device)` are in `TorchSharpModels`. `Pretrain.LoadManaged` keeps the
  vectors as a `float[]` (`Load`: a tensor; `CpuVectors()` reads either; `Embeddings` throws on a managed one), used when
  every pretrain reader is managed. After each managed model `Pipeline` runs `GC.Collect()` (weights are read into an
  array, then packed: load peak 1,035 → ~850 MB, +0.05 s). Proof: `tests/StanzaSharp.ManagedCheck` (no TorchSharp-cpu /
  libtorch-cpu-*, `StanzaSharpNoLibTorch`), run by `ManagedCheckTests` with an empty `NUGET_PACKAGES` (TorchSharp's
  fallback otherwise copies libtorch from the NuGet cache into `cpu/` next to the app), both packages byte-identical and
  no native torch module loaded; `verify-package.ps1 -Managed` compiles the same Program.cs against the packed package
  alone (golden, cross-os, linux-arm64), both on the default backend (no option set). A new managed code path must keep
  this: no `torch.*`, `Scalars`, `Weights.Device`, tensors.
- **Seam** (all internal): per processor network, arrays in and out: `ITokenizerNet` (`TokenizerNet` / `ManagedTokenizerNet`),
  `IMwtNet` (`MwtNet` / `ManagedMwtNet`). Batching, windows, argmax and decoding stay in the processor, shared, so both
  backends see the same batches. `Nn.Backend` { TorchSharp, Managed } is what tests and the benchmark pass to the
  processors' TorchSharp-side `Load(basePath, device, backend)`; the pipeline goes through `BackendModels`. With
  Managed, `Load` also sets `ManagedThreads.Count` = `Threads ?? min(Count, ManagedThreads.Default)`. Benchmark:
  `--backend managed|torch --processors ...` (managed by default; `--device`/`--no-tf32` select torch).
  - Tests that only make sense on TorchSharp (libtorch threads, the Scalar GC race, TF32) set
    `Backend = CudaBackend.Cpu`; `ManagedThreadsOptionTests` (in `ManagedKernelsCollection`) covers the managed `Threads`.
  - A new port: an `I…Net` with array I/O; the TorchSharp side wraps today's module code unchanged (default path stays
    byte-identical); the managed side composes `Nn.Managed` blocks. Its golden tests become theories over `bool managed`
    (`Repo.Backend(managed)`), and `ManagedBackendTests` gets a net-vs-net test per kernel path.
  - Discrete outputs on the managed backend must be byte-identical with the golden data; never loosen a test for it.
  - `INerNet` (`NerNet` / `ManagedNerNet`): ids, Viterbi, `fix_singleton_tags` and entities stay in `NerTagger`.
    `NerTagger.Load` (TorchSharp) / `LoadManaged` (managed charlms, or none for `_nocharlm`). The managed net builds each
    token's input row at its packed position, so input_transform, the biLSTM and the tag layer never see padding.
  - `IPosNet` (`PosNet` / `ManagedPosNet`) → `PosOutput` (UPOS scores; UPOS, XPOS, feats ids; the argmaxes are in the net
    since XPOS/feats read the argmax UPOS). simplify_punct, vocab lookups, batching, cache keys and decoding stay in
    `PosTagger`; `Load` / `LoadManaged`. The managed XPOS/feats biaffines are contracted with each of the 21 UPOS
    embeddings at load (one linear layer per scorer, all UPOS stacked). `upos_clf` sums in double: in float its
    400-term sums of logits up to |150| drifted 1.07e-4 from Stanza (Scalar path); in double ≤ 6.9e-5, at no
    measurable cost. Depparse sums its arcs in double too; its float label scores met 1e-4 as they are.
    UPOS logits (owner's decision, 2026-10-13, German): like sentiment's, a logit passes within 1e-4 of Stanza's float32
    **or** float64 value (`PosTests.AssertUposClose`, also FastPackageTests). German's logits reach |140| and Stanza's own
    float32 is 1.01e-4 from float64; managed was 1.37e-4 from float32 (6.3e-5 from the nearer value), TorchSharp alone
    1.22e-4. `upos_hid` in double changed nothing. The TorchSharp `PosNet` sums `upos_clf` in double too (round 26:
    on Intel AVX-512 runners libtorch's float sums put German `default_fast` UPOS 1.2e-4 from Stanza).
  - `IDepparseNet` (`DepparseNet` / `ManagedDepparseNet`) → `DepparseScores` (arc log-probs over the padded width, label
    argmax per pair, label scores for tests). `DependencyParser` keeps simplify_punct, vocab ids, ROOT, batching, MST and
    the deprel strings. The managed net scores no padding rows; padding columns all get in2 = ReLU(W2's bias) (LSTM
    output 0 there), so the log-softmax sees TorchSharp's exact width, and a row scores them once (round 29). Arcs (pair
    sums through `Act.Dot`, whose SIMD lanes are its four double accumulators; terms; log-softmax) in double;
    labels per dependent through `Gemm.Kernel` on a repacked T row, real pairs only, T in ≤ 32 MB chunks.
  - `ISentimentNet` (`SentimentNet` / `ManagedSentimentNet`) → [batch, classes] logits. `SentimentClassifier` keeps map_word,
    the delta vocab, `label_sentences` sorting, 5000-token batches, the padded width (≥ widest filter), cache keys (no MWT)
    and the argmax. Managed: input rows time-major, `ManagedLstm.ForwardPacked` with every batch size = n (the unpacked
    LSTM: padding runs through it; `ForwardUnpacked(lengths:)` projects the shared padding input once per direction, with
    the single-row kernel's row kept apart, round 28), full-width convolutions as one GEMM whose A rows overlap (lda = 600,
    K = Height·600), the (5,5)/(1,5) filter with its 8 channels in SIMD lanes (multiply then add, the scalar order: same
    bits on every path), FC layers through `Gemm`. Labels byte-identical. Logits: the sentiment
    golden also holds Stanza's float64 logits (owner's decision, 2026-10-08), and a logit passes within the usual
    tolerance of the float32 **or** the float64 value, on both backends. The classifier is ill-conditioned: on
    validation.txt sentence 41 Stanza's float32 logits are 1.49e-4 from float64 and the managed ones 1.2e-5 (1.38e-4
    from Stanza's). Double sums (FC, convolutions, LSTM input) only moved managed toward float64.
  - `ISeq2SeqNet` (`Seq2SeqNet` / `ManagedSeq2SeqNet`, in Nn since German: MWT uses it too, without the POS step and
    the edit head): `Encode` → an `ISeq2SeqDecoder` per batch (edit logits, `Step(previous)` →
    [batch, columns] log-probs). `Lemmatizer` keeps the dictionary, DeltaVocab, batches, the greedy loop (first maximum,
    like torch), edits and fallbacks; `Load(..., backend)`. Managed: packed encoder (`ManagedLstm.ForwardPacked` with
    `finalC`), per step four GEMMs (cell `[x | h]` K = 250, linear_in, linear_out, dec2vocab + copy gate column), attention
    and copy mix per row in double. Lemmas byte-identical; smallest top-2 margin 5.6e-3. Its log-probs are 4.8e-4 from
    TorchSharp, which is itself 2.7e-4 from Stanza in float64 (Stanza f32: 2.9e-4): `LemmaNet_ManagedMatchesTorchSharp`
    reports them without a tolerance (owner's decision, 2026-10-08: lemma is tested on exact lemmas, decoding and edits, as
    TorchSharp always was; no backend reaches 1e-4 on these log-probs, Stanza's float32 included).
    On UD EWT (`tools/lemma_divergence.py` + benchmark `lemma-divergence`; corpus not in the repo) no lemma differs among
    Stanza f32/f64, TorchSharp, managed and managed with double gates, also with every word through the seq2seq (1.25M
    steps, min margin 9.1e-4). It also found the tokenizer's `<UNK>` limit bug (1000 vs the config's `max_seqlen` 200), fixed in #41.
  - `IConstituencyNet` (`ConstituencyNet` / `ManagedConstituencyNet`): the net makes opaque handles (word vectors,
    constituent vectors, stack LSTM states; TorchSharp's are `IDisposable` and owned by the `ParserState`), and offers
    `EncodeWords`, `Word`, `Score`, `Open`, `Compose`, `PushTransitions`, `PushConstituents`. `ConstituencyParser` keeps the
    transition system, legality, unary_limit, the vocab lookups, the schedule (longest first, 50 in flight) and the trees.
    Managed: the word encoder is `ManagedLstm.ForwardPadded`; each stack LSTM layer is one GEMM over `[x | h]` (like the
    lemma's cell), whose tasks also run their panels' LSTM cells and copy the next layer's rows (`Gemm.Run`'s
    `epilogue`, round 30); scores, compositions and pushes run `Gemm.Run(..., rowInvariant: true)` (no 1-row kernel, whose split sum
    differs from the 6-row one), so a state's arithmetic is bitwise the same in any batch (tested). All float: scores 3.4e-5
    from Python (TorchSharp 3.1e-5; every path ≤ 4.6e-5, tolerance 1e-3 as before); trees byte-identical. Near-ties over
    845 golden sentences (22,569 steps): smallest decision margin 3.6e-4 (both backends), backends 7.3e-5 apart;
    `ConstituencyTests.NearTies_StayClearOfBackendDrift` fails if decisions differ or the smallest margin drops under 2× the
    largest difference between the backends' margins (owner's decision, 2026-10-31; it was 3× the score drift until the
    re-uploaded English tagger made validation_multilingual sentence 13 a near-tie at 9.8e-5, with 5.3e-5 drift and 3.8e-5
    margin difference; the drift is spread, ~1.5e-6 relative, so only double charlms would halve it). Since the decision
    review of the same day (owner's choice) every decision must clear 3× again, and only the reviewed near-ties listed in
    `ConstituencyTests.KnownNearTies` (that one today) keep the 2× floor; a new near-tie fails until reviewed and listed.
    On UD EWT (`tools/constituency_divergence.py` + benchmark `constituency-divergence`; 254,589 words, 684,407 decisions)
    no tree differs among Stanza f32, TorchSharp and managed (nor Stanza f64 on a 38k-word subset); smallest margin
    4.7e-5, but backend margins differ by 4.4e-6 median / 5.3e-5 at the 99.99th percentile (smallest margin/difference
    ratio 15). The gap is the float32 charlm (managed's summation order), amplified by the word encoder; per-step layers
    in double (`ManagedConstituencyNet.Double`, study only) change nothing, so no precision change. Run torch-based
    shards with `--threads 1`: parallel 8-thread libtorch processes crawl. Since then (2026-10-10) every managed LSTM
    step sums h·W_hhᵀ in blocks of 128 (`Gemm.KernelBlocked`): charlm error vs float64 halved (TorchSharp's level), golden
    max margin gap 7.3e-5 → 4.0e-5, median unchanged, +4% charlm time at 1 thread, +3% at 8 (docs/backends.md).
  - A pipeline runs every processor on one backend (`BackendModels`); it loads the charlms only if a `_charlm`
    processor reads them (`ManagedCharLanguageModel`: +31 MB of input tables). Managed processors read the shared `Pretrain` through `CpuVectors()` (its array, or a CPU tensor's own memory).
  - `CharlmCache` is backend-neutral: an entry holds its producer's form (tensors, or `[words, dim]` float arrays via the
    float `TryAdd`). `TryGet` gives tensors (a float entry is converted once, on the reader's device, and kept);
    `TryGetArrays` gives arrays (a tensor entry is copied per read). Same-backend pipelines never convert, so the default
    path is unchanged. With pos managed, the tagger adds arrays (only when `HasRoom`), the managed readers (constituency,
    sentiment, NER) read them as they are; a TorchSharp reader in a mixed pipeline would convert them to tensors once.
- **Code:** `src/StanzaSharp.Nn/Managed/` (`Gemm.cs`: `KernelPath`, `PackedMatrix`, `Gemm`, `Act`;
  `ManagedThreads.cs`; `PackedLstm.cs`; `ManagedModels.cs`: charlm, highway biLSTM, `ManagedLstm` = `nn.LSTM` over a
  padded batch like `Rnn.RunPacked` (`ForwardPadded`) or over packed rows (`ForwardPacked`), optional `h_init`/`c_init`,
  hidden padded to a multiple of 4 with zero units; `ManagedCharacterModel`, both variants: NER's bidirectional final
  states and the tagger/parser's unidirectional attention pooling, tested against `CharacterModel`). Tests:
  `ManagedBackendTests` (every path the machine runs natively). Harness: `StanzaSharp.Benchmark managed-spike
  check|speed|concurrent|gemm`. Numbers and method: `docs/managed-backend-spike.md`, `docs/backends.md`.
- **Rules for the managed code:**
  - `Gemm.Path` is detected at startup (Vector512 = AVX-512 where .NET accelerates `Vector512`, Vector256 = AVX2/FMA,
    Vector128 = NEON on Arm64, Scalar); every path reads the same packed weights. A new kernel or element-wise step needs
    a version per path and a test per path.
  - Vector512 (docs/performance/round-14-avx-512-kernels.md and round-19-vector512-intel-two-panels.md): one register per panel row, two panels per kernel call in
    `Gemm.Run` (`KernelPair512`) and in `PackedLstm.Step` (`KernelBlockedPair512`; a single panel only at a task's odd
    end); the LSTM cell's one exp over 16 gates. It must stay Vector256's bits (same FMAs in the same order,
    activations lane-identical; `Vector512_GivesVector256sBits`), so a CI runner's CPU draw can't change a result.
    Measured vs Vector256 on runners (round 19): +65–82% EPYC 9V45 (Zen 5), +8–16% EPYC 9V74 (Zen 4), +37–54% Xeon
    8573C (Emerald Rapids), +4–6% Xeon 8370C (Ice Lake; slower than Vector256 before the LSTM pairs). Faster on every
    CPU drawn, so no vendor rule. `DOTNET_PreferredVectorBitWidth=256` selects Vector256. No local AVX-512 (Ryzen
    5800X): test it with the Benchmark workflow's `avx512` jobs (`gh workflow run benchmark.yml --ref BRANCH -f
    avx512_only=true`; six ubuntu-26.04 and three windows-2025 jobs, those without `avx512f` stop with a notice; each
    also prints `managed-spike gemm` rates per path). Intel came up in 8 of 69 jobs. ci.yml's golden-shard and
    cross-os print the CPU and run Vector512 when they get AVX-512.
  - `ManagedThreads`: one process-wide pool of `Count` − 1 workers; each caller works its own region and idle workers
    help. Not per-caller teams: those oversubscribe (measured: 8 callers fall to TorchSharp's throughput). A region's
    split must not depend on how many threads help, so results stay bitwise identical under concurrency.
  - Contention from other processes (round 32): a join wait over 0.5 ms and over 8× the caller's mean item is a stall
    (a preempted worker holding an item). Per 100 ms window, with k threads per region, stalls over 1/k of the window
    put one more worker to sleep (`Pool._shed`, keeping max(1, Count/4) awake), under 1/(4k) one comes back. Concurrent
    callers (busy threads − awake workers > 1) reset `_shed` to 0 and void the window: shedding there starved long calls
    (owner's tuning request). `Work` lowers `_busy` before raising `_sleepers` and vice versa so that count never
    over-counts. Only who runs items changes, never the split. Internal, no option (a public one would be the owner's
    decision). `ParallelFor_ShedsWorkersWhileJoinsStallAndBringsThemBack` asserts that a shed worker comes back, not
    that all do (2026-10-17): with fewer free cores than its 4 threads (taskset 2 cores, the 3-core macOS runner at
    times) the pool's own threads stall for real and it settles at 1–2 shed by design; requiring 0 failed about one
    run in three there. Don't tighten it back.
  - `PackedLstm.Step` runs 4 tasks per thread (`TasksPerThread`; round 16): a late thread then holds a quarter of its
    share. Tried and dropped: fewer tasks for small steps (a work threshold, or timing each size class: biased by
    asleep workers, 25–50% slower on the Arm runners). A panel's arithmetic doesn't depend on its task.
  - Startup (round 17): code that runs per time step or per element (the pool's `For`/`Run`/`Drain`, `PackedLstm.Step`,
    `Act.LstmCell`/`AddGatedTanh`, the GEMM kernels) is `[MethodImpl(AggressiveOptimization)]`, so a fresh process's
    first calls don't run it at JIT tier 0; `Act.Sigmoid` uses `VectorExp` (the runtime's `VectorN.Exp`, copied op for
    op, inlined), since the runtime's own stays at tier 0 inside optimized callers. A new hot per-step method gets the
    attribute too. The managed `Pipeline.Load` loads two models at a time (`LoadLanes`; all at once: +95-152 MB load
    peak). Measure startup with `StanzaSharp.Benchmark startup` / `tools/startup.py` / benchmark.yml's `startup_only`.
  - Per-call buffers come from `ArrayPool<T>.Shared` and go back in `finally`; no buffer is shared between calls, and
    no mutable state lives on a model. Rented arrays hold stale data: write before reading. Measured (round 9): exact-size
    arrays instead raise the peak (+150 MB in depparse); the pool's power-of-two rounding costs no working set. Processors
    share the pool's buckets, so resizing one buffer can raise another processor's peak (sentiment's, round 9): measure
    the all-eight `--memory` peak too.
  - A GEMM column depends only on its 16-column panel: a GEMM split by whole panels (`new PackedMatrix(source, firstPanel,
    panels)`, or rows packed separately at a panel boundary) gives the same bits. Rows too, through `Gemm.RunRows`: every
    kernel but the single-row one gives a row the same arithmetic, and `Gemm.Run` uses that one only for row m − 1 when
    m % 6 == 1, so `RunRows` runs its chunk row-invariant and that row alone (round 10). Splitting a batch's sequences
    (fewer rows per LSTM or charlm step) moved the last bits in round 9; time steps and directions can be split.
  - Memory: a buffer per padded row multiplies (sentiment's 5000-token batch is 35,723 rows on the benchmark text). Build
    a GEMM's input and output a chunk of rows at a time (`ManagedLstm.ForwardUnpacked`, sentiment's convolutions) and
    keep only what a later stage reads whole; round 10 cut sentiment's +1.1 GB to +0.18 GB that way.
  - Check the `CancellationToken` per time step and per GEMM block.
  - `Gemm.Path` and `ManagedThreads.Count` are process-wide. Only `Pipeline.Load` and `LanguageIdentifier.Load` (managed backend: the count) and tests
    in `ManagedKernelsCollection` (parallelization off, so other classes never see a switched path) may set them.

## Native AOT and trimming

- Supported on the managed backend (owner's decision, 2026-10-16; docs/aot.md): Directory.Build.props sets
  `IsAotCompatible=true` on every `src/*` project but `StanzaSharp.Cuda` (TorchSharp isn't AOT-safe) and
  `StanzaSharp.Tool` (a .NET tool, reflection JSON). The trim/AOT analyzers then run on every build, and build-test's
  warnings as errors turns any IL2xxx/IL3xxx into a failure.
- New code in those projects must stay AOT-safe: no reflection-based `JsonSerializer` (use `JsonNode`/`Utf8JsonReader`,
  or a source-generated `JsonSerializerContext`), no `Type.GetType`/`Activator`/
  `MakeGenericType`/`Assembly.Load` or reflection over members, no `Reflection.Emit` or `RegexOptions.Compiled`
  (`[GeneratedRegex]`), no `[DynamicDependency]`/suppressions to silence a warning. P/Invoke stays blittable.
- AOT fixes the SIMD width at compile time: `Vector256.IsHardwareAccelerated` is false on the default x64 target, so
  `Gemm` runs Vector128 (~2x slower, same bits). Apps set `IlcInstructionSet=x86-64-v3` (docs, README, PACKAGE.md);
  v4 gives Vector512. `[MethodImpl(AggressiveOptimization)]` is a no-op there (no tiers). Arm64: NEON is baseline.
- CI: `tools/verify-aot.sh MODEL_DIR RID` publishes tests/StanzaSharp.ManagedCheck with `PublishAot`
  (`x86-64-v3`) and with `PublishTrimmed` (self-contained), fails on any IL2xxx/IL3xxx warning or torch file, and runs
  both (ManagedCheck compares both English packages with the golden files byte for byte). Golden shard 1 (linux-x64;
  installs clang/zlib1g-dev if missing) on full runs and on light runs with torch=full; cross-os windows-2025 shard 3
  on full runs (the runner's VS 2022 has the C++ linker). Locally AOT on Windows needs VS 2022's C++ workload; this
  machine's VS 2017 MSVC can't link .NET 10's runtime (`__tls_guard`, docs/aot.md).
