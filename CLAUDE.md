# StanzaSharp

A native C# port of the inference side of [Stanza](https://github.com/stanfordnlp/stanza) 1.15.0, Stanford's
PyTorch-based NLP library. It runs Stanza's own pretrained models on a managed backend (C# SIMD kernels, the default)
or through TorchSharp (the opt-in `StanzaSharp.Cuda` package), with output byte-identical to Python Stanza.

This file holds the rules and conventions. The decisions behind them, measurements and history are in the docs below;
read the one for the area you work in before changing it, and record a new decision there (dated, with who decided),
not here. Change this file only when a rule changes.

| Topic | Where |
|---|---|
| Languages, processors, packages, alternative models, transformers, coref | [docs/decisions/scope.md](docs/decisions/scope.md) |
| How each processor was ported, what it reproduces, input modes, options, performance rounds | [docs/decisions/status.md](docs/decisions/status.md), docs/performance/ |
| What each checkpoint uses (port only those paths) | [docs/checkpoint-findings.md](docs/checkpoint-findings.md) |
| Model formats, `.pt`/safetensors reading, the loader's failure contract | [docs/model-files.md](docs/model-files.md) |
| Golden data, `make_golden.py`, test conventions | [docs/decisions/validation.md](docs/decisions/validation.md), docs/verification.md (user-facing) |
| CI jobs, shards, caches, nightly, benchmark workflow | [docs/ci.md](docs/ci.md) |
| Packages, public API, downloads, versions, releases | [docs/decisions/packaging-and-releases.md](docs/decisions/packaging-and-releases.md) |
| Managed backend: decisions, the seam per processor, exactness, kernel background | [docs/decisions/managed-backend.md](docs/decisions/managed-backend.md), docs/backends.md |
| Native AOT, GPU, plans for larger features | docs/aot.md, docs/gpu.md, docs/*-plan.md |

## Working on this repository

- **Advisor**: when an advisor tool is available, call it before design decisions or recommendations, before splitting
  work across agents or sessions, and before declaring a larger task done.
- **Asking the owner** (owner's decision, 2026-10-17): whenever a decision or any input from the owner is needed, ask
  with the question prompt (`AskUserQuestion`: options, the recommended one first), not only in the reply text. New
  public API, a new required CI job, a loosened test bound, a new package/language/model family and anything that
  changes outputs are the owner's decisions.
- **Cloud sessions**: `.claude/hooks/session-start.sh` installs `dotnet-sdk-10.0` and `stylebro-migrate`, restores,
  builds the Cli and downloads `tools/ci-models.sh en` into `models/stanza/en` (`STANZASHARP_MODELS` points there). No
  Python, German or treebank models: fetch those with `tools/ci-models.sh download GROUP` (or the Cli's `download DIR
  --lang de --processors ...`) when a task needs them.
- **Where work runs**: long work (multi-step tasks, PRs, CI waits, releases) runs as cloud sessions; the owner's
  computer must stay free. Work that needs it (local timings, Windows-only checks) is asked for, never started unasked;
  a PR that needs such a step says so.
- Every change goes through a PR (`main` is protected). Agents open stacked PRs as drafts (drafts skip the heavy CI)
  and mark them ready once their base merges.

## Layout

```
src/StanzaSharp.Core           Document model, CoNLL-U/JSON I/O, Tree, safetensors/.pt checkpoint loading. No TorchSharp.
src/StanzaSharp.Nn             Pretrain, CharlmCache, transformer tokenizers, the managed kernels and layers (Managed/). No TorchSharp.
src/StanzaSharp.Cuda           The TorchSharp backend (package StanzaSharp.Cuda, `CudaBackend`): every torch net and layer,
                               processors' TorchSharp `Load` extensions. The only project referencing TorchSharp.
src/StanzaSharp.{Tokenize,Mwt,Pos,Lemma,Constituency,Depparse,Ner,Sentiment,Coref}   One processor each.
src/StanzaSharp                Pipeline facade (Pipeline, PipelineOptions, ModelDownloader, LanguageIdentifier, ...).
src/StanzaSharp.Tool           The `stanzasharp` .NET tool: download, compare, benchmark.
samples/                       Cli (experiments), Benchmark (speed/memory; tools/benchmark.py is its Python twin),
                               Example (API tour), docker (standalone app on the published package).
tests/StanzaSharp.Tests        xUnit; golden tests against Python Stanza output.
tests/StanzaSharp.ManagedCheck Proves the managed backend loads no libtorch (ManagedCheckTests, verify-package, verify-aot).
tests/golden/                  Golden data from Python Stanza (committed; keep it small).
tools/                         make_golden.py, stanza_convert.py, stanza_resources.py, ci-models.sh, changelog.py, ...
models/                        Downloaded/converted models (gitignored; never commit or redistribute model files).
```

- Only `StanzaSharp.Cuda` references TorchSharp (the managed `TorchSharp` package); everything else must build without
  it. Runnable projects that run the TorchSharp backend (Cli, Benchmark, Tests) reference `$(TorchSharpNative)`
  (`TorchSharp-cpu`; `TorchSharp-cuda-windows` with `STANZASHARP_CUDA=1`). Example and ManagedCheck set
  `StanzaSharpNoLibTorch`. `TorchSharpVersion`/`LibTorchVersion` are set only in Directory.Build.props and move together.
- The `StanzaSharp` package carries the facade, Core, Nn and the processors (`PrivateAssets="all"` references), so tests
  and samples reference the library projects they use directly. Every other project is `IsPackable=false`.
- `InternalsVisibleTo` comes from Directory.Build.props for all StanzaSharp assemblies, the tests, the benchmark and the
  Cli; don't add per-project ones.

## Reference implementation

The Python Stanza source (`tools/.venv/Lib/site-packages/stanza/`, version pinned in `tools/requirements.txt`) is the
source of truth. When porting, read its code rather than relying on memory, and port only the code paths the
checkpoints actually use (their configs say which; record them in docs/checkpoint-findings.md). Before adding a
language or model, run `stanzasharp compare` against Python Stanza: it finds what breaks.

## Correctness rules

- **Discrete outputs are byte-identical with Python Stanza** (tokens, offsets, words, tags, lemmas, trees, heads,
  entities, labels, chains) on every backend. Never loosen a test to make a backend pass.
- **Scores** use the same tolerances on both backends: 1e-4 (relative above |100|: within 1e-4 × max(1, |expected| /
  100), `TokenizerTests.Allowed`), 1e-3 only where TorchSharp already had it. Where Stanza's own float32 is
  ill-conditioned (UPOS logits, sentiment) a value passes within tolerance of Stanza's float32 **or** float64 value. A
  managed comparison that misses: find the drift's source, try cheap fixes (double accumulation, reduction order),
  measure their cost, and report to the owner if none is cheap. Never a managed-only tolerance, never a skipped test.
- **Near-ties**: `ConstituencyTests.NearTies_StayClearOfBackendDrift` holds every decision to 3 × the largest
  difference between the backends' margins; reviewed exceptions go in `KnownNearTies` (2 × floor) with the owner's
  sign-off. Add an entry; don't move the factor.
- **Deliberate differences from Stanza** are allowed only where Stanza warns and ignores, crashes or writes broken
  output, and are decided by the owner: we throw (unknown models, a processor the package lacks, empty pretokenized
  sentences). Document each in the area's doc.
- Use `Core.PyString` wherever Stanza calls `lower()`, `isupper()`, `isspace()` or `split()`, and run regexes Stanza runs
  over code points on `Paragraph.CodePoints` (`Tokenizer.PyRegex` for `\b`/`\w`). Model units are code points; offsets
  are UTF-16 indices.
- Bulk, pretokenized, `Process(Document)` and streaming follow Stanza's own batching, so results equal Stanza's for
  the same call shape (bulk ≠ one by one, in Stanza too).

## Tests

- Tests that need model files **skip, not fail**, when the models are missing (`[ModelFact]`, `[German]`, ...). CI's
  no-skip checks then make each job prove it ran what it should; a test class whose models a job lacks is exempted
  there by name (docs/ci.md).
- Golden tests are theories over `bool managed` (`Repo.Backend(managed)`, `Repo.PipelineBackendFor(managed)`) and, for
  German, a last `string lang` parameter (`[InlineData(..., "en")]` + `[German(...)]`). German-only tests have German
  in their name. A text with non-BMP characters is compared without offsets.
- Share loaded models: `SharedPipelines` for an options set more than one test uses, per-class `Fixture`s for models
  several tests read (lazy, so no-model runs only skip). Tests of Load and of options load their own.
- An expensive TorchSharp-only test without a `managed` parameter gets `[Trait("Backend", "TorchSharp")]`; a test that
  must run without libtorch (linux-arm64) gets `[Trait("Backend", "Managed")]` and must really need none.
- Tests that set `Gemm.Path` or `ManagedThreads.Count` go in `ManagedKernelsCollection`.
- Golden data comes only from `tools/make_golden.py` with the pinned versions (`tools/requirements.txt`; transformer
  extras in `tools/requirements-transformer.txt`); never hand-edit golden files. Add each new text fuzzing finding's
  shrunk text to `FUZZ_TEXTS`. Update docs/verification.md when golden data, tolerances or CI jobs change.

## Managed backend rules

- **No libtorch on the managed path**: no `torch.*` member (not even `torch.CPU`), `Scalars`, `Weights.Device` or
  tensor anywhere a managed pipeline runs; ManagedCheck fails if a TorchSharp assembly loads.
- A processor's network sits behind an internal `I…Net` with arrays in and out; batching, windows, decoding and vocab
  lookups stay in the shared processor code, so both backends see the same batches. A new port: the TorchSharp side
  wraps the module code unchanged, the managed side composes `Nn.Managed` blocks, its golden tests become `managed`
  theories and `ManagedBackendTests` gets a net-vs-net test per kernel path.
- `Gemm.Path` (Vector512, Vector256, Vector128, Scalar) is detected at startup: a new kernel or element-wise step needs
  a version and a test per path, and Vector512 must give Vector256's bits (`Vector512_GivesVector256sBits`).
- **Invariance**: a row's result must not depend on its batch, thread count or concurrent callers. Split a GEMM only by
  whole 16-column panels or through `Gemm.RunRows`; use `rowInvariant: true` where a state must have the same bits in
  any batch; a `ManagedThreads` region's split must not depend on how many threads help. Splitting a batch's sequences
  moves bits; time steps and directions may be split.
- Per-call buffers come from `ArrayPool<T>.Shared` and go back in `finally`; rented arrays hold stale data. Rent an output
  only where its one caller returns it. Build large GEMM inputs/outputs a chunk of rows at a time.
- Hot per-step code (pool loops, LSTM steps, GEMM kernels, activations) is `[MethodImpl(AggressiveOptimization)]`.
- Check the `CancellationToken` per time step and per GEMM block, outside dispose scopes or inside a `try/finally`.
- `Gemm.Path` and `ManagedThreads.Count` are process-wide: only `Pipeline.Load`, `LanguageIdentifier.Load` and tests in
  `ManagedKernelsCollection` set them.

## Thread safety and API rules

- Process keeps all mutable state per call; models are read-only after Load. Never add a mutable field to a processor
  (the one exception: the batch sizes, set on `WithBatchSize`'s fresh copy inside Load).
- The public API is the facade (`Pipeline`, `PipelineOptions`, `PipelineBackend`, `CharlmCacheOptions`,
  `ModelDownloader`, `Processor`, `ModelPath`, `LanguageIdentifier(Options)`, `MultilingualPipeline`,
  `MultilingualOptions`), `CudaBackend`, and Core's results (`Document`, `Sentence`, `Token`, `Word`, `Entity`, `Tree`,
  `Dependency`, `Conllu`, `Json`, `CorefChain`, `CorefMention`, `CorefAttachment`). New types default to `internal`;
  making one public is the owner's decision. Every public member needs an XML doc (CS1591 is an error in src/).
- **API compatibility**: packing StanzaSharp or StanzaSharp.Cuda checks the public API against the last release
  (package validation; build-test runs it). A binary break needs the owner's decision and a major version.
- Options go into `PipelineOptions` (no new optional parameters on `Load`); one options object downloads and loads the
  same files (`ModelDownloader.FilesFor` and `Pipeline` share `SelectModels`).
- Model files may come from anywhere: `Checkpoint.Load` and every later read succeed or throw `InvalidDataException`,
  bounded in time, memory and stack by the file's size (docs/model-files.md has the limits). Keep `CheckpointFuzzTests`
  passing and add a crafted input for each bug.
- `Pipeline.Load` never downloads; downloads are explicit and checked against pinned commits and MD5s
  (`ModelDownloader.Revisions`, `stanza_resources.json`). Adopting Stanford's re-uploaded models is the owner's decision
  and an output change: regenerate the golden sets that read them and say so in the changelog.

## TorchSharp rules

- Dispose tensors deterministically: every temporary tensor, at load time too, belongs to a `DisposeScope` or a
  `using` variable.
- Never pass TorchSharp a temporary `Scalar` (dotnet/TorchSharp#1583; `Nn.Scalars` explains the finalizer race): no
  tensor `+` or `add(Tensor)` (write `a.add(b, Scalars.One)`), no numbers where a Scalar is expected (`t * 2`,
  `masked_fill(m, 0)`, `eq(0)`, `arange(n)`): use `Scalars.*`, a `static readonly Scalar` or a `using var` local;
  `F.softplus` via `Scalars.Softplus`. Check a new TorchSharp function's source for internal conversions. On a TorchSharp
  upgrade, check whether #1583 is fixed and drop the workaround if so.
- A processor runs on CUDA through `Weights.On(device, ...)`, `_device = Weights.Device` for input tensors and
  `Weights.ToArray<T>()` for reads; pack_padded_sequence lengths stay on the CPU (docs/gpu.md).

## Native AOT

Every `src/*` project but StanzaSharp.Cuda and the Tool is `IsAotCompatible`, and warnings are errors in build-test. New
code there stays AOT-safe: no reflection-based `JsonSerializer` (use `JsonNode`/`Utf8JsonReader` or a source-generated
context), no `Type.GetType`/`Activator`/`MakeGenericType`/`Assembly.Load`/member reflection, no `Reflection.Emit` or
`RegexOptions.Compiled` (`[GeneratedRegex]`), no suppressions to silence a warning; P/Invoke stays blittable (docs/aot.md).

## Changes, releases and CI

- Each user-visible change adds one file `changes/<section>/<slug>.md` (added, changed, deprecated, removed, fixed,
  security) holding its CHANGELOG bullet; never edit CHANGELOG.md for an unreleased change (changes/README.md).
- Releases start from a merged "CHANGELOG: x.y.z" PR made by `python tools/changelog.py release x.y.z` (it also moves the
  API baseline); release.yml waits for main's CI, publishes and tags. Never push release tags by hand. A user-facing
  fix should not wait behind unrelated features: release it.
- CI shards split the test project by class; a new class lands in shard 1 automatically. A new model goes into a group
  line of `tools/ci-models.sh`, nowhere else. A new required CI job must also be added to branch protection (the
  owner's step). Keep paths forward-slash and file names case-exact; golden files are pinned binary/LF by
  `tests/golden/.gitattributes`. Details: docs/ci.md.
- Dependabot auto-merges minor/patch updates; library dependencies the package exposes (Logging.Abstractions) and
  TorchSharp/libtorch/StyleBro are excluded and moved by hand.

## Code style

StyleBro (`StyleBro.Analyzers` 0.7.0-alpha.1, preset 1.0) with `EnforceCodeStyleInBuild`; build-test fails on warnings
and on `stylebro-migrate format StanzaSharp.slnx --verify-no-changes`. Fix findings with `stylebro-migrate format`
(not plain `dotnet format`). Turned-off rules are explained in `.editorconfig`'s own section. Wrapped expressions put
the operator at the start of the continuation line (BRO1520). `stylebro-migrate init --write` breaks
`samples/docker/Directory.Build.props` (stylebro#177): revert that file after each run. Reformat commits go in
`.git-blame-ignore-revs`.

## Design

- Document → Sentence → Token → Word: tokens come from the tokenizer, words from MWT expansion; POS and the parsers
  work on words, NER and sentiment on tokens.
- Batch per sentence, in Stanza's batches; inference only (training stays in Python).

## Licensing

Apache 2.0 (`LICENSE`), like Stanza, whose code it ports; `NOTICE` credits Stanza and ships in the package. Model
licenses vary with the training data: never commit or redistribute model files.
