# Packaging, API and releases

Packages, the public API, downloads, versions and the release process, with the decisions behind them. Moved from CLAUDE.md on 2026-10-31.

- Dependabot (owner's decision, 2026-10-14; `.github/dependabot.yml`): GitHub Actions and NuGet, weekly.
  `dependabot-auto-merge.yml` turns on auto-merge (merge commit) for minor and patch updates, so they land once the
  required checks pass; majors wait for a person. Ignored: TorchSharp*/libtorch-* (move together, check
  dotnet/TorchSharp#1583), StyleBro.* (prerelease, upgraded by hand with a reformat), StanzaSharp* (the docker
  sample); pip is not watched (tools/requirements.txt pins the golden data). Microsoft.Extensions.Logging.Abstractions
  was included at first (the owner chose auto-merge for it too), and the bumps raised every user's minimum to 10.0.14;
  since the decision review of 2026-10-31 (owner's choice) it is ignored and pinned at 10.0.0, the lowest .NET 10
  version, raised by hand only when the library needs a newer API. The two actions that run with write or publish
  permissions (`NuGet/login`, `dependabot/fetch-metadata`) are pinned to commit SHAs (same review); Dependabot updates them.
  An auto-merge is made with `GITHUB_TOKEN`, whose push starts no workflow (#222, 2026-10-28; coverlet #115 had run
  main's CI because the owner merged it by hand), so the nightly runs all of ci.yml on such a commit (below; owner's
  decision, 2026-10-29). release.yml calls ci.yml itself when the commit has no push run.

- API compatibility (owner's choice, decision review 2026-10-31): `EnablePackageValidation` on StanzaSharp and
  StanzaSharp.Cuda (Directory.Build.props) compares each pack's public API with `PackageValidationBaselineVersion` on
  nuget.org and fails on a binary break. `tools/changelog.py release` sets the baseline to the newest version already
  released; build-test packs both with version `999.0.0-apicheck` (0.x versions skip the check: CP0003). An intentional
  break needs the owner's decision and a regenerated suppression file (`-p:ApiCompatGenerateSuppressionFile=true`).
- Public API (user's decision for 0.1.0, 2026-10-06: "pipeline only"). It is:
  - the facade: `Pipeline`, `PipelineOptions`, `CharlmCacheOptions`, `ModelDownloader`, `Processor` (0.4),
    `PipelineBackend` (0.5);
  - StanzaSharp.Cuda's `CudaBackend` (1.0);
  - Core's results: `Document`, `Sentence`, `Token`, `Word`, `Entity`, `Tree`, `Conllu`, `Json` (owner's decision,
    2026-10-14), `Dependency` (#174, 2026-10-18);
  - the facade's `ModelPath`, `LanguageIdentifier`, `LanguageIdentifierOptions`, `MultilingualPipeline`, `MultilingualOptions` (#174, 2026-10-19);
  - Core's `CorefChain`, `CorefMention`, `CorefAttachment` (coref PR b, owner's decision 2026-10-20: Stanza's shape);
  - StanzaSharp.DataIngestion's `SentenceChunker`, `EntityEnricher` (#233, owner's decision 2026-10-31: a package now);
  - StanzaSharp.TasmanianDevil's `StanzaNerRecognizer`, `StanzaTokenNormalizer`, `SePersonnummerRecognizer`,
    `SeOrganisationsnummerRecognizer` (#234, owner's decisions 2026-10-31).

  Everything else is `internal`.
  - Directory.Build.props gives every `src/*` assembly `InternalsVisibleTo` for the other StanzaSharp
    assemblies, the tests, the benchmark and the Cli sample. Don't add per-project `InternalsVisibleTo`.
  - New types default to `internal`. Making one public is an API decision for the user.
  - `src/*` builds docs with CS1591 as an error, so every public member needs an XML doc.
  - Each user-visible change adds one file, `changes/<section>/<slug>.md` (section: added, changed, deprecated, removed,
    fixed, security), holding its CHANGELOG bullet (user's decision, 2026-10-11: CHANGELOG.md's "Unreleased" list
    conflicted between parallel PRs). Never edit CHANGELOG.md for an unreleased change. `changes/README.md` has the format;
    `python tools/changelog.py preview` shows the next section.
- `Pipeline.Load(dir, PipelineOptions?)` (user's decision, 2026-10-06: an options object rather than more
  optional parameters).
  - `PipelineOptions` holds `Processors`, `Backend` and `CharlmCache { IsEnabled = true, MaxWords = 32_768 }`, among others.
  - `CudaBackend.Create(disableTf32: true)` (was `PipelineOptions.DisableTf32` up to 0.5; name and opt-in are the user's
    choices) sets both torch TF32 switches to false process-wide at Load and doesn't restore them.
  - `CharlmCache.TryAdd` keeps a sentence only if it fits, detaching its tensors from the caller's
    dispose scope. Otherwise it leaves them with the caller. Before this fix the cache disposed
    rejected tensors that the tagger was still using.
- Install (owner's 0.5 decisions, 2026-10-09): `dotnet add package StanzaSharp` is the whole default install (managed
  backend). `TorchSharp-cpu` appears in the docs only for the TorchSharp backend.
  - The `StanzaSharp.Cpu.*` platform packages (up to 0.4.2: code-free, StanzaSharp + one `libtorch-cpu-<rid>`) are no
    longer published from 0.5; their projects and `src/StanzaSharp.Cpu.props` are gone. TorchSharp-backend users add
    `TorchSharp-cpu` or one `libtorch-cpu-<rid>` 2.10.0 themselves (TorchSharp loads a single one fine; checked with
    `libtorch-cpu-win-x64` alone). Deprecating them on nuget.org is the owner's manual step.
  - `release.yml` packs `StanzaSharp`, `StanzaSharp.Cuda` (from `src/StanzaSharp.Cuda`), `StanzaSharp.DataIngestion`,
    `StanzaSharp.TasmanianDevil` and `StanzaSharp.Tool`.
  - `StanzaSharp.TasmanianDevil` (#234; owner's decisions 2026-10-31: TasmanianDevil adapters as a package, Deidentify.NET
    only if asked, ManagedCode.Presidio deferred; kept after learning of TasmanianDevil.Onnx, its zero-shot GLiNER NER): depends
    on StanzaSharp (same version) and TasmanianDevil 0.3.0 (stable, so no `-preview`). `StanzaNerRecognizer` = an
    `EntityRecognizer` per language (`supportedLanguage`), `DefaultTypeMap` over both NER conventions (OntoNotes PERSON/ORG/
    GPE/LOC/FAC/DATE/TIME, and PER/LOC/ORG/TME) to PERSON/ORGANIZATION/LOCATION/DATE_TIME, a fixed score (0.85, Presidio's
    spaCy recognizer's), `stanzaType` metadata; throws without ner. `StanzaTokenNormalizer` = an `ITokenNormalizer` for
    `LemmaContextAwareEnhancer`: Stanza's tokens; each word's lemma and UPOS from the word tagged alone (deterministic,
    cached up to 100,000 words, warmed per `Tokenize` with one bulk call); stop words = function UPOS (DET, ADP, PRON, AUX,
    CCONJ, SCONJ, PART), punctuation = PUNCT/SYM or no letter/digit; throws without lemma (`Pipeline.HasLemma`, internal).
    `SePersonnummerRecognizer`/`SeOrganisationsnummerRecognizer` (owner's decision 2026-10-31: ship them here, not wait for
    upstream): the Swedish country pack drafted for TasmanianDevil (`upstream-drafts/`, not in git), same entity names
    (`SE_PERSONNUMMER`, `SE_ORGANISATIONSNUMMER`), default language `sv`; date + Luhn (personnummer, samordningsnummer day
    61–91), third digit ≥ 2 + Luhn (organisationsnummer); a separated number promotes to full score, a bare one abstains and
    needs a context word (like NL_BSN). `TasmanianDevilTests` (en models) and `SwedishIdRecognizerTests` (no models: valid,
    invalid, odd input, PiiEngine before/after, 500 random ten-digit numbers unmasked). Swedish NER + both recognizers by hand
    on the masking example: everything masked but street numbers and the postal code.
  - `StanzaSharp.DataIngestion` (#233; owner's decisions 2026-10-31: a package now, chunker + NER enricher first): depends
    on StanzaSharp (same version) and Microsoft.Extensions.DataIngestion, still preview-only, so its `PackageVersion` is
    `$(Version)-preview` (owner's decision 2026-10-31; a stable package can't depend on a prerelease, NU5104) until that
    ships stable. `SentenceChunker`
    = HeaderChunker's sections (headers as context and prefix) cut into Stanza sentences (one bulk `Process` per document,
    sentences sliced from the element's text by token offsets, elements joined by `Environment.NewLine`, so with room to
    spare it gives HeaderChunker's chunks when each element's text starts and ends at a sentence), packed greedily against the caller's ML.Tokenizers tokenizer counted
    on the whole candidate chunk, whole-sentence overlap, tables whole, an oversized unit cut by
    `GetIndexByTokenCount`. `EntityEnricher` = distinct entity texts (`string[]`, `MetadataKey` "entities", like
    KeywordEnricher's; whitespace runs as one space), bulk `Process` per batch, a chunk starting with its `Context` tagged as
    two texts (one newline made "## Kessler Chemicals" and the body's first "Kessler Chemicals" one entity); throws at construction without ner (`Pipeline.HasNer`, internal).
    `DataIngestionTests` (golden shard 1; o200k from `Microsoft.ML.Tokenizers.Data.O200kBase`, offline; the sample's whole
    path through `VectorStoreWriter` and Semantic Kernel's `InMemoryVectorStore` with a hashing stub generator).
    `samples/StanzaSharp.Rag` (owner's choice 2026-10-31: a local ONNX embedding model): sentence-transformers'
    all-MiniLM-L6-v2 (pinned commit, sha256 per file, downloaded at first run) on ONNX Runtime with ML.Tokenizers'
    `BertTokenizer`, mean pooling + L2 norm; build-only in CI (no model download there). Dependabot
    offers each new DataIngestion preview (auto-merged as a minor bump once build-test passes).
  - On a Windows Arm64 machine Directory.Build.props gives the runnable projects (tests/, samples/)
    `libtorch-cpu-win-arm64` (`TorchSharp-cpu` lacks it), except those setting `StanzaSharpNoLibTorch`.
  - Supported platforms (README, PACKAGE.md): managed backend wherever .NET 10 runs; CI covers linux-x64, win-x64,
    win-arm64, osx-arm64 (full suite), linux-arm64 (managed tests) and Alpine (Docker). macOS Intel is not claimed.
    TorchSharp backend: where TorchSharp 0.107 has `runtimes/<rid>` and a `libtorch-cpu-<rid>` 2.10.0 exists
    (linux-x64, win-x64, win-arm64, osx-arm64); not musl, linux-arm64 or osx-x64.
  - `samples/docker` is a standalone app (its own empty Directory.Build.props) referencing only `StanzaSharp` (nothing
    of TorchSharp in the image since 1.0), on
    `mcr.microsoft.com/dotnet/runtime:10.0` (`BASE` arg; `RID` arg, linux-musl-x64 for Alpine).
  - New package IDs need no policy change: a nuget.org Trusted Publishing policy applies to all packages of its owner
    (optionally narrowed by a glob scope). StanzaSharp.Cpu.*, StanzaSharp.Tool and StanzaSharp.Cuda all published with
    the existing policy.
- `buildTransitive` (StanzaSharp.Cuda package since 1.0; `src/StanzaSharp.Cuda/buildTransitive/StanzaSharp.Cuda.targets`
  + a `StanzaSharp.Cuda.props` generated at pack time). Both checks below matter only for TorchSharp-backend users (owner, 2026-10-09: they stay). `TorchSharpVersion` (0.107.0) and `LibTorchVersion` (2.10.0) are set only in Directory.Build.props.
  - `STANZA001` (warning): a resolved `TorchSharp`, `TorchSharp-*` or `libtorch-*` package has another version. Versions
    come from `@(PackageDependencies)` + the assets file's `"<id>/<version>"` keys (copy-local items miss the
    asset-less `TorchSharp-cpu`; `TorchSharp-cpu` 0.106.0 resolves the same TorchSharp/libtorch as 0.107.0).
    `<NoWarn>STANZA001</NoWarn>` silences it. `verify-package.ps1` fails on any `STANZA` warning.
  - `StanzaSharpTrimNative=true` (opt-in) removes from `NativeCopyLocalItems`/`RuntimeTargetsCopyLocalItems` (so also
    from the .deps.json): libtorch_python, libshm, libnnapi_backend, libtorchbind_test, libjitbackend_test,
    libbackend_with_compiler, libaoti_custom_ops (.so; .dylib for python/shm). By DT_NEEDED/LC_LOAD_DYLIB nothing that
    LibTorchSharp → libtorch → libtorch_cpu → libc10/libgomp needs them, and TorchSharp loads only `LibTorchSharp`
    and `torch_cpu`/`torch_cuda` by name. Windows libtorch has no such files (its only unlinked ones, uv.dll,
    libiompstubs5md.dll and torch_global_deps.dll, 250 KB, are kept). A marker file in `obj/` is a .deps.json input,
    so switching the property regenerates it. Proven by the plain `verify-package.ps1` (it sets the property and
    checks the files are absent; Linux in golden, macOS in cross-os).
- `StanzaSharp.Tool` (`src/StanzaSharp.Tool`, `PackAsTool`, command `stanzasharp`): `download [DIR] [--lang CODE] [--package NAME]
  [--processors LIST]`. `DownloadCommand.cs` is compiled into samples/StanzaSharp.Cli too (one implementation). Since
  1.0 StanzaSharp has no TorchSharp, so the tool has no native files at all (0.5 MB with compare and benchmark). `tools/verify-tool.ps1` (golden) packs, `dotnet tool install --tool-path`s it from a local feed,
  checks for natives and downloads `tokenize,mwt`. release.yml packs it.
  - `compare FILE [--lang CODE] [--package] [--processors] [--models DIR] [--python PATH]` (`CompareCommand.cs`, owner's
    "proof for new users"; `--lang` 2026-10-14): runs the embedded `compare.py` (make_golden's bytes→str and `"{:C}\n"`)
    with `stanza.Pipeline(lang, dir=<parent of DIR>, resources_filepath=<the library's stanza_resources.json>,
    download_method=None)`, so no network and no resources.json in the model folder; C# loads with `Language`, DIR
    defaults to `models/stanza/<lang>`, and the processor list and files come from that language's package (a German
    treebank package runs tokenize to depparse). DIR may have any name: if it isn't
    `<lang>`, compare.py's `load_stanza` (benchmark.py imports it) links it as `<lang>` in its temp folder (a junction on
    Windows, which needs no privilege; a symlink elsewhere) while Stanza loads, then unlinks it itself (.NET's
    `Directory.Delete(recursive)` throws UnauthorizedAccessException on such a junction). Both sides
    get the full processor list (requirements added); Stanza's loaded `.pt` set must equal `ModelDownloader.FilesFor`,
    and C# loads with `VerifyChecksums`. Transformer models (2026-10-18, owner's choice): `load_stanza`'s
    `transformers_locally` has Stanza's `load_bert` read `<DIR>/transformer/<name>/` (local_files_only, like make_golden's
    `electra_locally`); a missing transformers/tokenizers/safetensors/peft exits 3 with `TRANSFORMER_REQUIREMENTS` (the
    pins of tools/requirements-transformer.txt, `ComparePy_TransformerRequirements_MatchRequirementsTransformerTxt`). Local
    only: CI's Python has none of them, so no test runs Stanza on these models (by hand, 2026-10-24, #209: `compare
    --processors tokenize,mwt,pos,coref` on corpus.txt, en and de, byte for byte). C# offsets are converted to code points when the text has non-BMP characters
    (`ConlluDiff.ToCodePointOffsets`). Exit 0/1/2. Stanza's resources come from the StanzaSharp assembly (`StanzaResources`,
    see "Models are downloaded only explicitly"), written next to compare.py as `stanza_resources.json`.
    `Run_PythonStanza_*` (CompareCommandTests, BenchmarkCommandTests) need tools/.venv (or `STANZASHARP_PYTHON`) and
    models/stanza/en (the German ones, `[PythonStanzaFact("de")]`, models/stanza/de next to it); CI runs them in golden-shard 2, the only job with Python Stanza; the other jobs' no-skip checks
    exempt them by name.
  - `benchmark [--lang CODE] [--package] [--processors] [--models DIR] [--threads N] [--words N] [--python PATH] [--quick] [--json FILE]`
    (`BenchmarkCommand.cs`, owner's decision 2026-10-10): managed backend vs the embedded `benchmark.py`, one Python process
    that loads Stanza once and answers JSON-line commands (`run`, `calls`) so runs alternate (warm-up runs on the whole text
    until each side settles, `WarmedUp`: no stage > 10% and 20 ms faster, 2–5 runs, round 16; 3 timed runs,
    medians; 50 one-sentence calls, median/p90; peak working set, macOS `ru_maxrss`). Text: the benchmark sample's golden
    texts (validation, corpus, tokenize_stress), embedded under `benchmark/` (`--lang de`: tests/golden/de's files of the
    same names, under `benchmark/de/`; the block's Run row names the language), cut after the paragraph that reaches 1,500
    whitespace words (`--quick`: 300, 1 run, 10 calls); the block reports words after tokenizing and MWT (2,009 by default). Threads default to each side's own (unless `--threads`). Python is optional: any failure
    (no Python/Stanza, models not `.pt`) gives StanzaSharp's column only, with the reason. With
    Python, the CoNLL-U of the last run is compared (`ConlluDiff`). Machine: CPU from the registry / `lscpu` (LC_ALL=C,
    else /proc/cpuinfo) / `sysctl`, cores from GetLogicalProcessorInformation / lscpu / `hw.physicalcpu`, SIMD
    `Gemm.Detect()`. Report formatting is invariant-culture ASCII (`x` for ratios). `BenchmarkCommandTests`
    (`Run_PythonStanza_Quick` and `_GermanQuick` run in golden-shard 2); `verify-tool.ps1` runs `benchmark --quick`.
- The `StanzaSharp` package, packed from `src/StanzaSharp` (user's decision, 2026-10-06).
  - It carries all ten assemblies (facade, Core, Nn, the processors) plus their XML docs: the facade's
    ProjectReferences are `PrivateAssets="all"`, and an `IncludeProjectReferences` target adds them.
  - Its only dependency is `Microsoft.Extensions.Logging.Abstractions`; no TorchSharp since 1.0.
- `StanzaSharp.Cuda` (1.0), packed from `src/StanzaSharp.Cuda` (assembly `StanzaSharp.Cuda.dll`): depends on
  `StanzaSharp` (same version) and managed `TorchSharp`; its other ProjectReferences are `PrivateAssets="all"` (their
  assemblies come with StanzaSharp). Users add `TorchSharp-cpu`/`-cuda-*` themselves. Readme: its `PACKAGE.md`. Its
  public API is `CudaBackend` (`Create(deviceIndex = 0, disableTf32 = false)`, `Cpu`). Publishing it needs the
  nuget.org Trusted Publishing policy to allow the ID (owner's step).
  - Every other project is `IsPackable=false` (Directory.Build.props, which also holds the shared
    package metadata and a `0.1.0-dev` default version).
  - Because of `PrivateAssets`, tests and samples reference the library projects they use directly.
  - `src/StanzaSharp/PACKAGE.md` is the package readme; `NOTICE` ships in the package.
- Models are downloaded only explicitly (user's decision): `ModelDownloader.DownloadAsync(dir)`, `stanzasharp download
  [DIR]` (the tool) or `StanzaSharp.Cli download [DIR]` (the same code).
  - Selective download (user's decision, 2026-10-06): `DownloadAsync(dir, processors)` / `download
    --processors LIST`.
  - Packages (user's decision, 2026-10-06: Stanza's package names as strings, so more can be added without
    API changes): `PipelineOptions.Package` (a package of `Pipeline.Packages[lang]`: `"default"`, `"default_fast"`, `"default_accurate"`, the treebank packages; anything else throws) and
    `DownloadAsync(dir, PipelineOptions)` / `download --package NAME`. The downloader takes the same options
    object as `Load`, so one object downloads and loads the same files; it ignores the other options.
    `Processors` is null by default: all the package's processors. A processor the package lacks throws.
  - `DownloadAsync(dir)` and `DownloadAsync(dir, processors)` (0.1.0 API) mean the English `default` package.
  - Languages (owner's decisions, 2026-10-13, docs/german-plan.md): `PipelineOptions.Language` (Stanza's code, default
    `"en"`; `Load` still takes the language folder); `DownloadAsync(dir, options)` honours it, `download --lang de`
    defaults to `models/stanza/<lang>`. German (`de`): `default`, `default_fast`, `default_accurate`, `combined`, `gsd`,
    `hdt` download and load. English also has the treebank packages (`combined`, `ewt`, ..., `eslspok`, `craft`, `genia`, `mimic`; 2026-10-15). Unknown languages
    and packages throw `ArgumentException` with the supported list. (German `default_accurate` was listed here as
  unsupported until #170 added it on 2026-10-18; ~100 more languages followed on 2026-10-30, see scope.md.)
  - `Pipeline.SelectModels` (language + package + processors → processor → model) and `Pipeline.SharedModels`
    (from `Pipeline.Dependencies`: each model's resources.json `dependencies`, since names don't tell them; German
    ner conll03 reads `fasttextwiki` + the charlms) are what both `Pipeline`'s constructor and `ModelDownloader.FilesFor` use,
    so they cannot drift apart.
  - Tables from resources.json (owner's decision, 2026-10-30, for more languages): `src/StanzaSharp/stanza_resources.json`
    (embedded, `StanzaResources`, read with `JsonNode` once) is Stanza 1.15.0's resources.json for the supported languages,
    `{"stanza_version", "<lang>": resources.json["<lang>"], ...}`, the file compare and benchmark hand Stanza.
    `Pipeline.Packages`, `Optional`, `Dependencies` and `ModelDownloader.Files` are built from it for the packages
    `Pipeline.SupportedPackages` names (hand-written: what is ported and tested) and the models of `Alternatives`
    (`Pipeline.SupportedModels`). Hand-written still: `SupportedPackages`, `Alternatives`, `Unread`, `Transformers`,
    `ModelDownloader.Revisions` and `Transformers`, `IdentityLemma`, `Requires`. A language's entry is a snapshot of
    its `Revisions` commit, never Stanford's latest resources.json (it follows their re-uploads; on 2026-10-30 it listed newer
    English pos and lemma models, with `alternate_md5`s, than en's commit had, which we then adopted). A new language: its entry from stanza-resources'
    `resources_1.15.0.json` of the day its `stanza-<lang>` commit is pinned, plus a `SupportedPackages` line and a
    `Revisions` line. `CompareCommandTests.StanzaResources_MatchModelDownloader` reads the file on its own and checks
    the derived tables against it, `StanzaResources_CoverEverySupportedLanguage` the languages and commits;
    `FilesFor_AreEnoughToLoadThePipeline`, `FileTableCoversEveryModelThePipelineLoads` and `LanguageTests` the file sets. CI downloads the packages per `tools/ci-models.sh` group (see "Model cache keys").
  - It fetches the `.pt` files from `huggingface.co/stanfordnlp/stanza-<lang>/resolve/<commit>/models/`
    (`ModelDownloader.Revisions`: en `d323be19`, de `a8ecd6c8`, multilingual `e9360ddf`) and checks each against the MD5
    from Stanza 1.15.0's resources.json (stanza-resources' `resources_1.15.0.json`), kept in `ModelDownloader.Files`. A
    commit, not the `v1.15.0` tag: on 2026-10-20 Stanford re-uploaded English's conll03 NER models (now on conll17,
    stanza#1698) and German's NER models (germeval2014 removed, germeval2014_charlm added, the dbmdz NER removed) and moved
    the tag. We pinned the old commits (`d7e0c740`, `6b928a48`; owner's decision 2026-10-22), then adopted the new models
    (#215, owner's decision 2026-10-24: these commits, new MD5s, resources and goldens). On 2026-10-28 and 2026-10-30
    Stanford re-uploaded English's pos (`combined_charlm`, `combined_nocharlm`, `combined_electra-large`) and lemma
    (`combined_nocharlm`, `combined_charlm`) models and moved the tag to `d323be19`; owner's decision 2026-10-30: adopt
    (en's entry from that day's resources_1.15.0.json, every English golden set reading pos or lemma regenerated; no code
    change: StanzaSharp ran them byte for byte as they were). About 2% of default's UPOS, 1.6% of lemmas and 1.9% of heads
    changed on the golden texts (default_fast about 3%); sentiment, NER, coref chains unchanged.
  - It keeps files that already match. `Pipeline.Load` never downloads.
  - Changing the Stanza version means new MD5s, new golden data and a new cache key.
- `tools/verify-package.ps1` packs a unique `0.0.0-verify.<time>` version and builds a fresh console app against it:
  plain, StanzaSharp.Cuda + `TorchSharp-cpu` and `StanzaSharpTrimNative`, running the README example with `Backend =
  CudaBackend.Cpu` and checking the parse; `-Managed`, only StanzaSharp, running ManagedCheck's Program.cs and failing
  if anything of TorchSharp or libtorch is in the output. It removes that version from the NuGet cache afterwards.
- Releases start from the CHANGELOG (user's decision, 2026-10-07; before, a pushed tag started them, and hand-made
  tags twice landed wrong). Merging a PR "CHANGELOG: x.y.z" (made by `python tools/changelog.py release x.y.z`, which writes
  `## [x.y.z] - date` from `changes/` and deletes those files) runs
  `.github/workflows/release.yml` (push to main touching CHANGELOG.md): its `version` job reads the first version
  heading and stops if tag `vx.y.z` exists; otherwise its `ci` job waits for main's ci.yml push run of the same commit
  (`gh run list --workflow ci.yml --commit $GITHUB_SHA --branch main --event push`, polled every minute, up to 170 min)
  and fails unless it succeeded (owner's decision, 2026-10-13; before, release.yml ran all of ci.yml a second time via
  `workflow_call`, and the two suites competing for runners made a release take 40+ min). A cancelled CI run (a newer
  push to main cancels it) or a failed one fails the release with a message: re-run that CI run, then the release run's
  failed jobs. Only if no such run turns up within 2 min does `checks` call ci.yml as before. Then it packs that version, pushes via NuGet
  Trusted Publishing (`NuGet/login@v1`, repo variable `NUGET_USER`), and only then creates the tag on that commit
  and the GitHub release (`gh release create --target`; a prerelease if the version has `-`). Never push release
  tags by hand. A failed publish is retried by re-running the run; `--skip-duplicate` keeps already-pushed packages.
  - Environment `prerelease` is for versions with `-`; `release` (approval required) for the rest. Both allow
    deployments from branch `main`.
  - One-time setup, done 2026-10-06:
    - nuget.org Trusted Publishing policies for both environments (owner bisforboman, repo
      StanzaSharp, workflow `release.yml`), added by the owner.
    - The repo variable `NUGET_USER` = `bisforboman`.
    - GitHub environment `release`: required reviewer bisforboman; deployments from branch `main` (and the old
      tag rule `v*`).
    - GitHub environment `prerelease`: no reviewers; branch `main` (and tag rule `v*-*`).
    - A 401 "No matching trust policy" on push means the nuget.org policy for that environment is
      missing or misnamed.
- `main` is protected like StyleBro since 2026-10-06 (user's decision): pull requests required (0
  approvals), required status checks (`build-test` and `golden` at first; all eight CI jobs now, see CI above),
  enforced for admins, auto-merge allowed,
  merged branches deleted. Every change goes through a PR. A new required CI job must also be added to the
  protection's required checks.
