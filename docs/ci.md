# CI

How `.github/workflows/` test and publish StanzaSharp, and the decisions behind it (dated in place). Moved from CLAUDE.md on 2026-10-31.

- CI (`.github/workflows/ci.yml`, github.com/bisforboman/StanzaSharp; pushes to `main`, PRs and
  `workflow_call` from release.yml; `ubuntu-26.04`, `ubuntu-26.04-arm`, `windows-2025`, `macos-26`, `windows-11-arm` pinned).
  - Branch protection on `main` requires `build-test`, `golden`, `docker`, `alpine`, `linux-arm64` and the three
    `cross-os (...)` checks (checked 2026-10-12). `build-test` and `golden` stay plain jobs; a matrix would rename them
    (`build-test (ubuntu-26.04)`) and block every PR. There is no merge queue (GitHub offers it only to organization-owned
    repositories; this one is user-owned): a PR merges after its light run, and the full run is main's push.
  - `build-test` runs the suite without models, then packs StanzaSharp and StanzaSharp.Cuda against the last release's
    public API (package validation, 2026-10-31; see docs/decisions/packaging-and-releases.md).
  - golden-shard's and cross-os-shard's Test steps export `ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE,STRICT` on x64
    (round 26): libtorch otherwise picks MKL/ATen kernels by CPU, and on Intel AVX-512 runners the TorchSharp German
    `default_fast` UPOS logits drifted 1.2e-4 (AMD 6.1e-5). With them every x64 runner gives the same bits. Test
    environment only; the managed backend never needed it.
  - Docs-only pull requests (`docs/`, `*.md`, `LICENSE`, `NOTICE`): the `changes` job makes `golden`, `docker`,
    `alpine`, `linux-arm64` and `cross-os` run with their steps skipped, so the required checks pass in about a minute. No job is skipped
    outright: a skipped job would count as passed. Pushes to main and
    release runs always run everything; if `changes` fails, everything runs.
  - Runner budget (user's decision, 2026-10-09): **draft** PRs skip every heavy step like docs-only ones (drafts can't
    merge; `ready_for_review` starts the real run), so agents open stacked PRs as drafts and mark them ready once their
    base merges. **Ready** PRs run level `light`: `docker` and `alpine` run everything; `cross-os` and
    `linux-arm64` run only `PipelineTests`, `ManagedBackendTests` and `ManagedCheckTests` (70 tests, ~2.5 min locally)
    and no package checks. Pushes to main and release runs are level `full`.
  - TorchSharp smoke (owner's decision, 2026-10-10): `changes` also outputs `torch`. A ready PR whose files miss
    `src/StanzaSharp.Cuda/`, `src/StanzaSharp.Nn/`, `src/StanzaSharp/{BackendModels,PipelineBackend}.cs`,
    `Directory.Build.props` and `.github/workflows/` gets `torch=smoke`: golden runs every test except the
    TorchSharp-backend cases (DisplayName `managed: False`, `[Trait("Backend", "TorchSharp")]`), keeping those marked
    `[Trait("Smoke", "TorchSharp")]` (`PipelineTests.Process_ReproducesGoldenConlluExactly`,
    `InputModeTests.Pretokenized_MatchesGolden`: all processors of both packages) and verify-package's plain mode.
    So ConcurrencyTests runs only its managed cases there. Filter
    `(DisplayName!~managed: False&Backend!=TorchSharp)|Smoke=TorchSharp`: 238 of 282 tests, ~31 of 78 min summed.
    Everything else (drafts/docs-only aside) is `torch=full`. A new expensive TorchSharp-only test without a
    `managed` parameter needs `[Trait("Backend", "TorchSharp")]` to stay out of light runs.

    | event | code | level | torch | golden runs |
    |---|---|---|---|---|
    | draft PR | false | light | – | steps skipped, passes in ~1 min |
    | docs-only ready PR | false | light | smoke | steps skipped, passes in ~1 min |
    | ready PR, only `src/StanzaSharp.Pos/**` | true | light | smoke | managed + smoke, 4 shards |
    | ready PR touching `src/StanzaSharp.Cuda/**` | true | light | full | everything, 4 shards |
    | push to main, release fallback (`workflow_call`) | true | full | full | everything, 4 shards |
    | `changes` failed | (empty) | (empty) | (empty) | everything (anything not clearly false/smoke) |
  - `concurrency` (`${{ github.workflow }}-${{ github.ref }}`): a newer push to the same PR or to main cancels the older
    CI run (a release waiting on that run then fails; see Packaging and releases). release.yml's fallback call gets the
    group `Release-refs/heads/main` (`github.workflow` is the caller's name), so it and main's own CI run don't cancel
    each other, and it is never cancelled.
  - `golden` (owner's decision, 2026-10-10) is an aggregate: it `needs` the matrix job `golden-shard` (1, 2, 3, 4), runs
    `if: !cancelled()` and fails unless `needs.golden-shard.result` is `success` (a skipped or failed shard fails it),
    so the required check keeps its name. The shards split the test project by class, balanced on the golden trx
    (full run, minutes summed; xunit runs a class's tests one at a time and ConcurrencyTests after everything else):
    - 2: `ConcurrencyTests.ConcurrentCalls_*` + FastPackage, GcPressure, ManagedBackend, Tokenizer: 87 tests, 17.9 min;
      plus CompareCommand and BenchmarkCommand (2026-10-12: the shard that finished first), with Python Stanza
      (`STANZASHARP_PYTHON` from setup-python; installing tools/requirements.txt adds ~40 s, the Python runs ~1 min).
    - 3: Pipeline, Ner, Sentiment, InputMode, Depparse, NoSsplit: 72 tests, 45.5 min (parallel; Pipeline 10.3 alone).
    - 4 (2026-10-18): the `Transformer*` classes (`SHARD_FOUR`), the only shard that downloads the transformer. On a
      pull request only with the transformer's paths (above); the matrix is `fromJSON(needs.changes.outputs.shards)`, so
      a shard left out is absent, not skipped (a skipped shard would fail `golden`).
    - 1: the complement of 2, 3 and 4 (ConcurrencyTests' cancellation cases and every other class, new ones too):
      123 tests, 14.6 min. So the union is always the whole suite, each test once (checked with `--list-tests`).
    - Expected wall per shard on full: ~12–15 min (was 38–44 for one job); light: ~4–6 min.
    Each shard, on a model-cache miss, downloads the models with the C# `ModelDownloader` (via the
    CLI) and converts them with CPU torch. The models are cached on the en group's file list (see "Model cache keys"),
    `tools/requirements.txt` and `tools/stanza_convert.py`; shard 1 writes the cache, shards 2–3 (like docker and
    alpine) only restore it. Each shard fails if any test is skipped
    (`outcome="NotExecuted"` in the trx; the trx `notExecuted` counter stays 0 for skips) or none ran. Shard 1 runs
    `tools/verify-package.ps1` against `models/stanza/en` twice: plain (StanzaSharp + TorchSharp-cpu, TorchSharp
    backend, `StanzaSharpTrimNative`; also on torch=smoke) and `-Managed` (only StanzaSharp, the default backend, no
    libtorch), and `verify-tool.ps1`. Artifacts: `test-results-golden-<shard>`.
  - German (owner's decision, 2026-10-13): one `models-de-treebanks-<hash of the de group's file list>` cache holding only Stanza's
    German `.pt` files (`models/stanza/de`, default, default_fast, gsd and hdt, about 1.3 GB; no conversion: `Repo.ModelsFor("de")`
    falls back to models/stanza/de), written by golden-shard 1 and restored by shards 2–3, downloaded with the CLI
    (`download models/stanza/de --lang de --package ...`) on a miss. Only the golden shards run German, so their no-skip
    check covers it. cross-os and linux-arm64 stay English: there the German rows skip, and their no-skip checks exempt
    test names matching `lang: "de"` or `German` (trx-escaped `lang: &quot;de&quot;`).
  - Model cache keys (owner's decision, 2026-10-16): each model cache's key hashes exactly the files and MD5s it holds,
    not `ModelDownloader.cs`, so a new or changed MD5 rotates only the caches holding that file. `tools/ci-models.sh` defines
    each group once as lines of `download` arguments: `en` (default + default_fast: `models-en-<hash>-<hash of
    requirements.txt and stanza_convert.py>` and `models-pt-<os>-<hash>`), `en-treebanks` (the treebank packages and
    `Pipeline.Alternatives`' English models, minus en's files) and `de` (all German). Each job's "Model cache keys" step
    builds the Cli and runs `tools/ci-models.sh keys`, which hashes the Cli's `download ... --list` output (`path md5`
    lines, no network) per group; the downloads on a miss are `tools/ci-models.sh download GROUP`, and the treebank
    cache's paths are the group's file list. A new model goes into a group line there, nowhere else in ci.yml.
  - English treebank packages (owner's decision, 2026-10-15): one `models-en-treebanks-<hash>` cache
    of only their own `.pt` files (the en-treebanks group's files that models-en lacks, about 2.8 GB with the domain NER models),
    restored into models/stanza/en next to the default package's, written by golden-shard 1 and restored by shards 2–3, downloaded with
    the CLI (`download models/stanza/en --package ...`) on a miss. So that models-en doesn't take them in, shard 1 restores
    models-en with `actions/cache/restore` and saves it with `actions/cache/save` right after the convert step, before the
    treebank models arrive. cross-os and linux-arm64 don't have them: their no-skip checks exempt `TreebankPackageTests`.
    The same cache holds `Pipeline.Alternatives`' English files (explicit paths, 501 MB: ptb3_charlm, the four NER
    models, fasttextcrawl), and models-de-treebanks German ner conll03 (51 MB); their rows (`ModelChoiceTests`) are
    exempt elsewhere the same way.
  - The treebanks' `_nocharlm` taggers and parsers and `_charlm` tokenizers and lemmatizers (owner's decisions, 2026-10-16):
    their own `models-treebank-choices-<key>` cache (tools/ci-models.sh `en-treebank-choices` + `de-treebank-choices`, their
    files only, explicit paths `treebank_choices_paths`; English 965 + 74 MB, German 269 + 41 MB), so adding them rotated no
    other key; written by golden-shard 1, restored by 2–3. `CharlmInputTests` (ewt's `_charlm` models) is exempt from the
    no-skip checks elsewhere, like `ModelChoiceTests`. So that
    models-de-treebanks (path `models/stanza/de`) doesn't take the German ones in, shard 1 restores it with
    `actions/cache/restore` and saves it with `actions/cache/save` right after the German download.
  - `cross-os (windows-2025)` / `cross-os (macos-26)` (Apple Silicon; `TorchSharp-cpu` brings
    `libtorch-cpu-osx-arm64`): build, full suite with `STANZASHARP_MODELS` = `models/stanza/en` (the
    `.pt` files, no Python or conversion), the same no-skip check, and both `verify-package.ps1` runs. Models are
    cached per OS on the en group's file list only; no NuGet cache, to stay inside the 10 GB cache budget.
    - Sharded like golden (owner's decision, 2026-10-13): the matrix job `cross-os-shard` (os × shard 1–3) uses golden's
      first three filters (`SHARD_TWO`/`SHARD_THREE` in the workflow `env`; shard 1 the complement, without `SHARD_FOUR`'s
      `Transformer*` classes, which only golden shard 4 can run). As one job the tests took
      29 / 40 / 61 min (windows-2025 / macos-26 / windows-11-arm), mostly ConcurrencyTests' two halves running alone
      at the end (7.6+7.6, 11.8+11.4, 25.4+17.6 min); the shards run them side by side. Each shard has every step
      (libomp, model cache, no-skip check, trx `test-results-<os>-<shard>`); the package checks run only in shard 3.
      Light runs have one shard per OS (`fromJSON` on `level`) with the light filter: PipelineTests doesn't split.
      3 macOS shards + benchmark.yml's 2 macOS jobs = the 5 concurrent macOS jobs free runners allow.
    - The required checks `cross-os (<os>)` are an aggregate matrix job over the three OSes (`if: !cancelled()`). It
      passes at once if `needs.cross-os-shard.result` is success; otherwise (that result covers all nine) it lists this
      run's jobs through the API and passes only if every `cross-os-shard (<os>, *)` job concluded success, so a
      failure fails only its own OS's check. No shard found fails.
    - macOS first needs `brew install libomp` for the TorchSharp backend's tests. `libtorch_cpu.dylib` from
      `libtorch-cpu-osx-arm64` 2.10.0 links `/opt/homebrew/opt/libomp/lib/libomp.dylib` by absolute path, not the
      `libomp.dylib` it ships. Without it TorchSharp reports only "doesn't contain a reference to
      libtorch-cpu-osx-arm64"; `otool -L` on the dylib shows the real cause. TorchSharp-backend users need it too
      (README, PACKAGE.md); the default managed backend doesn't.
  - `docker`: `tools/verify-docker.sh MODEL_DIR BASE...` packs StanzaSharp into `samples/docker/feed` (with a
    generated nuget.config), builds the sample per base image (runtime:10.0 and 10.0-noble-chiseled; `RID`
    linux-x64, linux-musl-x64 for a base named `*alpine*`), fails if any `libtorch*`/`libc10*`/`libgomp*` is in the
    image, and `cmp`s the container's CoNLL-U for corpus.txt with pipeline.conllu. Models from golden's cache via
    `actions/cache/restore` (no new entry; downloaded on a miss); no Docker layer cache (budget).
  - `alpine`: the same on `runtime:10.0-alpine` (musl). It replaced `alpine-experiment.yml`, whose verdict (PR #10)
    still holds for the TorchSharp backend: libtorch needs glibc-only symbols gcompat lacks (`__*_chk`, `backtrace`,
    `fcntl64`, `__res_init`, `pthread_attr_setaffinity_np`), so Alpine is managed-only.
  - `cross-os (windows-11-arm)`: Windows on Arm64 (GA runner label for public repos). It skips the plain
    `verify-package.ps1` (TorchSharp-cpu has no Arm64 libtorch) and runs `-Managed`.
  - `linux-arm64` (`ubuntu-26.04-arm`): no libtorch or LibTorchSharp exists for linux-arm64, so it builds with
    `-p:StanzaSharpNoLibTorch=true` (the Tests drop TorchSharp-cpu) and runs `--filter "DisplayName~managed: True|Backend=Managed"`:
    the managed cases of the both-backend theories plus tests marked `[Trait("Backend", "Managed")]` (they must need
    no libtorch: check with that build and an empty `NUGET_PACKAGES` locally). Then the no-skip check and
    `verify-package.ps1 -Managed`. Models cached like cross-os (`models-pt-Linux-…`).
  - `nightly.yml` (owner's decision, 2026-10-18): 01:00 UTC on main, and `workflow_dispatch`. `ci` calls ci.yml
    with `golden_shards: '[4]'` (every other ci.yml job skips on that input), or with it empty, all of ci.yml, when
    `main-ci` finds no ci.yml push run on the commit (an auto-merge's push starts none; owner's decision, 2026-10-29);
    `fuzz` runs `CheckpointFuzzTests` with
    `STANZASHARP_FUZZ_MODELS=all`, 3,000 iterations and the run number as seed, then `tools/text_fuzz.py run` (tokenize,mwt
    1,000 texts; default and default_fast 60 each) with Python Stanza (a decision that float64 Stanza makes as StanzaSharp
    does is a float32 near-tie, reported but not failing, owner's decision 2026-10-24 after #213); `benchmark` calls benchmark.yml `quick` without
    the avx512 jobs (`no_avx512`). Any failure comments on the open issue "Nightly run failed" (`report`, created if
    missing; closed by a person). Never required; release.yml waits for main's push run only, so it never sees these.
  - `benchmark.yml` (owner's decision 2026-10-10): `workflow_dispatch` (inputs `lang`, `package`, `quick`) and
    `workflow_call` (nightly.yml), never on pushes/PRs, never required; free standard runners only (ubuntu-26.04, ubuntu-26.04-arm, windows-2025,
    windows-11-arm, macos-26). Each job builds the tool, restores the `models-pt-<os>` cache (keyed like ci.yml's, by `tools/ci-models.sh keys`) (`lang=de`: golden-shard 1's
    `models-de-treebanks-…`, Linux only; restore only; downloads the package on a miss), installs Python 3.12 + `tools/requirements.txt` from the CPU wheel index (continue-on-error: without it the
    block has StanzaSharp only), runs `stanzasharp benchmark --json`, writes the block to its summary and artifact
    `benchmark-<os>`. `summary` combines them with `tools/benchmark_table.py` into `benchmarks.md` (summary + artifact).
    The tool is built with `-p:Version=<git describe>` (full-history checkout), so the block names e.g.
    `1.0.0-67-g97089bb` instead of Directory.Build.props' `0.1.0-dev`. First run: docs/performance.md "Run it yourself".
  - Keep paths forward-slash and file names case-exact (Linux). The root `.gitattributes` keeps sources
    LF; `tests/golden/.gitattributes` pins golden files (`eol=lf`, `binary`, `validation_whitespace.txt`
    `-text`), so Windows checkouts with `core.autocrlf=true` stay byte-exact. Bash steps run under
    `shell: bash` on every OS (Git Bash on Windows).
