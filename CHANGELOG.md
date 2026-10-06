# Changelog

All notable changes to the `StanzaSharp` NuGet package. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change the public API.

Output is verified against Python Stanza 1.15.0 and its English models (`ModelDownloader.StanzaVersion`).

## [Unreleased]

### Added
- Build warning `STANZA001` when the project resolves TorchSharp, `TorchSharp-cpu`/`TorchSharp-cuda-*` or a
  `libtorch-*` package in another version than StanzaSharp was built with (TorchSharp 0.107.0, libtorch 2.10.0).
- `<StanzaSharpTrimNative>true</StanzaSharpTrimNative>` leaves libtorch's Python bindings and test/mobile backends,
  which StanzaSharp never loads, out of build and publish output (CPU libtorch only): 35 MB less on Linux x64, 29 MB on macOS. The Docker
  sample uses it.
- The `stanzasharp` .NET tool (package `StanzaSharp.Tool`): `stanzasharp download [DIR] [--package NAME]
  [--processors LIST]` downloads the models without native libraries, e.g. in a Dockerfile.

## [0.3.0] - 2026-10-06

### Added
- Platform packages `StanzaSharp.Cpu.Linux`, `StanzaSharp.Cpu.Windows` and `StanzaSharp.Cpu.MacOS`. Each is
  StanzaSharp plus one platform's CPU libtorch, for deployments that don't want all three (as `TorchSharp-cpu`
  restores them). Each is verified in CI on its own OS.
- Platform package `StanzaSharp.Cpu.WindowsArm64`: Windows on Arm64, which `TorchSharp-cpu` doesn't cover. Verified
  in CI on a Windows Arm64 runner.
- Supported platforms in the README and package readme: Linux x64 (glibc), Windows x64 and Arm64, macOS on Apple
  Silicon. Alpine (musl), Linux Arm64 and Intel Macs are not supported: TorchSharp and libtorch have no builds for them.
- A sample Dockerfile (`samples/docker`) for a Linux x64 container on the .NET runtime image (plain or chiseled),
  verified in CI to give output identical to Stanza's.
- Bulk processing: `Pipeline.Process(IEnumerable<string>)` returns one `Document` per text, batching all their
  sentences together, like Stanza's `bulk_process`, and with identical output (which, as in Stanza, can differ
  slightly from processing each text alone: sentiment labels depend on their batch, and sentence ids continue
  across the documents). On 2,000 one-sentence texts it is 7.7× faster than one call per text (9× with `default_fast`).
- Pretokenized input: `Pipeline.Process(IEnumerable<IEnumerable<string>>)` takes sentences of tokens and skips the
  tokenizer, like Stanza's `tokenize_pretokenized=True`, with identical output.

### Changed
- Verified on Windows and macOS (Apple Silicon) in CI, besides Linux. On macOS, `brew install libomp` is
  required: TorchSharp-cpu's libtorch links OpenMP from Homebrew's path.

## [0.2.0] - 2026-10-06

### Added
- Stanza's English `default_fast` package: `PipelineOptions.Package = "default_fast"` (default
  `"default"`, also `Pipeline.DefaultPackage`). It runs tokenize, mwt, pos, lemma, depparse, sentiment and
  ner, with pos, depparse and ner models that have their own small character LSTMs instead of the large
  character language models. Output is identical to Python Stanza's `package='default_fast'`. On 8 CPU
  threads it is about 2.3× faster than the default package (whose output it does not match: these are
  different models).
  - An unknown package name throws `ArgumentException`, and so does a processor the package lacks, such as
    `constituency` in `default_fast`.
- `ModelDownloader.DownloadAsync(dir, PipelineOptions)` downloads exactly what `Pipeline.Load` reads with the
  same options (package and processors).
- The CLI's `--package NAME`, for running and for `download`.
### Changed
- `PipelineOptions.Processors` is now `string?` and null by default, meaning all processors of the package
  (for `default`, the same eight as before).
- The POS tagger no longer scores batch padding: pos is about 25% faster with the default package (16.1 →
  12.1 s on the 26k-word benchmark), with identical output.

## [0.1.0] - 2026-10-06

First stable release.

### Changed
- The public API is now only the pipeline and its results:
  - `Pipeline`, `PipelineOptions`, `CharlmCacheOptions` and `ModelDownloader`;
  - `Document`, `Sentence`, `Token`, `Word`, `Entity`, `Tree` and `Conllu`.

  The processor classes (`PosTagger`, `Tokenizer`, …), the shared model classes (`Pretrain`,
  `CharLanguageModel`, `CharlmCache`, `Weights`, …), `Checkpoint`, `SafeTensorFile` and `PyString` are
  now internal. Use `PipelineOptions.Processors` to run a subset of processors.
- `Entity.Build` and `Token.IsMwtCandidate` are internal.

### Added
- XML documentation for every public member.
- `samples/StanzaSharp.Example`, a commented tour of the API.
- Issue templates for bug reports and feature requests.

## [0.1.0-alpha.2] - 2026-10-06

### Added
- Named-entity recognition (`ner`, Stanza's OntoNotes model):
  - `Token.Ner` holds the BIOES tag;
  - `Sentence.Entities` and `Document.Entities` hold the spans;
  - CoNLL-U output gets `ner=` in MISC.
- Sentiment (`sentiment`): `Sentence.Sentiment` (0 negative, 1 neutral, 2 positive) and the `# sentiment`
  comment in CoNLL-U.
- `ModelDownloader.DownloadAsync(dir, processors)` downloads only the models those processors need.

### Changed
- The default pipeline runs all eight processors, in Stanza's order, like Stanza's English default:
  tokenize, mwt, pos, lemma, constituency, depparse, sentiment, ner.
- The full model download is about 600 MB.

## [0.1.0-alpha.1] - 2026-10-06

First release. Stanza's English pipeline in .NET, running Stanza's own models through TorchSharp, with
output byte-identical to Python Stanza on the golden test data.

### Added
- Processors: tokenize (with sentence splitting), mwt, pos (UPOS, XPOS, features), lemma, depparse
  and constituency.
- `Pipeline.Load(dir, PipelineOptions)` with:
  - `Processors`;
  - `Device`, for CUDA;
  - `DisableTf32`, for GPU output identical to the CPU;
  - `CharlmCache`, to share character-model outputs between processors.
- Loads Stanza's original `.pt` model files directly, with no Python needed, through a restricted
  unpickler that never runs code.
- `ModelDownloader` fetches the models from Stanza's Hugging Face repository and checks their MD5s.
- `Conllu.Read` and `Conllu.Write`, in Stanza's CoNLL-U dialect.

[Unreleased]: https://github.com/bisforboman/StanzaSharp/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.2...v0.1.0
[0.1.0-alpha.2]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.1...v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/bisforboman/StanzaSharp/releases/tag/v0.1.0-alpha.1
