# Changelog

All notable changes to the `StanzaSharp` NuGet package. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change the public API.

Output is verified against Python Stanza 1.15.0 and its English models (`ModelDownloader.StanzaVersion`).

## [Unreleased]

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

[Unreleased]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.2...v0.1.0
[0.1.0-alpha.2]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.1...v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/bisforboman/StanzaSharp/releases/tag/v0.1.0-alpha.1
