# Changelog

All notable changes to the `StanzaSharp` NuGet package. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change the public API.

Output is verified against Python Stanza 1.15.0 and its English models (`ModelDownloader.StanzaVersion`).

## [Unreleased]

### Added
- `stanzasharp compare FILE [--package NAME] [--processors LIST] [--models DIR] [--python PATH]` (`StanzaSharp.Tool`):
  runs Python Stanza 1.15.0 and StanzaSharp on your own text with the same, MD5-checked model files and reports
  whether the CoNLL-U output is identical, or the first difference (sentence, both lines), with sentence and word
  counts and both sides' load and processing times. Offsets after characters outside the BMP are converted to code
  points (Stanza's) before comparing. Exit code 0 identical, 1 different, 2 usage or setup error.

### Changed
- Less memory on long texts: the returned `Document` takes about 19% less (482 instead of 594 bytes per word with all
  eight processors: tokens hold room for one word, `SpaceAfter` and equal feats strings are shared, tree children are
  arrays), and NER frees the charlm outputs the tagger kept for it (up to 256 MB) as it goes. One call on 670,000
  words peaked at 3.0 GB instead of 3.6 GB. Outputs are unchanged. `Tree.Children` of a parsed tree is now an array
  (still an `IReadOnlyList<Tree>`).
- README: guidance for very long texts, including what splitting a text into parts changes (sentiment labels).

## [1.0.0] - 2026-10-10

1.0: the `StanzaSharp` package is fully managed, with no native or TorchSharp dependency; GPU and TorchSharp users add
`StanzaSharp.Cuda`. From here on the public API follows Semantic Versioning: breaking changes only in a new major version.

### Changed
- **The TorchSharp backend moved to its own package, `StanzaSharp.Cuda`.** `StanzaSharp` no longer depends on
  TorchSharp at all: the default install has no native code and no TorchSharp assemblies. For a GPU, or libtorch on the
  CPU, add `StanzaSharp.Cuda` and the native libtorch (`TorchSharp-cuda-windows`, `TorchSharp-cuda-linux` or
  `TorchSharp-cpu`):
  - `PipelineBackend.TorchSharp` → `CudaBackend.Cpu`.
  - `PipelineOptions { Device = torch.CUDA, DisableTf32 = true }` → `PipelineOptions { Backend = CudaBackend.Create(disableTf32: true) }`
    (`deviceIndex` picks another GPU).
  - `PipelineOptions.Device` and `PipelineOptions.DisableTf32` (obsolete since 0.5) are removed.
  - The `STANZA001` version check and `StanzaSharpTrimNative` come with `StanzaSharp.Cuda` now.
- Managed backend: each LSTM step now sums its recurrent matrix product in blocks of 128 terms, then adds the input
  projection, instead of one long chain. This halves the charlm's rounding error (now as close to float64 as TorchSharp)
  and the worst-case score gap between the backends on the golden parses (constituency margins: 7.3e-5 → 4.0e-5), at
  about 3–4% of the managed charlm's time. Outputs are unchanged.

## [0.6.0] - 2026-10-10

### Added
- `PipelineOptions.MaxConcurrentCalls` (opt-in; null, the default, means no limit): at most this many `Process` calls
  run on the pipeline at once, and further callers wait their turn (the wait honours the call's cancellation token).
  On the managed backend concurrent calls on short texts evict each other's model weights from the CPU cache, so with
  many callers 2 gives the most throughput (8 callers, one sentence each: 23.6 calls/s unlimited, 36.4 with one at a
  time). Output never depends on it.

### Changed
- Faster `Process` on a single sentence, the usual call of a service, on the managed backend (`default` package): the
  processors now share more of the character language models' work for a one-sentence text (the dependency parser
  continues the tagger's backward pass; NER reuses sentiment's on multi-word tokens). The output is the same bit for
  bit. One call on a 5–25-word sentence takes 27.9 ms instead of 30.0 (p90 44.6 instead of 50.7) on 8 threads, and
  83 ms instead of 94 (p90 141 instead of 170) on 1 thread, on a Ryzen 7 5800X; Python Stanza takes 189 and 319 ms.
  Details, and guidance on bulk processing and concurrent callers, in docs/performance.md (round 7).
- The TorchSharp backend's code moved into its own assembly, `StanzaSharp.TorchSharp.dll`, which the `StanzaSharp`
  package carries next to the others. Nothing else changes; this prepares the opt-in `StanzaSharp.Cuda` package of 1.0.
- The managed backend is about 30% faster on Arm64 (Windows on Arm, Linux Arm64, Apple Silicon): its matrix kernel
  now multiplies by vector element instead of broadcasting each value first, reaching 86–89% of the NEON peak instead
  of 58–60% on a Neoverse N2. Results are bitwise unchanged.

## [0.5.1] - 2026-10-09

### Fixed
- Loading Stanza's `.pt` files from a slow filesystem, such as a Docker Desktop bind mount of a Windows folder, took
  minutes instead of seconds (issue #48), on both backends. The checkpoint reader asked the file for its length once
  per pickle opcode, millions of `fstat` calls for the pretrain's vocabulary, each about 0.3 ms on such a mount. It
  now reads the length once per pickle. Loading `tokenize,mwt,pos,constituency` from a bind mount on a busy machine
  took 13–26 minutes and now takes 8–9 s; from the container's own disk, 12 s became 6 s.

## [0.5.0] - 2026-10-09

### Added
- `PipelineOptions.Backend` and `PipelineBackend`: the implementation that runs the models.
  `PipelineBackend.Managed` (the new default) or `PipelineBackend.TorchSharp` (libtorch, as up to 0.4). Both give the
  same tags, lemmas, parses, entities and sentiment.
- Alpine (musl) and Linux on Arm64 run StanzaSharp now, on the managed backend, and CI checks both: the Docker sample
  on `runtime:10.0-alpine` reproduces the golden CoNLL-U byte for byte, and Linux Arm64 runs the managed tests and the
  package check.

### Changed
- **The models run on a managed backend by default**: C# kernels using the CPU's SIMD instructions (AVX2/FMA on x64,
  NEON on Arm64), with no libtorch or other native library. `dotnet add package StanzaSharp` is the whole install;
  `TorchSharp-cpu` is needed only for `PipelineBackend.TorchSharp`. The output is byte-identical to the TorchSharp
  backend and to Python Stanza on all the golden data (both packages). On a Ryzen 7 5800X with 8 threads all eight
  processors run about 2.2x faster (26,264 words: 25.9 s against 56.4 s); on 1 thread every processor is faster too,
  by 13–52%. Memory: one `Process` call on 8,000 words peaks at 2.5 GB instead of 3.8–4.3 GB; loading peaks at about
  850 MB (TorchSharp: 820 MB). Details in docs/backends.md.
- `Threads` sets the managed backend's thread pool (process-wide, shared by concurrent `Process` calls) or, on
  TorchSharp, libtorch's intra-op threads as before. A managed pipeline never calls into libtorch, so `Load` no longer
  changes libtorch's thread count unless TorchSharp is selected.
- The Docker sample references only `StanzaSharp`: no libtorch in the image, which now works on any .NET 10 runtime
  image, including chiseled and Alpine. The image is 209 MB on `runtime:10.0` (was 707 MB) and 103 MB on
  `runtime:10.0-noble-chiseled` (was 601 MB); 100 MB on `runtime:10.0-alpine`.
- Cancellation inside the POS tagger and the sentiment classifier is faster: the token is now also checked inside
  each batch (the tagger: after each charlm pass or the character model, between the LSTM layers and between the
  heads; sentiment: after each charlm pass, after the LSTM and between the convolutions), so a call stops within one
  such step instead of up to a whole batch (0.8 s for POS, 1.2 s for sentiment on an 8-core machine). Any call now
  stops within about half a second there. Output is unchanged.

### Deprecated
- `PipelineOptions.Device` and `PipelineOptions.DisableTf32` are `[Obsolete]`; GPU support moves to a separate
  `StanzaSharp.Cuda` package, and both options leave the main package in 1.0, with `PipelineBackend.TorchSharp`. They
  keep working: with `Backend` unset, setting either selects the TorchSharp backend, so existing GPU code runs as
  before (with a warning). Setting them together with `Backend = PipelineBackend.Managed` throws an
  `ArgumentException`.

### Removed
- The platform packages `StanzaSharp.Cpu.Linux`, `.Windows`, `.WindowsArm64` and `.MacOS` are no longer published
  (0.4.2 is their last version): the default backend needs no libtorch. For the TorchSharp backend, reference
  `TorchSharp-cpu`, or one platform's `libtorch-cpu-<rid>` package in version 2.10.0 (`libtorch-cpu-linux-x64`,
  `-win-x64`, `-win-arm64`, `-osx-arm64`), next to `StanzaSharp`, and set `Backend = PipelineBackend.TorchSharp`.
  `STANZA001` and `StanzaSharpTrimNative` still apply to those packages.

### Fixed
- Tokens longer than the tokenizer's `max_seqlen` (200 characters for English, read from the model's config) now
  become `<UNK>`, as in Stanza; before, only tokens over 1,000 characters did, so tokens of 201–1,000 characters
  (long URLs) kept their text and could get different tags, lemmas, parses and entities than in Stanza. Only the text
  changes (offsets stay), in bulk input too; pretokenized input keeps its tokens, as in Stanza.

## [0.4.2] - 2026-10-07

### Changed
- On Linux (glibc), a `Process` call of at least 1,000 words now returns the memory it freed to the OS
  (`malloc_trim`). glibc kept it for reuse, so a long-running process stayed near its peak: with
  `tokenize,mwt,pos,constituency` the RSS after a 6,761-word call falls from 930 to 650 MB (after loading: 590 MB;
  Python Stanza: 890 MB). The trim takes 25–40 ms after such a call; output is unchanged. Other platforms are
  unchanged; on Windows, the environment variable `MIMALLOC_PURGE_DELAY=0` has the same effect for about 20% more
  time (see the README).
- The POS tagger no longer pads its batches: its input and its highway LSTM layers work on the real words only, so
  one long sentence in a batch of 250 no longer makes every sentence as long. With `tokenize,mwt,pos,constituency`
  the peak on 5,000 words falls from 1.57 to 1.02 GB (15,000 words: 1.66 to 1.25 GB), and pos is about 20% faster
  on such text. Tags, lemmas and parses are unchanged; logits move by at most 5e-5 (docs/performance.md, round 4).
- Cancellation inside the dependency parser is about 8x faster: the token is now also checked inside each 5,000-word
  batch (after each charlm pass, between the LSTM layers, between the scorers' chunks and between the sentences' tree
  decodes), so a call stops within about an eighth of a batch (one charlm pass or LSTM layer) instead of up to a
  whole batch (1.7 s on an 8-core machine, 6-7 s on CI runners). Output is unchanged.

## [0.4.1] - 2026-10-07

### Changed
- Loading Stanza's `.pt` files no longer reads each file whole and copies its tensors into a second buffer: tensors
  are read from the file as the models load, as for converted models. With `tokenize,mwt,pos,constituency` the load
  peak falls from 970 MB to 540 MB (1,090 to 510 MB with Server GC), and the memory no longer waits for a GC
  (issue #19).
- The POS tagger and the dependency parser free their padded batch tensors as soon as they are used. A batch padded
  to one long sentence peaked much higher: on 5,000 words of the validation texts the peak falls from 2.36 to
  1.57 GB (Python Stanza: 2.36 GB). Output is unchanged.
- The README describes what sets the memory peak and how to bound it (parts of about 1,000 words per call).

## [0.4.0] - 2026-10-06

### Added
- `PipelineOptions.Threads`: libtorch's intra-op thread count, set at `Pipeline.Load` (process-wide, like
  `torch.set_num_threads`). By default it is now capped at `Environment.ProcessorCount`, which .NET limits to a
  container's CPU quota; before, libtorch used one thread per physical core of the host.
- Cancellation: `Pipeline.Process(text, cancellationToken)`, and the same for bulk and pretokenized input. The token
  is checked between processors and between batches inside them; a canceled call throws `OperationCanceledException`,
  returns no document, and leaves the pipeline usable.
- `Pipeline.Process` is thread-safe: one pipeline can serve concurrent calls from many threads, with each call's output
  identical to a sequential call's (verified by a stress test). One loaded pipeline is about 950 MB of private memory
  (`default`) or 700 MB (`default_fast`), plus each call's working memory; see the README.
- `PipelineOptions.SplitSentences`: false gives Stanza's `tokenize_no_ssplit`, one sentence per paragraph, with output
  identical to Stanza's, also in bulk.
- `Processor` constants (`Processor.Tokenize`, ..., `Processor.Ner`) for the processor list.
- `PipelineOptions.VerifyChecksums`: checks every model file against Stanza's published MD5 before loading.
  Only Stanza's `.pt` files have checksums; with converted models it throws.
- `PipelineOptions.Logger` (`ILogger`): model load times and the thread count at Information, each processor's time
  per call at Debug. StanzaSharp now depends on `Microsoft.Extensions.Logging.Abstractions` (10.0.0, which brings
  `Microsoft.Extensions.DependencyInjection.Abstractions`; about 130 KB of assemblies, already in ASP.NET Core).
- Build warning `STANZA001` when the project resolves TorchSharp, `TorchSharp-cpu`/`TorchSharp-cuda-*` or a
  `libtorch-*` package in another version than StanzaSharp was built with (TorchSharp 0.107.0, libtorch 2.10.0).
- `<StanzaSharpTrimNative>true</StanzaSharpTrimNative>` leaves libtorch's Python bindings and test/mobile backends,
  which StanzaSharp never loads, out of build and publish output (CPU libtorch only): 35 MB less on Linux x64, 29 MB on macOS. The Docker
  sample uses it.
- The `stanzasharp` .NET tool (package `StanzaSharp.Tool`): `stanzasharp download [DIR] [--package NAME]
  [--processors LIST]` downloads the models without native libraries, e.g. in a Dockerfile.

### Changed
- `Sentence.Text` documents that every `Process` overload sets it; it is null only for CoNLL-U read without `# text`.

### Fixed
- A possible crash or wrong result when a garbage collection ran during a model call, more likely with concurrent
  calls and on Arm64: TorchSharp frees temporary Scalars (such as the hidden `alpha = 1` of a tensor addition) in a
  finalizer, which could run while libtorch was still reading them. StanzaSharp now keeps every Scalar it passes alive.

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

[Unreleased]: https://github.com/bisforboman/StanzaSharp/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.6.0...v1.0.0
[0.6.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/bisforboman/StanzaSharp/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/bisforboman/StanzaSharp/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/bisforboman/StanzaSharp/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.2...v0.1.0
[0.1.0-alpha.2]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.1...v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/bisforboman/StanzaSharp/releases/tag/v0.1.0-alpha.1
