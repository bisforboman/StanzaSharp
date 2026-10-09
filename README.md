# StanzaSharp

C# port of Stanza's English inference pipeline (tokenize, mwt, pos, lemma, depparse, ner, sentiment, constituency).
It runs Stanza's own pretrained models in plain .NET code, with no native dependencies. On the golden test corpus, the
output is byte-identical to Python Stanza 1.15.0.
See CLAUDE.md for scope, layout, decisions and build order.

## Install

```
dotnet add package StanzaSharp
```

That is the whole install. Requires .NET 10. The models run on StanzaSharp's **managed backend**: C# kernels using the
CPU's SIMD instructions (AVX2 or Arm64 NEON where present), with no libtorch or other native library. On the full
pipeline it is about twice as fast as the TorchSharp backend on 8 threads and peaks lower in memory (see
[Performance](#performance)), and its output is the same, byte for byte.

**The TorchSharp backend** (libtorch) is still there in 0.5 for those who want it; it is the way to a GPU (see
[GPU](#gpu)) and leaves the main package in 1.0. Select it and add the native libtorch yourself:

```
dotnet add package TorchSharp-cpu
```

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Backend = PipelineBackend.TorchSharp });
```

- `TorchSharp-cpu`'s version must match the `TorchSharp` version StanzaSharp depends on (0.107.0); otherwise the build
  warns with `STANZA001`. Silence it with `<NoWarn>STANZA001</NoWarn>`.
- `TorchSharp-cpu` restores libtorch for Linux x64, Windows x64 and macOS (about 265 MB of downloads). To restore one
  platform's only, reference the matching `libtorch-cpu-<rid>` 2.10.0 package (`libtorch-cpu-linux-x64`,
  `-win-x64`, `-win-arm64`, `-osx-arm64`) instead; on Windows on Arm64 that is the only way, since `TorchSharp-cpu` has
  no Arm64 libtorch. (The `StanzaSharp.Cpu.*` platform packages did this up to 0.4 and are no longer published.)
- On macOS (Apple Silicon), also run `brew install libomp`: libtorch loads OpenMP from Homebrew's path.
- `<StanzaSharpTrimNative>true</StanzaSharpTrimNative>` in your project drops the libtorch files StanzaSharp never
  loads from build and publish output: the Python bindings (`libtorch_python`, `libshm`) and test and mobile backends
  (`libtorchbind_test`, `libjitbackend_test`, `libbackend_with_compiler`, `libaoti_custom_ops`, `libnnapi_backend`).
  That is 35 MB less on Linux x64 (a `-r linux-x64` publish goes from 503 to 468 MB) and 29 MB on macOS; the Windows
  libtorch packages ship none of them. Nothing TorchSharp loads links to them. It applies to the CPU libtorch only;
  CUDA builds are left whole.

Smaller models help either way: `Package = "default_fast"` and downloading only the processors you use.

### Supported platforms

The managed backend runs wherever .NET 10 runs. These run it in CI, against the golden data:

| Platform | Managed backend (default) | TorchSharp backend |
|---|---|---|
| Linux x64 (glibc: Ubuntu, Debian, RHEL, ...) | Yes, full suite in CI | Yes: `TorchSharp-cpu` or `libtorch-cpu-linux-x64` |
| Windows x64 | Yes, full suite in CI | Yes: `TorchSharp-cpu` or `libtorch-cpu-win-x64` |
| Windows on Arm64 | Yes, full suite in CI | Yes: `libtorch-cpu-win-arm64` only (`TorchSharp-cpu` has no Arm64 libtorch) |
| macOS on Apple Silicon | Yes, full suite in CI | Yes, after `brew install libomp`: `TorchSharp-cpu` or `libtorch-cpu-osx-arm64` |
| Linux Arm64 (glibc) | Verified in CI: the managed tests and the package | No: no `libtorch-cpu-linux-arm64` package, and TorchSharp has no `linux-arm64` native layer |
| Alpine and other musl Linux | Verified in CI: the Docker check on `runtime:10.0-alpine` | No: libtorch and TorchSharp are built for glibc only. Not even with `gcompat`: libtorch needs glibc-only symbols (`__memcpy_chk`, `backtrace`, `fcntl64`, …) it doesn't provide |
| macOS on Intel (x64) | Not tested | No: TorchSharp has no `osx-x64` native layer, and `libtorch-cpu-osx-x64` stops at 2.2 |

GPU: see [GPU](#gpu) (CUDA on Windows and Linux x64, TorchSharp backend).

### Docker

Any .NET 10 runtime image works: the image needs nothing but the .NET runtime, so `mcr.microsoft.com/dotnet/runtime:10.0`
(Ubuntu), the chiseled `runtime:10.0-noble-chiseled` and Alpine's `runtime:10.0-alpine` (publish with
`-r linux-musl-x64` there) all do. [samples/docker](samples/docker/Dockerfile) is a small app that reads text on stdin
and writes CoNLL-U:

```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY . .
RUN dotnet publish -c Release -r linux-x64 --no-self-contained -o /app

FROM mcr.microsoft.com/dotnet/runtime:10.0
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "StanzaSharpDocker.dll", "/models"]
```

```
docker build -t stanzasharp-sample samples/docker
docker run -i --rm -v /path/to/models/stanza/en:/models:ro stanzasharp-sample < input.txt
```

Mount the models rather than copying them into the image: they are about 600 MB, several times the image itself
(DOCKER_SIZES). CI builds this image on the plain, chiseled and Alpine base images, checks that no libtorch is in it,
and that its output is byte-identical to the golden file. (TorchSharp's 2 MB `LibTorchSharp` interop library is in
the publish output, as StanzaSharp still references TorchSharp; the managed backend never loads it.)

To download the models in a Dockerfile (on the SDK image):

```dockerfile
RUN dotnet tool install --tool-path /tools StanzaSharp.Tool && /tools/stanzasharp download /models --processors tokenize,mwt,pos
```

## Models

Download Stanza's English models (about 600 MB, from Stanza's Hugging Face repository) once. Every file is
checked against its published MD5; `Pipeline.Load` itself never touches the network.

```csharp
await ModelDownloader.DownloadAsync("models/stanza/en");                            // all, ~600 MB
await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Package = "default_fast" });
```

or from the command line with the `stanzasharp` tool (package `StanzaSharp.Tool`, no native libraries):
`dotnet tool install -g StanzaSharp.Tool`, then `stanzasharp download models/stanza/en`
(add `--processors LIST` for a subset, `--package default_fast` for that package). In this repository,
`dotnet run --project samples/StanzaSharp.Cli -- download models/stanza/en` does the same.
`DownloadAsync(dir, options)` fetches exactly what `Pipeline.Load(dir, options)` reads. A subset includes the models of the processors it requires and
the shared word vectors and character models when used.
A directory you already have from Python Stanza (`<processor>/<name>.pt`, e.g. `~/stanza_resources/en`)
works too.

## Usage

```csharp
using StanzaSharp;

using var nlp = Pipeline.Load("models/stanza/en");
var doc = nlp.Process("Barack Obama was born in Hawaii. He was elected president in 2008.");

foreach (var sentence in doc.Sentences)
{
    foreach (var word in sentence.Words)
        Console.WriteLine($"{word.Id}\t{word.Text}\t{word.Lemma}\t{word.Upos}\t{word.Feats}\t{word.Head}\t{word.Deprel}");
    Console.WriteLine(sentence.Constituency); // (ROOT (S (NP (NNP Barack) (NNP Obama)) ...))
}
```

`Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,mwt" })` runs only the listed processors; each
needs the ones before it. `PipelineOptions` also holds `Backend` (`PipelineBackend.Managed`, the default, or
`PipelineBackend.TorchSharp`; see [Install](#install)) and `CharlmCache`
(`IsEnabled`, `MaxWords`: the tagger's character-model outputs reused by the constituency parser and sentiment, at most
32,768 words / ~256 MB by default; output is identical either way).
The default is all eight processors, like Stanza's English default:
`tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner`. List fewer to run less; `ner` and `sentiment`
need only `tokenize`. NER gives named entities: `Token.Ner` holds the BIOES tag (`B-PERSON`, `O`, ...), and `sentence.Entities` /
`doc.Entities` the spans (`Text`, `Type`, `StartChar`, `EndChar`), from Stanza's OntoNotes model (18 types).
Sentiment gives `sentence.Sentiment`: 0 negative, 1 neutral, 2 positive.
`Conllu.Write(doc)` gives Stanza-style CoNLL-U, with `ner=` in MISC when NER ran.

### Many texts, and pretokenized text

```csharp
// Bulk: one Document per text, in order; sentences of all texts are batched together (Stanza's bulk_process).
List<Document> docs = nlp.Process(tweets); // any IEnumerable<string>

// Pretokenized: sentences of tokens; the tokenizer model is skipped and the tokens kept as given
// (Stanza's tokenize_pretokenized=True). doc.Text is the tokens joined by spaces: "Hello world . Bye ."
var doc = nlp.Process(new[] { new[] { "Hello", "world", "." }, new[] { "Bye", "." } });
```

Both match Python Stanza byte for byte. Bulk is much faster than one call per text for short texts: 7.7× on
2,000 one-sentence texts ([docs/performance.md](docs/performance.md#bulk-processing-many-short-texts)). Its output
can differ slightly from processing each text alone, exactly as in Stanza: the sentiment classifier sees its batch's padding (172 of 854
golden labels change), and sentence ids continue across the documents. Pretokenized tokens are never split into
multi-word tokens (`"don't"` stays one word), as in Stanza.

### Packages

`PipelineOptions.Package` picks one of Stanza's English packages by its Stanza name:

- `"default"` (the default): all eight processors above.
- `"default_fast"`: tokenize, mwt, pos, lemma, depparse, sentiment and ner, without constituency. Its pos,
  depparse and ner models have their own small character LSTMs instead of the large character language
  models, so they are faster and need less memory, at slightly lower accuracy. Output is byte-identical to
  Python Stanza's `package='default_fast'`.

```csharp
var options = new PipelineOptions { Package = "default_fast" };
await ModelDownloader.DownloadAsync("models/stanza/en", options);
using var fast = Pipeline.Load("models/stanza/en", options);
```

An unknown package name throws, and so does listing a processor the package lacks (`constituency` with
`default_fast`). See [docs/performance.md](docs/performance.md) for the speed of both.

### Threads, concurrency and cancellation

```csharp
using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions
{
    Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos}", // or "tokenize,mwt,pos"
    Threads = 2,              // threads per operation; null: at most Environment.ProcessorCount
    VerifyChecksums = true,   // check the .pt files against Stanza's MD5s first
    Logger = loggerFactory.CreateLogger("StanzaSharp"), // load and processing times
});
var doc = nlp.Process(text, cancellationToken); // OperationCanceledException within about one batch
```

- **Threads.** On the managed backend `Threads` is the size of StanzaSharp's thread pool; on the TorchSharp backend it
  is libtorch's `set_num_threads`. Both are **process-wide**: the last pipeline loaded sets it. Null (the default)
  caps the current count at `Environment.ProcessorCount`, which .NET limits to a container's CPU quota, and keeps a
  lower count set earlier. (libtorch applies a new count to the loading thread and to threads that haven't run a
  tensor operation yet, so with TorchSharp load before processing starts.)
- **Concurrency.** `Process` is thread-safe: one pipeline can serve many threads, and each call's output is identical
  to a sequential call's. The models are only read; everything a call changes is its own. On the managed backend
  concurrent calls share the one pool: each call works its own part and idle threads help, so N callers don't start
  N teams. On the TorchSharp backend each call runs its operations on up to `Threads` threads of its own, so N
  callers on C cores do best with `Threads` about C / N. Throughput is bounded by the CPU, not the number of callers.
  Sharing one pipeline saves memory: a loaded pipeline takes about 0.8–1 GB (`default`; about 0.7 GB
for `default_fast`), and each call in flight adds
  its own working memory on top (a few hundred MB for a page of text, more for long documents).
- **Memory.** The numbers below were measured on the TorchSharp backend; the managed backend peaks lower (one call
  on 8,000 words: 2.5 GB against 3.8–4.3 GB, [docs/backends.md](docs/backends.md#constituency)). A call's peak is set by its largest batch more than by the length of the text: the tagger pads up to
  250 sentences to the longest one among them. With `tokenize,mwt,pos,constituency` (8 threads) loading peaks at about
  0.5 GB, and one call on 500, 5,000 and 15,000 words peaks at 0.65, 1.6 and 1.7 GB (Python Stanza: 0.75, 2.4 and
  2.5 GB). To bound it in a memory-limited container, call `Process` on parts of about 1,000 words, split at blank
  lines: on 15,000 words that peaks at 1.2 GB instead of 1.7 GB and takes about 15% longer. The annotations are the
  same; only sentence ids, offsets (each part's own) and the whitespace at the cuts differ. One call per paragraph
  peaks at 0.6 GB but is 5× slower. Bulk `Process(texts)` batches all texts together, so it peaks like one call.
  `CharlmCache` changes the peak by less than 50 MB up to its 32k-word cap (turning it off costs 30–40% more time),
  and neither the GC mode nor converting the models changes it much. Measurements: [docs/performance.md](docs/performance.md#results-round-3-memory-of-a-short-lived-process).
- **Memory between calls** (long-running services; measured on the TorchSharp backend, whose native allocator this
  is about: the managed backend's buffers live on the .NET heap). On Linux (glibc) a call of 1,000 words or more gives the
  memory it freed back to the OS (`malloc_trim`, 25–40 ms), so with `tokenize,mwt,pos,constituency` the RSS drops
  back to about 650 MB after each call (590 MB after loading; Python Stanza: 850–950 MB). No `MALLOC_*` setting is
  needed in a container, and `MALLOC_ARENA_MAX` does not help. On Windows libtorch's built-in allocator (mimalloc)
  keeps freed memory committed, so the working set stays near the peak (1.4–1.6 GB). The environment variable
  `MIMALLOC_PURGE_DELAY=0` makes it give memory back as it is freed (about 580 MB after a call, and a 370 MB lower
  peak) for about 20% more time. It must be set before libtorch loads: in the environment, or with
  `Environment.SetEnvironmentVariable` before the first TorchSharp call. The .NET GC heap is 100–180 MB, so GC
  settings matter little. Measurements: [docs/performance.md](docs/performance.md#results-round-4-memory-after-process-returns-linux-and-windows).
- **Cancellation.** Every `Process` overload takes a `CancellationToken`. It is checked between processors and
  between batches inside each (inside the tagger's, dependency parser's and sentiment classifier's 5,000-word batches
  too, between their stages), so a call stops within about half a second on an 8-core machine; no document is returned and
  the pipeline stays usable.
- **One sentence per paragraph.** `SplitSentences = false` is Stanza's `tokenize_no_ssplit`: the tokenizer still
  splits tokens, but each paragraph (text between blank lines) is one sentence. Bulk input works the same way;
  pretokenized input keeps its sentences, as in Stanza.
- **Checksums.** `VerifyChecksums` reads every model file once more before loading (about 1.3 s for `default`'s 700 MB, 0.8 s for `default_fast`, with the files in the OS cache) and throws an
  `InvalidDataException` naming a file that doesn't match. Only Stanza's `.pt` files have published checksums, so with
  converted models it throws instead of skipping the check.
- **Logging.** `Logger` gets each model's load time and the thread count at Information, and each processor's time
  per call at Debug. Without a logger nothing is measured.
- `Sentence.Text` is nullable because CoNLL-U read without `# text` has none, but `Process` always sets it.

`samples/StanzaSharp.Example` is a commented tour of the whole API: `dotnet run --project samples/StanzaSharp.Example`.

From the command line, this writes CoNLL-U for a file (or standard input):

```powershell
dotnet run --project samples/StanzaSharp.Cli -- input.txt
dotnet run --project samples/StanzaSharp.Cli -- --processors tokenize,mwt,pos input.txt
dotnet run --project samples/StanzaSharp.Cli -- --package default_fast input.txt
dotnet run --project samples/StanzaSharp.Cli -- --backend torch input.txt
```

## Development setup

The .NET solution needs only the .NET 10 SDK. Python 3 is used for the reference environment: Stanza
itself (to generate golden data), the optional converter to safetensors + JSON (slightly faster to load
than `.pt`), and the benchmark's Python side.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Models
```

`-Models` creates `tools\.venv` with Stanza, downloads the English models to `models\stanza` and
converts them to `models\converted\en`. Use `-Python` for the environment only, or no switch for just the
.NET solution. Add `-Cuda` for GPU libtorch (see GPU below).

## Performance

The managed backend runs all eight processors about 2.2x faster than the TorchSharp backend on 8 threads (26,264
words on a Ryzen 7 5800X: 25.9 s against 56.4 s) and peaks lower (2.3 GB against 2.5 GB; one `Process` call on 8,000
words: 2.5 GB in 11.5 s against 3.8–4.3 GB in 24 s), with byte-identical output. On 1 thread every processor is
faster too, by 13–52%. Loading takes about as long (1.1 s) and peaks at about 850 MB (TorchSharp: 820 MB). Details per
processor: [docs/backends.md](docs/backends.md#phase-2-progress).

On the TorchSharp backend, on 8 CPU threads the six-processor pipeline is about 1.7x faster than Python Stanza and peaks
at 2.7 GB of memory (Python: 4.3 GB); see [docs/performance.md](docs/performance.md). The `default_fast` package runs
its seven processors about 2.3x faster than the default's eight, and 1.5x faster than Python's `default_fast`. On a
machine busy with other work, fewer threads (`Threads`) are often faster.

## GPU

The GPU runs on the TorchSharp backend. `Pipeline.Load(dir, new PipelineOptions { Backend = PipelineBackend.TorchSharp,
Device = torch.CUDA })` runs every model on an NVIDIA GPU. Reference a CUDA libtorch package such as
`TorchSharp-cuda-windows` (several GB), in the same version as StanzaSharp's `TorchSharp` (0.107.0).
`Device` and `DisableTf32` are obsolete from 0.5: GPU support moves to a separate `StanzaSharp.Cuda` package
(`Backend = CudaBackend.Create(...)`), and both leave the main package in 1.0. Until then they keep working; setting
either selects the TorchSharp backend, and setting them with `Backend = PipelineBackend.Managed` throws.

For output identical to the CPU (and so to Python Stanza), turn off TF32, which libtorch enables for cuDNN by
default on Ampere and newer GPUs:

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Backend = PipelineBackend.TorchSharp, Device = torch.CUDA, DisableTf32 = true });
```

`DisableTf32` sets `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` to false. These
are process-wide torch settings, so it is opt-in and affects all TorchSharp code in the process. With TF32 off, the GPU matched
every golden file exactly on an RTX 3080; with TF32 on, 3 of 845 sentences differed (near-ties). Results
are deterministic from run to run either way. The pipeline is roughly 4x faster on an RTX 3080 than on 8 CPU
threads (Ryzen 7 5800X), mostly in pos and depparse; the constituency parser gains least. Details in
[docs/gpu.md](docs/gpu.md).

In this repository, set `STANZASHARP_CUDA=1` to build the Cli, Benchmark and Tests with
`TorchSharp-cuda-windows`; that also compiles `GpuTests`, which skip without a CUDA device:

```powershell
$env:STANZASHARP_CUDA = '1'
dotnet test tests/StanzaSharp.Tests --filter FullyQualifiedName~GpuTests --logger "console;verbosity=detailed"
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --device cuda --no-tf32
```

Unset it (and let `dotnet` restore again) to go back to the CPU build.

## Tests

```powershell
dotnet test
```

Tests that need models skip when `models/converted/en` (or, for the `.pt` loader tests,
`models/stanza/en`) is missing. To regenerate the golden data, run
`tools\.venv\Scripts\python tools\make_golden.py`.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests (.NET 10; Ubuntu 24.04 x64 and Arm64,
Windows x64 and Arm64, macOS).

- `build-test` builds the solution and runs the tests without models (the model tests skip).
- `golden` downloads the English models with `ModelDownloader` (through the CLI), converts them with
  `tools/stanza_convert.py` (CPU-only torch wheel), and runs the full suite (both backends). It fails if any test was
  skipped. Then `tools/verify-package.ps1` packs StanzaSharp and runs two fresh apps that reference the package: one
  with only StanzaSharp (the default install; it must reproduce the golden CoNLL-U with no libtorch loaded), one with
  `TorchSharp-cpu` on the TorchSharp backend (no `STANZA` build warning allowed, with `StanzaSharpTrimNative`). Then
  `tools/verify-tool.ps1` installs the `stanzasharp` tool from a local feed and downloads `tokenize,mwt` with it. The
  original and converted models are cached, keyed on `ModelDownloader.cs`, `tools/requirements.txt` and
  `tools/stanza_convert.py`.
- `docker` runs `tools/verify-docker.sh`: it packs StanzaSharp, builds [samples/docker](samples/docker/Dockerfile) on
  `mcr.microsoft.com/dotnet/runtime:10.0` and on its chiseled variant, and checks that no libtorch is in the image and
  that each container turns `corpus.txt` into `pipeline.conllu` byte for byte. It reports the image sizes in the job
  summary. `alpine` does the same on `runtime:10.0-alpine` (musl).
- `cross-os (windows-2025)`, `cross-os (macos-15)` (Apple Silicon) and `cross-os (windows-11-arm)` run the same
  full suite, the no-skip check and both `tools/verify-package.ps1` runs (on Windows Arm64 only the managed one:
  `TorchSharp-cpu` has no Arm64 libtorch), against Stanza's `.pt` files without converting them (no Python). Their
  models are cached per OS, keyed on `ModelDownloader.cs`.
- `linux-arm64` (`ubuntu-24.04-arm`): no libtorch exists for it, so the tests build without one and it runs those
  that need none (the managed cases of the both-backend theories, and the tests marked
  `[Trait("Backend", "Managed")]`), the no-skip check and the managed `verify-package.ps1`.

`build-test` and `golden` stay separate jobs, not a matrix, because branch protection requires checks
by those exact names (the others are not required). All upload their `.trx` test results. To reproduce `golden` locally, run `setup.ps1 -Models`, then
`dotnet test --logger trx --results-directory TestResults` and check that nothing was skipped.

## Releasing

Merge a pull request that moves the "Unreleased" entries in CHANGELOG.md under a new version heading, such as
`## [0.5.0] - 2026-10-08`. `.github/workflows/release.yml` sees a version without a tag, runs all CI checks, waits
for approval, packs that version and publishes it to nuget.org through NuGet Trusted Publishing. Only then does it
create the tag (`v0.5.0`) and a GitHub release (a prerelease when the version has a `-`).

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0, the same as Stanza, which this ports. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Stanza's pretrained models are not part of this repository; their licenses vary with the training
data, so check them before redistributing models.
