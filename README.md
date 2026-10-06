# StanzaSharp

C# port of Stanza's English inference pipeline (tokenize, mwt, pos, lemma, depparse, ner, sentiment, constituency) on TorchSharp.
It runs Stanza's own pretrained models. On the golden test corpus, the output is byte-identical to
Python Stanza 1.15.0.
See CLAUDE.md for scope, layout, decisions and build order.

## Install

```
dotnet add package StanzaSharp
dotnet add package TorchSharp-cpu
```

`TorchSharp-cpu` (or a `TorchSharp-cuda-*` package) brings the native libtorch; its version must match the
`TorchSharp` version StanzaSharp depends on (0.107.0). Requires .NET 10. Tested on Linux, Windows and macOS.
On macOS (Apple Silicon), also run `brew install libomp`: TorchSharp-cpu's libtorch loads OpenMP from Homebrew's path.
On Windows on Arm64, use `StanzaSharp.Cpu.WindowsArm64` instead (below). See [Supported platforms](#supported-platforms).

**Deploying to one platform** (e.g. a small Linux service): `TorchSharp-cpu` restores libtorch for Linux,
Windows and macOS (about 265 MB of downloads). A platform package restores StanzaSharp plus only that
platform's CPU libtorch instead. It's the only package you need:

| Package | Platform | libtorch download |
|---|---|---:|
| `StanzaSharp.Cpu.Linux` | Linux x64 | 128 MB |
| `StanzaSharp.Cpu.Windows` | Windows x64 | 80 MB |
| `StanzaSharp.Cpu.WindowsArm64` | Windows on Arm64 | 42 MB |
| `StanzaSharp.Cpu.MacOS` | macOS on Apple Silicon | 57 MB |

```
dotnet add package StanzaSharp.Cpu.Linux
```

Publish with that platform's runtime identifier (`dotnet publish -r linux-x64`) so only its native files are
copied. Smaller models help too: `Package = "default_fast"` and downloading only the processors you use.

### Supported platforms

StanzaSharp runs wherever TorchSharp 0.107 has its native layer (`LibTorchSharp`) and a CPU libtorch 2.10.0
package exists. Each supported platform runs the full test suite in CI.

| Platform | Supported | Package |
|---|---|---|
| Linux x64 (glibc: Ubuntu, Debian, RHEL, ...) | Yes | `StanzaSharp.Cpu.Linux`, or `StanzaSharp` + `TorchSharp-cpu` |
| Windows x64 | Yes | `StanzaSharp.Cpu.Windows`, or `StanzaSharp` + `TorchSharp-cpu` |
| Windows on Arm64 | Yes | `StanzaSharp.Cpu.WindowsArm64` only: `TorchSharp-cpu` has no Arm64 libtorch |
| macOS on Apple Silicon | Yes, after `brew install libomp` | `StanzaSharp.Cpu.MacOS`, or `StanzaSharp` + `TorchSharp-cpu` |
| Alpine and other musl Linux | No | None: libtorch and TorchSharp are built for glibc only, with no `linux-musl` build |
| Linux Arm64 | No | None: no `libtorch-cpu-linux-arm64` package, and TorchSharp has no `linux-arm64` native layer |
| macOS on Intel (x64) | No | None: TorchSharp has no `osx-x64` native layer, and `libtorch-cpu-osx-x64` stops at 2.2 |

GPU: see [GPU](#gpu) (CUDA on Windows and Linux x64).

### Docker

Use a glibc-based image, such as `mcr.microsoft.com/dotnet/runtime:10.0` (Ubuntu). libtorch needs only glibc,
libstdc++ and libgcc_s, and ships its own OpenMP, so the chiseled `mcr.microsoft.com/dotnet/runtime:10.0-noble-chiseled`
works too. Not Alpine (see above). [samples/docker](samples/docker/Dockerfile) is a small app that reads text
on stdin and writes CoNLL-U:

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

Mount the models rather than copying them into the image: they are about 600 MB, and the image is already
600 to 700 MB depending on the base, of which the published app is 500 MB, nearly all of it libtorch. CI builds this image on both base images and checks that its output
is byte-identical to the golden file.

## Models

Download Stanza's English models (about 600 MB, from Stanza's Hugging Face repository) once. Every file is
checked against its published MD5; `Pipeline.Load` itself never touches the network.

```csharp
await ModelDownloader.DownloadAsync("models/stanza/en");                            // all, ~600 MB
await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Package = "default_fast" });
```

or from the command line: `dotnet run --project samples/StanzaSharp.Cli -- download models/stanza/en`
(add `--processors LIST` for a subset, `--package default_fast` for that package).
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
needs the ones before it. `PipelineOptions` also holds `Device`, `DisableTf32` and `CharlmCache`
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

`samples/StanzaSharp.Example` is a commented tour of the whole API: `dotnet run --project samples/StanzaSharp.Example`.

From the command line, this writes CoNLL-U for a file (or standard input):

```powershell
dotnet run --project samples/StanzaSharp.Cli -- input.txt
dotnet run --project samples/StanzaSharp.Cli -- --processors tokenize,mwt,pos input.txt
dotnet run --project samples/StanzaSharp.Cli -- --package default_fast input.txt
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

On 8 CPU threads the six-processor pipeline is about 1.7x faster than Python Stanza and peaks at 2.7 GB of memory
(Python: 4.3 GB); see [docs/performance.md](docs/performance.md). The `default_fast` package runs its seven
processors about 2.3x faster than the default's eight, and 1.5x faster than Python's `default_fast`. Libtorch uses one thread per physical core by
default. On a machine busy with other work, fewer threads (`torch.set_num_threads(n)`) are often faster.

## GPU

`Pipeline.Load(dir, new PipelineOptions { Device = torch.CUDA })` runs every model on an NVIDIA GPU. The
default is the CPU. Reference a CUDA libtorch package such as
`TorchSharp-cuda-windows` (several GB) instead of `TorchSharp-cpu`, in the same version (0.107.0). StanzaSharp
itself depends only on the managed `TorchSharp` package either way.

For output identical to the CPU (and so to Python Stanza), turn off TF32, which libtorch enables for cuDNN by
default on Ampere and newer GPUs:

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Device = torch.CUDA, DisableTf32 = true });
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

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests (.NET 10; Ubuntu 24.04, plus Windows and macOS).

- `build-test` builds the solution and runs the tests without models (the model tests skip).
- `golden` downloads the English models with `ModelDownloader` (through the CLI), converts them with
  `tools/stanza_convert.py` (CPU-only torch wheel), and runs the full suite. It fails if any test was
  skipped. Then `tools/verify-package.ps1` packs StanzaSharp and runs a fresh app that references the
  package. The original and converted models are cached, keyed on `ModelDownloader.cs`,
  `tools/requirements.txt` and `tools/stanza_convert.py`.
- `docker` runs `tools/verify-docker.sh`: it packs StanzaSharp and `StanzaSharp.Cpu.Linux`, builds
  [samples/docker](samples/docker/Dockerfile) on `mcr.microsoft.com/dotnet/runtime:10.0` and on its chiseled
  variant, and checks that each container turns `corpus.txt` into `pipeline.conllu` byte for byte. It reports
  the image sizes in the job summary.
- `cross-os (windows-2025)`, `cross-os (macos-15)` (Apple Silicon) and `cross-os (windows-11-arm)` run the same
  full suite, the no-skip check and `tools/verify-package.ps1` (each with its platform package), against Stanza's
  `.pt` files without converting them (no Python). Their models are cached per OS, keyed on `ModelDownloader.cs`.
- `.github/workflows/alpine-experiment.yml` (manual, or on PRs touching the Docker sample; not required) runs
  the same Docker check on Alpine with `gcompat`.

`build-test` and `golden` stay separate jobs, not a matrix, because branch protection requires checks
by those exact names. All upload their `.trx` test results. To reproduce `golden` locally, run `setup.ps1 -Models`, then
`dotnet test --logger trx --results-directory TestResults` and check that nothing was skipped.

## Releasing

Push a tag such as `v0.1.0-alpha.1`. `.github/workflows/release.yml` runs all CI checks, packs that version,
publishes it to nuget.org through NuGet Trusted Publishing and creates a GitHub release (a prerelease
when the version has a `-`).

## Changes

See [CHANGELOG.md](CHANGELOG.md).

## License

Apache License 2.0, the same as Stanza, which this ports. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Stanza's pretrained models are not part of this repository; their licenses vary with the training
data, so check them before redistributing models.
