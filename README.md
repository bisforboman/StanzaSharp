# StanzaSharp

C# port of Stanza's English inference pipeline (tokenize, mwt, pos, lemma, depparse, constituency) on TorchSharp.
It runs Stanza's own pretrained models. On the golden test corpus, the output is byte-identical to
Python Stanza 1.15.0.
See CLAUDE.md for scope, layout, decisions and build order.

## Install

```
dotnet add package StanzaSharp
dotnet add package TorchSharp-cpu
```

`TorchSharp-cpu` (or a `TorchSharp-cuda-*` package) brings the native libtorch; its version must match the
`TorchSharp` version StanzaSharp depends on (0.107.0). Requires .NET 10.

## Models

Download Stanza's English models (about 450 MB, from Stanza's Hugging Face repository) once. Every file is
checked against its published MD5; `Pipeline.Load` itself never touches the network.

```csharp
await ModelDownloader.DownloadAsync("models/stanza/en");
```

or from the command line: `dotnet run --project samples/StanzaSharp.Cli -- download models/stanza/en`.
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
(`IsEnabled`, `MaxWords`: the tagger's character-model outputs reused by the constituency parser, at most
32,768 words / ~256 MB by default; output is identical either way).
The default is all six processors: `tokenize,mwt,pos,lemma,depparse,constituency`.
`Conllu.Write(doc)` gives Stanza-style CoNLL-U.

From the command line, this writes CoNLL-U for a file (or standard input):

```powershell
dotnet run --project samples/StanzaSharp.Cli -- input.txt
dotnet run --project samples/StanzaSharp.Cli -- --processors tokenize,mwt,pos input.txt
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
(Python: 4.3 GB); see [docs/performance.md](docs/performance.md). Libtorch uses one thread per physical core by
default. On a machine busy with other work, fewer threads (`torch.set_num_threads(n)`) are often faster.

## GPU

`Pipeline.Load(dir, new PipelineOptions { Device = torch.CUDA })` runs every model on an NVIDIA GPU; each
processor's `Load` takes an optional `device` too. The default is the CPU. Reference a CUDA libtorch package such as
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

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests (Ubuntu 24.04, .NET 10).

- `build-test` builds the solution and runs the tests without models (the model tests skip).
- `golden` downloads the English models with `ModelDownloader` (through the CLI), converts them with
  `tools/stanza_convert.py` (CPU-only torch wheel), and runs the full suite. It fails if any test was
  skipped. Then `tools/verify-package.ps1` packs StanzaSharp and runs a fresh app that references the
  package. The original and converted models are cached, keyed on `ModelDownloader.cs`,
  `tools/requirements.txt` and `tools/stanza_convert.py`.

Both upload their `.trx` test results. To reproduce `golden` locally, run `setup.ps1 -Models`, then
`dotnet test --logger trx --results-directory TestResults` and check that nothing was skipped.

## Releasing

Push a tag such as `v0.1.0-alpha.1`. `.github/workflows/release.yml` runs all CI checks, packs that version,
publishes it to nuget.org through NuGet Trusted Publishing and creates a GitHub release (a prerelease
when the version has a `-`).

## License

Apache License 2.0, the same as Stanza, which this ports. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Stanza's pretrained models are not part of this repository; their licenses vary with the training
data, so check them before redistributing models.
