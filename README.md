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

`Pipeline.Load(dir, "tokenize,mwt")` runs only the listed processors; each needs the ones before it.
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
.NET solution. Add `-Cuda` for GPU libtorch.

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
