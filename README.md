# StanzaSharp

C# port of Stanza's English inference pipeline (tokenize, mwt, pos, constituency) on TorchSharp.
It runs Stanza's own pretrained models. On the golden test corpus, the output is byte-identical to
Python Stanza 1.15.0.
See CLAUDE.md for scope, layout, decisions and build order.

## Setup

Requires the .NET 10 SDK. The library reads Stanza's original `.pt` model files directly, so using
it needs no Python. Python 3 is only needed for the reference environment: downloading models with
Stanza, the optional converter, and regenerating golden data.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Models
```

`-Models` creates `tools\.venv` with Stanza, downloads the English models to `models\stanza` and
converts them to `models\converted\en` (safetensors + JSON, slightly faster to load).
Use `-Python` for the environment only, or no switch for just the .NET solution. Add `-Cuda` for GPU libtorch.
If you already have Stanza's English models (e.g. `~/stanza_resources/en`), point `Pipeline.Load` at that
directory instead.

## Usage

```csharp
using StanzaSharp;

using var nlp = Pipeline.Load("models/converted/en"); // or Stanza's own "models/stanza/en"
var doc = nlp.Process("Barack Obama was born in Hawaii. He was elected president in 2008.");

foreach (var sentence in doc.Sentences)
{
    foreach (var word in sentence.Words)
        Console.WriteLine($"{word.Text}\t{word.Upos}\t{word.Xpos}\t{word.Feats}");
    Console.WriteLine(sentence.Constituency); // (ROOT (S (NP (NNP Barack) (NNP Obama)) ...))
}
```

`Pipeline.Load(dir, "tokenize,mwt")` runs only the listed processors; each needs the ones before it.
`Conllu.Write(doc)` gives Stanza-style CoNLL-U.

From the command line, this writes CoNLL-U for a file (or standard input):

```powershell
dotnet run --project samples/StanzaSharp.Cli -- input.txt
dotnet run --project samples/StanzaSharp.Cli -- --processors tokenize,mwt,pos input.txt
```

## Tests

```powershell
dotnet test
```

Tests that need models skip when `models/converted/en` (or, for the `.pt` loader tests,
`models/stanza/en`) is missing. To regenerate the golden data, run
`tools\.venv\Scripts\python tools\make_golden.py`.

## CI

`.github/workflows/ci.yml` runs on every push and pull request (Ubuntu, .NET 10). It has not run
on GitHub yet, since the repository has no remote.

- `build-test` builds the solution and runs the tests without models (the model tests skip).
- `golden` installs `tools/requirements.txt` with the CPU-only torch wheel, downloads and converts
  the English models, and runs the full suite. It fails if any test was skipped. The original and
  converted models are cached, keyed on `tools/requirements.txt` and `tools/stanza_convert.py`.

Both upload their `.trx` test results. To reproduce `golden` locally, run `setup.ps1 -Models`, then
`dotnet test --logger trx --results-directory TestResults` and check that nothing was skipped.

## License

Apache License 2.0, the same as Stanza, which this ports. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Stanza's pretrained models are not part of this repository; their licenses vary with the training
data, so check them before redistributing models.
