# StanzaSharp

C# port of Stanza's English inference pipeline (tokenize, mwt, pos, constituency) on TorchSharp.
It runs Stanza's own pretrained models. On the golden test corpus, the output is byte-identical to
Python Stanza 1.15.0.
See CLAUDE.md for scope, layout, decisions and build order.

## Setup

Requires the .NET 10 SDK and, for the reference environment, Python 3.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Models
```

`-Models` creates `tools\.venv` with Stanza, downloads the English models and converts them.
Use `-Python` for the environment only, or no switch for just the .NET solution. Add `-Cuda` for GPU libtorch.

## Usage

```csharp
using StanzaSharp;

using var nlp = Pipeline.Load("models/converted/en");
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

Tests that need models skip when `models/converted/en` is missing. To regenerate the golden data,
run `tools\.venv\Scripts\python tools\make_golden.py`.

## CI

`.github/workflows/ci.yml` runs on every push and pull request (Ubuntu, .NET 10). It has not run
on GitHub yet, since the repository has no remote.

- `build-test` builds the solution and runs the tests without models (the model tests skip).
- `golden` installs `tools/requirements.txt` with the CPU-only torch wheel, downloads and converts
  the English models, and runs the full suite. It fails if any test was skipped. The converted
  models are cached, keyed on `tools/requirements.txt` and `tools/stanza_convert.py`.

Both upload their `.trx` test results. To reproduce `golden` locally, run `setup.ps1 -Models`, then
`dotnet test --logger trx --results-directory TestResults` and check that nothing was skipped.
