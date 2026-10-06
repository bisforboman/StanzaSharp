# StanzaSharp

[Stanza](https://github.com/stanfordnlp/stanza)'s English NLP pipeline in .NET:
- tokenization and sentence splitting
- multi-word token expansion ("don't" → "do" + "n't")
- part-of-speech tags and morphological features
- lemmas
- dependency parsing
- named entities (OntoNotes types: PERSON, ORG, GPE, DATE, ...)
- constituency parsing
- sentence sentiment (negative, neutral, positive)

It runs Stanza's own pretrained models through TorchSharp, with no Python needed. The output is identical
to Python Stanza 1.15.0 on its golden test data.

## Install

```
dotnet add package StanzaSharp
dotnet add package TorchSharp-cpu
```

Add `TorchSharp-cpu` (or a `TorchSharp-cuda-*` package for GPU) for the native libtorch. Its version must
match the `TorchSharp` version StanzaSharp depends on.

## Download the models

The models (about 600 MB) come from Stanza's Hugging Face repository. Download them once:

```csharp
await ModelDownloader.DownloadAsync("models/stanza/en");                            // all, ~600 MB
await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
```

Each file is checked against its published MD5, and files already present are kept. `Pipeline.Load` itself
never touches the network.

## Use

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

- `Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,mwt" })` runs only the listed processors; each needs the ones before it. The default is all eight, like Stanza's English default: `tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner`.
- Add `ner` to the list for named entities: `doc.Entities` (text, type, character offsets) and a BIOES tag per token (`Token.Ner`).
- Add `sentiment` to the list for `sentence.Sentiment`: 0 negative, 1 neutral, 2 positive.
- `Conllu.Write(doc)` gives CoNLL-U in Stanza's format.

## Faster: the default_fast package

`new PipelineOptions { Package = "default_fast" }` selects Stanza's `default_fast` package: no constituency
parser, and pos, depparse and ner with their own small character models instead of the large character
language models. It is faster and smaller, and its output is identical to Python Stanza's
`package='default_fast'`.

```csharp
var options = new PipelineOptions { Package = "default_fast" };
await ModelDownloader.DownloadAsync("models/stanza/en", options); // what Load reads with these options
using var nlp = Pipeline.Load("models/stanza/en", options);
```

## GPU

Reference `TorchSharp-cuda-windows` (or `TorchSharp-cuda-linux`) instead of `TorchSharp-cpu`, then
`Pipeline.Load(dir, new PipelineOptions { Device = torch.CUDA, DisableTf32 = true })`. `DisableTf32` gives
output identical to the CPU by turning TF32 off process-wide; with TF32 on (libtorch's default), a few
near-tie decisions can differ. See the repository's docs/gpu.md for measurements.

## License

Apache 2.0, like Stanza, whose code this ports (see NOTICE). Stanza's models are not included. Their
licenses vary with the training data.

Source, issues and docs: https://github.com/bisforboman/StanzaSharp
