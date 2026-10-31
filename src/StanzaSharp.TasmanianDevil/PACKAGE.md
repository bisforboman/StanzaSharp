# StanzaSharp.TasmanianDevil

[StanzaSharp](https://www.nuget.org/packages/StanzaSharp) for [TasmanianDevil](https://github.com/filipw/tasmaniandevil),
a .NET library for detecting and de-identifying PII. It adds two components.

- `StanzaNerRecognizer` finds `PERSON`, `LOCATION`, `ORGANIZATION` and `DATE_TIME`, the entity types that regular
  expressions can't reach.
  - It uses Stanza's supervised NER model for the pipeline's language: English (OntoNotes), Swedish, German and many more.
  - It runs in managed code and gives the same spans on every run.
  - It joins TasmanianDevil's analyzer like any other recognizer, so overlap resolution and anonymization treat its spans
    like the regex matches.
- `StanzaTokenNormalizer` gives TasmanianDevil's context matching (`LemmaContextAwareEnhancer`) real lemmas, in any
  language Stanza has a lemmatizer for, instead of English Porter stems.

## Use

```csharp
using StanzaSharp;
using StanzaSharp.TasmanianDevil;
using TasmanianDevil.Analyzer;

// Once: await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Processors = "tokenize,ner" });
using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions { Processors = "tokenize,ner" });
registry.AddRecognizer(new StanzaNerRecognizer(nlp));                    // English
registry.AddRecognizer(new StanzaNerRecognizer(swedishNlp, "sv"));       // a pipeline per language
```

`StanzaNerRecognizer` needs only `tokenize,ner`. `StanzaTokenNormalizer` needs `tokenize,mwt,pos,lemma`; pass it to
`LemmaContextAwareEnhancer`'s `normalizer` parameter.

### Entity types

`StanzaNerRecognizer.DefaultTypeMap` maps Stanza's entity types to TasmanianDevil's.

| Stanza | TasmanianDevil |
|---|---|
| PERSON, PER | PERSON |
| ORG | ORGANIZATION |
| GPE, LOC, FAC | LOCATION |
| DATE, TIME, TME | DATE_TIME |

The map covers both naming conventions of Stanza's models: English's OntoNotes types, and the PER/LOC/ORG types most
other languages use. Pass your own map to change it. Every span gets the same score, 0.85 by default; Stanza gives no
confidence, and 0.85 is what Python Presidio's spaCy recognizer uses.

### Compared with TasmanianDevil.Onnx

TasmanianDevil's own NER add-on uses a zero-shot GLiNER model: one multilingual model of 580 MB (fp16), which you
prompt with labels.

This package uses Stanza's supervised model for each language instead, with a fixed set of types per language. For
English, `tokenize` and `ner` are about 210 MB. It needs no native ONNX runtime, and it gives the same output as Python
Stanza.

## License

Apache 2.0, like Stanza, whose code StanzaSharp ports; see NOTICE. TasmanianDevil is MIT. The models have their own
licenses, which vary with their training data.
