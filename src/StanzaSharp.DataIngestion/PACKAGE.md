# StanzaSharp.DataIngestion

[StanzaSharp](https://www.nuget.org/packages/StanzaSharp) for
[Microsoft.Extensions.DataIngestion](https://learn.microsoft.com/dotnet/ai/conceptual/data-ingestion), .NET's building
blocks for RAG ingestion pipelines. Both run Stanza's models locally: no service and no API key.

- `SentenceChunker` (an `IngestionChunker<string>`): chunks of whole sentences, found by Stanza's sentence splitter
  (100+ languages), each at most `MaxTokensPerChunk` tokens of your embedding model's tokenizer, with whole-sentence
  overlap. Headers become each chunk's context, as in `HeaderChunker`; tables stay whole.
- `EntityEnricher` (an `IngestionChunkProcessor<string>`): each chunk's named entities under the metadata key
  `entities` (a `string[]`), for filtered or hybrid search.

This package is a prerelease while Microsoft.Extensions.DataIngestion is in preview.

## Use

```csharp
using Microsoft.Extensions.DataIngestion;
using Microsoft.ML.Tokenizers;
using StanzaSharp;
using StanzaSharp.DataIngestion;

// Once: await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Processors = "tokenize,ner" });
using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions { Processors = "tokenize,ner" });
var tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o"); // your embedding model's tokenizer
var chunker = new SentenceChunker(nlp, new IngestionChunkerOptions(tokenizer) { MaxTokensPerChunk = 512, OverlapTokens = 64 });

using var pipeline = new IngestionPipeline<string>(reader, chunker, writer);
pipeline.ChunkProcessors.Add(new EntityEnricher(nlp, types: new HashSet<string> { "PERSON", "ORG", "GPE" }));
```

The chunker needs only `tokenize` (load a pipeline with `Processors = "tokenize"` if you don't enrich); the enricher
needs `ner`. One pipeline can serve both. Use the documents' language (`PipelineOptions.Language`); the entity types
are its NER model's (English: OntoNotes' PERSON, ORG, GPE, DATE, ...; German and most others: PER, LOC, ORG, MISC).

## License

Apache 2.0, like Stanza, whose code StanzaSharp ports; see NOTICE. The models have their own licenses, which vary
with their training data.
