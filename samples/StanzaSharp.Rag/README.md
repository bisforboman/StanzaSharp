# RAG ingestion sample

Markdown files go through StanzaSharp.DataIngestion into an in-memory vector store, then the sample runs a semantic
search filtered by a named entity. Everything runs locally; there is no service and no API key.

1. `MarkdownReader` (Microsoft.Extensions.DataIngestion.Markdig) reads each `.md` file.
2. `SentenceChunker` cuts it into chunks of whole sentences. Each chunk is at most 200 tokens of the embedding model's own
   tokenizer, overlaps the previous one by up to 40 tokens of whole sentences, and starts with its headers.
3. `EntityEnricher` adds each chunk's people, organisations and places (`PERSON`, `ORG`, `GPE`, `LOC`, `FAC`) under the
   metadata key `entities`.
4. DataIngestion's `VectorStoreWriter` stores the chunks in Semantic Kernel's `InMemoryVectorStore`.
   [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeds them. It is Apache 2.0 and runs
   on ONNX Runtime with mean pooling, as sentence-transformers does (`MiniLmEmbeddingGenerator.cs`).
5. The sample prints the most frequent entities, then the top 3 chunks for a query, first unfiltered, then only those whose
   `entities` contain the chosen name.

```
dotnet run --project samples/StanzaSharp.Rag -- [DOCS] [--query TEXT] [--entity NAME] [--models DIR]
```

- `DOCS` defaults to the sample's own `docs/`, three short notes from an invented freight company.
- `--entity ""` skips the filtered search.
- `DIR` defaults to `models`. The first run downloads two sets of models there:
  - English `tokenize` and `ner` (about 210 MB) to `DIR/stanza/en`;
  - the embedding model (90 MB, a pinned revision checked by sha256) to `DIR/all-MiniLM-L6-v2`.

## What it shows

The default query is "What does the customer want in the new building?". Ranked by similarity alone, the top 3 are the
notes about the building and its budget. The customer's request for a cold store is in a chunk about Nordlys Seafood that
says "warehouse", not "building", so it ranks lower. Filtered to chunks that name Nordlys Seafood, that chunk comes first:

```
"What does the customer want in the new building?", only chunks naming Nordlys Seafood
  0.149  [customers.md] # Customer notes ## Nordlys Seafood Nordlys Seafood moves frozen fish from Bergen to Rotterdam. ...
         entities: Nordlys Seafood, Bergen, Rotterdam, Sigrid Aune, Tomas Venn
```

In an application the entity usually comes from the question. Run the same NER over the query and filter on what it
finds.

## Notes

- Each chunk starts with its headers, so a header's entities count for every chunk under it: "Rotterdam" is in all
  three chunks of the warehouse note.
- `VectorStoreWriter` disposes the vector store when the ingestion pipeline is disposed, so the sample searches inside
  that scope.
- Swap in another store (Qdrant, Postgres, Azure AI Search, ...) or another `IEmbeddingGenerator`. Keep the chunker's
  tokenizer the embedding model's, so chunks stay under the model's input limit (256 tokens here).
- Microsoft.Extensions.DataIngestion and the Semantic Kernel connector are previews, so their APIs may still change.
