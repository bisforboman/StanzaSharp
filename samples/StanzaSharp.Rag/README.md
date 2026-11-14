# RAG ingestion sample

Markdown files go through StanzaSharp.DataIngestion into an in-memory vector store, then the sample runs a semantic
search filtered by a named entity. Everything runs locally; there is no service, no API key and no native library.

1. `MarkdownReader` (Microsoft.Extensions.DataIngestion.Markdig) reads each `.md` file.
2. `SentenceChunker` cuts it into chunks of whole sentences. Each chunk is at most 200 tokens of the embedding model's own
   tokenizer, overlaps the previous one by up to 40 tokens of whole sentences, and starts with its headers.
3. `EntityEnricher` adds each chunk's people, organisations and places (`PERSON`, `ORG`, `GPE`, `LOC`, `FAC`) under the
   metadata key `entities`.
4. DataIngestion's `VectorStoreWriter` stores the chunks in Semantic Kernel's `InMemoryVectorStore`.
   [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embeds them (or another model, see
   `--model` below). It runs on StanzaSharp's managed transformer encoder with the model's own tokenizer and mean
   pooling, as sentence-transformers does (`SentenceEmbeddingGenerator.cs`).
5. The sample prints the most frequent entities, then the top 3 chunks for a query, first unfiltered, then only those whose
   `entities` contain the chosen name.

```
dotnet run --project samples/StanzaSharp.Rag -- [DOCS] [--query TEXT] [--entity NAME] [--model NAME] [--models DIR]
```

- `DOCS` defaults to the sample's own `docs/`, three short notes from an invented freight company.
- `--entity ""` skips the filtered search.
- `DIR` defaults to `models`. The first run downloads two sets of models there:
  - English `tokenize` and `ner` (about 210 MB) to `DIR/stanza/en`;
  - the embedding model (a pinned revision, each file checked by sha256) to its own folder in `DIR`.
- `--model` chooses the embedding model. Each gives the vectors sentence-transformers gives
  ([docs/embedding-spike.md](../../docs/embedding-spike.md)):

  | `--model` | Model | Languages | Dimensions | Token limit | Download | Licence |
  |---|---|---|---|---|---|---|
  | `minilm` (default) | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | English | 384 | 256 | 91 MB | Apache 2.0 |
  | `e5-small` | [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | about 100 | 384 | 512 | 488 MB | MIT |
  | `e5-base` | [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | about 100 | 768 | 512 | 1.1 GB | MIT |
  | `swedish` | [sentence-bert-swedish-cased](https://huggingface.co/KBLab/sentence-bert-swedish-cased) | Swedish, English | 768 | 384 | 500 MB | Apache 2.0 |

  The licences are the model cards'. The sample's notes and its NER stay English whatever the model.

## What it shows

The default query is "What does the customer want in the new building?". Ranked by similarity alone, the top 3 are the
notes about the building and its budget. The customer's request for a cold store is in a chunk about Nordlys Seafood that
says "warehouse", not "building", so it ranks lower. Filtered to chunks that name Nordlys Seafood, that chunk comes first:

```
"What does the customer want in the new building?", only chunks naming Nordlys Seafood
  0.141  [customers.md] # Customer notes ## Nordlys Seafood Nordlys Seafood moves frozen fish from Bergen to Rotterdam. ...
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
  tokenizer the embedding model's, so chunks stay under the model's input limit (256 tokens for `minilm`).
- The chunker and the encoder share one tokenizer: StanzaSharp's own WordPiece or SentencePiece Unigram, read from the
  model's `tokenizer.json`, behind a small `Microsoft.ML.Tokenizers.Tokenizer` that only counts. Its ids are
  transformers'. Microsoft.ML.Tokenizers' `BertTokenizer`, which the sample used before, differs on some words
  ("Tomas" is two other pieces, and "café" is `[UNK]`), so the scores moved a little from earlier versions.
- E5 expects `query: ` before a question and `passage: ` before a stored text. The store embeds the chunks through
  `GenerateAsync` (the passage prefix); the sample embeds the question itself with `EmbedQueryAsync` and searches by
  vector. The other models have no prefixes.
- The scores are cosine similarities and each model has its own scale: E5's lie close together (0.75 to 0.80 here), so
  compare ranks, not values, between models.
- The embedding model runs on the encoder StanzaSharp uses for Stanza's transformer models, in place of ONNX Runtime
  (#268). That encoder and the tokenizers are not in the package's public API yet: `SentenceEmbeddingGenerator.cs`
  builds only inside this repository, which gives the sample access to StanzaSharp's internals.
- Microsoft.Extensions.DataIngestion and the Semantic Kernel connector are previews, so their APIs may still change.
