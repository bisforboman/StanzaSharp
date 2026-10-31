- The `StanzaSharp.DataIngestion` package (#233) plugs StanzaSharp into Microsoft.Extensions.DataIngestion, for RAG
  ingestion. It is published as a prerelease (`x.y.z-preview`) while that library is in preview.
  - `SentenceChunker` makes chunks of whole sentences up to `MaxTokensPerChunk` of your embedding model's tokens, with
    sentence-aligned overlap. Headers become each chunk's context, as in `HeaderChunker`.
  - `EntityEnricher` adds each chunk's named entities as metadata (`entities`).
