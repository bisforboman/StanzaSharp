- `samples/StanzaSharp.Rag` (#233) runs RAG ingestion locally: Markdown, `SentenceChunker`, `EntityEnricher`, an
  in-memory vector store with all-MiniLM-L6-v2 embeddings on ONNX Runtime, then a search filtered to the chunks that name
  an entity.
