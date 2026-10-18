- `Pipeline.Stream(texts, batchSize = 50, cancellationToken)`: Stanza's `Pipeline.stream`, a lazy sequence of documents
  over any number of texts, processed `batchSize` at a time like `Process(IEnumerable<string>)`, with sentence ids
  counted across the stream; byte for byte with Stanza's `stream` on both backends.
