- Stanza's batch sizes as `PipelineOptions` properties: `TokenizeBatchSize`, `MwtBatchSize`, `PosBatchSize` and
  `PosBatchMaximumTokens`, `LemmaBatchSize`, `ConstituencyBatchSize`, `DepparseBatchSize` and
  `DepparseMinLengthToBatchSeparately`, `SentimentBatchSize`, `NerBatchSize` (Stanza's `{processor}_batch_size`,
  `pos_batch_maximum_tokens`, `depparse_min_length_to_batch_separately`). As in Stanza, sentiment labels change with
  `SentimentBatchSize`, and parses can change with the depparse sizes; the others trade only speed against memory.
  `TokenizeBatchSize` works although Stanza ignores `tokenize_batch_size`.
