- `PipelineOptions.TokenizePostprocessor`: Stanza's `tokenize_postprocessor`, a function that gets the tokenizer's
  sentences (each token's text and multi-word flag, `PostprocessorToken`) and returns the sentences and tokens to go on
  with: merged, split or regrouped, flagged for the MWT model or expanded by hand. Byte for byte with Stanza on text and
  bulk input, with and without sentence splitting, on both backends.
