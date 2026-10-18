- `PipelineOptions.Package = "default_accurate"` (and `stanzasharp download --package default_accurate`): Stanza's most
  accurate package, all eight processors, byte for byte with Stanza on both backends. English: pos, constituency,
  depparse, ner and sentiment on `google/electra-large-discriminator`. German (`Language = "de"`): pos and depparse on
  `german-nlp-group/electra-base-german-uncased`, constituency and sentiment on the same with their LoRA adapters.
