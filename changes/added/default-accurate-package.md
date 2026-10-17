- `PipelineOptions.Package = "default_accurate"` (and `stanzasharp download --package default_accurate`): Stanza's most
  accurate English package, all eight processors, with pos, constituency, depparse, ner and sentiment on
  `google/electra-large-discriminator`, byte for byte with Stanza on both backends. German's `default_accurate` (another
  transformer) still throws.
