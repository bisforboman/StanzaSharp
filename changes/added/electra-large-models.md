- `PipelineOptions.Models`: Stanza's `default_accurate` tagger, parser and NER model, pos and depparse
  `combined_electra-large` and ner `ontonotes-ww-multi_electra-large`, byte for byte with Stanza on both backends. They
  read the frozen transformer `google/electra-large-discriminator` (1.3 GB, from Hugging Face at a pinned revision, checked
  by sha256), which `ModelDownloader.DownloadAsync(dir, options)` and `stanzasharp download --model ...` fetch into
  `transformer/` next to the models; a pipeline runs it once per sentence for all three.
