- `PipelineOptions.Pretagged`, Stanza's `pretagged=True`: lemma, depparse and constituency load without the processors
  they require (`Processors = "depparse"` loads the parser without the tagger), for `Process(Document)` on documents
  that already carry tags and lemmas. The downloader follows it. Byte-identical to Python Stanza on both backends.
