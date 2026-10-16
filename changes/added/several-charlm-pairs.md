- A pipeline can read several pairs of character language models: `PipelineOptions.Models` may choose a model whose
  pair differs from the package's (e.g. the default package with ner `i2b2`, which reads mimic's), which no longer
  throws; the pipeline loads each pair once (about 60 MB more for the second on the managed backend), and the tagger's
  cached outputs are reused only by models that read the same pair. Byte-identical to Python Stanza on both backends.
