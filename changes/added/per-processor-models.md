- `PipelineOptions.Models`: one model per processor instead of the package's, like Python Stanza's
  `processors={'ner': 'i2b2'}` (a model's or a package's name per processor; the keys' processors are added to the
  package's), honoured by `ModelDownloader.DownloadAsync(dir, options)` and `--model PROCESSOR=NAME` in the tool's
  download, compare and benchmark. New models: English constituency `ptb3_charlm`, English ner `ontonotes_charlm`,
  `ontonotes_nocharlm`, `combined_charlm` and `i2b2` (clinical, with the `mimic` package), German ner `conll03`, all
  byte-identical to Python Stanza on both backends.
