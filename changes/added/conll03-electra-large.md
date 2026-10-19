- `PipelineOptions.Models`: Stanza's NER model `conll03_electra-large` (PER, LOC, ORG, MISC), which reads `roberta-base`
  (a byte-level BPE tokenizer and a RoBERTa encoder, downloaded by `DownloadAsync(dir, options)`), byte for byte with Stanza on
  both backends. A pipeline may now read two transformers, e.g. with the `default_accurate` package's models. It reads
  conll17's word vectors, which it was trained on: Stanza 1.15.0's resources.json lists GloVe, with which Stanza fails to load it.
