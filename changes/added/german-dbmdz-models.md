- `PipelineOptions.Models` (German): ner `germeval2014_dbmdz-bert-german-cased` and sentiment `sb10k_dbmdz-bert-german-cased`,
  on `dbmdz/bert-base-german-cased` (a BERT, downloaded by `DownloadAsync(dir, options)`), byte for byte with Stanza on both
  backends. The NER model reads conll17's word vectors, which it was trained on: Stanza 1.15.0's resources.json lists
  fasttextwiki, with which Stanza fails to load it.
