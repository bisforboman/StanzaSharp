- German: `Pipeline.Load(dir, new PipelineOptions { Language = "de" })` runs Stanza's German models, all eight processors,
  byte-identical to Python Stanza 1.15.0 on both backends, with the packages `default`, `default_fast` and the treebank
  packages `combined`, `gsd` and `hdt` (tokenize to depparse; gsd's and hdt's taggers are Stanza's older checkpoint
  format, read as Stanza reads it). German MWT is a character seq2seq model (shared with the
  lemmatizer), and German NER (GermEval 2014: `PER`, `LOC`, `ORG`, `MISC`) reads fastText's German vectors, so a German
  pipeline loads two word-vector files. The samples' `StanzaSharp.Cli` takes `--lang de`.
