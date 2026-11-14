- `samples/StanzaSharp.Rag` takes `--model`: all-MiniLM-L6-v2 (the default), multilingual-e5-small, multilingual-e5-base
  or KBLab's sentence-bert-swedish-cased, each on the managed encoder with sentence-transformers' vectors. It now sizes
  and encodes with StanzaSharp's own tokenizers, whose ids are transformers'; Microsoft.ML.Tokenizers' `BertTokenizer`
  differed on some words, so the sample's scores moved slightly.
