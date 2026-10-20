- Coreference resolution: `Processor.Coref`, Stanza's `udcoref_xlm-roberta-lora` on XLM-RoBERTa large (English and
  German). It is optional, as in Stanza: name it in `Processors` (e.g. `"tokenize,mwt,pos,coref"`; with pos, ties between
  representative mentions go to proper nouns) and it fills `Document.Coref`, every word's `CorefChains`, and an empty word
  for each zero mention. The model (136 MB) and the transformer (2.24 GB) download with `ModelDownloader` or `stanzasharp
  download --processors tokenize,coref`. On both backends; torch.topk's own candidate order is reproduced.
