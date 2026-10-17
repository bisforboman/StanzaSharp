- `PipelineOptions.Models`: the rest of Stanza's `default_accurate` models, constituency `ptb3-revised_electra-large`
  and sentiment `sstplus_electra-large`, byte for byte with Stanza on both backends. The parser reads the frozen
  `google/electra-large-discriminator` (sharing the tagger's encoder pass); the sentiment classifier applies its LoRA
  adapters to that transformer at run time, so no second 1.3 GB file is needed.
