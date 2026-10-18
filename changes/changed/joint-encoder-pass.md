- `default_accurate` (and other combinations of a frozen transformer model with a LoRA one, such as `sstplus_electra-large`)
  runs short calls about 15% faster on the managed backend: the frozen transformer and the LoRA adapters' pass share one
  sweep over the transformer's weights. Results are byte-identical.
