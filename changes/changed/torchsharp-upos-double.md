- TorchSharp backend (`StanzaSharp.Cuda`): the tagger's UPOS head (`upos_clf`) sums in double, as the managed backend
  does, so its logits come closer to Stanza's float64 values and no longer depend on libtorch's CPU kernels in the last
  layer (German `default_fast` on an Intel AVX-512 CPU: 1.2e-4 from Stanza before). Tags are unchanged.
