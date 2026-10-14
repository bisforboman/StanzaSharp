- TorchSharp backend (`StanzaSharp.Cuda`): the constituency parser is about 20% faster on the CPU (each step reads and
  writes its states with a constant number of tensor ops), and the dependency parser's memory peak is about 280 MB
  lower on a 27k-word text (no padded input, no full label-score tensor). Outputs and scores are bit-identical.
