- Long texts allocate less garbage (managed backend): the tokenizer keeps each paragraph as one string and builds the
  model's inputs per batch, constituency reuses its stack-state objects, dependency decoding no longer allocates an
  array per tree node, NER's emission scores are rented, and lowercasing an ASCII word allocates nothing when it has no
  capital. Outputs are bit-identical.
