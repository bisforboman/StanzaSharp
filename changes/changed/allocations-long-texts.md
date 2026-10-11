- Less memory on long texts (managed backend): the parsers and classifiers reuse their per-batch arrays instead of
  leaving them to the garbage collector (constituency recycles its parser states, the dependency parser decodes in
  rented buffers), so a call allocates about 18 KB per word instead of 54, and one call on 201,313 words with all eight
  processors peaks at about 1.9 GB instead of 2.2–2.3 GB. Outputs are unchanged.
