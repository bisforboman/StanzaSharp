- Managed backend: short texts (up to 4 sentences per tagger batch) are about 15% faster with the `default` package
  (one 5–25-word sentence: 83 → 70 ms median, 143 → 117 ms p90 on a 4-core cloud Xeon). The tagger's charlm passes
  now also compute, as extra rows of the same steps, the dependency parser's forward pass and the token passes that
  sentiment and NER need for sentences with multi-word tokens. Output is unchanged, bit for bit.
