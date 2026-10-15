- Managed backend: the dependency parser scores the padding columns of a batch's arc matrix once per row instead of
  once per column, and sums its arc dot products with SIMD. Output, arc log-probs and label scores are bit for bit the
  same.
