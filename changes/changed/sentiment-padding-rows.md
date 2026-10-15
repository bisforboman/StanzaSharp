- Managed backend: sentiment computes its biLSTM's first-layer input projection once for all padding rows of a batch
  instead of once per row, and runs its strided (5, 5) filter with SIMD across the 8 channels. Results are bit for bit
  the same; sentiment is faster on short and mid-length documents (docs/performance/round-28-sentiment-padding.md).
