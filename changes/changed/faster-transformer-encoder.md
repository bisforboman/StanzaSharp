- The `default_accurate` package (and the `_electra-large` models) runs about 1.5x faster on the managed backend: the
  transformer encoder's GELU, attention sum and LayerNorm are vectorized, with byte-identical results on the golden texts.
