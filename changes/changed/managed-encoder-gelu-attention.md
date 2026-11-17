- The managed backend's transformer encoder (the `default_accurate` package, `_electra-large` models, coref) runs its
  GELU and attention faster and gives the same results: ELECTRA-large's encoder takes about 11% less time.
