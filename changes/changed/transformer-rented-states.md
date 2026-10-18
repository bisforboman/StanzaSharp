- The managed transformer encoder rents its per-piece states, so a `default_accurate` call allocates about 86 KB per word
  instead of 150, and peaks about 200 MB lower.
