- Managed backend: the dependency parser reuses the tagger's backward character-LM outputs for every sentence the
  charlm cache holds, not only for one-sentence documents, and runs only ROOT's two characters on from the tagger's
  final state. Output is bit for bit the same. The cache keeps about 4 KB more per sentence for it (not counted in
  `CharlmCacheOptions.MaxWords`).
