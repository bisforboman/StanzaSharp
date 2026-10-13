- Managed backend: the character language models look up their character ids without a string per character, so a
  long `Process` call allocates about 17% less (201,313 words, all eight processors: 3.7 → 3.1 GB, 18 → 15 KB per
  word). The peak memory and the output are unchanged, bit for bit.
