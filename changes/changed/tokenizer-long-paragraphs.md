- The tokenizer is about 2.4x faster on a text with a paragraph over 1,000 characters (the benchmark's 2,009 words:
  144 → 60 ms on 8 threads, 257 → 145 ms on 1): after such a paragraph's first window it no longer runs the other,
  finished paragraphs of its batch through the network as padding. The output is unchanged, bit for bit.
