- MWT seq2seq models (German, English `partut`) are decoded with a beam search of 5, as Stanza does (its MWT trainer
  never passes its `beam_size` of 1 through), instead of greedily, which could expand a token differently.
