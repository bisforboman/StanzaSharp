- Coreference results in the document model, as Stanza has them: `Document.Coref` (`CorefChain`: index, mentions,
  representative text and index; `CorefMention`: sentence, start and end word, or a zero mention's empty word),
  `Word.CorefChains` (`CorefAttachment`: chain, start, end, representative) and empty words (`Sentence.EmptyWords`,
  `Word.EmptyIndex`). `Conllu` and `Json` write them byte for byte like Stanza (`coref_chains=`, empty nodes `i.m`) and
  read them back, rebuilding the chains. No processor sets them yet.
