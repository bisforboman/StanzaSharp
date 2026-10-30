- 15 more languages whose Stanza `default` tagger and parser read no word vectors (their `_nopretrain` models: Stanza
  leaves out the pretrained input when a model's config says `pretrain` is false, and so does StanzaSharp now): Bororo,
  Coptic, Egyptian, Middle French, Khoekhoe, Old East Slavic, Naija, Old Occitan, Maghrebi Arabic French, Pomak, Turkish
  German, Zaar, Southern Kurdish, Shanghainese and Nheengatu (`bor`, `cop`, `egy`, `frm`, `naq`, `orv`, `pcm`, `pro`,
  `qaf`, `qpm`, `qtd`, `say`, `sdh`, `wuu`, `yrl`), byte for byte against Python Stanza. StanzaSharp now has 101
  languages (docs/languages.md).
