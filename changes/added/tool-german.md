- `stanzasharp compare` and `stanzasharp benchmark` take `--lang de` for German: Python Stanza runs its German pipeline
  with the German entry of Stanza 1.15.0's resources.json (shipped with the tool), StanzaSharp loads with
  `Language = "de"`, and `--models` defaults to `models/stanza/de`. `--package` takes German's packages too (`combined`,
  `gsd` and `hdt` run only their processors, tokenize to depparse). `benchmark --lang de` times a built-in German text
  (our German test texts, cut like the English one), and the block names the language. The manual Benchmark workflow
  has a `lang` input.
