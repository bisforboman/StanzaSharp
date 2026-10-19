- `MultilingualPipeline`: Stanza's `MultilingualPipeline`. The language identifier routes each text to its language's
  pipeline (English and German by default, all loaded at `Load`), each language's texts in one bulk call, and sets the new
  `Document.Lang`. A text in a language without a configured pipeline comes back with only `Text` and `Lang`;
  `MultilingualOptions.Restrict` (Stanza's `restrict`) has the identifier answer only the configured languages.
