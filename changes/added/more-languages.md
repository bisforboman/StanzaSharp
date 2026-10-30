- Spanish, French, Italian, Portuguese and Dutch (`PipelineOptions.Language = "es"`, `"fr"`, `"it"`, `"pt"`, `"nl"`):
  each language's Stanza `default` package, and `default_fast` except in Spanish (Stanford's Spanish `default_fast`
  parser file is truncated), byte for byte against Python Stanza; `stanzasharp download` and `compare` take the new
  `--lang` codes.
