- StanzaSharp.TasmanianDevil: `StanzaNerRecognizer.LinkNames` links each person's mentions across a text (`Karin
  Lindqvist`, `Karins`, `K. Lindqvist`, `LINDQVIST, KARIN`) and finds the capitalized mentions NER missed;
  `StanzaNerRecognizer.PersonPlaceholders` turns that into numbered placeholders (`<PERSON_1>`, `<PERSON_2>`) for
  TasmanianDevil's anonymizer instead of one `<PERSON>` for everyone. Off by default.
