- Stanza's document helpers: `Document.TokenCount` and `WordCount`, `Sentence.Dependencies` (a list of `Dependency`:
  head word, relation, dependent; the head null for the root), `Sentence.DependenciesString()`, `TokensString()` and
  `WordsString()`, and `Token.ToString()` / `Word.ToString()`, each giving exactly what Stanza's `num_tokens`,
  `num_words`, `dependencies`, `dependencies_string()`, `tokens_string()`, `words_string()` and `pretty_print()` give.
