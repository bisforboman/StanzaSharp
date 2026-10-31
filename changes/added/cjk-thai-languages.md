- Japanese, Thai, Simplified and Traditional Chinese, Korean and Classical Chinese (`ja`, `th`, `zh-hans`, `zh-hant`,
  `ko`, `lzh`), `default` package, byte for byte against Python Stanza. Their tokenizers read Stanza's dictionary
  features (whether the characters around each one form a word of the model's lexicon), Chinese drops a line break
  inside a paragraph, and Chinese, Japanese and Korean tokens keep no spaces, as in Stanza. StanzaSharp now has 107
  languages (docs/languages.md).
