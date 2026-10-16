- `tokenize` now loads and runs the package's `mwt` too, as Stanza does (`add_mwt`): `Processors = "tokenize,ner"`
  (or `"tokenize,sentiment"`, `"tokenize"`) expands multi-word tokens into words like Stanza's output, and
  `ModelDownloader` and `stanzasharp compare` select the files Stanza loads. A model folder downloaded on 1.x for such a
  list lacks the mwt model: `Pipeline.Load` then throws a `FileNotFoundException` naming the file; re-run
  `stanzasharp download` (or `ModelDownloader.DownloadAsync(dir, options)`), which fetches only the missing files.
