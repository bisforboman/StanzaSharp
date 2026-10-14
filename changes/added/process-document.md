- `Pipeline.Process(Document)` (and a `CancellationToken` overload) runs the pipeline's processors on a document
  that already has its tokens, such as one from `Conllu.Read` or `Json.Read`, like Stanza's `process_conllu`:
  tokenize and mwt are skipped, every other processor overwrites what it sets, and the rest is kept.
