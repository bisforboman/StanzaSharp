# Language identification and a multilingual pipeline (#174)

Stanza's `langid` processor and `MultilingualPipeline`, the last item of #174 the owner chose (2026-10-18). This is the
plan; the owner's decisions are at the end. Done: PR 1 `LanguageIdentifier`, PR 2 `MultilingualPipeline` (per-language
`PipelineOptions` as a list, each keyed by its `Language`; `Restrict` off by default, as in Stanza; no shared default `Processors`).

## What Stanza does (1.15.0)

- **Model** (`multilingual/langid/ud.pt`, 9 MB, MD5 `84cd508b9702b31cc236ba2b6d94e41a`; `models/langid/model.py`
  `LangIDBiLSTM`): a character embedding (8,704 characters, 150-d, `<PAD>` as padding), a 2-layer biLSTM of 150 per
  direction, a Linear 300 → 68 languages at every position, the sum over all positions, then argmax. `char_to_idx` keys
  are single code points plus the literal `UNK` and `<PAD>`; an unknown character is `UNK`. The languages are UD codes
  (`en`, `de`, `fr`, `zh-hans`, `zh-hant`, `grc`, `lzh`, ...). `loss_train.weight` is training only.
- **Processor** (`pipeline/langid_processor.py`): documents are grouped by text length, and each group is one tensor,
  so no row is ever padded; `langid_batch_size` is passed to the model, which never reads it. `langid_lang_subset` adds
  -inf to every other language's score. `langid_clean_text` (off by default) strips URLs, handles, hashtags, `!`/`.`
  runs and emoji (through the `emoji` package). `MAX_SEQ_LENGTH_DEFAULT` (1000) is defined but unused: a text is one
  sequence however long. The result is `doc.lang`, which neither CoNLL-U nor `to_dict` writes.
- **Measured edges:** an empty text crashes (`Expected sequence length to be larger than 0`); whitespace, a single
  letter or digits get an arbitrary language (`"   "` → `mr`, `"12345"` → `ja`). With the subset `en,de` a French text
  becomes `en` and Japanese `de`: the subset changes predictions, not just routing.
- **`MultilingualPipeline`** (`pipeline/multilingual.py`): runs langid over the documents, groups them by language, and
  hands each group to that language's `Pipeline` in one bulk call (so sentiment and depparse batch as in bulk). The
  pipelines are built lazily per detected language, in an LRU cache of 10 (`max_cache_size`), from `lang_configs`
  (per-language keyword arguments) or the default `processors`. `restrict=True` sets the subset to `lang_configs`' keys.

## The port

- **Nets:** `ILangIdNet` with `LangIdNet` (TorchSharp) and `ManagedLangIdNet`: the embedding, `ManagedLstm` over the
  equal-length group (no padding), the Linear per position, the sum over positions in position order, in float as
  Stanza does. No new kernel. Language codes byte-identical; the summed scores against Stanza's float32 and float64 sums
  with the relative 1e-4 tolerance (sums over a long text pass |100|), as for UPOS and sentiment.
- **Files:** the model belongs to no language StanzaSharp has, so a small table of its own in `ModelDownloader` (path,
  MD5), downloaded into `<model root>/multilingual/langid/ud.pt` by `DownloadAsync` when asked for, and per CI run like
  the transformer group (9 MB) rather than in a cache key.
- **API:** `Document.Lang` (Core, public: Stanza's `doc.lang`); `MultilingualPipeline.Load(modelRoot,
  MultilingualOptions)` with per-language `PipelineOptions`, a default `Processors`, `Restrict`/`LanguageSubset`; `Process(string)`
  and `Process(IEnumerable<string>)` with `CancellationToken` overloads, each document's `Lang` set; perhaps a public
  `LanguageIdentifier` for callers who route themselves.
- **Out of scope:** `clean_text` (it needs the `emoji` package's tables; off by default), `langid_batch_size` (unread),
  compare and benchmark.

## Golden

`make_golden.py --langid-only`:
- `langid/langid.json`: a hand-written list of short texts across many scripts (whitespace, one character, digits, mixed
  scripts, long), each with the predicted code and both sums, unrestricted and with the `en,de` subset;
- `langid/multilingual.json`: a mixed English and German list through `MultilingualPipeline(restrict=True)`, each
  document's language and CoNLL-U.

## The owner's decisions (2026-10-19)

1. A text detected as a language StanzaSharp has no models for comes back with only `Lang` set (no sentences),
   documented, so the caller can route it elsewhere. Stanza would load that language's pipeline.
2. `LanguageIdentifier` is public too: `Load(dir)` and `Identify(texts)` → codes, for callers who route texts themselves.
3. `MultilingualPipeline.Load` loads every configured language's pipeline (English and German by default); `Process`
   changes nothing shared, like `Pipeline`.
4. Two PRs: the identifier (nets, downloader, golden), then `MultilingualPipeline`.

## The owner's decision (2026-10-30), with more languages

With 86 languages, the owner chose to load like Stanza ("Lazy LRU, like Stanza"), replacing decision 3: a language's
pipeline loads at its first text, and `MultilingualOptions.MaxLoadedPipelines` (Stanza's `max_cache_size`, 10) bounds
how many stay loaded, the least recently used one freed first (once no call is using it: a lease count per pipeline, so
calls may still run concurrently). `Languages` null means every supported language with a folder under the model root
(default package), no longer English and German; explicit `Languages` are both the options and the set (Stanza's
`lang_configs` only override). Still never a download: a language without a folder comes back with only `Lang` (decision 1);
a missing model file throws at the language's first text, a missing folder of a listed language at `Load`. The
identifier's `no` is routed to `nb` (Stanza's alias); with `Restrict` and no `Languages`, the folders the identifier has
no label for stay out of the subset.
