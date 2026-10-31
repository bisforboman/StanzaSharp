# Validation and golden data

How StanzaSharp is checked against Python Stanza: golden data, what `tools/make_golden.py` writes, test conventions. Moved from CLAUDE.md on 2026-10-31 (the CI part is in [docs/ci.md](../ci.md)); docs/verification.md is the user-facing summary.

- Golden tests: run Python Stanza (same version, same models) on a fixed English corpus and save
  each stage's output under `tests/golden/`:
  - token and sentence boundaries
  - MWT expansions
  - UPOS/XPOS/feats
  - bracketed trees
- Also save selected intermediate tensors so numeric drift can be localized, for example charlm
  hidden states and tagger logits for a few sentences. Compare floats with a tolerance and
  discrete outputs exactly.
- Tests that need model files must skip, not fail, when `models/converted/en` (or `STANZASHARP_MODELS`) is missing.
- `docs/verification.md` is the user-facing summary of all this (counts, tolerances, CI jobs, test counts per runner).
  Update it when golden data, tolerances or CI jobs change.
- Loaded models are shared between tests (2026-10-11): a pipeline is read-only after Load, so sharing is safe.
  - `SharedPipelines` (class fixture): `pipelines.Get(package, managed, processors, splitSentences)`. Each pipeline loads on
    first use and is disposed when the last class using it ends, so classes running in parallel share it without being
    serialized, and only what running classes need stays in memory. Use it for an options set more than one test uses.
  - Per-class `Fixture`s (Pos, Constituency, Depparse, Sentiment: one `Models` per backend; ManagedBackend: pretrain and
    both backends' charlms) hold models that several tests of the class read. Fixtures load lazily, so no-model runs
    still only skip.
  - Still loading their own: tests of Load itself, options (threads, logger, checksums, cache settings, max calls),
    `ConcurrencyTests` (it runs alone), and one-off options sets (sharing those would only keep them alive longer).
- Versions are pinned in `tools/requirements.txt`; golden data depends on them.
- `tools/make_golden.py` regenerates golden data from `tests/golden/corpus.txt`:
  - `pipeline.conllu`: the full pipeline output (all eight processors), including offsets, MWT ranges, tags, heads,
    `ner=`, and the `# constituency` and `# sentiment` comments.
  - `intermediates.safetensors` + `.json`, covering the first 3 sentences, each run alone:
    - tokenizer logits
    - charlm per-word representations
    - UPOS logits, and `s{i}.pos.upos_logits64`: the same in float64 (fast/ too; `--upos64-only` adds them)
    - per-step parser transition logits
  - `tokenize_stress.txt` + `.conllu`: tokenizer output for a paragraph over 1000 characters and
    for more paragraphs than fit in one batch.
  - `lemma/`: its own folder, written by `write_lemma_golden` (`--lemma-only`): `<name>.conllu` from
    tokenize,mwt,pos,lemma for corpus.txt and every validation*.txt, plus `words.json` (pipeline lemma,
    raw seq2seq output and edit class for hand-picked words: dictionary hits, all edit types,
    unknown and non-BMP characters, a word past `max_dec_len`).
  - `mwt.json`: expansions of a word list, both through the pipeline and classifier-only.
  - `depparse/` (`make_golden.py --depparse-only`): `tokenize,mwt,pos,lemma,depparse` output for
    `corpus.txt` (as `corpus.conllu`) and each `validation*.txt`; arc/label log-probs for the
    first 3 sentences; `mst.json`, Stanza's trees on tie-heavy integer score matrices.
  - `ner/` (`make_golden.py --ner-only`): `tokenize,mwt,pos,lemma,depparse,ner` output for `corpus.txt`
    (as `corpus.conllu`) and each `validation*.txt`, `<name>.json` with each sentence's entities, and
    the emission scores of the first 3 sentences, each tagged alone.
  - `sentiment/` (`--sentiment-only`): `tokenize,mwt,sentiment` `<name>.conllu` + `<name>.json`
    (label and 3 logits per sentence, plus `logits64`: the same batches run in float64, because the classifier is
    ill-conditioned and Stanza's own float32 error can exceed 1e-4; `SentimentTests` accepts a logit within tolerance of
    either) for corpus.txt, each validation*.txt and `sentiment/reviews.txt`
    (opinionated sentences); `all.json` for all of them as one document (two batches).
  - `fast/` (`--fast-only`): `package='default_fast'` output (all seven of its processors) for `corpus.txt`
    (as `corpus.conllu`) and each `validation*.txt`; for the first 3 sentences, each run alone, UPOS logits,
    arc/label log-probs (from the run's tags and lemmas) and NER emissions.
  - `json/` (`--json-only`): `inputs.json` (`JSON_TEXT`: every field of Stanza's JSON) and Stanza's `str(doc)` of it
    (`Document.__repr__`, `json.dumps(to_dict(), indent=2, ensure_ascii=False)`, no final newline) for `default.json`,
    `default_fast.json` and `pos.json` (`tokenize,mwt,pos`). `Json.Write` matches it: field order DEFAULT_OUTPUT_FIELDS,
    a field only when not None, an MWT's range line first with `id` `[a, b]`, `ner`/`multi_ner` (`[ner]`, one model)
    on the token's line, `misc` = the spacing (`Conllu.Misc`), appended last. No text, sent_id, tree or sentiment (not
    in `to_dict`). `Json.Read` = `Document(json)`: sent_id = index, no text; entities rebuilt like `Conllu.Read`.
  - `process_document/` (`--process-document-only`): `input.conllu` (tokenize,mwt on corpus.txt with every annotation
    wrong and a `(ROOT (X wrong))` tree), `corpus.conllu` / `corpus.fast.conllu` = `nlp.process(CoNLL.conll2doc(...),
    processors=` all loaded but tokenize, mwt`)`. The default output equals `pipeline.conllu`.
  - `pretagged/` (`--pretagged-only`): `PRETAGGED_CASES`, `stanza.Pipeline(processors=..., pretagged=True)` on
    fast/corpus.conllu, fast/validation.conllu and process_document/input.conllu, some with lemma/XPOS/UPOS removed (and
    `# sentiment` comments dropped where constituency runs: Stanza would keep them first).
  - `options/` (`--options-only`, both languages): `OPTION_CASES`, Stanza with `ner_predict_tagset`, `lemma_dict_only` or
    `mwt_dict_only` (owner's decision, 2026-10-18: typed `PipelineOptions.NerTagSet`, `LemmaDictionaryOnly`,
    `MwtDictionaryOnly`), `options.json` maps them; `OptionTests` (shard 1; exempt from the no-skip checks elsewhere,
    whose rows lack the chosen models). NER decodes the chosen tag set's head (a missing one throws at Load), the
    lemmatizer's `LemmatizeByDictionary` is trainer.predict_dict (no seq2seq, no contextual classifiers), MWT's
    `dictOnly` loads no net. The batch sizes (#174, owner's decision 2026-10-18: typed properties `MwtBatchSize`,
    `PosBatchSize`, `PosBatchMaximumTokens`, `LemmaBatchSize`, `ConstituencyBatchSize`, `DepparseBatchSize`,
    `DepparseMinLengthToBatchSeparately`, `SentimentBatchSize`, `NerBatchSize`; null = the checkpoint's `batch_size`,
    or Stanza's processor default where it has its own: pos 5,000 tokens, constituency 50, depparse 150, sentiment 5,000)
    are rows `sentiment_batch_size=20`, `batch_sizes` (every one small, `SMALL_BATCHES`) and `default_fast+batch_sizes`;
    only sentiment's changes the golden texts (depparse's moved no head at any size tried). Each processor's internal
    `WithBatchSize` returns a copy sharing its nets, made in Load, so nothing changes after Load. `TokenizeBatchSize`
    (owner's choice) has no working Stanza twin: TokenizeProcessor reads `trainer.args['batch_size']`, the checkpoint's,
    whatever `tokenize_batch_size` says; `TokenizeBatchSize_GivesTheSameTokens` checks it against the default tokens.
    Model files of one's own (#174, owner's decisions 2026-10-19): `PipelineOptions.ModelPaths`
    (`IReadOnlyDictionary<string, ModelPath>`, `Processor` keys; `ModelPath(Model, Pretrain, ForwardCharlm, BackwardCharlm)`,
    implicit from a string) = Stanza's `{processor}_model_path` / `_pretrain_path` / `_forward_charlm_path` /
    `_backward_charlm_path`. `Pipeline.CustomPaths` normalizes them (full base paths without `.pt`/`.json`/`.safetensors`).
    The processor keeps the replaced model's name, so every name-keyed table (`Dependencies`, `Transformers`, `Unread`)
    still says what it reads; only the file changes, and each named dependency replaces the one of its kind. A path for
    a processor the package lacks (or for the identity lemmatizer) gets `Pipeline.CustomModel` and reads only what its
    ModelPath names (as Stanza). Pretrains and charlms are keyed by dependency name or a custom file's full path, so a
    custom pair loads apart and gets its own `CharlmCache` scope. Both Models and ModelPaths naming one processor throws;
    `FilesFor`/`ModelFiles`/`Verify` leave the custom files out (VerifyChecksums skips them); a missing custom file names
    its ModelPath; a custom model's ArgumentException at load is rethrown naming its ModelPath. The tokenizer and the
    lemmatizer now ignore charlms they don't read, like the tagger and parsers. Rows `pos_model_path`, `ner_model_path`
    (nocharlm models in the default package) and `default_fast+constituency_model_path` (`{lang_dir}` in options.json);
    `ModelPathTests` (renamed copies, default_fast + a `_charlm` tagger, the throws, `FilesFor`). compare/benchmark take no
    paths. `LemmaBeamSize` (#174, 2026-10-19) = Stanza's `lemma_beam_size` (overrides the checkpoint's `beam_size`, 1):
    the lemmatizer's `WithBeamSize` copy decodes with `Seq2SeqConfig.Beam` (MWT's beam search: the edit logits from the
    unrepeated encoding, only the first batch rows mixing in the copy distribution, as in Stanza); dictionary and contextual
    classifiers unchanged. Rows `lemma_beam_size=5` (English, with `options/lemma_rare.txt`, and German) and
    `lemma=combined_charlm+lemma_beam_size=3`; with the models of 2026-10-30 the beam changes 1 lemma (`combined_charlm`, validation_contractions; before, 4 of `combined_nocharlm`, all in validation_multilingual).
  - `langid/langid.json` (`--langid-only`; #174, owner's decisions 2026-10-19, docs/langid-plan.md): `LANGID_TEXTS` (47
    texts, 36 languages, edges) + validation.txt's first paragraph through Stanza's langid processor (a minimal
    resources.json of its own), alone and in bulk, unrestricted and with `langid_lang_subset` en,de, with
    `LangIDBiLSTM.forward`'s sums in float32 and float64; `LanguageIdentifierTests`. `LanguageIdentifier` (public, in the
    facade: `LangIdModel` groups texts by code-point length so no row is padded, `UNK` for unknown characters, the subset as a
    mask, argmax; an empty text throws, Stanza crashes) over `ILangIdNet` (`ManagedLangIdNet`: embedding, `ManagedLstm`
    2 × 150 bi, Linear per position, summed in double; `LangIdNet` in StanzaSharp.Cuda), `BackendModels.LangId`. The model is
    `multilingual/langid/ud.pt` (`ModelDownloader.LanguageIdentifierFile`, `DownloadLanguageIdentifierAsync`, the tool's
    `download DIR --lang multilingual`), not in `Files`; CI downloads it in golden shard 1 on every run (`tools/ci-models.sh
    langid`), and the other jobs' no-skip checks exempt `LanguageIdentifierTests`. `clean_text` is not ported.
  - `langid/multilingual.json` (`--langid-only` too, `MULTILINGUAL_TEXTS`): Stanza's `MultilingualPipeline(restrict=True)`
    with en and de (default packages, all processors), each document's lang and `"{:C}"`. `MultilingualPipeline` (public,
    PR 2; owner's decisions 2026-10-19, loading 2026-10-30: "Lazy LRU, like Stanza"): `Load(modelRoot, MultilingualOptions)`
    loads the identifier from `<root>/multilingual`; each language's `Pipeline` loads from `<root>/<lang>` at its first
    text, at most `MaxLoadedPipelines` (10, Stanza's `max_cache_size`) stay loaded, least recently used freed first, and a
    lease count per pipeline frees an evicted one only when no call is using it (so the live count can briefly exceed the
    bound; one lease per language group of a call, so a call may span more languages; a failed load isn't cached).
    `MultilingualOptions.Languages`: a list of `PipelineOptions`, each keyed by its `Language`, duplicates throw, validated
    at Load through `FilesFor` plus a folder check; null = every `SupportedPackages` language with a folder (default
    package); unlike Stanza's `lang_configs` the list is also the set; no shared default processors, unlike Stanza's
    `processors=`. The identifier's `no` routes to `nb` (`Document.Lang` stays `no`, as Stanza). `Restrict` (default false,
    as Stanza) sets the identifier's subset to them (nb as `no`; with `Languages` null, codes the identifier lacks are left
    out: `LanguageIdentifier.LoadSubsetOfKnown`) unless `LanguageIdentifier.Languages` names one. A lazy load sets
    `ManagedThreads.Count` like any Load (null Threads: a no-op after the first). `Process` identifies all texts, then one
    bulk `Pipeline.Process` per language (a single text too: Stanza wraps it in a list), `Document.Lang` set (Core; not in
    CoNLL-U or JSON); a language without a pipeline gives a document with only `Text` and `Lang`. `MultilingualPipelineTests`
    (en, de and multilingual next to `Repo.StanzaModels`: golden shard 1; exempt from the other jobs' no-skip checks).
  - `postprocess/postprocess.json` (`--postprocess-only`): `tokenize_postprocessor=postprocess` (merges "New York", splits
    "cannot", hand-expands "gonna", flags "wanna", merges the last two sentences; `PostprocessorTests.Postprocess` is the
    same in C#) on `POSTPROCESS_TEXTS` alone, in one bulk call and with `tokenize_no_ssplit` (#174:
    `PipelineOptions.TokenizePostprocessor`, `PostprocessorToken(Text, IsMultiWord, Expansion)`). `Tokenizer.Postprocess` is
    utils.postprocess_doc: after `Decode` (long tokens replaced only after it, as TokenizeProcessor.process does), before
    bulk's `Split`; offsets by `IndexOf` after the previous token; a hand expansion through `Token.SetWords` (MWT's
    set_mwt_expansions, moved to Core). Stanza shifts a hand-expanded token's word ids by the automatic MWT expansions
    earlier in the whole document (SKIP branch, `+ idx_e`) and then crashes building dependencies; we number them as
    written, so the golden texts put every hand expansion before any automatic one.
  - `stream/<package>.json` (`--stream-only`): Stanza's `Pipeline.stream` over `BULK_TEXTS` and validation.txt's first 20
    paragraphs with `batch_size` 4, each document's `"{:C}"`; `InputModeTests.Stream_MatchesStanzasStream` (#174:
    `Pipeline.Stream(texts, batchSize = 50, ct)`, lazy, one bulk `Process` per batch, sent_ids counted across the stream).
  - `helpers/` (`--helpers-only`, no models): Stanza's document helpers (`num_tokens`, `num_words`, `dependencies_string()`,
    `tokens_string()`, `words_string()`) on `CoNLL.conll2doc` of `HELPER_FILES` and a generated `crafted.conllu` (texts whose
    Python `repr` needs quotes or escapes, a sentence without heads); `DocumentHelperTests` (#174, owner's decision
    2026-10-18: `Document.TokenCount`/`WordCount`, `Sentence.Dependencies`/`DependenciesString()`/`TokensString()`/
    `WordsString()`, `Token`/`Word.ToString()` = `pretty_print()`; `PyString.Repr` is Python's `repr(str)`).
  - `fuzz.json` (`--fuzz-only`): `FUZZ_TEXTS` (German: `GERMAN["FUZZ_TEXTS"]`), the minimized differences
    `tools/text_fuzz.py` found (docs/text-fuzzing.md), each processed alone by both packages; compared with code-point
    offsets (`TokenizerTests.Pipeline_FuzzFindings_MatchGolden`). Add a new finding's shrunk text there.
  - `pretokenized/` (`--pretokenized-only`): `inputs.json` (cases of token lists, plus `corpus`: corpus.txt's
    Stanza tokens) and `<name>.conllu` / `<name>.fast.conllu` for both packages.
  - `bulk/<package>.json` (`--bulk-only`): one `bulk_process` call over `BULK_TEXTS` (tiny, empty and
    whitespace-only texts; stored whole) + corpus.txt + validation*.txt (stored as the golden file processed alone,
    a sent_id offset and the sentences bulk changes), so it stays small.
  - `validation*.conllu`: full pipeline output for each hand-written `validation*.txt` (news,
    academic, instructions, social, dialogue/poetry, tech, multilingual, contractions, long,
    whitespace, nonbmp; 835 sentences, 845 with corpus.txt). They are read as bytes, so CR/CRLF reach Stanza as-is.
    `validation_whitespace.txt` is `-text` in `.gitattributes` to keep its CRLF. The test compares
    each file and reports its first differing sentence. `validation_nonbmp` is compared without
    `start_char`/`end_char`, because offsets differ by design after non-BMP characters.
  - `tests/golden/de/` (`make_golden.py --lang de`, 2.9 MB): the same files for 13 hand-written German texts (181
    sentences, 2,227 words; the validation set adds poetry, umlauts/compounds with decomposed umlauts, quotes) and
    German word lists (`GERMAN`: MWT_WORDS with non-dictionary contractions for the seq2seq, LEMMA_WORDS, PRETOKENIZED_CASES,
    BULK_TEXTS, LONG_TOKEN_TEXT at max_seqlen 400). No pt/ or mst.json (language-independent). The full run now also
    writes no_ssplit/ (both languages). Sentiment's float64 pass sets the default dtype only around the sentiment
    processor (German's seq2seq MWT breaks under float64). The `combined` package's golden equals `depparse/` (same
    models; `TreebankPackageTests` checks it). German tests are rows of the English golden theories: a `string lang` last
    parameter, `[InlineData(..., "en")]` plus `[German(...)]` (Repo.cs: appends `"de"`, skips without German models),
    and `Repo.ModelsFor(lang)` (models/converted/de next to the English models, else models/stanza/de),
    `Repo.GoldenFor(lang)`, `Repo.ModelOf(lang, processor, package)`, `Repo.Charlm(lang, forward)`; class fixtures keep
    models per (backend, language) in `ModelCache`. German-only tests have German in their name (`[GermanFact]`,
    `[GermanTheory]`). A text with characters outside the BMP (`PipelineTests.HasNonBmp`; German `validation_social`
    too) is compared without offsets.
- Python `str` semantics that .NET lacks live in `Core.PyString` (`Lower` with İ and final sigma,
  `IsUpper`/`IsLower` with Other_Uppercase/Other_Lowercase, and `IsSpace` with U+001C..U+001F).
  Use it wherever Stanza calls `lower()`, `isupper()`, `isspace()` or `split()`. Regexes ported
  from Python need care too: Python's `\b`/`\w` count No/Nl characters such as `²` as word
  characters and combining marks as non-word (see `Tokenizer.PyRegex`).
