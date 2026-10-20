# StanzaSharp

A native C# port of the inference side of [Stanza](https://github.com/stanfordnlp/stanza), Stanford's
PyTorch-based NLP library, running the original pretrained models on a managed backend (C# SIMD kernels, the
default from 0.5) or through TorchSharp (see "Managed backend").

## Scope

- **Language:** English and German (`PipelineOptions.Language = "de"`, 2026-10-13; docs/german-plan.md, "Checkpoint
  findings (German)"). German runs every processor of `default`, `default_fast`, `combined`, `gsd` and `hdt` byte for
  byte on both backends.
- **Processors:** all eight of Stanza's English default, in Stanza's order (`PIPELINE_NAMES`): tokenize,
  mwt, pos, lemma, constituency, depparse, sentiment, ner. This is `Pipeline.AllProcessors`, the default
  (user's decision, 2026-10-06: like Stanza; it costs about 36% more time than without ner and sentiment).
  - `Pipeline.Process` runs them in that order whatever order they are listed in. Only the CoNLL-U
    comment order is observable (`# sentiment` after `# constituency`).
  - `pipeline.conllu` and `validation*.conllu` are generated with all eight (`PROCESSORS` in
    make_golden.py).
- **Inference only.** Training stays in Python; we load Stanza's released weights.
- **Package:** Stanza's English *default* package, which needs no transformer:
  - tokenize: `combined_nocharlm`
  - mwt: `combined`
  - pos: `combined_charlm`
  - lemma: `combined_nocharlm`
  - constituency: `ptb3-revised_charlm`
  - depparse: `combined_charlm`
  - ner: `ontonotes-ww-multi_charlm`
  - sentiment: `sstplus_charlm`
- **Package `default_fast`** (user's decision, 2026-10-06: selected by Stanza's package name,
  `PipelineOptions.Package`): tokenize, mwt and lemma as above; pos `combined_nocharlm`, depparse
  `combined_nocharlm`, ner `ontonotes-ww-multi_nocharlm`; sentiment `sstplus_charlm`; no constituency.
  Its default processors are those seven. `Pipeline.Packages` maps package → processor → model.
- **Package `default_accurate`** (owner's decision, 2026-10-17; docs/transformer-plan.md, PR 4; German below): tokenize and lemma `combined_charlm`, mwt `combined`, pos and depparse
  `combined_electra-large`, constituency `ptb3-revised_electra-large`, ner `ontonotes-ww-multi_electra-large`, sentiment
  `sstplus_electra-large`; all eight processors (see "Transformers"). Golden row `default_accurate` of `MODEL_CHOICES`
  (no models chosen: `package=default_accurate`), `TransformerModelTests`. German's (#170, 2026-10-18;
  docs/german-accurate-plan.md): tokenize and lemma `combined_charlm`, mwt `combined`, pos and depparse
  `combined_german-nlp-electra` (frozen, one shared pass), constituency `spmrl_german-nlp-electra` and sentiment
  `sb10k_german-nlp-electra` (each with its LoRA adapters, a pass each; constituency's `bert_lora` is a top-level key of
  the checkpoint, `ConstituencyParser` builds its `WithLora` net like the classifier), ner `germeval2014`; transformer
  `german-nlp-group/electra-base-german-uncased`. Golden rows `pos+depparse=german-nlp-electra`,
  `constituency=spmrl_german-nlp-electra`, `sentiment=sb10k_german-nlp-electra` and `default_accurate` in
  `tests/golden/de/models/` (`de/models/transformer_long.txt`: one sentence of 558 pieces, two windows).
- **English treebank packages** (owner's decision, 2026-10-15; no new API, `PipelineOptions.Package`): `combined` (the
  default package's models), `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`, `eslspok` (no mwt; lemma `identity`)
  and the biomedical `craft`, `genia`, `mimic` (no mwt; own pretrains, mimic own charlms):
  tokenize, mwt, pos, lemma, depparse only, byte for byte on both backends (`TreebankPackageTests` against
  `tests/golden/<package>/`). "Checkpoint findings (English treebank packages)" has what they needed.
- **One model per processor** (owner's decisions, 2026-10-16; docs/per-processor-models-plan.md has the inventory and
  the decisions): `PipelineOptions.Models` (`IReadOnlyDictionary<string, string>?`, `Processor` keys), Stanza's
  `processors={'ner': 'i2b2'}`. A value is a model of `Pipeline.ModelsOf(lang, processor)` (the packages' models plus
  `Pipeline.Alternatives`) or a package's name (its model). With `Processors` null the set is the package's processors
  plus the keys (requirements added); with `Processors` set a key outside it throws. Like Stanza's add_mwt, a replaced
  tokenizer brings its treebank's mwt unless Models names one. Unknown, transformer and unported names throw
  `ArgumentException` listing the supported ones (Stanza warns and ignores).
  - Several charlm pairs (owner's decision, 2026-10-16, part 2): like the pretrains, the pipeline loads every charlm
    `Dependencies` names (`BackendModels.Charlm(path, ...)`, keyed by path, e.g. `forward_charlm/mimic`) and each
    processor gets its model's own pair (`Charlms`). default + `ner: i2b2` loads 1billion and mimic. A call's
    `CharlmCache` serves only the tagger's pair: a processor whose model reads another pair gets null (`CacheFor` in
    `AnnotateWithModels`), so no entry, single-sentence table entry or backward continuation crosses pairs.
    `CharlmCacheOptions.MaxWords` therefore bounds that one cache (per pair and total alike). Tested by
    `ModelChoiceTests` (`default+ner=i2b2` golden, cache off, one-sentence calls against the uncached pipeline).
  All in `SelectModels`, so Load, `ModelDownloader.FilesFor`/`DownloadAsync(dir, options)` and the tool (`--model
  P=NAME`, repeatable or comma-separated, on download, compare, benchmark and the Cli sample) agree. compare/benchmark
  pass Stanza the resolved processor → model map as a JSON processors dict (package None). Alternatives (first PR):
  English constituency `ptb3_charlm`, ner `ontonotes_charlm`, `ontonotes_nocharlm` (fasttextcrawl pretrain, 300-d),
  `combined_charlm` (3 tag sets, predict_tagset 0), `i2b2` (old format with a `PretrainedWordVocab` delta: lowercased
  word looked up as is, spaces as U+00A0; mimic pretrain and charlms); German ner `conll03` (germeval2014's format).
  Byte for byte on both backends (`ModelChoiceTests` against `tests/golden/models/<name>/`, `make_golden.py
  --models-only`, `MODEL_CHOICES`; i2b2 on the mimic package also on a hand-written `models/clinical.txt`). A new
  alternative needs `Alternatives`, `Dependencies` (a transformer model also `Transformers`), `ModelDownloader.Files`, a `MODEL_CHOICES` row and a
  `ModelChoiceTests` row (`EveryAlternative_HasItsFilesAndDependencies`, `EmbeddedResources_MatchModelDownloader`
  check the tables). The domain NER models (owner's decision, 2026-10-16, part 2): `conll03_charlm` (glove pretrain),
  `radiology` (mimic) and the biomedical anatem, bc4chemd, bc5cdr, bionlp13cg, jnlpba, linnaeus, ncbi_disease, s800
  (biomed pretrain, pubmed charlms; `lowercase: false`). `NerTagger` reads `lowercase` (the word as written for both
  vocabs when off) and `char_lowercase` (radiology: each character lowercased alone, `LowerChars`, a multi-character
  result being one unknown unit like Python's; then neither cache reads nor writes, since its charlm input differs from
  the tagger's). Golden on hand-written `models/biomedical.txt` / `radiology.txt` (craft + each biomedical model,
  default + bc5cdr, mimic + radiology). The treebanks' `_nocharlm` pos and depparse (owner's decision, 2026-10-16, part
  3): English `{ewt,gum,lines,partut,atis,childes,eslspok}_nocharlm` pos and depparse, depparse `mimic_nocharlm`
  (mimic pretrain, so the head-constraint repair), German `{gsd,hdt}_nocharlm`; no new code (the `_charlm` twins' legacy
  heads and composite XPOS + `combined_nocharlm`'s character model). Golden rows `pos+depparse=<tb>_nocharlm` on their
  package (`depparse=mimic_nocharlm` on mimic), corpus.txt only; their own CI cache (`models-treebank-choices`, about 1.2 GB,
  tools/ci-models.sh `en-treebank-choices` + `de-treebank-choices`). The `_charlm` tokenizers and lemmatizers (owner's
  decision, 2026-10-16, part 4): English tokenize `{combined,ewt,gum,lines,partut,atis,childes,eslspok}_charlm`, lemma
  `{ewt,gum,lines,partut,atis,childes}_charlm`, German tokenize and lemma `{combined,gsd,hdt}_charlm`, and (owner's
  decision, 2026-10-17) English lemma `combined_charlm` with its contextual `LemmaClassifier`s ("Checkpoint findings",
  lemma; golden row `lemma=combined_charlm` on the default package, corpus + validation + validation_contractions: 33
  decisions, 4 changing the lemma, `contextual.json` logits, `ContextualLemmaClassifiers_MatchGoldenLogits`). They read per-character charlm
  states (`Nn.IPerCharLanguageModel`, "Checkpoint findings (English treebank packages)"); golden rows
  `tokenize+lemma=<tb>_charlm` (tokenize only for eslspok and `combined`, which also runs `tokenize_stress.txt`), each lemma
  row with `seq2seq.json` (the model alone on every corpus.txt word, `CharlmLemmatizer_Seq2Seq_MatchesGolden`); the same
  CI cache; `CharlmInputTests` (per path). Byte for byte on both backends.
- **Shared dependencies:** `pos`, `constituency`, `depparse`, `ner` and `sentiment` all depend on the
  pretrained word-vector file (pretrain). Every `_charlm` model also depends on the forward and backward
  character LMs (charlm); the `_nocharlm` ones have their own character LSTM (`Nn.CharacterModel`) instead.
  The charlm is shared infrastructure and lives in `StanzaSharp.Nn`. The pipeline loads only what the
  selected models read (`Pipeline.SharedModels`): in `default_fast` the charlms only with sentiment.
- **Transformers (`default_accurate`, owner's decisions 2026-10-17; docs/transformer-plan.md):** its pos,
  depparse, ner and constituency read the stock, frozen `google/electra-large-discriminator`, sentiment the same with
  LoRA adapters (applied at run time). The plan: PR 1a (done) the input side, 1b (done) the encoder nets, 2 (done)
  pos/depparse/ner with one shared encoder pass (exact only with `Gemm.Run(rowInvariant: true)`), 3 (done)
  constituency/sentiment, 4 (done) the package.
  - Files: not Stanza resources, so `ModelDownloader.Transformers` (repository → pinned revision, files, sha256),
    `DownloadTransformerAsync` (internal; the Cli sample's `download-transformer DIR NAME [FILE]...`) into
    `<model dir>/transformer/<repository>/` (`TransformerDirectory`); `Checkpoint.Load` reads the `pytorch_model.bin`
    (torch's zip format) by that name. Kept out of `Files`, `FilesFor` and every `tools/ci-models.sh` group. A file's
    sha256 is Hugging Face's LFS hash, the `x-linked-etag` header of `resolve/<revision>/<file>` (or `sha256sum` of the
    download), not the `etag` of the CDN response the redirect ends at: #182 stored that one for German's
    `pytorch_model.bin`, and the download failed its check (fixed in #184).
  - `Nn.WordPiece` = tokenizer.json's BertNormalizer + BertPreTokenizer + WordPiece per word (`is_split_into_words`): the
    added tokens' texts matched first in the raw word (leftmost, longest), control = Cc/Cf/Co/Cs but tab/newline/CR
    removed (unassigned Cn kept: tokenizers gives [UNK]), White_Space → space, NFD minus Mn (`strip_accents`; null follows
    lowercase), lowercase (Rust's per-character `to_lowercase`: İ → i + U+0307). `LoadDirectory` takes a directory's
    tokenizer.json, else (German's `german-nlp-group/electra-base-german-uncased`, #170 PR a, 2026-10-18) what
    transformers' converter builds from `vocab.txt` + tokenizer_config.json (`do_lower_case`/`tokenize_chinese_chars` true
    unless set, `strip_accents` false there; the five special tokens added; 100 characters per word); `ReadMaxLength`
    reads `model_max_length` or the older `max_len`, `TransformerConfig` defaults `position_embedding_type` to absolute.
    `tests/golden/de/transformer/tokenizer.json` (`make_golden.py --lang de --transformer-only`, plus
    `GERMAN_TRANSFORMER_TOKENIZER_CASES`); golden shard 4 downloads its tokenizer files (`tools/ci-models.sh transformer`).
    `Nn.TransformerInput` = Stanza's fix_blank_tokens (a word without pieces → "-") + convert_to_position_list.
    `tests/golden/transformer/tokenizer.json` (`make_golden.py --transformer-only`, `TRANSFORMER_TOKENIZER_CASES`;
    `tools/requirements-transformer.txt` pins transformers/tokenizers, apart from requirements.txt, which the cache keys
    hash), `TransformerTokenizerTests`.
  - The encoder (PR 1b): `Nn.ITransformerNet` (`Encode(ids, lastLayers)` → per sentence [lastLayers, pieces, hidden]),
    `TransformerConfig` from the transformer's `config.json` (ElectraModel only; anything else throws).
    `Nn.Managed.ManagedTransformer`: the dense layers as row-invariant `Gemm.Run` over every sentence's pieces, attention
    per sentence and head without padding, attention/LayerNorm/GELU in double (round 45: lane-wise vectors, the same bits
    on every path and width; GELU's erf from Numerical Recipes' erfc, 1.2e-7 relative, with its own mul/add-only `Exp`;
    LayerNorm's sums `Act.Dot`'s four accumulators), pooled buffers, rented outputs (round 47, `RentsOutputs`: the call's
    `TransformerCache` returns those it keeps when the call ends, `Embed` the others once mixed), a cancellation check per layer; so a sentence's
    states have the same bits alone, batched and on any thread count (`TransformerKernelTests`). `TransformerNet`
    (StanzaSharp.Cuda) pads each call like Stanza, with transformers' additive mask and `scaled_dot_product_attention`.
    `Nn.TransformerEmbedder` = extract_bert_embeddings: the windows over `model_max_length` (tokenizer_config.json, 512;
    stride 492, later windows from row 20), the legacy mix ((h[-4] + h[-3] + h[-2]) / 4) or a `bert_layer_mix` (mean of
    the last N plus their weighted sum; float32 in Python's order), each word's last piece, endpoints optional.
    `transformer/hidden.safetensors` (+ `.json`: the sentences, the long one, the mix weights, each sentence's bound):
    Stanza's own extract_bert_embeddings in one batch, float32 and float64 (stored rounded to float32), checked by
    `TransformerEncoderTests` on both backends.
  - The models (PR 2): `Pipeline.Alternatives` pos `combined_electra-large`, depparse `combined_electra-large`, ner
    `ontonotes-ww-multi_electra-large` (`--model pos=combined_electra-large`, ...): the `_charlm` twins' configs and
    weights plus 1,024 input columns, the transformer's per word (last for pos and depparse; ner: word | transformer |
    charlm). pos and ner use the legacy mix, depparse its `bert_layer_mix` (all zero weights: the mean of h21–h24), with
    the endpoints ([CLS] is ROOT's row, [SEP] dropped). pos and depparse send the simplify_punct'ed words (both data.py's),
    ner the token texts. `TransformerEmbedder.LayerMix` reads a model's settings (fine-tuned or LoRA transformers throw).
    `Pipeline.Transformers` (`<processor>/<model>` → repository) is apart from `Dependencies`, `ModelDownloader.Files` and
    the CI caches (`FilesFor` and `download --list` leave it out); the pipeline loads it once (`BackendModels.Transformer`),
    `ModelDownloader.DownloadAsync(dir, options)` fetches it (`DownloadTransformerAsync`), and `Load` names a missing file.
    Per call, a `TransformerCache` (by id sequence, h21–h24 of every piece, at most 16,384 pieces ≈ 256 MB; internal) when
    two or more processors read it: depparse reuses pos's states, ner those of sentences whose tokens are the words.
    Joint pass (#169, round 48, managed only): `ITransformerNet.EncodeJoint` runs groups of (net, sentences), the frozen net
    and its `WithLora` nets, in one sweep over the layers (base GEMMs over all rows, each group's adapters over its rows, in
    `WithLora`'s order, so every sentence keeps its own net's bits; `JointPass_GivesEachGroupItsOwnNetsBits`). After mwt the
    pipeline asks every reader for its sentences (`TransformerReader(doc)`: pos/depparse the simplify_punct'ed words,
    constituency the words, ner/sentiment the token texts) and `TransformerEmbedder.Prefill` encodes them into one cache per
    net, only when 2+ readers read a 4-layer mix and every distinct sentence fits one call (`BatchPieces`, 8,192 pieces);
    otherwise nothing changes. Logged as the `transformer` stage. TorchSharp (`EncodesJointly` false) keeps a pass per net.
    Golden row `pos+depparse+ner=electra-large` (`MODEL_CHOICES`, corpus + validation, contractions, social, tech and
    `models/transformer_long.txt`, a sentence of 706 pieces in two windows;
    make_golden's `electra_locally` has Stanza read `<models>/en/transformer/`), `TransformerModelTests` (both backends),
    `TransformerCacheTests` (a fake encoder, no models).
  - The models (PR 3): constituency `ptb3-revised_electra-large` (frozen, raw words, its `bert_layer_mix` of 4 layers,
    endpoints: each sentence's rows are [word_start | CLS], words, [word_end | SEP], the transformer columns last; it
    shares the call's `TransformerCache`) and sentiment `sstplus_electra-large` (no charlms although resources.json lists
    them, `UsesCharlm` false; `Pipeline.Unread` drops them from what Load loads and checks, `ReadDependencies`, while
    downloads keep them, owner's decision 2026-10-18; word 100 + transformer 1,024; filters 3, 4, 5; the legacy mix over token texts, no
    endpoints). Sentiment's LoRA (`bert_lora`, peft's eval forward: base(x) + B(A(x))·alpha/rank after the bias, on query,
    value, attention.output.dense, intermediate.dense before GELU, output.dense before the LayerNorm): `Nn.LoraWeights`,
    `ITransformerNet.WithLora` = a net sharing the base's weights plus adapters (managed: two row-invariant `Gemm.Run`s
    per adapter, so the invariances hold; TorchSharp: `F.linear` twice). A `TransformerCache` belongs to one net
    (`Net`): `Embed` ignores it for any other, so the LoRA pass never reads or fills the frozen states.
    `TransformerEmbedder.LayerMix(..., lora: true)` allows `use_peft`/`bert_finetune` when no `bert_model.*` weights are in
    the checkpoint. Golden rows `constituency=ptb3-revised_electra-large`, `sentiment=sstplus_electra-large` (+ reviews.txt)
    and `electra-large` (all five, `PROCESSORS`); make_golden needs `peft` (requirements-transformer.txt).
  - The treebanks' (#168, owner's decision 2026-10-18): pos and depparse `{ewt,gum,lines,partut,atis,childes,eslspok}_electra-large`
    and ner `ontonotes_electra-large` (fasttextcrawl, word 300): `combined_electra-large`'s / `ontonotes-ww-multi_electra-large`'s
    configs on the `_charlm` twins' vocabularies and heads (legacy tag heads, lines/partut composite XPOS); no new code. Golden
    rows `pos+depparse=<tb>_electra-large` on their package (corpus.txt only) and `ner=ontonotes_electra-large` (+ validation).
    Constituency `ptb3_electra-large` (#171, owner's choice 2026-10-19): LoRA-adapted like German's spmrl parser, and
    `transition_scheme` TOP_DOWN (plain: no compound or unary transitions), so `ParserState.IsLegalTopDown` (parse_transitions.py's
    top-down branches: Open(ROOT) only first, no Shift into the root's Open alone, no Close right after an Open, the last two
    closes only at the end) and a Close that takes only the children above the open marker; scores, composition, nets and
    scheduling as IN_ORDER. Golden row `constituency=ptb3_electra-large` (corpus, validation, contractions,
    transformer_long.txt; 45 of validation's 104 trees differ from ptb3-revised_electra-large's), downloaded by golden shard 4.
  - RoBERTa (#172, owner's decisions 2026-10-19: two PRs, two transformers per pipeline, download per run; docs/transformer-plan.md
    "RoBERTa"): ner `conll03_electra-large` reads `roberta-base`. PR (a): `Nn.ByteLevelBpe` (GPT-2's byte-level BPE with Stanza's
    `add_prefix_space=True`: added tokens first, `<mask>` lstrip; a space before every other piece not starting with U+0020;
    the split pattern scanned by code point; merges by rank) behind `ITransformerTokenizer` with `WordPiece`, picked when
    tokenizer.json's model is BPE (`type` may be absent); `TransformerConfig` reads roberta (`Prefix` `roberta.`,
    `PositionOffset` 2 = pad + 1, `MaxPositions` 512; the nets slice the position table at load, one token-type row);
    `ModelDownloader.Transformers["roberta-base"]`; goldens `transformer/tokenizer_roberta-base.json` (1,093 cases,
    `ROBERTA_TOKENIZER_CASES`) and `hidden_roberta-base.*` (`--transformer-only`); `TransformerTokenizerTests`,
    `TransformerEncoderTests.Roberta_*`, `TransformerKernelTests.Roberta_*`. PR (b): the NER model (old checkpoint format like
    germeval2014's, input word 100 | roberta 768 | charlm 2048, legacy mix over token texts) and every transformer the models
    read loaded once by name (`Pipeline._transformers`; the shared pass per transformer, its readers found by tokenizer);
    it reads conll17 although resources.json lists glove (owner's decision 2026-10-19: its word vocabulary is conll17's,
    and Stanza fails its embedding check with glove): `Pipeline.ListedPretrains` (EmbeddedResources_MatchModelDownloader's
    exception), make_golden's and compare.py's `PRETRAIN_FIXES` (Stanza's `ner_pretrain_path`; `PretrainFixes_MatchListedPretrains`);
    reported upstream as stanfordnlp/stanza#1698 (owner, 2026-10-19): drop the entries once a Stanza version's resources.json is fixed.
    Golden rows `ner=conll03_electra-large` (+ validation, validation_social) and `electra-large+ner=conll03_electra-large`
    (two transformers), `TransformerModelTests`; golden shard 4 downloads it (`tools/ci-models.sh transformer`).
  - German's on the same transformer (owner's choice 2026-10-19): pos and depparse `{gsd,hdt}_german-nlp-electra` and ner
    `germeval2014_german-nlp-electra` (fasttextwiki, word 300): `combined_german-nlp-electra`'s / germeval2014's configs on
    their twins' vocabularies and heads (gsd/hdt legacy tag heads; depparse's 4-layer `bert_layer_mix`); no new code. Golden
    rows `pos+depparse=<tb>_german-nlp-electra` on their package (corpus.txt) and `ner=germeval2014_german-nlp-electra`
    (+ validation, transformer_long) in `tests/golden/de/models/`; golden shard 4 downloads them. On a third transformer,
    `dbmdz/bert-base-german-cased` (BERT: `TransformerConfig` reads `model_type` bert, `Prefix` `bert.`; a cased vocab.txt
    WordPiece; `de/transformer/tokenizer_dbmdz_bert-base-german-cased.json`, `DBMDZ_TOKENIZER_CASES`, `DbmdzFact`): ner
    `germeval2014_dbmdz-bert-german-cased` (conll17, not the listed fasttextwiki: `ListedPretrains`, `PRETRAIN_FIXES`) and
    sentiment `sb10k_dbmdz-bert-german-cased` (LoRA; no charlms although listed: `Unread`). Golden rows
    `ner=germeval2014_dbmdz-bert-german-cased`, `sentiment=sb10k_dbmdz-bert-german-cased`.
  - Coref's XLM-R (docs/coref-plan.md, PR a; owner's choice 2026-10-19; #176 stays parked for b and c): `Nn.Unigram`
    (tokenizers' Precompiled normalizer: per extended grapheme cluster via `StringInfo`, a cluster under 6 UTF-8 bytes
    replaced whole by its *shortest* trie match, else per character; WhitespaceSplit + Metaspace prepend always + split;
    Unigram best path, unknown = min score − 10, a later start wins only when strictly better, adjacent unknowns fused),
    picked by `TransformerEmbedder.Load` (`IsUnigram`: `type`, else a vocab array); `TransformerConfig` reads
    `xlm-roberta` (prefix `roberta.`, offset 2); `ModelDownloader.Transformers["FacebookAI/xlm-roberta-large"]` (legacy-format
    `pytorch_model.bin`, 2.24 GB). Goldens `transformer/tokenizer_FacebookAI_xlm-roberta-large.json` (1,095 cases,
    `XLMR_TOKENIZER_CASES`) and `hidden_FacebookAI_xlm-roberta-large.*`; tests `XlmRoberta_*` (`XlmRobertaFact`); golden
    shard 4 downloads it (`tools/ci-models.sh transformer`). Owner's decisions (2026-10-20): Q3 Stanza's API shape, Q4
    write and read `coref_chains=`, Q5 reproduce libstdc++'s `nth_element` order for topk(sorted=False); Q1, Q2, Q6–Q8 open.
    PR (b), the document model (Core, `Coref.cs`): `Document.Coref` (its setter attaches like `_attach_coref_mentions`:
    every word gets a list, every word of the representative mention `repr`, a zero mention's empty word start + end,
    never representative), `CorefChain`/`CorefMention` (0-based, a zero's start = end = its empty word's `Id`, plus
    `EmptyIndex`)/`CorefAttachment`, `Word.CorefChains` (null until set), `Word.EmptyIndex`, `Sentence.EmptyWords` (not in
    `Words`, `WordCount` or `Dependencies`). `Conllu`/`Json` write empty words after word `i` (inside an MWT too, `0.m`
    first; no dummy head) and `coref_chains=` last in MISC / `coref_chains` after `end_char`, and read both back:
    `CorefLinks.Rebuild` makes each chain's mentions from first/last-word marks (document order), the representative from
    its last word's `repr` mark, its text as `extract_text` (pieces joined by single spaces: Stanza reads the spacing from
    an empty `token.misc`). A word whose text is `_` keeps lemma `_` (Stanza's lemma setter). Golden `tests/golden/coref/`
    (`make_golden.py --coref-only`; `electra_locally` also serves XLM-R and coref's own `load_tokenizer`): corpus +
    validation texts and `crafted.*` (chains set by hand with empty words); `CorefDocumentTests`, no models.
  - CI: no cache (the model caches are near GitHub's 10 GB; owner's decision): golden shard 4 downloads the
    `default_accurate` package, the treebanks' transformer models (their packages' other files from the en-treebanks cache)
    and the transformer's files on every run (`tools/ci-models.sh transformer`: `download --package default_accurate`, ...,
    with retries); the `Transformer*` classes (`SHARD_FOUR`, `.Transformer`) run only there
    (owner's decision, 2026-10-18: in shard 1 they had made its job 82 min), and the other jobs' no-skip checks exempt
    `Transformer`. A pull request runs shard 4 only when it touches the torch=full files, `ModelDownloader.cs`,
    `tools/ci-models.sh`, `tests/golden/**/transformer/` or `Transformer*.cs` tests (`changes`' `shards` output); pushes
    to main and the nightly always run it (owner's decision, 2026-10-18).

## Status

All 7 steps of the build order are done. The bullets below describe the original TorchSharp port; its torch classes
(`Weights`, `Rnn`, `CharLanguageModel`, `HighwayLstm`, `Biaffine`, `CharacterModel`, the nets) live in `StanzaSharp.Cuda`
since 1.0, and every processor also has a managed net (see "Managed backend"), the default.
- Models are converted, golden data is in `tests/golden/`, and the checkpoint configs are summarized
  under "Checkpoint findings".
- `Core` has the document model, `Conllu` (round-trips the golden file byte for byte), `Tree`,
  `SafeTensorFile` and `Checkpoint`.
- `Nn.Weights.LoadFrom` fills any TorchSharp module from a checkpoint by parameter-name prefix.
- `Tokenize.Tokenizer` matches Stanza's tokens, offsets, spacing and sentences on the golden corpus
  and the stress set.
- `Mwt.MwtExpander` uses the dictionary, then the character classifier. It matches the golden
  words, word offsets, and `mwt.json`, which covers dictionary and classifier-only output.
- `Nn.Rnn.RunPacked` wraps pack/LSTM/unpack for models that pack by real length (MWT, POS,
  and the parser's word LSTM).
- `Nn.CharLanguageModel.BuildCharRepresentation` gives one vector per word, taken at the space
  after it. It matches the golden charlm tensors within 1e-4. The tagger and the parser both use it.
- `Nn.Pretrain` provides `UnitToId` (spaces stored as U+00A0) and `Embeddings`. Callers decide
  on lowercasing: POS lowercases every word, while the parser tries the word as written, then
  lowercased.
- `Pos.PosTagger` takes the shared `Pretrain` and both charlms, which the caller owns. It matches the
  golden UPOS/XPOS/feats exactly after tokenize → mwt → pos. `Nn` gained `HighwayLstm` and
  `Biaffine`. Like Stanza's `simplify_punct`, the tagger sees runs such as `??` or `!?!` as `?`
  or `!`. Only the tagger does this, not the parser.
- `Constituency.ConstituencyParser` also takes the shared `Pretrain` and charlms. It reproduces the
  golden trees and the per-step transition scores. It schedules like Stanza's `parse_sentences`:
  sentences sorted longest first, 50 states in flight, each finished state replaced by the next.
  Each step is one forward pass plus batched stack pushes. Stack LSTMs are stepped without
  padding, so batching does not change results. On TorchSharp (round 25) the net's handles are rows of per-thread
  "bank" tensors (vectors, each stack's h and c; a free list, freed when the last row comes back), so a step costs a
  constant number of ops (`index_select`/`index_copy_`), not a few per state.
- `Nn.CharlmCache`: the pipeline passes one to `PosTagger.Process` and `ConstituencyParser.Process`,
  so the parser reuses the tagger's charlm outputs (the tagger keeps a sentence only if
  `simplify_punct` left its words unchanged). The charlm is not bitwise batch-invariant, so cached
  values can differ from a fresh computation in the last bits; outputs are still identical.
- Performance: `docs/performance.md` has the benchmark (`samples/StanzaSharp.Benchmark`,
  `tools/benchmark.py`), the method, the latest runner results and remaining ideas.
  Each measurement round is its own file, `docs/performance/round-NN-<slug>.md` (NN zero-padded, the number the
  brief assigns; owner's decision, 2026-10-13, as parallel PRs kept conflicting in one file). A new round adds
  its file and leaves performance.md alone (no index there); only "Run it yourself" changes go there. Open ideas are one file each in `docs/performance/ideas/` (user's decision,
  2026-10-13): a new idea adds a file, finishing or dropping one deletes its file (the round file is the record).
  Round 2 (memory): checkpoints are read tensor by tensor straight into TorchSharp memory (no
  whole-file buffers), depparse's biaffine scorers run a few sentences at a time, POS batches are
  cut at 5000 words like Stanza, and `CharlmCache` keeps at most 32k words (`MaxWords`).
  Round 3 (issue #19, `--memory`): `.pt` files are read tensor by tensor too (the unpickler reads a stream; storages
  are read from the file on demand), and `HighwayLstm` frees each layer's padded tensors as soon as they are used.
  Round 7 (per call, `--per-call N`, `--concurrent T`): one short sentence takes 28 ms managed (Python 189); the rest is
  memory bandwidth (weights streamed per call), not setup. Past ~2 concurrent callers managed throughput drops
  (L3 evictions); `PipelineOptions.MaxConcurrentCalls` (0.6, opt-in) is the gate.
  After a Process call of 1,000+ words, `NativeHeap.Trim` calls glibc's `malloc_trim(0)` on Linux (user's decision,
  2026-10-07: always on, no public API; `PipelineOptions.TrimNativeHeap` is internal, for the benchmark). On Windows
  libtorch's built-in mimalloc keeps freed pages; only `MIMALLOC_PURGE_DELAY=0` releases them (documented, not set).
  Round 11 (garbage on long texts, managed): constituency's stack states, word vectors and Compose outputs come from a
  thread-static `Recycler` (per call, since Parse is synchronous; emptied when every array is back) and go back when
  `ParserState.Dispose` disposes them; Chu-Liu/Edmonds works on flat rented n × n buffers; `DepparseBatch.RentScores`
  lets the managed net rent the arc/label outputs, which `DependencyParser.Parse` returns. Rent an output only where its
  one caller gives it back; the Cuda nets keep plain arrays. A call now allocates ~18 KB/word instead of ~54 (docs/performance/round-11-garbage-long-texts.md).
  Round 18: the pool's trimming costs nothing (`DOTNET_SYSTEM_BUFFERS_ARRAYPOOL_TRIMSHARED=false`: same bytes); the
  managed charlm's per-character strings (`EnumerateRunes().Select(ToString)`) were ~3 KB/word, now looked up by span
  (`GetAlternateLookup<ReadOnlySpan<char>>`): 15 KB/word. Depparse's forward charlm from a stored "\n\n " state was
  exact but saved 0.13–0.2 ms per sentence, under a call's noise: not shipped.
  Round 22: 11 KB/word (NER's emissions rented through `INerNet.Forward`'s span, Tarjan's per-node arrays, ASCII
  `PyString.Lower`, the `Recycler` keeping `StackState`/`Composed` objects, the tokenizer's `Paragraph` as one string with
  inputs rented per batch). GC allocation ticks over-credit small objects; trust the benchmark's per-processor bytes.
  Round 23: the `ArrayPool` "misses" (`PoolExhausted`) left in the LSTMs are the pool's first fill on a fresh process
  (~0.6 GB, live and reused; a second call allocates 8 KB/word, 7 since round 38: rented Chu-Liu/Edmonds contractions, constituency step buffers), not garbage; no thread, partition or size limit is hit.
  Round 12 (concurrent calls, `--with-long`): `ManagedThreads.Pool` keeps workers asleep while callers in a region plus
  awake workers number max(ProcessorCount, Count) (`_busy`/`_limit`); with Threads = ProcessorCount and 8 callers, spinning
  oversubscription had stalled calls for seconds. Calls don't serialize or starve; >2 callers cost throughput (L3/DRAM,
  same on 1 thread). macOS runner one-sentence slowness was Spotlight indexing (benchmark.yml turns it off). Neither
  physical cores nor all logical processors won everywhere (Ryzen vs the 2-core runners). Round 13: the default is the
  physical cores from 4 up, else all logical processors (Ryzen 8 vs 16: big texts -9%/-6%, one-sentence p90 -16%/-36%,
  2 callers +11% calls/s; runners unchanged at 4/4/4/4/3).
  Round 6 (managed backend, docs/performance/round-06-memory-managed-backend.md): no native heap (malloc_trim returns nothing there); after a call the
  GC heap holds the models plus `ArrayPool<T>.Shared`'s scratch buffers (all eight, 6,761 words: 2.4 GB working set
  then, 1.3 GB since round 10 shrank sentiment's buffers), released only by a gen2 GC a minute later; the README tells services to run an aggressive GC when idle. The library
  runs no GC.
- `Tree.ToString` prints `(`/`)` in labels and words as `-LRB-`/`-RRB-`, like Stanza. The tree
  itself keeps the raw text.
- Measured float drift vs Python: charlm < 1e-7, tokenizer logits ≈ 4e-6, UPOS logits ≈ 2e-5,
  parser scores ≈ 2e-5. Test tolerances are 1e-4 and 1e-3. A 1e-4 tolerance is relative above |100| (owner's decision,
  2026-10-14, round 26): a value passes within 1e-4 × max(1, |expected| / 100) of its reference
  (`TokenizerTests.Allowed`, used by `AssertClose`, `PosTests.AssertUposClose` and `SentimentTests`), since float32
  can't hold 1e-4 above |x| ≈ 300 (one ulp is 3.05e-5 from 256 up). Only UPOS logits exceed |100| in the golden data
  (German up to |380|, English |139|); 1e-3 tolerances stay absolute.
- `StanzaSharp.Pipeline` ties the processors together: `Load(modelDir, "tokenize,mwt,...")`, then
  `Process(text)`. It loads the shared pretrain/charlms once. `Conllu.Write(pipeline.Process(corpus))`
  is byte-identical to `tests/golden/pipeline.conllu`.
  - The `samples/StanzaSharp.Cli` sample writes the same CoNLL-U from a file or stdin.
  - `Conllu.Write` fills a missing HEAD with `id - 1`, like Stanza's writer.
- `Depparse.DependencyParser` (processor `depparse`, requires tokenize, mwt, pos, lemma; runs after
  lemma and constituency, as in Stanza) sets `Word.Head`/`Deprel` with the pipeline's shared pretrain/charlms.
  `Pipeline` with `tokenize,mwt,pos,lemma,depparse` reproduces every `tests/golden/depparse/*.conllu`
  byte for byte (13 files, 8401 words), and so does the parser alone on Stanza's own tags and lemmas.
  Arc/label log-prob drift ≈ 2e-5 (tolerance 1e-4).
  - The parser feeds the charlms `"\n"` + the words (a ROOT word), so every forward state and the backward ROOT
    state differ from the tagger's. But backward, ROOT comes last: the tagger's backward pass is the parser's prefix.
    Since `Gemm.KernelBlocked` a charlm step gives a row the same arithmetic in any batch, so (managed only) the tagger
    keeps each cached sentence's backward final cell state (`CharlmCache.TryAdd`'s `backwardFinalC`, 4 KB per sentence,
    not counted in `MaxWords`) and the parser takes every such sentence's backward rows and continues from the tagger's
    final (h, c) over ROOT only, in its own batches (`CharlmCache.TryGetBackward`: the entry by `Sentence`, else the
    single-sentence table by texts; `ManagedCharLanguageModel.Continue` takes per-sentence start states). Bit for bit
    (`Batches_ContinueTheTaggers...`, `OneSentenceDocument_ContinuesTheTaggers...`; round 20 hashed every path).
  - `CharlmCache`'s single-sentence table (`AddAlone`/`TryGetAlone`, keyed by the texts): a managed charlm call on one
    sentence depends on its texts only, so `ManagedCharLanguageModel.BuildBoth` (pos, sentiment, ner) reuses any earlier
    one on the same texts exactly. Besides depparse, NER takes a multi-word-token sentence's token charlms from sentiment.
    `PipelineTests.OneSentenceDocuments_EqualTheUncachedPipeline` checks the whole pipeline against the cache turned off.
  - A missing lemma is read as `_`, as in Stanza, so without the lemmatizer the parses differ.
  - `Conllu.Write` needs no change: with heads set it matches Stanza (DEPS stays `_`).
- `Lemma.Lemmatizer` (processor `lemma`, requires tokenize, mwt, pos) reproduces `tests/golden/lemma/`
  byte for byte: 13 files, 8401 words, 868 of them through the seq2seq model. It mirrors
  LemmaProcessor: dictionary skip, DeltaVocab, `batch_size` 50 batches sorted like `sort_all`, greedy
  decoding, edits, `<UNK>` fallback, and the `Word.lemma` setter turning `_` into null.
- `Ner.NerTagger` (processor `ner`, requires only tokenize: it reads token text, like Stanza) sets
  `Token.Ner` (BIOES) and `Sentence.Entities` (`Entity`: Text, Type, StartChar, EndChar, Tokens;
  `Document.Entities` concatenates them). `tests/golden/ner/` is reproduced byte for byte: 13 files,
  845 sentences, 706 entities, with `tokenize,mwt,pos,lemma,depparse,ner`, with the cache off, and with
  `tokenize,ner` alone. Emission drift ≈ 2.4e-6 (tolerance 1e-4).
  - `Entity.Build` is `build_ents`/`decode_from_bioes`; `Conllu.Read` calls it too (text from the
    tokens, since a read document has no `Text`).
  - `Conllu` writes/reads `ner=` after `end_char` in MISC, on the MWT range line and on single-word
    tokens' word lines, never on MWT words. `multi_ner` is not written by Stanza and not ported (with
    one model it is just `(ner,)`).
  - `CharlmCache`: NER's charlm input (`"\n"`, each token + `" "`, same vocab) equals the tagger's
    when a sentence's tokens are its words (no MWT), so it reuses those; others are computed.
- `Checkpoint.Load` also reads Stanza's original `.pt` files (`Core/TorchCheckpoint.cs` +
  `Core/Pickle.cs`), so `Pipeline.Load("models/stanza/en")` works without Python. For all fourteen
  checkpoints the result equals the converter's: identical JSON and byte-identical tensors.
- `Sentiment.SentimentClassifier` (processor `sentiment`, requires tokenize; runs last, as in Stanza)
  sets `Sentence.Sentiment` (0 negative, 1 neutral, 2 positive). `Conllu` reads and writes
  `# sentiment = N` after `# constituency`, Stanza's comment order. `tokenize,mwt,sentiment`
  reproduces `tests/golden/sentiment/` byte for byte (14 files, 886 sentences), with every label and
  logit (drift ≤ 2.2e-5, tolerance 1e-4), also as one document in two 5000-token batches (`all.json`).
  - Batches must match Stanza's: padding runs through the unpacked biLSTM and the convolutions, so a
    sentence's result depends on its batch (one document vs per file: 173 of 886 labels change in Stanza itself).
  - `CharlmCache`: the classifier reads token texts, so it reuses the tagger's outputs for sentences
    without multi-word tokens (641 of 886). Labels stay identical; logits drift up to ~1e-4 (vs
    Stanza), as the network amplifies the cached charlm's last-bit differences; tested at 1e-3.
- `default_fast` (`PipelineOptions.Package`): `PosTagger`, `DependencyParser` and `NerTagger` switch on the
  checkpoint's `charlm` flag and take null charlms for `_nocharlm` checkpoints. `tests/golden/fast/` is
  reproduced byte for byte (13 files, 845 sentences, 8,401 words, all seven processors). Drift: UPOS
  logits 1.5e-5, arc log-probs 7.6e-6, label log-probs 1.5e-5, NER emissions 1.9e-6 (tolerance 1e-4).
  - No `CharlmCache` there: the nocharlm tagger has no charlm outputs to share (`PosTagger.UsesCharlm`).
  - Stanza silently skips a processor its package lacks (`processors='...,constituency'` with
    `default_fast` loads no parser); we throw instead (user's decision), naming the `default` package.
- Input modes (2026-10-06, the user's approved examples): `Process(IEnumerable<string>)` (bulk) and
  `Process(IEnumerable<IEnumerable<string>>)` (pretokenized). Collection expressions pick the right overload;
  `Process([])` and `Process(null)` are ambiguous (compile errors), never a silent wrong pick.
  - Bulk = Stanza's `bulk_process`: `Tokenizer.Process(texts)` tokenizes `"\n\n".Join(texts)` and `Tokenizer.Split`
    deals sentences out like `TokenizeProcessor.bulk_process` (a sentence goes to the current text while its last
    token ends inside it; offsets shifted; the last token's SpaceAfter and first token's SpacesBefore from the text
    itself). The other processors run on one combined `Document` (UDProcessor.bulk_process), then entities are
    rebuilt per document. Sentence ids continue across documents, as in Stanza.
  - Bulk ≠ one by one, in Stanza too: sentiment batches change labels (172 of 854 in the golden; depparse could too, via
    its padded log-softmax, but no head changed), and sent_ids continue. Without those two processors the output equals one call per text apart from
    sent_id (`Bulk_EqualsProcessingAloneExceptBatchedProcessors`).
  - Pretokenized = `process_pre_tokenized_text` with a list: text = tokens joined by single spaces, offsets into
    it, `IsMwtCandidate` false, so mwt changes nothing ("don't" stays one word, as in Stanza). Empty sentences and
    empty/whitespace tokens throw (Stanza fails on the first two; whitespace tokens give broken CoNLL-U there).
    Stanza's string form (whitespace tokens, newline sentences) is not exposed; make_golden asserts it equals the list form.
  - `Process(Document)` (owner's decision, 2026-10-14) = Stanza's `process_conllu`: `nlp.process(doc, processors=` the
    loaded ones minus tokenize and mwt`)`. Same `Annotate` as the other overloads with `mwt: false`, gated and cancelable
    like them; the document is changed in place and returned. Each processor overwrites what it sets (Stanza does too);
    the rest is kept (a tree in default_fast). Stanza keeps comments in input order, so a document
    whose CoNLL-U had `# sentiment` before a new `# constituency` writes them the other way round there; `Conllu.Write`
    always writes constituency first. `Process(null)` stays ambiguous; `Document` is no collection type, so collection
    expressions never pick it.
  - `PipelineOptions.Pretagged` (owner's decisions, 2026-10-17; docs/pretagged-plan.md) = Stanza's `pretagged=True`:
    lemma, depparse and constituency need no processor at Load (`SelectModels`' `pretagged`; `FilesFor` follows), so
    `Processors = "depparse"` loads the parser without the tagger and parses a `Process(Document)` document's own tags.
    Tokenize and the package's mwt always load (1.2 MB), so the text overloads run and fail in the parsers like Stanza's
    `tokenize,depparse` pretagged pipeline; pos, ner and sentiment keep their requirements. Missing annotations as in
    Stanza (depparse throws for a word with neither UPOS nor XPOS, reads a missing XPOS or lemma as `_`; the lemmatizer
    a missing UPOS as `_`), except that constituency throws for a word without its tag (Stanza parses an empty tag).
    `tests/golden/pretagged/` (`make_golden.py --pretagged-only`, `PRETAGGED_CASES`), `PretaggedTests`.
- 0.4 API (issue #12, items chosen by the owner, 2026-10-06):
  - `PipelineOptions.Threads` (int?): at Load, process-wide, `ManagedThreads.Count` on the managed backend and
    `torch.set_num_threads` on TorchSharp. Managed null = min(current count, `ManagedThreads.Default`), the pool's initial
    count: `CpuCores.DefaultThreads` = the physical cores if ≥ 4, else `Environment.ProcessorCount`, never more than
    ProcessorCount (owner's decision, 2026-10-12; docs/performance/round-13-physical-cores.md). `CpuCores.Physical()` (Nn/Managed;
    the benchmark's core count too) never throws, null → logical: Linux distinct sysfs `thread_siblings_list`, Windows
    GetLogicalProcessorInformationEx records, macOS sysctlbyname `hw.physicalcpu`; no process spawned. A caller that
    sets `ManagedThreads.Count` and then loads with null gets it capped at the default, so the benchmark sample passes
    `Threads` explicitly. The rest of this bullet is TorchSharp's. Null = min(libtorch's current
    value, `Environment.ProcessorCount`), so a container's CPU quota is respected and a lower count the caller set is
    kept. Inter-op threads are untouched (eager mode never uses that pool; libtorch allows setting it only once).
    libtorch's OpenMP gives each calling thread its own team of `Threads` threads (measured: 8 callers × 8 threads
    = 83 OS threads), and a thread that already ran an op keeps its old count after a later Load.
    Measured (benchmark, default, 6,566 words): 1 thread 57.8 s, 2: 33.2 s, 8: 19.1 s, 16: 19.7 s. One loaded pipeline
    ≈ 950 MB private (default), 700 MB (default_fast). VerifyChecksums ≈ +1.3 s / +0.8 s load.
  - `CancellationToken` overloads of all three `Process` (separate overloads, binary compatible). Checked in
    `Pipeline.Step` before each processor and at the head of each batch loop (tokenizer batches and windows, pos,
    lemma, constituency per step, depparse, sentiment, ner; mwt is one batch). Depparse also checks inside a batch:
    after each charlm pass, between `HighwayLstm` layers (its optional token), before each `DeepBiaffine` chunk and
    per Chu-Liu/Edmonds decode; the longest gap is one charlm pass or LSTM layer (~1/8 of a batch). Pos does the same
    (after each charlm pass or the char model, between `HighwayLstm` layers, between the heads), and so does sentiment
    (after each charlm pass, after the LSTM, between the convolutions). Worst measured latency ≈ 0.6 s on the desktop;
    `ConcurrencyTests.MaxLatency` is 5 s for CI. Every check sits outside the inner
    dispose scopes or inside a try/finally that disposes, so nothing leaks (`ConcurrencyTests`).
  - Thread safety: Process keeps all mutable state per call (the Document, `CharlmCache`, the lemmatizer's
    DeltaVocab copy, the parser states); the models are read-only after Load; TorchSharp's dispose scopes are
    thread-static and libtorch's grad mode thread-local. The stress test (8 threads, mixed modes, both packages)
    needed no code change. Keep it that way: never add mutable fields to a processor (the one exception: the batch sizes,
    set only on `WithBatchSize`'s fresh `MemberwiseClone` inside Load, before the pipeline is returned).
  - TorchSharp counts the two undefined index tensors of `pack_padded_sequence(enforce_sorted: true)` (lemmatizer)
    as live forever (`DisposeScopeManager.Statistics`); they hold no memory. Leak tests leave lemma out.
  - `SplitSentences = false` = `tokenize_no_ssplit`: `Tokenizer.Decode` ends sentences only at paragraph ends.
    `tests/golden/no_ssplit/` (`make_golden.py --no-ssplit-only`). Pretokenized input ignores it, as in Stanza.
  - `MaxConcurrentCalls` (int?, owner's decision 2026-10-09: opt-in, null = no limit): a `SemaphoreSlim` in `Pipeline`
    around each Process overload's work (`Gated`); the wait honours the call's token. Managed throughput peaks at ~2 calls
    in flight (cache eviction of the charlm weights; docs/performance/round-07-per-call-short-texts.md).
  - `Processor` constants; `VerifyChecksums` (MD5 of each `.pt` Load reads, `ModelDownloader.Verify`, and the sha256 of each
    transformer file, 2026-10-18; throws for converted models, which have no published MD5); `Logger` (`Microsoft.Extensions.Logging.Abstractions` 10.0.0,
    since 1.0 the StanzaSharp package's only dependency): load times at Information, per-processor times at Debug, nothing measured
    without a logger.

Tokenizer notes:
- `Tokenizer.Predict` batches like Stanza: sort paragraphs by length, batch by 32, pad to max+1,
  and use 1000-char windows. Both LSTMs are packed at the length of each row's raw units. In
  the normal case that is the row's own length + 1, since collate appends one `<PAD>`. In the
  first long-paragraph window, the length is cut to the window. After `advance_old_batch`, every
  row runs at the full window width, so padding then reaches the backward direction.
  - Those later windows run only the unfinished rows (`Tokenizer.LiveRows`, round 15: 2.4x on the benchmark text), plus
    finished (all-padding) rows placed before the last live row until `Gemm.Run`'s single-row kernel (row m − 1 when
    m % 6 is 1) falls on the same row as with the whole batch, so every used logit keeps its bits on every path.
- `FixLabels` runs the e-mail/URL regex (`MaskRe`) only on paragraphs containing one of `MaskLiterals` (`@`, `://`,
  `www.`, `.gov` ... read off the pattern's alternatives; round 16). If the pattern changes, re-derive them; the
  `MaskLiterals_*` tests check every alternative and a fuzz.
- Regexes Stanza runs over a paragraph's code points (`STRUCTURAL_FEATURES`, `MASK_RE`) run on `Paragraph.CodePoints`,
  one char per unit (a non-BMP digit → `٠`, letter/number → `ª`, else U+FFFD), so match indices are unit indices; on
  the UTF-16 text a surrogate pair broke `\b` and counted twice in `{2,}` (found by `tools/text_fuzz.py`).
- Model units are code points, like Python. Offsets are UTF-16 indices, so they differ from
  Stanza's only after non-BMP characters.
- MWT-flagged tokens (`Token.IsMwtCandidate`) carry a single word until the MWT stage runs.
- Two lengths from the config: windows are `max(1000, max_seqlen)` units (output_predictions), and tokens longer
  than `max_seqlen` (200 for English; default 1000) in code points become `<UNK>` (TokenizeProcessor.process: text
  and word text only, offsets and MWT flag kept; bulk too, never pretokenized). `tests/golden/long_token*`
  (`make_golden.py --long-token-only`), both packages.

## Working on this repository

- **Advisor**: when an advisor tool is available, call it before design decisions or recommendations, before splitting
  work across agents or sessions, and before declaring a larger task done.
- **Asking the owner** (owner's decision, 2026-10-17): whenever a decision or any input from the owner is needed, ask
  with the question prompt (`AskUserQuestion`: options, the recommended one first), not only as a question in the reply
  text.
- **Cloud sessions**: `.claude/hooks/session-start.sh` (SessionStart, cloud only) installs Ubuntu's `dotnet-sdk-10.0`
  (the network policy blocks `dotnet-install.sh`'s host), `stylebro-migrate`, restores, builds the Cli and downloads
  `tools/ci-models.sh en` into `models/stanza/en`, with `STANZASHARP_MODELS` pointing there (no Python, no German or
  treebank models: download those with `tools/ci-models.sh download GROUP` when a task needs them).
- **Where work runs**: long work (multi-step tasks, PRs, CI waits, releases) runs as cloud sessions; the owner's
  computer must stay free to restart or turn off. Work that needs it (local timings, Windows-only checks) is asked for,
  never started unasked; a cloud session that needs such a step says so in its PR.

## Layout

```
src/StanzaSharp.Core           Document model, CoNLL-U I/O, constituency Tree, safetensors reader,
                               checkpoint loading (converted JSON or .pt). No TorchSharp dependency.
src/StanzaSharp.Nn             Pretrain vocab/vectors, CharlmCache, the managed kernels and layers (Managed/). No TorchSharp.
src/StanzaSharp.Cuda           The TorchSharp backend (the StanzaSharp.Cuda package, `CudaBackend`): every processor's
                               torch net, the torch layers (charlm, highway LSTM, Weights, ...), the tensor halves of
                               Pretrain/CharlmCache, processors' TorchSharp `Load`. The only project referencing TorchSharp.
src/StanzaSharp.Tokenize       Tokenizer + sentence splitting.
src/StanzaSharp.Mwt            Multi-word token expansion.
src/StanzaSharp.Pos            POS / feature tagger.
src/StanzaSharp.Lemma          Lemmatizer (dictionary + character seq2seq).
src/StanzaSharp.Constituency   Constituency parser.
src/StanzaSharp.Depparse       Dependency parser (biaffine graph parser + Chu-Liu/Edmonds).
src/StanzaSharp.Ner            Named-entity recognizer (biLSTM + CRF Viterbi).
src/StanzaSharp.Sentiment      Sentence sentiment (CNN classifier over biLSTM states).
src/StanzaSharp                Pipeline facade wiring the processors together.
src/StanzaSharp.Tool           The `stanzasharp` .NET tool: download, compare, benchmark (no native libtorch).
samples/StanzaSharp.Cli        Console runner for quick experiments.
samples/StanzaSharp.Benchmark  Per-stage speed/memory benchmark; tools/benchmark.py is the Python twin.
samples/StanzaSharp.Example    Commented tour of the public API (download, load, every result, CoNLL-U).
tests/StanzaSharp.Tests        xUnit; golden tests against Python Stanza output.
tests/StanzaSharp.ManagedCheck Console app without libtorch: proves the managed backend loads none (ManagedCheckTests).
tests/golden/                  Golden data generated from Python Stanza (committed, keep it small).
tools/stanza_convert.py        Checkpoint inspector/converter (.pt -> .safetensors + .json); optional.
tools/.venv/                   Python env with stanza installed (gitignored).
models/stanza/                 Original downloaded models (gitignored).
models/converted/en/           Converted models (gitignored); C# prefers them over the .pt files.
```

Package policy: only `StanzaSharp.Cuda` (the StanzaSharp.Cuda package) references TorchSharp, and only the managed
`TorchSharp` package; the facade, Core, Nn, the processors and the Tool reference no TorchSharp at all and must build
without it. Runnable projects that test or run the TorchSharp backend (Cli, Benchmark, Tests) reference `$(TorchSharpNative)` (Directory.Build.props): `TorchSharp-cpu`,
or `TorchSharp-cuda-windows` when built with `STANZASHARP_CUDA=1`, which brings the native libtorch, for the
TorchSharp backend. The Example and ManagedCheck set `StanzaSharpNoLibTorch` (no libtorch: the default backend needs
none); the Tests do too when built with `-p:StanzaSharpNoLibTorch=true` (the linux-arm64 job). The
managed and native TorchSharp versions must match. CI and the default build stay on CPU.

GPU (TorchSharp backend only, the `StanzaSharp.Cuda` package): `Pipeline.Load(dir, new PipelineOptions { Backend =
CudaBackend.Create(deviceIndex, disableTf32) })` and every processor's `Load(..., device)`. `Weights.On(device, ...)`
scopes the load; `ToTensor`/`LoadFrom` place weights on `Weights.Device`, and each model keeps
`_device = Weights.Device` for its input tensors (`torch.tensor(..., device: _device)`). Read tensors with
`Weights.ToArray<T>()`, not `data<T>()`. pack_padded_sequence lengths stay on the CPU. A new processor
needs exactly these three things to run on CUDA. GPU tests (`GpuTests.cs`) compile only with
`STANZASHARP_CUDA=1`. Measurements and exactness: docs/gpu.md.

## Reference implementation

The Python Stanza source installed in `tools/.venv/Lib/site-packages/stanza/` is the source of
truth. When porting a model, read its code there (`stanza/models/tokenization`, `mwt`, `pos`,
`constituency`, `common/char_model.py`, `common/pretrain.py`, `pipeline/`) rather than relying on
memory of how Stanza works. Port the code paths the English checkpoints actually use, which the
checkpoint configs tell you; skip unused options.

## Model files

C# loads either format. `Checkpoint.Load(basePath)` reads `basePath.json` + `basePath.safetensors`
when they exist, else the original `basePath.pt`, so `Pipeline.Load` accepts both `models/converted/en`
and Stanza's own download directory `models/stanza/en` (`<processor>/<name>.pt`). Converting is
optional: it makes the checkpoints easy to inspect, but since round 21 it no longer loads faster than the `.pt`
files (round 34; still slower after round 36). A packed-weight cache was measured and dropped (owner, 2026-10-16: 0.1–0.2 s for 250–420 MB
per package).

Use `setup.ps1 -Models`, or run the steps by hand:

```powershell
tools\.venv\Scripts\python -c "import stanza; stanza.download('en', model_dir=r'models\stanza', processors='tokenize,mwt,pos,constituency')"
tools\.venv\Scripts\python tools\stanza_convert.py inspect models\stanza\en\constituency\ptb3-revised_charlm.pt
tools\.venv\Scripts\python tools\stanza_convert.py convert models\stanza\en --out models\converted\en
```

Converter output for each checkpoint is a `<name>.safetensors` file holding all tensors, plus a
`<name>.json` file mirroring the checkpoint's structure. In the JSON:

- Each tensor is replaced by `{"$tensor": key, "dtype", "shape"}`, keyed into the safetensors file.
- Non-JSON Python types are tagged:
  - `$tuple`, `$set`
  - `$dict` (a list of `[key, value]` pairs, for dicts with non-string keys)
  - `$bytes` (base64)
  - `$float` (`nan`/`inf`)
  - `$repr`/`$type` (unknown objects)
- `optimizer` and `scheduler` are dropped by default.

Safetensors layout: a u64 little-endian header length, a JSON header (`dtype`, `shape`,
`data_offsets` relative to the data section, optional `__metadata__`), padded to 8-byte alignment,
then raw little-endian tensor data. Read it with a small hand-written reader in `StanzaSharp.Core`;
no package is needed.

All fourteen checkpoints convert with `torch.load(weights_only=True)`; no unsafe pickling is needed.

### Reading .pt files in C#

All fourteen Stanza checkpoints use torch's legacy format (`_use_new_zipfile_serialization=False`), a
sequence of protocol-2 pickles: magic number, protocol version (1001), sys_info (little-endian),
the checkpoint, the list of storage keys; then per key an int64 element count and the raw bytes.
Tensors are `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)`
over persistent ids `('storage', torch.FloatStorage|LongStorage, key, location, numel, None)`; LSTM
weights share one storage at different offsets. The only globals are `collections.OrderedDict`,
`torch._utils._rebuild_tensor_v2`, the storage types, and `_codecs.encode` (protocol 2's encoding of
`bytes`, used by the lemma's `dicts`). The zip format (`archive/data.pkl` +
`archive/data/<key>`) is read too, for checkpoints saved by newer trainers.

Large plain containers (a str→int dict or list of str with ≥ 1,024 items, `PyContainer.MinCount`) skip the JSON
round trip when loading a `.pt` (round 21) and a converted `.json` (round 36: built in the validating reader pass, the
rest parsed with them as `null`): the node is a `JsonValue` holding the unpickled object, not a `JsonArray`/
`JsonObject`, and `ToJsonString()` still equals the converter's JSON. Read vocabularies with `Checkpoint.UnitToId` and
long string lists with `Checkpoint.Strings`; `AsArray()`/`AsObject()` on them throws.

`Unpickler` is restricted like `weights_only=True`: torch's protocol-2 opcodes only, plain data
(None/bool/int/float/str/bytes/list/tuple/dict), and an allowlist of OrderedDict, `_rebuild_tensor_v2`,
`_codecs.encode(str, 'latin1')` and the typed storage classes. Anything else throws `InvalidDataException`; nothing is instantiated by
name. Not supported (none of the models need them): sets, bfloat16, big-endian files,
`$repr` objects. `TorchCheckpoint` then mirrors the converter's `Splitter` exactly, including
Python's float `repr` in the JSON, key collisions (`key#2`) and numpy turning 0-d tensors into
shape `[1]`. `tests/golden/pt/` holds tiny fixtures in both formats with their converter output
(`make_golden.py --pt-only` regenerates them).

Failure contract (model files may come from anywhere): `Checkpoint.Load` and every later tensor lookup or read
either succeed or throw `InvalidDataException` for a malformed, truncated or hostile file; `NotSupportedException`
only for a well-formed big-endian `.pt`; OS errors (missing file) stay `IOException`s. Time, memory and stack are
bounded by the file's size. The limits that make it so:
- Unpickler: MARK records a stack height (O(1)); items below an open MARK can't be popped; dict keys are scalars only
  (str/int/bool/float; a nested-tuple key was hashed recursively); a length is checked against the stream before
  anything is allocated.
- Splitter: JSON depth ≤ 60 (`JsonNode.Parse` reads 64; also stops cyclic lists) and JSON size ≤ 16 × the file + 1 MB
  (the memo lets one object be referenced many times, and the JSON copies it each time).
- Storages and views: sizes compared by division, never `numel * size` (overflow); a storage key listed once; a tensor
  no larger than its storage (bounds stride-0 broadcasts) with non-negative sizes/strides/offset, its extent checked
  step by step before it can overflow. Zip entries must be stored uncompressed (as torch.save writes them) and fit in
  the file, so no claimed size is allocated.
- safetensors: header ≤ 100 MB (the library's limit), known dtype, `numel × size == end - begin`, and the tensors tile
  the data section exactly (no gaps, overlaps or trailing bytes), as the safetensors library checks.
- Converted `.json`: valid UTF-8, no escaped lone surrogate and no property name twice in one object, checked in the
  validating pass (`JsonNode` decodes strings and builds an object's dictionary lazily, so these would throw
  `InvalidOperationException` or `ArgumentException` wherever a model first read them; the duplicate names were found by
  the first nightly fuzz, 2026-10-18, about 10 ms on conll17's 250k-word JSON).
- `CheckpointFuzzTests`: seeded mutations (bit flips, truncations, extreme ints, inserted/deleted spans) of the fixtures
  and three small Stanza checkpoints in CI (their `.pt` files, and their converted `.json` where present; ~10 s), plus
  crafted inputs for each bug. Each load also reads every property name, string and vocabulary (`ReadNodes`). Longer runs:
  `STANZASHARP_FUZZ_ITERATIONS`, `STANZASHARP_FUZZ_SEED`, `STANZASHARP_FUZZ_MODELS=all`, `STANZASHARP_FUZZ_REPORT=FILE`.
  It does not validate the model code's reading of the JSON (a missing config key there is still a model-specific error).

## Checkpoint findings

What the English checkpoints actually use (Stanza 1.15.0). Port only these paths.

- **tokenize** (`combined_nocharlm`): `models/tokenization/model.py`.
  - Char embedding 32, plus 9 feature functions (`feat_funcs` in the config).
  - A 1-layer biLSTM with hidden size 64. `conv_res` is null, so there are no conv layers;
    `conv_filters` and `residual` are unused by the model.
  - `hierarchical`: a second biLSTM (`rnn2`) and `*_clf2` heads, with `hier_invtemp` 0.5.
  - `use_mwt`, so the output has 5 classes.
  - No dictionary and no charlm.
- **mwt** (`combined`): a dictionary (`dict`, consulted first because `ensemble_dict`), then
  `models/mwt/character_classifier.py`. This is **not a seq2seq model**: it is a per-character
  binary split classifier with a 2-layer biLSTM (50 per direction) and an MLP.
- **charlm** (forward/backward `1billion`): char embedding 100, a 1-layer LSTM with hidden size 1024,
  and learned `h_init`/`c_init`. The decoder is unused at inference.
- **pretrain** (`conll17`): `emb` [250000, 100] plus a vocabulary.
- **pos** (`combined_charlm`):
  - Inputs: word embedding 75, pretrain → `trans_pretrained` 125, and both charlms (2048).
    There is no own char model and no BERT.
  - A 2-layer highway biLSTM with hidden size 200 and learned `h_init`/`c_init`.
  - Outputs: UPOS through an MLP. XPOS and the 21 feats use biaffine classifiers conditioned on
    UPOS (`tag_columns`).
- **lemma** (`combined_nocharlm`): `models/lemma/trainer.py`, `models/common/seq2seq_model.py`.
  - `dicts` (v3: gzip + JSON) `{upos: {word: lemma}}`, `"*"` POS-independent. With `ensemble_dict`,
    words found by (UPOS, word) then `*` skip the model.
  - Seq2seq: char embedding 50 (vocab 234), UPOS embedding 50 prepended as an extra encoder step,
    1-layer biLSTM 2×100, `LSTMCell` decoder 200 with `soft` (dot) attention, greedy (`beam_size` 1),
    `max_dec_len` 50. No charlm, not caseless, no contextual lemmatizers.
  - The alternative `combined_charlm` (`Lemma.LemmaClassifier`): the `_charlm` seq2seq (below) plus two `contextual`
    `LemmaClassifierLSTM`s (stanza/models/lemma_classifier/lstm_model.py, trainer.predict_contextual), applied last
    first after the dictionary/seq2seq lemmas: `'s` AUX → be/have, `her` PRON → she/her (`target_words` matched
    lowercased, `target_upos`). Each runs **once per document**, every target word a row with its whole sentence,
    padded (ids 0) to the longest; argmax replaces the lemma. Per word [conll17 row of the lowercased word +
    `delta_embedding` row | `upos_emb` 20 | forward charlm | backward charlm] (2168; build_char_representation over the
    word texts, no simplify_punct), a 1-layer biLSTM 2×128 **unpacked** (Stanza packs every row at the padded width:
    `lengths` iterates the padded tensor), the output at the target, MLP 256 → 64 → ReLU → 2. Delta ids are 0-based
    (`enumerate(known_words)`, so the first word shares the padding row with unknowns; a repeated word's last index
    wins). The lemma model's resources.json dependencies add conll17, so `BackendModels.Lemma` takes a pretrain. Net
    seam `ILemmaClassifierNet` (`LemmaClassifierNet` / `ManagedLemmaClassifierNet`: `ManagedLstm.ForwardUnpacked`,
    `Gemm`; no new kernel). Not ported (unused): attention (`num_heads` > 0), no-charlm, the transformer variant.
  - `copy`: a copy gate mixes the vocab distribution with attention over the source chars (minus the
    POS step), scattered onto char ids. Characters missing from the vocab get ids past it (DeltaVocab,
    built per document over the model's words in code point order), so they can be copied.
  - `edit`: a 3-way classifier on the final encoder state (`cat(hn[-1], hn[-2])`): 0 use the decoded
    string, 1 the word, 2 the lowercased word. Empty or `<UNK>`-containing output falls back to the word.
  - Stanza's LemmaProcessor only requires tokenize; we require pos too since the model reads UPOS.
- **constituency** (`ptb3-revised_charlm`):
  - `IN_ORDER` transitions, with LSTM transition and constituent stacks.
  - `MAX` composition (`reduce_linear`), ReLU, and 2 output layers. ReLU is applied *before*
    every output layer, including the first.
  - No attention: `use_lattn` is false and `pattn_num_layers` is 0. No BERT.
  - Word input is 2268 = delta 100 + pretrain 100 + tag 20 + charlm 2048. It feeds a 2-layer
    biLSTM with hidden size 512.
  - `combined_dummy_embedding` is on, and `unary_limit` is 4.
  - The parser consumes **XPOS** tags (`retag_method: xpos`) from the POS stage.
  - Stanza prints bracket words as `-LRB-`/`-RRB-` in trees, so `(` and `)` appear that way in
    golden trees.
- **depparse** (`combined_charlm`): `models/depparse/model.py` `GraphParser`.
  - A ROOT word (id 3 in every vocab, `"\n"` for the charlms) is prepended to each sentence.
  - Input 2423 = trans_pretrained 125 + word 75 + lemma 75 + (UPOS+XPOS) 50 *twice* + charlm 2048.
    Stanza appends the tag embedding a second time where it computes the UFeats one, so the
    `ufeats_emb` weights are never used. Words and lemmas are lowercased for their vocabs and
    `simplify_punct`ed, as in the tagger.
  - A 3-layer highway biLSTM, hidden 400, learned `h_init`/`c_init` (the tagger's `HighwayLstm`).
  - Deep biaffine (pairwise) arc and label scorers (hidden 400, 49 labels), plus `linearization`
    and `distance` terms added to the arc scores. No arc embedding.
  - Decoding: log-softmax over heads, then `chuliu_edmonds_one_root` in float64. The log-softmax
    includes the batch's padding columns, so batches must match Stanza's: sorted longest first,
    5000 words (with ROOT) per batch, over 150 alone. `resolve_head_constraints` is false.
- **ner** (`ontonotes-ww-multi_charlm`): `models/ner/model.py` `NERTagger`, `trainer.py` `predict`.
  - Works on **tokens** (`doc.get(TEXT, from_token=True)`), so an MWT is tagged once.
  - Input 2148 = word 100 + charlm 2048, through `input_transform` (Linear 2148→2148). The word
    embedding is the pretrain matrix (the checkpoint's `word` vocab is conll17's) plus a fine-tuned
    `delta_emb` (48309 words); both look up the lowercased token. A word the pretrain knows but delta
    doesn't uses delta's PAD row (zeros). The charlm input is built exactly like `build_char_representation`.
  - A 1-layer biLSTM, hidden 256, `h_init`/`c_init` (zeros).
  - "multi": two tag sets, each a Linear + CRF: OntoNotes (77 BIOES tags, 18 types) and WorldWide
    (41 tags). `predict_tagset` is 0, and Stanza unmaps both but keeps only column 0, so only the
    OntoNotes head is run.
  - Decoding: numpy `viterbi_decode` in float32 (ties to the lowest id), ids < 4 become `O`, then
    `fix_singleton_tags`. `merge_tags` is the identity with one model. Batches of 32 sentences in
    document order; everything is packed, so batching only moves the last float bits.
- **sentiment** (`sstplus_charlm`): `models/classifiers/cnn_classifier.py` `CNNClassifier`
  (`model_type` CNN), in `params` of the checkpoint with `extra_vocab` and `labels` `0,1,2`.
  - Input: the sentence's **token** texts (`extract_sentences`), not words, with no lowercasing or
    `simplify_punct`. Each batch is padded at the end to its longest sentence, at least 5 (the widest filter).
  - Per token 2148 = pretrain 100 + delta embedding 100 summed (`extra_wordvec_method` SUM) + charlm
    2048. Pretrain lookup (`map_word`): the token, else without a trailing `'`, else lowercased;
    still unknown gets the learned `unk` vector (padding keeps row 0). The delta vocab (31014) looks
    the token up as written, else id 1. The charlms run over the tokens like the tagger's.
  - A 2-layer biLSTM (300 per direction) over the **unpacked** padded batch, then Conv2d filters 3, 4
    and 5 tokens × 600 (1000 channels each) and one (5, 5) filter with stride (1, 5) (8 channels ×
    120 = 960), each ReLU then max over all positions (`maxpool_width` 1): 3960 → FC 400 → 100 → 3,
    ReLU between. Dropout is off at inference.
  - `label_sentences`: sentences sorted longest first (stable), `split_into_batches` of 5000 tokens
    (longer alone), argmax.
- **The `_nocharlm` models** (`default_fast`). Each config differs from its `_charlm` twin only in
  `charlm: false`; vocabularies, batching and every other layer are the same. Instead of the charlms they
  carry `charmodel.*`, a `common/char_model.py` `CharacterModel` (`Nn.CharacterModel`):
  - Its own char vocab (`vocab.char`, not lowercased, `char_lowercase` false), code points, unknown → UNK 1.
    Char embedding 100 (padding_idx 0), a 1-layer LSTM with learned `charlstm_h_init`/`c_init`, run over
    every word of the batch as its own packed sequence (sorted by length), so batching only moves float bits.
  - **pos** (`combined_nocharlm`): unidirectional (`char_bidirectional` false), hidden 400, attention
    pooling `sum_t sigmoid(char_attn(h_t)) * h_t`, then `trans_char` (Linear 400→125, no bias). Input
    325 = word 75 + trans_pretrained 125 + char 125, in that order. Same `simplify_punct`ed words.
  - **depparse** (`combined_nocharlm`): the same character model (always unidirectional with attention in
    `GraphParser`) + `trans_char` 400→125. ROOT's characters are the single id `ROOT_ID` (3). Input 500 =
    trans_pretrained 125 + word 75 + lemma 75 + tags 50 twice + char 125.
  - **ner** (`ontonotes-ww-multi_nocharlm`): bidirectional, hidden 100, **no** attention: the final forward
    and backward states (`h[-2:]`), 200. Input 300 = word 100 + char 200 → `input_transform` 300→300.
    Reads tokens, as the charlm model does.

## Checkpoint findings (German)

What the German `default`/`default_fast` checkpoints use (Stanza 1.15.0; docs/german-plan.md has the full diff). Every
model but MWT and NER is English's architecture with other vocabularies and sizes, read from the checkpoint:
- **tokenize** (`combined_nocharlm`): 5 `feat_funcs` (no `labeled_field`/`phone_id`/`date_pattern`/`currency`), so
  `rnn.weight_ih_l0` is [256, 37]; `max_seqlen` 400 (tokens over 400 code points become `<UNK>`, windows stay 1000).
- **mwt** (`combined`): a `Seq2SeqModel` (`force_exact_pieces` null), not the classifier. `MwtExpander` runs it through
  the lemmatizer's net, now `Nn.ISeq2SeqNet` (`Seq2SeqNet` in StanzaSharp.Cuda / `ManagedSeq2SeqNet` in Nn; a POS step
  and the edit head are optional, `Seq2SeqConfig.UsesPos`/`Edit`), decoded by `Seq2SeqConfig.Beam` (the lemmatizer:
  `Greedy`):
  - emb 50, vocab 25, biLSTM 2×50, `LSTMCell` 100, soft attention, copy (over every source position: no POS step to
    skip), `max_dec_len` 10 (hdt 12), `never_decode_unk` (`<UNK>` is -inf).
  - A beam search of 5, though the config says `beam_size` 1: mwt/trainer.py passes it positionally where
    `Seq2SeqModel.predict` takes `pos`, so predict's default 5 applies. Rows are beam k · batch + b (h_in, mask, h, c
    repeated); the first step expands beam 0 alone; scores are float32 sums, torch.topk (ties to the lower index here);
    update_state reorders h and c (`ISeq2SeqDecoder.Reorder`). decode scatters the *unrepeated* source into the repeated
    rows, so only beam 0 (rows below batch) mixes in the copy distribution; the other beams keep vocab + log(1 − σ(copy)).
    Greedy decoding had matched German's golden by luck; English partut's `'90s'` needed the beam (found 2026-10-15).
  - Input `<SOS>` + code points + `<EOS>`; DeltaVocab over all the document's candidates (code point order); batches
    of `batch_size` 50 in document order, each sorted like `sort_all`. **Every** candidate goes through the model,
    then the dictionary (17 entries: `im`, `zum`, `ins`, ...) overrides it (`trainer.ensemble`), as in Stanza.
  - Output ids < 4 dropped, unmapped; blank or space-less output keeps the token. Model-only output can be odd: `IM`
    → `In deM` (mwt.json).
- **pos/lemma/depparse** (`combined_charlm`/`combined_nocharlm`, newswiki charlms): as English. STTS XPOS, 23 feats
  (Conllu.Write sorts feats case-insensitively, like Stanza's writer: the tagger's raw order differs).
- **constituency** (`spmrl_charlm`): `retag_method: upos` (the parser reads UPOS), `unary_limit` 8, SPMRL labels,
  trees `(ROOT (VROOT ...))`.
- **ner** (`germeval2014`, also in `default_fast`): the old checkpoint format. No `delta` vocab or `delta_emb` (the word
  vector is the pretrain row alone), one tag set as a plain `TagVocab` (21 BIOES tags over PER/LOC/ORG/MISC), heads named
  `tag_clf.*`/`crit._transitions` (Stanza renames them to `tag_clfs.0`/`crits.0` on load; `NerTagger.HeadPrefix`).
  Word vectors from its own pretrain, `fasttextwiki` [250000, 300]: input 2348 = 300 + 2048 → `input_transform`
  2348→2348, biLSTM 256. The pipeline loads every pretrain `Pipeline.Dependencies` names (`_pretrains`), and each
  processor gets its model's own.
- **sentiment** (`sb10k_charlm`): as English, `extra_vocab` 24,235.
- **gsd / hdt** (treebank packages: tokenize, mwt, pos, lemma, depparse; `<pkg>_nocharlm` tokenizers with their own
  vocabularies, `gsd`/`hdt` seq2seq MWT, hdt's with `max_dec_len` 12 and 33 dictionary entries):
  - Their taggers predate `tag_columns`: config without `tag_columns`/`tag_column_link` (Stanza's defaults: the three
    native columns, `tag_emb`) and heads named `xpos_hid`/`xpos_clf`/`ufeats_hid`/`ufeats_clf`, which `PosTagger`
    renames like model.py's `remap_legacy_state_dict` before either net reads them.
  - hdt's depparse batches are 1300 words (`batch_size` in its checkpoint, read like every other parser's).
  - `tests/golden/de/gsd/`, `hdt/` (`make_golden.py --lang de --treebank-only`): corpus + validation and `mwt.json`;
    `TreebankPackageTests`.
- **UD divergence study** (2026-10-16, docs/german-divergence.md; `default` and `default_fast` on UD GSD + HDT test,
  652k words): Stanza f32, TorchSharp and managed give identical CoNLL-U (every field), trees, lemmas (also vs Stanza
  float64). Smallest parser margin 1.9e-5, smallest margin / backend difference 11 (EWT 15); lemma margins ≥ 1.8e-3.
  The tools take `--lang`/`--package`/`--processors` and diff whole-pipeline CoNLL-U (`constituency_divergence.py`).
  Benchmark: docs/performance/round-33-german-benchmark.md (2.4–7.4x Stanza; the macOS smoke run's 0.90x was quick-mode
  noise).

## Checkpoint findings (English treebank packages)

What the English treebank packages' checkpoints add to the default package's paths (Stanza 1.15.0; each package is
`<pkg>_nocharlm` tokenizer and lemmatizer, `<pkg>` MWT, `<pkg>_charlm` tagger and parser on conll17 + 1billion):
- **tokenize**: `max_seqlen` 300 (ewt, lines), 400 (partut), 100 (childes); vocabularies of their own. eslspok has
  `use_mwt` false: no `mwt_clf`/`mwt_clf2`, three classes; both nets pad the two MWT classes with -inf, so the shared
  decoding sees five as before.
- **mwt**: ewt, gum, lines, atis, childes are character classifiers like combined; partut's is a seq2seq (1 layer,
  `max_dec_len` 19), decoded by the beam search German's needs too (see German's mwt above).
- **pos**: all predate `tag_columns` (legacy heads, like gsd/hdt). lines and partut have a composite XPOS vocabulary
  (`XPOSVocab`: a CompositeVocab, not keyed; lines `sep` "-", 3 parts; partut `sep` "", one part per character, 4 parts),
  so `tag_hid.xpos` has `composite_deep_biaff_hidden_dim` (100) outputs and `tag_clf.xpos.{i}` one biaffine per part;
  the tag is the parts' argmaxes joined with `sep` without `<EMPTY>` ones, "" or "_" stored as null (`Nn.XposVocab`;
  `PosOutput.Xpos` is one id array per part). A WordVocab's `<EMPTY>` id is "_" (its `ignore`), null too.
- **lemma**: atis is `caseless`: the seq2seq input, DeltaVocab and dictionary lookups use the lowercased word; the edits
  and the fallback use the word as written (trainer.postprocess). eslspok's lemma is `identity` (LemmaProcessor's
  use_identity): lemma = the word's text, no model file (`Pipeline.IdentityLemma`, skipped by `Pipeline.ModelFiles`),
  requiring only tokenize, as in Stanza.
- **depparse**: lines and partut embed the composite XPOS one table per part (`xpos_emb.{i}`), added after UPOS in part
  order (pos_emb += ...); `DepparseBatch.Xpos` is [size, width, `XposParts`], ROOT all ROOT; an unknown part is UNK, a
  missing one EMPTY (CompositeVocab.unit2id). Everything else as combined (5000-word batches).
- **biomedical** (craft, genia: `_nocharlm` tagger and parser on `pretrain/biomed` [100000, 200]; mimic: `_charlm` on
  `pretrain/mimic` [100000, 200] and its own forward/backward charlms, 1024 hidden, 101-char vocab): no mwt; tokenizers
  without `use_mwt` and with 5 `feat_funcs` (like German), `max_seqlen` 400 (craft, genia) and 300 (mimic). Their
  lemmatizers are older: no `copy` key (Seq2SeqModel's default False: no copy gate, `Seq2SeqConfig.Copy`, log-probs
  over the vocabulary alone), no `SOS_tensor`, no `contextual`. Their parsers lack `resolve_head_constraints`, and
  depparse_processor's default is True: after Chu-Liu/Edmonds, head_constraints.py's repair runs
  (`DependencyParser.ResolveHeadConstraints`: labels by each arc's best float32 log_softmax label; while a head has two
  dependents of one group — nsubj/csubj/nsubj:pass/csubj:pass, nsubj:outer/csubj:outer, obj — each choice of the one
  that keeps it reruns CLE with the others' rows plus their best allowed label, and the best tree by float32 summed
  arc + label log-prob wins; at most 5 rounds, exclusions kept). The parser then asks the net for label scores
  (`labelScores`), [size, width, width, relations] per batch. The default package's parsers say false.
- `SelectModels` requires only the processors the package has (eslspok's pos and depparse need no mwt).
- **The `_charlm` tokenizers and lemmatizers** (Alternatives; English and German): the `_nocharlm` twin's config and
  weights plus `CharacterLanguageModelWordAdapter`, whose `per_char_representation` gives the charlm's state after every
  unit (`Nn.IPerCharLanguageModel.PerChar`, both backends), each sequence from (h_init, c_init) **left to right even for the
  backward charlm** (no reversal there), packed (padding `" "`), zeros past its length.
  - tokenize (forward charlm only, `Dependencies`): input `[emb 32 | charlm 1024 | feats]` (English `rnn.weight_ih_l0`
    [256, 1065], German 1061). The charlm reads trainer.predict's raw units, the normalized `Paragraph` units: per row the
    units, then charlm `<PAD>` units (id 0) up to the row's packed length (collate's one, advance_old_batch's fill;
    `Tokenizer.Run`), zeros past it.
  - lemma (both charlms): encoder input `[emb 50 | forward 1024 | backward 1024]` (2098). Per word the states over
    `"\n" + word + " "` (the source text: lowercased for caseless atis), n + 2 of them, at encoder steps 0 to n + 1: the
    UPOS step gets the state after `"\n"`, `<SOS>` the next, and the last step (`<EOS>`) zeros (embed appends its
    raw_zeros row at the end, not where the UPOS step is). `Seq2SeqConfig.Contextual`, `ISeq2SeqNet.Encode`'s `contextual`.
- `tests/golden/<package>/` (`make_golden.py --treebank-only [--packages ...]`): corpus + validation (13 files) and
  `mwt.json` where the package has mwt; `combined` is checked against `depparse/` (same models). The tests read the .pt
  files from `Repo.StanzaModels` (`Repo.ModelsFor(lang, package)`); rows skip when a package's files are missing.

## Validation

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
    `lemma=combined_charlm+lemma_beam_size=3`; the beam changes 4 lemmas, all in validation_multilingual.
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
    PR 2; owner's decisions 2026-10-19): `Load(modelRoot, MultilingualOptions)` loads the identifier from
    `<root>/multilingual` and every configured language's `Pipeline` from `<root>/<lang>` up front
    (`MultilingualOptions.Languages`: a list of `PipelineOptions`, each keyed by its `Language`, duplicates throw; null = en
    and de; no shared default processors, unlike Stanza's `processors=`); `Restrict` (default false, as Stanza) sets the
    identifier's subset to them unless `LanguageIdentifier.Languages` names one. `Process` identifies all texts, then one
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
- CI (`.github/workflows/ci.yml`, github.com/bisforboman/StanzaSharp; pushes to `main`, PRs and
  `workflow_call` from release.yml; `ubuntu-26.04`, `ubuntu-26.04-arm`, `windows-2025`, `macos-26`, `windows-11-arm` pinned).
  - Branch protection on `main` requires `build-test`, `golden`, `docker`, `alpine`, `linux-arm64` and the three
    `cross-os (...)` checks (checked 2026-10-12). `build-test` and `golden` stay plain jobs; a matrix would rename them
    (`build-test (ubuntu-26.04)`) and block every PR. There is no merge queue (GitHub offers it only to organization-owned
    repositories; this one is user-owned): a PR merges after its light run, and the full run is main's push.
  - `build-test` runs the suite without models.
  - golden-shard's and cross-os-shard's Test steps export `ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE,STRICT` on x64
    (round 26): libtorch otherwise picks MKL/ATen kernels by CPU, and on Intel AVX-512 runners the TorchSharp German
    `default_fast` UPOS logits drifted 1.2e-4 (AMD 6.1e-5). With them every x64 runner gives the same bits. Test
    environment only; the managed backend never needed it.
  - Docs-only pull requests (`docs/`, `*.md`, `LICENSE`, `NOTICE`): the `changes` job makes `golden`, `docker`,
    `alpine`, `linux-arm64` and `cross-os` run with their steps skipped, so the required checks pass in about a minute. No job is skipped
    outright: a skipped job would count as passed. Pushes to main and
    release runs always run everything; if `changes` fails, everything runs.
  - Runner budget (user's decision, 2026-10-09): **draft** PRs skip every heavy step like docs-only ones (drafts can't
    merge; `ready_for_review` starts the real run), so agents open stacked PRs as drafts and mark them ready once their
    base merges. **Ready** PRs run level `light`: `docker` and `alpine` run everything; `cross-os` and
    `linux-arm64` run only `PipelineTests`, `ManagedBackendTests` and `ManagedCheckTests` (70 tests, ~2.5 min locally)
    and no package checks. Pushes to main and release runs are level `full`.
  - TorchSharp smoke (owner's decision, 2026-10-10): `changes` also outputs `torch`. A ready PR whose files miss
    `src/StanzaSharp.Cuda/`, `src/StanzaSharp.Nn/`, `src/StanzaSharp/{BackendModels,PipelineBackend}.cs`,
    `Directory.Build.props` and `.github/workflows/` gets `torch=smoke`: golden runs every test except the
    TorchSharp-backend cases (DisplayName `managed: False`, `[Trait("Backend", "TorchSharp")]`), keeping those marked
    `[Trait("Smoke", "TorchSharp")]` (`PipelineTests.Process_ReproducesGoldenConlluExactly`,
    `InputModeTests.Pretokenized_MatchesGolden`: all processors of both packages) and verify-package's plain mode.
    So ConcurrencyTests runs only its managed cases there. Filter
    `(DisplayName!~managed: False&Backend!=TorchSharp)|Smoke=TorchSharp`: 238 of 282 tests, ~31 of 78 min summed.
    Everything else (drafts/docs-only aside) is `torch=full`. A new expensive TorchSharp-only test without a
    `managed` parameter needs `[Trait("Backend", "TorchSharp")]` to stay out of light runs.

    | event | code | level | torch | golden runs |
    |---|---|---|---|---|
    | draft PR | false | light | – | steps skipped, passes in ~1 min |
    | docs-only ready PR | false | light | smoke | steps skipped, passes in ~1 min |
    | ready PR, only `src/StanzaSharp.Pos/**` | true | light | smoke | managed + smoke, 4 shards |
    | ready PR touching `src/StanzaSharp.Cuda/**` | true | light | full | everything, 4 shards |
    | push to main, release fallback (`workflow_call`) | true | full | full | everything, 4 shards |
    | `changes` failed | (empty) | (empty) | (empty) | everything (anything not clearly false/smoke) |
  - `concurrency` (`${{ github.workflow }}-${{ github.ref }}`): a newer push to the same PR or to main cancels the older
    CI run (a release waiting on that run then fails; see Packaging and releases). release.yml's fallback call gets the
    group `Release-refs/heads/main` (`github.workflow` is the caller's name), so it and main's own CI run don't cancel
    each other, and it is never cancelled.
  - `golden` (owner's decision, 2026-10-10) is an aggregate: it `needs` the matrix job `golden-shard` (1, 2, 3, 4), runs
    `if: !cancelled()` and fails unless `needs.golden-shard.result` is `success` (a skipped or failed shard fails it),
    so the required check keeps its name. The shards split the test project by class, balanced on the golden trx
    (full run, minutes summed; xunit runs a class's tests one at a time and ConcurrencyTests after everything else):
    - 2: `ConcurrencyTests.ConcurrentCalls_*` + FastPackage, GcPressure, ManagedBackend, Tokenizer: 87 tests, 17.9 min;
      plus CompareCommand and BenchmarkCommand (2026-10-12: the shard that finished first), with Python Stanza
      (`STANZASHARP_PYTHON` from setup-python; installing tools/requirements.txt adds ~40 s, the Python runs ~1 min).
    - 3: Pipeline, Ner, Sentiment, InputMode, Depparse, NoSsplit: 72 tests, 45.5 min (parallel; Pipeline 10.3 alone).
    - 4 (2026-10-18): the `Transformer*` classes (`SHARD_FOUR`), the only shard that downloads the transformer. On a
      pull request only with the transformer's paths (above); the matrix is `fromJSON(needs.changes.outputs.shards)`, so
      a shard left out is absent, not skipped (a skipped shard would fail `golden`).
    - 1: the complement of 2, 3 and 4 (ConcurrencyTests' cancellation cases and every other class, new ones too):
      123 tests, 14.6 min. So the union is always the whole suite, each test once (checked with `--list-tests`).
    - Expected wall per shard on full: ~12–15 min (was 38–44 for one job); light: ~4–6 min.
    Each shard, on a model-cache miss, downloads the models with the C# `ModelDownloader` (via the
    CLI) and converts them with CPU torch. The models are cached on the en group's file list (see "Model cache keys"),
    `tools/requirements.txt` and `tools/stanza_convert.py`; shard 1 writes the cache, shards 2–3 (like docker and
    alpine) only restore it. Each shard fails if any test is skipped
    (`outcome="NotExecuted"` in the trx; the trx `notExecuted` counter stays 0 for skips) or none ran. Shard 1 runs
    `tools/verify-package.ps1` against `models/stanza/en` twice: plain (StanzaSharp + TorchSharp-cpu, TorchSharp
    backend, `StanzaSharpTrimNative`; also on torch=smoke) and `-Managed` (only StanzaSharp, the default backend, no
    libtorch), and `verify-tool.ps1`. Artifacts: `test-results-golden-<shard>`.
  - German (owner's decision, 2026-10-13): one `models-de-treebanks-<hash of the de group's file list>` cache holding only Stanza's
    German `.pt` files (`models/stanza/de`, default, default_fast, gsd and hdt, about 1.3 GB; no conversion: `Repo.ModelsFor("de")`
    falls back to models/stanza/de), written by golden-shard 1 and restored by shards 2–3, downloaded with the CLI
    (`download models/stanza/de --lang de --package ...`) on a miss. Only the golden shards run German, so their no-skip
    check covers it. cross-os and linux-arm64 stay English: there the German rows skip, and their no-skip checks exempt
    test names matching `lang: "de"` or `German` (trx-escaped `lang: &quot;de&quot;`).
  - Model cache keys (owner's decision, 2026-10-16): each model cache's key hashes exactly the files and MD5s it holds,
    not `ModelDownloader.cs`, so a new or changed MD5 rotates only the caches holding that file. `tools/ci-models.sh` defines
    each group once as lines of `download` arguments: `en` (default + default_fast: `models-en-<hash>-<hash of
    requirements.txt and stanza_convert.py>` and `models-pt-<os>-<hash>`), `en-treebanks` (the treebank packages and
    `Pipeline.Alternatives`' English models, minus en's files) and `de` (all German). Each job's "Model cache keys" step
    builds the Cli and runs `tools/ci-models.sh keys`, which hashes the Cli's `download ... --list` output (`path md5`
    lines, no network) per group; the downloads on a miss are `tools/ci-models.sh download GROUP`, and the treebank
    cache's paths are the group's file list. A new model goes into a group line there, nowhere else in ci.yml.
  - English treebank packages (owner's decision, 2026-10-15): one `models-en-treebanks-<hash>` cache
    of only their own `.pt` files (the en-treebanks group's files that models-en lacks, about 2.8 GB with the domain NER models),
    restored into models/stanza/en next to the default package's, written by golden-shard 1 and restored by shards 2–3, downloaded with
    the CLI (`download models/stanza/en --package ...`) on a miss. So that models-en doesn't take them in, shard 1 restores
    models-en with `actions/cache/restore` and saves it with `actions/cache/save` right after the convert step, before the
    treebank models arrive. cross-os and linux-arm64 don't have them: their no-skip checks exempt `TreebankPackageTests`.
    The same cache holds `Pipeline.Alternatives`' English files (explicit paths, 501 MB: ptb3_charlm, the four NER
    models, fasttextcrawl), and models-de-treebanks German ner conll03 (51 MB); their rows (`ModelChoiceTests`) are
    exempt elsewhere the same way.
  - The treebanks' `_nocharlm` taggers and parsers and `_charlm` tokenizers and lemmatizers (owner's decisions, 2026-10-16):
    their own `models-treebank-choices-<key>` cache (tools/ci-models.sh `en-treebank-choices` + `de-treebank-choices`, their
    files only, explicit paths `treebank_choices_paths`; English 965 + 74 MB, German 269 + 41 MB), so adding them rotated no
    other key; written by golden-shard 1, restored by 2–3. `CharlmInputTests` (ewt's `_charlm` models) is exempt from the
    no-skip checks elsewhere, like `ModelChoiceTests`. So that
    models-de-treebanks (path `models/stanza/de`) doesn't take the German ones in, shard 1 restores it with
    `actions/cache/restore` and saves it with `actions/cache/save` right after the German download.
  - `cross-os (windows-2025)` / `cross-os (macos-26)` (Apple Silicon; `TorchSharp-cpu` brings
    `libtorch-cpu-osx-arm64`): build, full suite with `STANZASHARP_MODELS` = `models/stanza/en` (the
    `.pt` files, no Python or conversion), the same no-skip check, and both `verify-package.ps1` runs. Models are
    cached per OS on the en group's file list only; no NuGet cache, to stay inside the 10 GB cache budget.
    - Sharded like golden (owner's decision, 2026-10-13): the matrix job `cross-os-shard` (os × shard 1–3) uses golden's
      first three filters (`SHARD_TWO`/`SHARD_THREE` in the workflow `env`; shard 1 the complement, without `SHARD_FOUR`'s
      `Transformer*` classes, which only golden shard 4 can run). As one job the tests took
      29 / 40 / 61 min (windows-2025 / macos-26 / windows-11-arm), mostly ConcurrencyTests' two halves running alone
      at the end (7.6+7.6, 11.8+11.4, 25.4+17.6 min); the shards run them side by side. Each shard has every step
      (libomp, model cache, no-skip check, trx `test-results-<os>-<shard>`); the package checks run only in shard 3.
      Light runs have one shard per OS (`fromJSON` on `level`) with the light filter: PipelineTests doesn't split.
      3 macOS shards + benchmark.yml's 2 macOS jobs = the 5 concurrent macOS jobs free runners allow.
    - The required checks `cross-os (<os>)` are an aggregate matrix job over the three OSes (`if: !cancelled()`). It
      passes at once if `needs.cross-os-shard.result` is success; otherwise (that result covers all nine) it lists this
      run's jobs through the API and passes only if every `cross-os-shard (<os>, *)` job concluded success, so a
      failure fails only its own OS's check. No shard found fails.
    - macOS first needs `brew install libomp` for the TorchSharp backend's tests. `libtorch_cpu.dylib` from
      `libtorch-cpu-osx-arm64` 2.10.0 links `/opt/homebrew/opt/libomp/lib/libomp.dylib` by absolute path, not the
      `libomp.dylib` it ships. Without it TorchSharp reports only "doesn't contain a reference to
      libtorch-cpu-osx-arm64"; `otool -L` on the dylib shows the real cause. TorchSharp-backend users need it too
      (README, PACKAGE.md); the default managed backend doesn't.
  - `docker`: `tools/verify-docker.sh MODEL_DIR BASE...` packs StanzaSharp into `samples/docker/feed` (with a
    generated nuget.config), builds the sample per base image (runtime:10.0 and 10.0-noble-chiseled; `RID`
    linux-x64, linux-musl-x64 for a base named `*alpine*`), fails if any `libtorch*`/`libc10*`/`libgomp*` is in the
    image, and `cmp`s the container's CoNLL-U for corpus.txt with pipeline.conllu. Models from golden's cache via
    `actions/cache/restore` (no new entry; downloaded on a miss); no Docker layer cache (budget).
  - `alpine`: the same on `runtime:10.0-alpine` (musl). It replaced `alpine-experiment.yml`, whose verdict (PR #10)
    still holds for the TorchSharp backend: libtorch needs glibc-only symbols gcompat lacks (`__*_chk`, `backtrace`,
    `fcntl64`, `__res_init`, `pthread_attr_setaffinity_np`), so Alpine is managed-only.
  - `cross-os (windows-11-arm)`: Windows on Arm64 (GA runner label for public repos). It skips the plain
    `verify-package.ps1` (TorchSharp-cpu has no Arm64 libtorch) and runs `-Managed`.
  - `linux-arm64` (`ubuntu-26.04-arm`): no libtorch or LibTorchSharp exists for linux-arm64, so it builds with
    `-p:StanzaSharpNoLibTorch=true` (the Tests drop TorchSharp-cpu) and runs `--filter "DisplayName~managed: True|Backend=Managed"`:
    the managed cases of the both-backend theories plus tests marked `[Trait("Backend", "Managed")]` (they must need
    no libtorch: check with that build and an empty `NUGET_PACKAGES` locally). Then the no-skip check and
    `verify-package.ps1 -Managed`. Models cached like cross-os (`models-pt-Linux-…`).
  - `nightly.yml` (owner's decision, 2026-10-18): 01:00 UTC on main, and `workflow_dispatch`. `transformer` calls ci.yml
    with `golden_shards: '[4]'` (every other ci.yml job skips on that input); `fuzz` runs `CheckpointFuzzTests` with
    `STANZASHARP_FUZZ_MODELS=all`, 3,000 iterations and the run number as seed, then `tools/text_fuzz.py run` (tokenize,mwt
    1,000 texts; default and default_fast 60 each) with Python Stanza; `benchmark` calls benchmark.yml `quick` without
    the avx512 jobs (`no_avx512`). Any failure comments on the open issue "Nightly run failed" (`report`, created if
    missing; closed by a person). Never required; release.yml waits for main's push run only, so it never sees these.
  - `benchmark.yml` (owner's decision 2026-10-10): `workflow_dispatch` (inputs `lang`, `package`, `quick`) and
    `workflow_call` (nightly.yml), never on pushes/PRs, never required; free standard runners only (ubuntu-26.04, ubuntu-26.04-arm, windows-2025,
    windows-11-arm, macos-26). Each job builds the tool, restores the `models-pt-<os>` cache (keyed like ci.yml's, by `tools/ci-models.sh keys`) (`lang=de`: golden-shard 1's
    `models-de-treebanks-…`, Linux only; restore only; downloads the package on a miss), installs Python 3.12 + `tools/requirements.txt` from the CPU wheel index (continue-on-error: without it the
    block has StanzaSharp only), runs `stanzasharp benchmark --json`, writes the block to its summary and artifact
    `benchmark-<os>`. `summary` combines them with `tools/benchmark_table.py` into `benchmarks.md` (summary + artifact).
    The tool is built with `-p:Version=<git describe>` (full-history checkout), so the block names e.g.
    `1.0.0-67-g97089bb` instead of Directory.Build.props' `0.1.0-dev`. First run: docs/performance.md "Run it yourself".
  - Keep paths forward-slash and file names case-exact (Linux). The root `.gitattributes` keeps sources
    LF; `tests/golden/.gitattributes` pins golden files (`eol=lf`, `binary`, `validation_whitespace.txt`
    `-text`), so Windows checkouts with `core.autocrlf=true` stay byte-exact. Bash steps run under
    `shell: bash` on every OS (Git Bash on Windows).

## Packaging and releases

- Dependabot (owner's decision, 2026-10-14; `.github/dependabot.yml`): GitHub Actions and NuGet, weekly.
  `dependabot-auto-merge.yml` turns on auto-merge (merge commit) for minor and patch updates, so they land once the
  required checks pass; majors wait for a person. Ignored: TorchSharp*/libtorch-* (move together, check
  dotnet/TorchSharp#1583), StyleBro.* (prerelease, upgraded by hand with a reformat), StanzaSharp* (the docker
  sample); pip is not watched (tools/requirements.txt pins the golden data). Microsoft.Extensions.Logging.Abstractions
  is included (the owner chose auto-merge for it too), so a bump raises the package's minimum dependency.
  A Dependabot merge to main runs main's full CI like any other (seen 2026-10-14, coverlet #115).

- Public API (user's decision for 0.1.0, 2026-10-06: "pipeline only"). It is:
  - the facade: `Pipeline`, `PipelineOptions`, `CharlmCacheOptions`, `ModelDownloader`, `Processor` (0.4),
    `PipelineBackend` (0.5);
  - StanzaSharp.Cuda's `CudaBackend` (1.0);
  - Core's results: `Document`, `Sentence`, `Token`, `Word`, `Entity`, `Tree`, `Conllu`, `Json` (owner's decision,
    2026-10-14), `Dependency` (#174, 2026-10-18);
  - the facade's `ModelPath`, `LanguageIdentifier`, `LanguageIdentifierOptions`, `MultilingualPipeline`, `MultilingualOptions` (#174, 2026-10-19);
  - Core's `CorefChain`, `CorefMention`, `CorefAttachment` (coref PR b, owner's decision 2026-10-20: Stanza's shape).

  Everything else is `internal`.
  - Directory.Build.props gives every `src/*` assembly `InternalsVisibleTo` for the other StanzaSharp
    assemblies, the tests, the benchmark and the Cli sample. Don't add per-project `InternalsVisibleTo`.
  - New types default to `internal`. Making one public is an API decision for the user.
  - `src/*` builds docs with CS1591 as an error, so every public member needs an XML doc.
  - Each user-visible change adds one file, `changes/<section>/<slug>.md` (section: added, changed, deprecated, removed,
    fixed, security), holding its CHANGELOG bullet (user's decision, 2026-10-11: CHANGELOG.md's "Unreleased" list
    conflicted between parallel PRs). Never edit CHANGELOG.md for an unreleased change. `changes/README.md` has the format;
    `python tools/changelog.py preview` shows the next section.
- `Pipeline.Load(dir, PipelineOptions?)` (user's decision, 2026-10-06: an options object rather than more
  optional parameters).
  - `PipelineOptions` holds `Processors`, `Backend` and `CharlmCache { IsEnabled = true, MaxWords = 32_768 }`, among others.
  - `CudaBackend.Create(disableTf32: true)` (was `PipelineOptions.DisableTf32` up to 0.5; name and opt-in are the user's
    choices) sets both torch TF32 switches to false process-wide at Load and doesn't restore them.
  - `CharlmCache.TryAdd` keeps a sentence only if it fits, detaching its tensors from the caller's
    dispose scope. Otherwise it leaves them with the caller. Before this fix the cache disposed
    rejected tensors that the tagger was still using.
- Install (owner's 0.5 decisions, 2026-10-09): `dotnet add package StanzaSharp` is the whole default install (managed
  backend). `TorchSharp-cpu` appears in the docs only for the TorchSharp backend.
  - The `StanzaSharp.Cpu.*` platform packages (up to 0.4.2: code-free, StanzaSharp + one `libtorch-cpu-<rid>`) are no
    longer published from 0.5; their projects and `src/StanzaSharp.Cpu.props` are gone. TorchSharp-backend users add
    `TorchSharp-cpu` or one `libtorch-cpu-<rid>` 2.10.0 themselves (TorchSharp loads a single one fine; checked with
    `libtorch-cpu-win-x64` alone). Deprecating them on nuget.org is the owner's manual step.
  - `release.yml` packs `StanzaSharp`, `StanzaSharp.Cuda` (from `src/StanzaSharp.Cuda`) and `StanzaSharp.Tool`.
  - On a Windows Arm64 machine Directory.Build.props gives the runnable projects (tests/, samples/)
    `libtorch-cpu-win-arm64` (`TorchSharp-cpu` lacks it), except those setting `StanzaSharpNoLibTorch`.
  - Supported platforms (README, PACKAGE.md): managed backend wherever .NET 10 runs; CI covers linux-x64, win-x64,
    win-arm64, osx-arm64 (full suite), linux-arm64 (managed tests) and Alpine (Docker). macOS Intel is not claimed.
    TorchSharp backend: where TorchSharp 0.107 has `runtimes/<rid>` and a `libtorch-cpu-<rid>` 2.10.0 exists
    (linux-x64, win-x64, win-arm64, osx-arm64); not musl, linux-arm64 or osx-x64.
  - `samples/docker` is a standalone app (its own empty Directory.Build.props) referencing only `StanzaSharp` (nothing
    of TorchSharp in the image since 1.0), on
    `mcr.microsoft.com/dotnet/runtime:10.0` (`BASE` arg; `RID` arg, linux-musl-x64 for Alpine).
  - New package IDs need no policy change: a nuget.org Trusted Publishing policy applies to all packages of its owner
    (optionally narrowed by a glob scope). StanzaSharp.Cpu.*, StanzaSharp.Tool and StanzaSharp.Cuda all published with
    the existing policy.
- `buildTransitive` (StanzaSharp.Cuda package since 1.0; `src/StanzaSharp.Cuda/buildTransitive/StanzaSharp.Cuda.targets`
  + a `StanzaSharp.Cuda.props` generated at pack time). Both checks below matter only for TorchSharp-backend users (owner, 2026-10-09: they stay). `TorchSharpVersion` (0.107.0) and `LibTorchVersion` (2.10.0) are set only in Directory.Build.props.
  - `STANZA001` (warning): a resolved `TorchSharp`, `TorchSharp-*` or `libtorch-*` package has another version. Versions
    come from `@(PackageDependencies)` + the assets file's `"<id>/<version>"` keys (copy-local items miss the
    asset-less `TorchSharp-cpu`; `TorchSharp-cpu` 0.106.0 resolves the same TorchSharp/libtorch as 0.107.0).
    `<NoWarn>STANZA001</NoWarn>` silences it. `verify-package.ps1` fails on any `STANZA` warning.
  - `StanzaSharpTrimNative=true` (opt-in) removes from `NativeCopyLocalItems`/`RuntimeTargetsCopyLocalItems` (so also
    from the .deps.json): libtorch_python, libshm, libnnapi_backend, libtorchbind_test, libjitbackend_test,
    libbackend_with_compiler, libaoti_custom_ops (.so; .dylib for python/shm). By DT_NEEDED/LC_LOAD_DYLIB nothing that
    LibTorchSharp → libtorch → libtorch_cpu → libc10/libgomp needs them, and TorchSharp loads only `LibTorchSharp`
    and `torch_cpu`/`torch_cuda` by name. Windows libtorch has no such files (its only unlinked ones, uv.dll,
    libiompstubs5md.dll and torch_global_deps.dll, 250 KB, are kept). A marker file in `obj/` is a .deps.json input,
    so switching the property regenerates it. Proven by the plain `verify-package.ps1` (it sets the property and
    checks the files are absent; Linux in golden, macOS in cross-os).
- `StanzaSharp.Tool` (`src/StanzaSharp.Tool`, `PackAsTool`, command `stanzasharp`): `download [DIR] [--lang CODE] [--package NAME]
  [--processors LIST]`. `DownloadCommand.cs` is compiled into samples/StanzaSharp.Cli too (one implementation). Since
  1.0 StanzaSharp has no TorchSharp, so the tool has no native files at all (0.5 MB with compare and benchmark). `tools/verify-tool.ps1` (golden) packs, `dotnet tool install --tool-path`s it from a local feed,
  checks for natives and downloads `tokenize,mwt`. release.yml packs it.
  - `compare FILE [--lang CODE] [--package] [--processors] [--models DIR] [--python PATH]` (`CompareCommand.cs`, owner's
    "proof for new users"; `--lang` 2026-10-14): runs the embedded `compare.py` (make_golden's bytes→str and `"{:C}\n"`)
    with `stanza.Pipeline(lang, dir=<parent of DIR>, resources_filepath=<embedded stanza_resources_<lang>.json>,
    download_method=None)`, so no network and no resources.json in the model folder; C# loads with `Language`, DIR
    defaults to `models/stanza/<lang>`, and the processor list and files come from that language's package (a German
    treebank package runs tokenize to depparse). DIR may have any name: if it isn't
    `<lang>`, compare.py's `load_stanza` (benchmark.py imports it) links it as `<lang>` in its temp folder (a junction on
    Windows, which needs no privilege; a symlink elsewhere) while Stanza loads, then unlinks it itself (.NET's
    `Directory.Delete(recursive)` throws UnauthorizedAccessException on such a junction). Both sides
    get the full processor list (requirements added); Stanza's loaded `.pt` set must equal `ModelDownloader.FilesFor`,
    and C# loads with `VerifyChecksums`. Transformer models (2026-10-18, owner's choice): `load_stanza`'s
    `transformers_locally` has Stanza's `load_bert` read `<DIR>/transformer/<name>/` (local_files_only, like make_golden's
    `electra_locally`); a missing transformers/tokenizers/safetensors/peft exits 3 with `TRANSFORMER_REQUIREMENTS` (the
    pins of tools/requirements-transformer.txt, `ComparePy_TransformerRequirements_MatchRequirementsTransformerTxt`). Local
    only: CI's Python has none of them, so no test runs Stanza on these models. C# offsets are converted to code points when the text has non-BMP characters
    (`ConlluDiff.ToCodePointOffsets`). Exit 0/1/2. `stanza_resources_en.json` is `{"stanza_version", "en": resources.json["en"]}`
    of Stanza 1.15.0 (`stanza_resources_de.json` likewise with `de`); `CompareCommandTests.EmbeddedResources_MatchModelDownloader` fails when its version or an MD5 of
    `ModelDownloader.Files` disagrees. Regenerate it after a Stanza version change (Stanza's own downloader, run from
    the repo root; `stanza.__version__` is the new version):
    `tools/.venv/Scripts/python -c "import json, stanza; from stanza.resources.common import download_resources_json as d; d('models/stanza'); r = json.load(open('models/stanza/resources.json', encoding='utf-8')); open('src/StanzaSharp.Tool/stanza_resources_en.json', 'w', encoding='utf-8', newline='\n').write(json.dumps({'stanza_version': stanza.__version__, 'en': r['en']}, indent=1, ensure_ascii=False) + '\n')"`.
    `Run_PythonStanza_*` (CompareCommandTests, BenchmarkCommandTests) need tools/.venv (or `STANZASHARP_PYTHON`) and
    models/stanza/en (the German ones, `[PythonStanzaFact("de")]`, models/stanza/de next to it); CI runs them in golden-shard 2, the only job with Python Stanza; the other jobs' no-skip checks
    exempt them by name.
  - `benchmark [--lang CODE] [--package] [--processors] [--models DIR] [--threads N] [--words N] [--python PATH] [--quick] [--json FILE]`
    (`BenchmarkCommand.cs`, owner's decision 2026-10-10): managed backend vs the embedded `benchmark.py`, one Python process
    that loads Stanza once and answers JSON-line commands (`run`, `calls`) so runs alternate (warm-up runs on the whole text
    until each side settles, `WarmedUp`: no stage > 10% and 20 ms faster, 2–5 runs, round 16; 3 timed runs,
    medians; 50 one-sentence calls, median/p90; peak working set, macOS `ru_maxrss`). Text: the benchmark sample's golden
    texts (validation, corpus, tokenize_stress), embedded under `benchmark/` (`--lang de`: tests/golden/de's files of the
    same names, under `benchmark/de/`; the block's Run row names the language), cut after the paragraph that reaches 1,500
    whitespace words (`--quick`: 300, 1 run, 10 calls); the block reports words after tokenizing and MWT (2,009 by default). Threads default to each side's own (unless `--threads`). Python is optional: any failure
    (no Python/Stanza, models not `.pt`) gives StanzaSharp's column only, with the reason. With
    Python, the CoNLL-U of the last run is compared (`ConlluDiff`). Machine: CPU from the registry / `lscpu` (LC_ALL=C,
    else /proc/cpuinfo) / `sysctl`, cores from GetLogicalProcessorInformation / lscpu / `hw.physicalcpu`, SIMD
    `Gemm.Detect()`. Report formatting is invariant-culture ASCII (`x` for ratios). `BenchmarkCommandTests`
    (`Run_PythonStanza_Quick` and `_GermanQuick` run in golden-shard 2); `verify-tool.ps1` runs `benchmark --quick`.
- The `StanzaSharp` package, packed from `src/StanzaSharp` (user's decision, 2026-10-06).
  - It carries all nine assemblies (facade, Core, Nn, the processors) plus their XML docs: the facade's
    ProjectReferences are `PrivateAssets="all"`, and an `IncludeProjectReferences` target adds them.
  - Its only dependency is `Microsoft.Extensions.Logging.Abstractions`; no TorchSharp since 1.0.
- `StanzaSharp.Cuda` (1.0), packed from `src/StanzaSharp.Cuda` (assembly `StanzaSharp.Cuda.dll`): depends on
  `StanzaSharp` (same version) and managed `TorchSharp`; its other ProjectReferences are `PrivateAssets="all"` (their
  assemblies come with StanzaSharp). Users add `TorchSharp-cpu`/`-cuda-*` themselves. Readme: its `PACKAGE.md`. Its
  public API is `CudaBackend` (`Create(deviceIndex = 0, disableTf32 = false)`, `Cpu`). Publishing it needs the
  nuget.org Trusted Publishing policy to allow the ID (owner's step).
  - Every other project is `IsPackable=false` (Directory.Build.props, which also holds the shared
    package metadata and a `0.1.0-dev` default version).
  - Because of `PrivateAssets`, tests and samples reference the library projects they use directly.
  - `src/StanzaSharp/PACKAGE.md` is the package readme; `NOTICE` ships in the package.
- Models are downloaded only explicitly (user's decision): `ModelDownloader.DownloadAsync(dir)`, `stanzasharp download
  [DIR]` (the tool) or `StanzaSharp.Cli download [DIR]` (the same code).
  - Selective download (user's decision, 2026-10-06): `DownloadAsync(dir, processors)` / `download
    --processors LIST`.
  - Packages (user's decision, 2026-10-06: Stanza's package names as strings, so more can be added without
    API changes): `PipelineOptions.Package` (a package of `Pipeline.Packages[lang]`: `"default"`, `"default_fast"`, `"default_accurate"`, the treebank packages; anything else throws) and
    `DownloadAsync(dir, PipelineOptions)` / `download --package NAME`. The downloader takes the same options
    object as `Load`, so one object downloads and loads the same files; it ignores the other options.
    `Processors` is null by default: all the package's processors. A processor the package lacks throws.
  - `DownloadAsync(dir)` and `DownloadAsync(dir, processors)` (0.1.0 API) mean the English `default` package.
  - Languages (owner's decisions, 2026-10-13, docs/german-plan.md): `PipelineOptions.Language` (Stanza's code, default
    `"en"`; `Load` still takes the language folder); `DownloadAsync(dir, options)` honours it, `download --lang de`
    defaults to `models/stanza/<lang>`. German (`de`): `default`, `default_fast`, `combined`, `gsd`, `hdt` download and
    load. English also has the treebank packages (`combined`, `ewt`, ..., `eslspok`, `craft`, `genia`, `mimic`; 2026-10-15). Unknown languages
    and packages (German `default_accurate`: another transformer) throw `ArgumentException` with the supported list.
  - `Pipeline.SelectModels` (language + package + processors → processor → model) and `Pipeline.SharedModels`
    (from `Pipeline.Dependencies`: each model's resources.json `dependencies`, since names don't tell them; German
    NER reads `fasttextwiki` + the charlms) are what both `Pipeline`'s constructor and `ModelDownloader.FilesFor` use,
    so they cannot drift apart. A new package needs an entry in `Pipeline.Packages[lang]`, its models'
    `Pipeline.Dependencies[lang]` and its files' MD5s in `ModelDownloader.Files[lang]`;
    `CompareCommandTests.EmbeddedResources_MatchModelDownloader` checks all three against the tool's embedded
    `stanza_resources_<lang>.json`, and `FilesFor_AreEnoughToLoadThePipeline`, `FileTableCoversEveryModelThePipelineLoads`
    and `LanguageTests` the file sets. CI downloads the packages per `tools/ci-models.sh` group (see "Model cache keys").
  - It fetches the `.pt` files from `huggingface.co/stanfordnlp/stanza-<lang>/resolve/v1.15.0/models/`
    and checks each against the MD5 from Stanza 1.15.0's resources.json, kept in
    `ModelDownloader.Files`.
  - It keeps files that already match. `Pipeline.Load` never downloads.
  - Changing the Stanza version means new MD5s, new golden data and a new cache key.
- `tools/verify-package.ps1` packs a unique `0.0.0-verify.<time>` version and builds a fresh console app against it:
  plain, StanzaSharp.Cuda + `TorchSharp-cpu` and `StanzaSharpTrimNative`, running the README example with `Backend =
  CudaBackend.Cpu` and checking the parse; `-Managed`, only StanzaSharp, running ManagedCheck's Program.cs and failing
  if anything of TorchSharp or libtorch is in the output. It removes that version from the NuGet cache afterwards.
- Releases start from the CHANGELOG (user's decision, 2026-10-07; before, a pushed tag started them, and hand-made
  tags twice landed wrong). Merging a PR "CHANGELOG: x.y.z" (made by `python tools/changelog.py release x.y.z`, which writes
  `## [x.y.z] - date` from `changes/` and deletes those files) runs
  `.github/workflows/release.yml` (push to main touching CHANGELOG.md): its `version` job reads the first version
  heading and stops if tag `vx.y.z` exists; otherwise its `ci` job waits for main's ci.yml push run of the same commit
  (`gh run list --workflow ci.yml --commit $GITHUB_SHA --branch main --event push`, polled every minute, up to 170 min)
  and fails unless it succeeded (owner's decision, 2026-10-13; before, release.yml ran all of ci.yml a second time via
  `workflow_call`, and the two suites competing for runners made a release take 40+ min). A cancelled CI run (a newer
  push to main cancels it) or a failed one fails the release with a message: re-run that CI run, then the release run's
  failed jobs. Only if no such run turns up within 2 min does `checks` call ci.yml as before. Then it packs that version, pushes via NuGet
  Trusted Publishing (`NuGet/login@v1`, repo variable `NUGET_USER`), and only then creates the tag on that commit
  and the GitHub release (`gh release create --target`; a prerelease if the version has `-`). Never push release
  tags by hand. A failed publish is retried by re-running the run; `--skip-duplicate` keeps already-pushed packages.
  - Environment `prerelease` is for versions with `-`; `release` (approval required) for the rest. Both allow
    deployments from branch `main`.
  - One-time setup, done 2026-10-06:
    - nuget.org Trusted Publishing policies for both environments (owner bisforboman, repo
      StanzaSharp, workflow `release.yml`), added by the owner.
    - The repo variable `NUGET_USER` = `bisforboman`.
    - GitHub environment `release`: required reviewer bisforboman; deployments from branch `main` (and the old
      tag rule `v*`).
    - GitHub environment `prerelease`: no reviewers; branch `main` (and tag rule `v*-*`).
    - A 401 "No matching trust policy" on push means the nuget.org policy for that environment is
      missing or misnamed.
- `main` is protected like StyleBro since 2026-10-06 (user's decision): pull requests required (0
  approvals), required status checks (`build-test` and `golden` at first; all eight CI jobs now, see CI above),
  enforced for admins, auto-merge allowed,
  merged branches deleted. Every change goes through a PR. A new required CI job must also be added to the
  protection's required checks.

## Managed backend (#29)

- **Goal:** the main `StanzaSharp` package runs on a pure managed backend (C# SIMD kernels, no native dependencies).
- **Owner's decisions** (2026-10-07; GPU revised 2026-10-08):
  - GPU stays, as a separate opt-in package `StanzaSharp.Cuda` carrying the TorchSharp backend. So the backend seam
    (Phase 1) is permanent with two implementations, managed and TorchSharp; the TorchSharp one is also the CPU test
    reference.
  - Tolerances (revised 2026-10-08): the managed backend uses the **same score tolerances as TorchSharp** (1e-4 where
    tests use 1e-4; 1e-3 only where TorchSharp already has it, e.g. sentiment with the cache), and discrete outputs are
    byte-identical. A managed comparison that misses 1e-4: find the drift's source, try cheap fixes (higher-precision
    accumulation, reduction order) and measure their cost; if none is cheap, report the drift, source and options with
    costs to the owner. Never a managed-only tolerance, never a skipped test.
  - Accepted: a slower Arm64 path at first; up to ~15% slower at 1 thread.
  - 0.5: managed backend the default, TorchSharp still selectable. 1.0: TorchSharp leaves the main package (not the
    project).
  - Public backend option (2026-10-08): **A**. `PipelineOptions.Backend` takes a sealed public `PipelineBackend` class whose
    constructor and members are internal: `PipelineBackend.Managed`, `PipelineBackend.TorchSharp`; later
    `CudaBackend.Create(device, disableTf32)` in `StanzaSharp.Cuda`, created through internals (the main package never
    references it; device and TF32 travel inside the backend). No enum.
    - 0.5: Managed is the default; `Device`/`DisableTf32` become `[Obsolete]` and mean "TorchSharp on that device". A
      managed pipeline must load with no native libtorch package: `Load` must not call `torch.set_num_threads` (or any
      other torch function) unless TorchSharp is used.
    - 1.0 (done): `Device`, `DisableTf32` and `PipelineBackend.TorchSharp` are gone. `PipelineBackend` (internal
      constructor: a name and a `Func<BackendModels>`; `ToString` = the name) has only `Managed` in the main package;
      `CudaBackend.Create(deviceIndex, disableTf32)` ("Cuda:N") and `CudaBackend.Cpu` ("TorchSharp") are in
      StanzaSharp.Cuda (owner, 2026-10-09: the CPU option stays public). Tests: `Repo.PipelineBackendFor`.
- **Status:** Phase 0 (kernels), Phase 1 (seam; tokenize and mwt ported) and Phase 2 (ner, pos, depparse, sentiment,
  lemma, constituency; `CharlmCache` backend-neutral) are done: every processor of both packages runs managed, and a
  managed pipeline loads no TorchSharp charlms. `docs/backends.md` has the design, the Cuda split plan, exactness, speed and
  the Phase 2 progress table. A managed pipeline needs no native libtorch (below). 0.5: the public `PipelineBackend`
  option, managed the default, platform packages discontinued, Docker sample managed, `alpine` and `linux-arm64` CI jobs.
  The `StanzaSharp.Cuda` split (owner's decisions, 2026-10-09: two PRs) is done. Step 1 (#55): all torch code is in
  `StanzaSharp.Cuda`. Processors take a net factory in an internal constructor and have `LoadManaged`; their
  TorchSharp `Load` is a C# 14 static extension in `StanzaSharp.Cuda` (`PosTaggerLoad` etc., in the processor's
  namespace), so `PosTagger.Load(...)` call sites are unchanged. `Pretrain.Native` (`IPretrainVectors`) and
  `CharlmCache` entries' `ICharlmReps` hold the tensor forms; `Embeddings`, `Pretrain.Load` and the tensor
  `TryAdd`/`TryGet` are extensions. Step 2 (1.0): the facade references no TorchSharp; `Pipeline` loads through an
  internal `BackendModels` (`Configure` threads/TF32, `Load` wrapper for the device, pretrain, charlms, each processor;
  `ManagedModels` in the facade, `TorchSharpModels` in StanzaSharp.Cuda), made by `PipelineBackend.Models`.
- **No libtorch on the managed backend** (0.5 requirement, done): TorchSharp's `torch` static constructor loads libtorch, so
  the managed path calls no `torch.*` member at all (not even `torch.CPU`) and makes no TorchSharp object. Since 1.0
  the main package has no TorchSharp at all, and ManagedCheck fails if a TorchSharp assembly loads. Managed
  `Pipeline.Load` sets only `ManagedThreads.Count` (`ManagedModels.Configure`); libtorch's threads, TF32 and
  `Weights.On(Device)` are in `TorchSharpModels`. `Pretrain.LoadManaged` keeps the
  vectors as a `float[]` (`Load`: a tensor; `CpuVectors()` reads either; `Embeddings` throws on a managed one), used when
  every pretrain reader is managed. After each managed model `Pipeline` runs `GC.Collect()` (weights are read into an
  array, then packed: load peak 1,035 → ~850 MB, +0.05 s). Proof: `tests/StanzaSharp.ManagedCheck` (no TorchSharp-cpu /
  libtorch-cpu-*, `StanzaSharpNoLibTorch`), run by `ManagedCheckTests` with an empty `NUGET_PACKAGES` (TorchSharp's
  fallback otherwise copies libtorch from the NuGet cache into `cpu/` next to the app), both packages byte-identical and
  no native torch module loaded; `verify-package.ps1 -Managed` compiles the same Program.cs against the packed package
  alone (golden, cross-os, linux-arm64), both on the default backend (no option set). A new managed code path must keep
  this: no `torch.*`, `Scalars`, `Weights.Device`, tensors.
- **Seam** (all internal): per processor network, arrays in and out: `ITokenizerNet` (`TokenizerNet` / `ManagedTokenizerNet`),
  `IMwtNet` (`MwtNet` / `ManagedMwtNet`). Batching, windows, argmax and decoding stay in the processor, shared, so both
  backends see the same batches. `Nn.Backend` { TorchSharp, Managed } is what tests and the benchmark pass to the
  processors' TorchSharp-side `Load(basePath, device, backend)`; the pipeline goes through `BackendModels`. With
  Managed, `Load` also sets `ManagedThreads.Count` = `Threads ?? min(Count, ManagedThreads.Default)`. Benchmark:
  `--backend managed|torch --processors ...` (managed by default; `--device`/`--no-tf32` select torch).
  - Tests that only make sense on TorchSharp (libtorch threads, the Scalar GC race, TF32) set
    `Backend = CudaBackend.Cpu`; `ManagedThreadsOptionTests` (in `ManagedKernelsCollection`) covers the managed `Threads`.
  - A new port: an `I…Net` with array I/O; the TorchSharp side wraps today's module code unchanged (default path stays
    byte-identical); the managed side composes `Nn.Managed` blocks. Its golden tests become theories over `bool managed`
    (`Repo.Backend(managed)`), and `ManagedBackendTests` gets a net-vs-net test per kernel path.
  - Discrete outputs on the managed backend must be byte-identical with the golden data; never loosen a test for it.
  - `INerNet` (`NerNet` / `ManagedNerNet`): ids, Viterbi, `fix_singleton_tags` and entities stay in `NerTagger`.
    `NerTagger.Load` (TorchSharp) / `LoadManaged` (managed charlms, or none for `_nocharlm`). The managed net builds each
    token's input row at its packed position, so input_transform, the biLSTM and the tag layer never see padding.
  - `IPosNet` (`PosNet` / `ManagedPosNet`) → `PosOutput` (UPOS scores; UPOS, XPOS, feats ids; the argmaxes are in the net
    since XPOS/feats read the argmax UPOS). simplify_punct, vocab lookups, batching, cache keys and decoding stay in
    `PosTagger`; `Load` / `LoadManaged`. The managed XPOS/feats biaffines are contracted with each of the 21 UPOS
    embeddings at load (one linear layer per scorer, all UPOS stacked). `upos_clf` sums in double: in float its
    400-term sums of logits up to |150| drifted 1.07e-4 from Stanza (Scalar path); in double ≤ 6.9e-5, at no
    measurable cost. Depparse sums its arcs in double too; its float label scores met 1e-4 as they are.
    UPOS logits (owner's decision, 2026-10-13, German): like sentiment's, a logit passes within 1e-4 of Stanza's float32
    **or** float64 value (`PosTests.AssertUposClose`, also FastPackageTests). German's logits reach |140| and Stanza's own
    float32 is 1.01e-4 from float64; managed was 1.37e-4 from float32 (6.3e-5 from the nearer value), TorchSharp alone
    1.22e-4. `upos_hid` in double changed nothing. The TorchSharp `PosNet` sums `upos_clf` in double too (round 26:
    on Intel AVX-512 runners libtorch's float sums put German `default_fast` UPOS 1.2e-4 from Stanza).
  - `IDepparseNet` (`DepparseNet` / `ManagedDepparseNet`) → `DepparseScores` (arc log-probs over the padded width, label
    argmax per pair, label scores for tests). `DependencyParser` keeps simplify_punct, vocab ids, ROOT, batching, MST and
    the deprel strings. The managed net scores no padding rows; padding columns all get in2 = ReLU(W2's bias) (LSTM
    output 0 there), so the log-softmax sees TorchSharp's exact width, and a row scores them once (round 29). Arcs (pair
    sums through `Act.Dot`, whose SIMD lanes are its four double accumulators; terms; log-softmax) in double;
    labels per dependent through `Gemm.Kernel` on a repacked T row, real pairs only, T in ≤ 32 MB chunks.
  - `ISentimentNet` (`SentimentNet` / `ManagedSentimentNet`) → [batch, classes] logits. `SentimentClassifier` keeps map_word,
    the delta vocab, `label_sentences` sorting, 5000-token batches, the padded width (≥ widest filter), cache keys (no MWT)
    and the argmax. Managed: input rows time-major, `ManagedLstm.ForwardPacked` with every batch size = n (the unpacked
    LSTM: padding runs through it; `ForwardUnpacked(lengths:)` projects the shared padding input once per direction, with
    the single-row kernel's row kept apart, round 28), full-width convolutions as one GEMM whose A rows overlap (lda = 600,
    K = Height·600), the (5,5)/(1,5) filter with its 8 channels in SIMD lanes (multiply then add, the scalar order: same
    bits on every path), FC layers through `Gemm`. Labels byte-identical. Logits: the sentiment
    golden also holds Stanza's float64 logits (owner's decision, 2026-10-08), and a logit passes within the usual
    tolerance of the float32 **or** the float64 value, on both backends. The classifier is ill-conditioned: on
    validation.txt sentence 41 Stanza's float32 logits are 1.49e-4 from float64 and the managed ones 1.2e-5 (1.38e-4
    from Stanza's). Double sums (FC, convolutions, LSTM input) only moved managed toward float64.
  - `ISeq2SeqNet` (`Seq2SeqNet` / `ManagedSeq2SeqNet`, in Nn since German: MWT uses it too, without the POS step and
    the edit head): `Encode` → an `ISeq2SeqDecoder` per batch (edit logits, `Step(previous)` →
    [batch, columns] log-probs). `Lemmatizer` keeps the dictionary, DeltaVocab, batches, the greedy loop (first maximum,
    like torch), edits and fallbacks; `Load(..., backend)`. Managed: packed encoder (`ManagedLstm.ForwardPacked` with
    `finalC`), per step four GEMMs (cell `[x | h]` K = 250, linear_in, linear_out, dec2vocab + copy gate column), attention
    and copy mix per row in double. Lemmas byte-identical; smallest top-2 margin 5.6e-3. Its log-probs are 4.8e-4 from
    TorchSharp, which is itself 2.7e-4 from Stanza in float64 (Stanza f32: 2.9e-4): `LemmaNet_ManagedMatchesTorchSharp`
    reports them without a tolerance (owner's decision, 2026-10-08: lemma is tested on exact lemmas, decoding and edits, as
    TorchSharp always was; no backend reaches 1e-4 on these log-probs, Stanza's float32 included).
    On UD EWT (`tools/lemma_divergence.py` + benchmark `lemma-divergence`; corpus not in the repo) no lemma differs among
    Stanza f32/f64, TorchSharp, managed and managed with double gates, also with every word through the seq2seq (1.25M
    steps, min margin 9.1e-4). It also found the tokenizer's `<UNK>` limit bug (1000 vs the config's `max_seqlen` 200), fixed in #41.
  - `IConstituencyNet` (`ConstituencyNet` / `ManagedConstituencyNet`): the net makes opaque handles (word vectors,
    constituent vectors, stack LSTM states; TorchSharp's are `IDisposable` and owned by the `ParserState`), and offers
    `EncodeWords`, `Word`, `Score`, `Open`, `Compose`, `PushTransitions`, `PushConstituents`. `ConstituencyParser` keeps the
    transition system, legality, unary_limit, the vocab lookups, the schedule (longest first, 50 in flight) and the trees.
    Managed: the word encoder is `ManagedLstm.ForwardPadded`; each stack LSTM layer is one GEMM over `[x | h]` (like the
    lemma's cell), whose tasks also run their panels' LSTM cells and copy the next layer's rows (`Gemm.Run`'s
    `epilogue`, round 30); scores, compositions and pushes run `Gemm.Run(..., rowInvariant: true)` (no 1-row kernel, whose split sum
    differs from the 6-row one), so a state's arithmetic is bitwise the same in any batch (tested). All float: scores 3.4e-5
    from Python (TorchSharp 3.1e-5; every path ≤ 4.6e-5, tolerance 1e-3 as before); trees byte-identical. Near-ties over
    845 golden sentences (22,569 steps): smallest decision margin 3.6e-4 (both backends), backends 7.3e-5 apart;
    `ConstituencyTests.NearTies_StayClearOfBackendDrift` fails if decisions differ or the margin drops under 3× that drift.
    On UD EWT (`tools/constituency_divergence.py` + benchmark `constituency-divergence`; 254,589 words, 684,407 decisions)
    no tree differs among Stanza f32, TorchSharp and managed (nor Stanza f64 on a 38k-word subset); smallest margin
    4.7e-5, but backend margins differ by 4.4e-6 median / 5.3e-5 at the 99.99th percentile (smallest margin/difference
    ratio 15). The gap is the float32 charlm (managed's summation order), amplified by the word encoder; per-step layers
    in double (`ManagedConstituencyNet.Double`, study only) change nothing, so no precision change. Run torch-based
    shards with `--threads 1`: parallel 8-thread libtorch processes crawl. Since then (2026-10-10) every managed LSTM
    step sums h·W_hhᵀ in blocks of 128 (`Gemm.KernelBlocked`): charlm error vs float64 halved (TorchSharp's level), golden
    max margin gap 7.3e-5 → 4.0e-5, median unchanged, +4% charlm time at 1 thread, +3% at 8 (docs/backends.md).
  - A pipeline runs every processor on one backend (`BackendModels`); it loads the charlms only if a `_charlm`
    processor reads them (`ManagedCharLanguageModel`: +31 MB of input tables). Managed processors read the shared `Pretrain` through `CpuVectors()` (its array, or a CPU tensor's own memory).
  - `CharlmCache` is backend-neutral: an entry holds its producer's form (tensors, or `[words, dim]` float arrays via the
    float `TryAdd`). `TryGet` gives tensors (a float entry is converted once, on the reader's device, and kept);
    `TryGetArrays` gives arrays (a tensor entry is copied per read). Same-backend pipelines never convert, so the default
    path is unchanged. With pos managed, the tagger adds arrays (only when `HasRoom`), the managed readers (constituency,
    sentiment, NER) read them as they are; a TorchSharp reader in a mixed pipeline would convert them to tensors once.
- **Code:** `src/StanzaSharp.Nn/Managed/` (`Gemm.cs`: `KernelPath`, `PackedMatrix`, `Gemm`, `Act`;
  `ManagedThreads.cs`; `PackedLstm.cs`; `ManagedModels.cs`: charlm, highway biLSTM, `ManagedLstm` = `nn.LSTM` over a
  padded batch like `Rnn.RunPacked` (`ForwardPadded`) or over packed rows (`ForwardPacked`), optional `h_init`/`c_init`,
  hidden padded to a multiple of 4 with zero units; `ManagedCharacterModel`, both variants: NER's bidirectional final
  states and the tagger/parser's unidirectional attention pooling, tested against `CharacterModel`). Tests:
  `ManagedBackendTests` (every path the machine runs natively). Harness: `StanzaSharp.Benchmark managed-spike
  check|speed|concurrent|gemm`. Numbers and method: `docs/managed-backend-spike.md`, `docs/backends.md`.
- **Rules for the managed code:**
  - `Gemm.Path` is detected at startup (Vector512 = AVX-512 where .NET accelerates `Vector512`, Vector256 = AVX2/FMA,
    Vector128 = NEON on Arm64, Scalar); every path reads the same packed weights. A new kernel or element-wise step needs
    a version per path and a test per path.
  - Vector512 (docs/performance/round-14-avx-512-kernels.md and round-19-vector512-intel-two-panels.md): one register per panel row, two panels per kernel call in
    `Gemm.Run` (`KernelPair512`) and in `PackedLstm.Step` (`KernelBlockedPair512`; a single panel only at a task's odd
    end); the LSTM cell's one exp over 16 gates. It must stay Vector256's bits (same FMAs in the same order,
    activations lane-identical; `Vector512_GivesVector256sBits`), so a CI runner's CPU draw can't change a result.
    Measured vs Vector256 on runners (round 19): +65–82% EPYC 9V45 (Zen 5), +8–16% EPYC 9V74 (Zen 4), +37–54% Xeon
    8573C (Emerald Rapids), +4–6% Xeon 8370C (Ice Lake; slower than Vector256 before the LSTM pairs). Faster on every
    CPU drawn, so no vendor rule. `DOTNET_PreferredVectorBitWidth=256` selects Vector256. No local AVX-512 (Ryzen
    5800X): test it with the Benchmark workflow's `avx512` jobs (`gh workflow run benchmark.yml --ref BRANCH -f
    avx512_only=true`; six ubuntu-26.04 and three windows-2025 jobs, those without `avx512f` stop with a notice; each
    also prints `managed-spike gemm` rates per path). Intel came up in 8 of 69 jobs. ci.yml's golden-shard and
    cross-os print the CPU and run Vector512 when they get AVX-512.
  - `ManagedThreads`: one process-wide pool of `Count` − 1 workers; each caller works its own region and idle workers
    help. Not per-caller teams: those oversubscribe (measured: 8 callers fall to TorchSharp's throughput). A region's
    split must not depend on how many threads help, so results stay bitwise identical under concurrency.
  - Contention from other processes (round 32): a join wait over 0.5 ms and over 8× the caller's mean item is a stall
    (a preempted worker holding an item). Per 100 ms window, with k threads per region, stalls over 1/k of the window
    put one more worker to sleep (`Pool._shed`, keeping max(1, Count/4) awake), under 1/(4k) one comes back. Concurrent
    callers (busy threads − awake workers > 1) reset `_shed` to 0 and void the window: shedding there starved long calls
    (owner's tuning request). `Work` lowers `_busy` before raising `_sleepers` and vice versa so that count never
    over-counts. Only who runs items changes, never the split. Internal, no option (a public one would be the owner's
    decision). `ParallelFor_ShedsWorkersWhileJoinsStallAndBringsThemBack` asserts that a shed worker comes back, not
    that all do (2026-10-17): with fewer free cores than its 4 threads (taskset 2 cores, the 3-core macOS runner at
    times) the pool's own threads stall for real and it settles at 1–2 shed by design; requiring 0 failed about one
    run in three there. Don't tighten it back.
  - `PackedLstm.Step` runs 4 tasks per thread (`TasksPerThread`; round 16): a late thread then holds a quarter of its
    share. Tried and dropped: fewer tasks for small steps (a work threshold, or timing each size class: biased by
    asleep workers, 25–50% slower on the Arm runners). A panel's arithmetic doesn't depend on its task.
  - Startup (round 17): code that runs per time step or per element (the pool's `For`/`Run`/`Drain`, `PackedLstm.Step`,
    `Act.LstmCell`/`AddGatedTanh`, the GEMM kernels) is `[MethodImpl(AggressiveOptimization)]`, so a fresh process's
    first calls don't run it at JIT tier 0; `Act.Sigmoid` uses `VectorExp` (the runtime's `VectorN.Exp`, copied op for
    op, inlined), since the runtime's own stays at tier 0 inside optimized callers. A new hot per-step method gets the
    attribute too. The managed `Pipeline.Load` loads two models at a time (`LoadLanes`; all at once: +95-152 MB load
    peak). Measure startup with `StanzaSharp.Benchmark startup` / `tools/startup.py` / benchmark.yml's `startup_only`.
  - Per-call buffers come from `ArrayPool<T>.Shared` and go back in `finally`; no buffer is shared between calls, and
    no mutable state lives on a model. Rented arrays hold stale data: write before reading. Measured (round 9): exact-size
    arrays instead raise the peak (+150 MB in depparse); the pool's power-of-two rounding costs no working set. Processors
    share the pool's buckets, so resizing one buffer can raise another processor's peak (sentiment's, round 9): measure
    the all-eight `--memory` peak too.
  - A GEMM column depends only on its 16-column panel: a GEMM split by whole panels (`new PackedMatrix(source, firstPanel,
    panels)`, or rows packed separately at a panel boundary) gives the same bits. Rows too, through `Gemm.RunRows`: every
    kernel but the single-row one gives a row the same arithmetic, and `Gemm.Run` uses that one only for row m − 1 when
    m % 6 == 1, so `RunRows` runs its chunk row-invariant and that row alone (round 10). Splitting a batch's sequences
    (fewer rows per LSTM or charlm step) moved the last bits in round 9; time steps and directions can be split.
  - Memory: a buffer per padded row multiplies (sentiment's 5000-token batch is 35,723 rows on the benchmark text). Build
    a GEMM's input and output a chunk of rows at a time (`ManagedLstm.ForwardUnpacked`, sentiment's convolutions) and
    keep only what a later stage reads whole; round 10 cut sentiment's +1.1 GB to +0.18 GB that way.
  - Check the `CancellationToken` per time step and per GEMM block.
  - `Gemm.Path` and `ManagedThreads.Count` are process-wide. Only `Pipeline.Load` and `LanguageIdentifier.Load` (managed backend: the count) and tests
    in `ManagedKernelsCollection` (parallelization off, so other classes never see a switched path) may set them.

## Native AOT and trimming

- Supported on the managed backend (owner's decision, 2026-10-16; docs/aot.md): Directory.Build.props sets
  `IsAotCompatible=true` on every `src/*` project but `StanzaSharp.Cuda` (TorchSharp isn't AOT-safe) and
  `StanzaSharp.Tool` (a .NET tool, reflection JSON). The trim/AOT analyzers then run on every build, and build-test's
  warnings as errors turns any IL2xxx/IL3xxx into a failure.
- New code in those projects must stay AOT-safe: no reflection-based `JsonSerializer` (use `JsonNode`/`Utf8JsonReader`,
  or a source-generated `JsonSerializerContext`), no `Type.GetType`/`Activator`/
  `MakeGenericType`/`Assembly.Load` or reflection over members, no `Reflection.Emit` or `RegexOptions.Compiled`
  (`[GeneratedRegex]`), no `[DynamicDependency]`/suppressions to silence a warning. P/Invoke stays blittable.
- AOT fixes the SIMD width at compile time: `Vector256.IsHardwareAccelerated` is false on the default x64 target, so
  `Gemm` runs Vector128 (~2x slower, same bits). Apps set `IlcInstructionSet=x86-64-v3` (docs, README, PACKAGE.md);
  v4 gives Vector512. `[MethodImpl(AggressiveOptimization)]` is a no-op there (no tiers). Arm64: NEON is baseline.
- CI: `tools/verify-aot.sh MODEL_DIR RID` publishes tests/StanzaSharp.ManagedCheck with `PublishAot`
  (`x86-64-v3`) and with `PublishTrimmed` (self-contained), fails on any IL2xxx/IL3xxx warning or torch file, and runs
  both (ManagedCheck compares both English packages with the golden files byte for byte). Golden shard 1 (linux-x64;
  installs clang/zlib1g-dev if missing) on full runs and on light runs with torch=full; cross-os windows-2025 shard 3
  on full runs (the runner's VS 2022 has the C++ linker). Locally AOT on Windows needs VS 2022's C++ workload; this
  machine's VS 2017 MSVC can't link .NET 10's runtime (`__tls_guard`, docs/aot.md).

## Code style

StyleBro is on (user's decision, 2026-10-11): `StyleBro.Analyzers` 0.7.0-alpha.1 in Directory.Build.props (PrivateAssets,
no package dependency; `StyleBroPreset` 1.0, so a newer release's rules stay off until it is raised),
`EnforceCodeStyleInBuild`, and `GenerateDocumentationFile` in every project (IDE0005 needs it; outside src/ CS1591 is
off). Fix findings with `stylebro-migrate format StanzaSharp.slnx` (`dotnet tool install --global StyleBro.Migrate
--version 0.7.0-alpha.1`), not plain `dotnet format`, which also applies every other analyzer's fixes. CI's
`build-test` builds with warnings as errors and runs `format --verify-no-changes`.
- `.editorconfig`: `stylebro-migrate init`'s block (between its markers; `init --write` rewrites it; since 0.6 it also
  writes `end_of_line = lf` and `dotnet_sort_system_directives_first`, which the preset no longer sets), then our own
  section, where each turned-off rule says why: BRO1001 (members stay in reading order), BRO1114/BRO1142 (grouped
  declarations like `int rows, cols`), BRO1108 (lists wrap at 120 columns), BRO1601 (no `<inheritdoc/>` on internal
  implementations), BRO1604 (property summaries are noun phrases). It also keeps one-line statements and turns on
  IDE0005/IDE0051/IDE0052 so unused usings and private members fail the build.
- BRO1520 is on (owner's decision, 2026-10-13): where an expression wraps, the operator (`+`, `&&`, `??`, `?`/`:`)
  starts the continuation line (`dotnet_style_operator_placement_when_wrapping = beginning_of_line`, in init's block,
  where init had turned the rule off because the code mixed both forms). The rule skips an operator next to a comment.
- `init --write` (0.7.0-alpha.1) also puts `<StyleBroPreset>` into `samples/docker/Directory.Build.props`, after its
  `<Project />` (invalid XML; stylebro#177): revert that file after each run.
- The reformat commits are in `.git-blame-ignore-revs` (`git config blame.ignoreRevsFile .git-blame-ignore-revs`).
- The SDK formatter puts the `)` of a tuple ending in a block lambda on its own line (`}` newline `);`, in
  TorchCheckpoint); that is IDE0055, not StyleBro.

## Design decisions

- Model the document as Document → Sentence → Token → Word from the start. Tokens come from the
  tokenizer, Words come out of MWT expansion, and POS and the parser operate on Words. Keep this
  split even before MWT is implemented.
- Batch inference per sentence from the start.
- Run inference under no-grad / eval mode, and dispose tensors deterministically (TorchSharp
  `DisposeScope`). A temporary tensor, at load time too, belongs to a scope (or a `using` variable): left to its
  finalizer, it can be freed by a GC during the native call that reads it.
- Never pass TorchSharp a temporary `Scalar` (`Nn.Scalars` explains the finalizer race). No tensor `+` or
  `add(Tensor)` (hidden `alpha = 1`): write `a.add(b, Scalars.One)`. No numbers where a Scalar is expected
  (`t * 2`, `1 + t`, `masked_fill(m, 0)`, `eq(0)`, `pow(2)`, `arange(n)`): use `Scalars.*`, a `static readonly Scalar`,
  or a `using var s = value.ToScalar()` local. `F.softplus` goes through `Scalars.Softplus`. Tensor `-` tensor is
  fine (no alpha). Check a new TorchSharp function's source for internal conversions before using it.
  Reported upstream as dotnet/TorchSharp#1583 (open PRs #1434/#1496 would fix it); when upgrading TorchSharp, check
  whether it is fixed and, if so, drop the workaround, including `Scalars.Softplus`'s private `softplus1` call.

## Build order

1. Generate golden data with Python, and inspect every checkpoint's config (start with constituency;
   its config decides which optional layers the parser needs).
2. `Core`: document model, safetensors reader, checkpoint JSON loading.
3. Tokenizer (character-level labeling + post-processing into tokens and sentences).
4. MWT: dictionary lookup first, then the character split classifier.
5. Charlm + pretrain in `Nn`; validate against saved hidden states.
6. POS tagger.
7. Constituency parser: in-order transition system, parser state, Tree type, plus the neural scoring.

## Licensing

StanzaSharp is Apache 2.0 (`LICENSE`), like Stanza, whose code it ports. `NOTICE` credits Stanza;
keep it in any redistribution. Model licenses vary with the training data, so do not commit or
redistribute model files. `models/` is gitignored.
