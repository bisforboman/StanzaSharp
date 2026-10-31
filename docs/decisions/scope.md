# Scope: languages, processors, packages, models

What StanzaSharp ports and supports, with the owner's decisions behind each part (dated in place). Moved from CLAUDE.md on 2026-10-31; CLAUDE.md keeps only the rules.

- **Language:** English and German (`PipelineOptions.Language = "de"`, 2026-10-13; docs/german-plan.md, "Checkpoint
  findings (German)"). German runs every processor of `default`, `default_fast`, `combined`, `gsd` and `hdt` byte for
  byte on both backends.
- **More languages** (owner's decisions, 2026-10-30: a European batch first, tables generated from resources.json, small
  golden + nightly): Spanish (`es`, `default` only: its `default_fast` parser `depparse/combined_nocharlm.pt`, uploaded
  2026-10-27, is truncated, so neither Stanza nor StanzaSharp loads it), French, Italian, Portuguese and Dutch (`default`
  and `default_fast`), each Stanza's own package with no new model code. What they needed: Portuguese's `oscar2023`
  charlms are saved by the charlm trainer (`{model, epoch, global_step}`; `Nn.CharlmCheckpoint.Model` unwraps it as
  char_model.py's load does), and its `cintil_charlm` parser has a label `A'`, whose transition repr Python writes in
  double quotes (`OpenConstituent(("A'",))`, `ConstituencyParser.OpenRepr`). Spanish and French packages list coref as
  optional; `Optional` keeps only models `Alternatives` lists, so it is not offered there (untested). Golden:
  `tests/golden/languages/<lang>/` (a hand-written `text.txt` per language and `<package>.conllu`, `make_golden.py
  --languages-only`, `LANGUAGES`), `LanguageGoldenTests` (both backends); the models only in nightly.yml's `languages` job
  (no cache); ci.yml's shards leave the class out (`SHARD_NIGHTLY`) and the other no-skip checks exempt it. A new
  language: `python tools/stanza_resources.py add LANG` (its entry and the `Revisions` commit, the same day), a
  `SupportedPackages`/`DefaultOnly` entry, its `text.txt` and golden (`--languages-only --languages LANG`), `docs`;
  run `stanzasharp compare` against Python Stanza first (it finds what breaks). `benchmark` has texts for en and de only.
  The sweep (2026-10-30; owner's decisions: the zero-code languages first, then a PR per fix group; nightly all of them,
  a batch at a time; the sweep's short texts as golden): every other Stanza language's `default` through `compare`. 79
  matched byte for byte and ship with `default` only (`Pipeline.DefaultOnly`; their text 2–3 sentences, the historical
  ones public-domain passages). The 15 `_nopretrain` taggers/parsers followed (bor, cop, egy, frm, naq, orv, pcm, pro,
  qaf, qpm, qtd, say, sdh, wuu, yrl; owner's choice 2026-10-30): config `pretrain` false, so no trans_pretrained columns
  (Stanza's `if self.args['pretrain']`); `PosTagger`/`DependencyParser.UsesPretrain`, the nets take a null `Pretrain`, the
  pipeline passes `PretrainIfAny` (resources.json lists no pretrain for them); identical in `compare`, 101 languages. Not
  yet: tokenizers with extra features (ja, th, zh-hans,
  zh-hant: `feat_dim` over the 5 we build), a loader error on ko and lzh (`The node must be of type 'JsonArray'`), or
  (MuRIL transformer), and el (its lemmatizer's contextual classifiers load word vectors from a path on Stanford's
  machine: Python Stanza fails too). `LanguageGoldenTests` builds its rows from `SupportedPackages` (every language but en
  and de), only those whose models are present, and `STANZASHARP_LANGUAGES` narrows them; `tools/test-languages.sh` is
  the nightly loop (download 8 languages' packages, test, fail on a skip, delete; about 25 GB a night); `make_golden.py
  --languages-only` takes every folder of tests/golden/languages (`LANGUAGES` names packages beyond `default`);
  `python tools/stanza_resources.py docs` regenerates docs/languages.md. A `languages_only` dispatch of nightly.yml runs
  just that job.
- **Processors:** all eight of Stanza's English default, in Stanza's order (`PIPELINE_NAMES`): tokenize,
  mwt, pos, lemma, constituency, depparse, sentiment, ner; plus the optional coref (after constituency; see "Coref"). This is `Pipeline.AllProcessors`, the default
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
  the checkpoint, `ConstituencyParser` builds its `WithLora` net like the classifier), ner `germeval2014_german-nlp-electra`
  (since #215; it was default's `germeval2014`); transformer
  `german-nlp-group/electra-base-german-uncased`. Golden rows `pos+depparse=german-nlp-electra`,
  `constituency=spmrl_german-nlp-electra`, `sentiment=sb10k_german-nlp-electra`, `default_accurate` and
  `default_accurate+sentiment=sb10k_dbmdz-bert-german-cased` (two transformers in one pipeline) in `tests/golden/de/models/`
  (`de/models/transformer_long.txt`: one sentence of 558 pieces, two windows).
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
  word looked up as is, spaces as U+00A0; mimic pretrain and charlms); German ner `conll03` (the old format: no delta, a
  plain `TagVocab`, `tag_clf` heads; fasttextwiki, 300-d).
  Byte for byte on both backends (`ModelChoiceTests` against `tests/golden/models/<name>/`, `make_golden.py
  --models-only`, `MODEL_CHOICES`; i2b2 on the mimic package also on a hand-written `models/clinical.txt`). A new
  alternative needs `Alternatives` (its dependencies and MD5 come from `stanza_resources.json`; a transformer model also
  `Transformers`), a `MODEL_CHOICES` row and a `ModelChoiceTests` row (`EveryAlternative_HasItsFilesAndDependencies`,
  `StanzaResources_MatchModelDownloader` check the tables). The domain NER models (owner's decision, 2026-10-16, part 2): `conll03_charlm` (conll17 since Stanford's re-upload, #215),
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
    per sentence and head without padding (round 49: four query rows at a time through register-blocked kernels over the
    head's keys and values widened once, the row loop's bits; `BlockedAttention_GivesTheRowLoopsBits`), attention/LayerNorm/GELU in double (round 45: lane-wise vectors, the same bits
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
    "RoBERTa"): ner `conll03_electra-large` read `roberta-base` until Stanford's re-upload (#215, 2026-10-24: retrained on
    ELECTRA-large, new format, conll17, input 100 + 1024 + 2048); no model reads roberta-base since, and the owner kept PR (a)
    and the encoder tests as tested infrastructure (golden shard 4 still downloads it, `download-transformer`). PR (a): `Nn.ByteLevelBpe` (GPT-2's byte-level BPE with Stanza's
    `add_prefix_space=True`: added tokens first, `<mask>` lstrip; a space before every other piece not starting with U+0020;
    the split pattern scanned by code point; merges by rank) behind `ITransformerTokenizer` with `WordPiece`, picked when
    tokenizer.json's model is BPE (`type` may be absent); `TransformerConfig` reads roberta (`Prefix` `roberta.`,
    `PositionOffset` 2 = pad + 1, `MaxPositions` 512; the nets slice the position table at load, one token-type row);
    `ModelDownloader.Transformers["roberta-base"]`; goldens `transformer/tokenizer_roberta-base.json` (1,093 cases,
    `ROBERTA_TOKENIZER_CASES`) and `hidden_roberta-base.*` (`--transformer-only`); `TransformerTokenizerTests`,
    `TransformerEncoderTests.Roberta_*`, `TransformerKernelTests.Roberta_*`. PR (b): every transformer the models read loaded
    once by name (`Pipeline._transformers`; the shared pass per transformer, its readers found by tokenizer). Its
    `ListedPretrains`/`PRETRAIN_FIXES` workaround (resources.json listed glove/fasttextwiki for models trained on conll17,
    stanfordnlp/stanza#1698) went with #215, which fixed resources.json. Golden row `ner=conll03_electra-large` (+ validation,
    validation_social), `TransformerModelTests`; two transformers in one pipeline are German's
    `default_accurate+sentiment=sb10k_dbmdz-bert-german-cased` row.
  - German's on the same transformer (owner's choice 2026-10-19): pos and depparse `{gsd,hdt}_german-nlp-electra` and ner
    `germeval2014_german-nlp-electra` (conll17 since #215, word 100): `combined_german-nlp-electra`'s / germeval2014_charlm's configs on
    their twins' vocabularies and heads (gsd/hdt legacy tag heads; depparse's 4-layer `bert_layer_mix`); no new code. Golden
    rows `pos+depparse=<tb>_german-nlp-electra` on their package (corpus.txt) and `ner=germeval2014_german-nlp-electra`
    (+ validation, transformer_long) in `tests/golden/de/models/`; golden shard 4 downloads them. On a third transformer,
    `dbmdz/bert-base-german-cased` (BERT: `TransformerConfig` reads `model_type` bert, `Prefix` `bert.`; a cased vocab.txt
    WordPiece; `de/transformer/tokenizer_dbmdz_bert-base-german-cased.json`, `DBMDZ_TOKENIZER_CASES`, `DbmdzFact`): sentiment
    `sb10k_dbmdz-bert-german-cased` (LoRA; no charlms although listed: `Unread`); Stanford removed its ner
    `germeval2014_dbmdz-bert-german-cased` on 2026-10-20 (#215). Golden rows `sentiment=sb10k_dbmdz-bert-german-cased` and
    `default_accurate+sentiment=sb10k_dbmdz-bert-german-cased`.
  - Coref's XLM-R (docs/coref-plan.md, PR a; owner's choice 2026-10-19; PRs b and c followed on 2026-10-20): `Nn.Unigram`
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
  - Coref PR (c), the processor (owner's decisions 2026-10-20: Q1 main package, Q2 opt-in processor, Q6 the default
    model only, Q8 float32; Q7 by the existing rule, a per-run download in golden shard 4): `Processor.Coref`
    (`StanzaSharp.Coref`: `CorefResolver` + `ICorefNet`, `ManagedCorefNet` / TorchSharp `CorefNet`), Stanza's
    `udcoref_xlm-roberta-lora` for en and de. Not in `AllProcessors` (eight, Stanza's default list); `Pipeline.StanzaOrder` has
    all nine, coref after constituency. `Pipeline.Optional` = resources.json's packages' `optional` (default, default_fast,
    default_accurate): `SelectModels` adds its model only when Processors names coref; `Models` may choose it (`Alternatives`)
    on any package. `Requires` tokenize; UPOS only breaks representative ties (PROPN). `CorefResolver.Run` = CorefModel.run:
    each word through the tokenizer (`Encode` of all words: Stanza's `tokenize(word)` per word gives the same pieces), windows
    of 510 backed up to a sentence end, the LoRA net (`WithLora`, `LoraWeights` reads peft's `base_model.model.` keys and
    skips a saved pooler) on each, the special ids dropped; WordEncoder; rough scores masked for j ≥ i and `Nn.TopK` (ATen's
    topk_impl_loop: libstdc++ 13's nth_element when k·64 > n, else partial_sort, comparator NaN-first then >; ported from
    stl_algo.h / stl_heap.h; `TopKTests` against torch on Linux, `coref/topk.json`, `make_golden.py --topk-only`, no models)
    on both backends, so TorchSharp on Windows doesn't get MSVC's order; distance buckets; pair scores; start = start_map over
    the k pair scores in topk's order (masked ones too); `_clusterize` (dummy wins its ties, start wins its tie with ε,
    components sorted, lists ordered like Python's); SpanPredictor with every head padded to the longest of their sentences
    (padding rows ffnn(0), then both convolutions); zeros predictor; then coref_processor.process (zeros before the word,
    numbered per sentence; spans sorted, paired with the cluster's words by position). Managed: the anaphoricity and span
    layers split by input part (per word, per bucket/distance, per pair). Measured: every golden text's CoNLL-U byte for byte on
    both backends (en 13 texts, 1,674 chains; de 13, 272), clusters, spans and zero scores exact on the intermediates;
    the top-k order differs from torch's in 2 of 549 rows of validation_dialogue (same candidates; near-ties between rough
    scores closer than the float drift rearrange nth_element), on both backends, without changing a decision. Bulk: coref
    runs once over the combined document, so the returned documents' `Coref` stays empty and their words keep the
    attachments (Stanza's bulk_process). Empty document: `Coref` = [] (Stanza crashes). A word without subwords throws
    (Stanza too). Tests `TransformerCorefTests` (golden shard 4; `Run_MatchesStanzasSteps`, `Pipeline_ReproducesStanzasCoref`
    en + de); golden `coref/intermediates.*` (windows, words, top-k in topk's order, scores, clusters; corpus and
    validation_dialogue), `coref/zeros/` (zero mentions forced: `COREF_ZERO_SHIFT` added to the zero scores, #208;
    `Process_WithZeroMentions_MatchesStanza`, a read-back exact on them; `CopiedCorefModel_UnderAnotherName_GivesStanzasChains`:
    a coref `ModelPath` keeps `CorefModel`'s name, so XLM-R, on a package without optional coref too, #209; a zero differs on read-back only inside a chain's
    overlapping mentions, which the model didn't predict in about 600 zeros) and `de/coref/` (texts only). CI: `tools/ci-models.sh transformer` downloads `--processors
    tokenize,coref` (en) and, with XLM-R linked from en's folder, de. compare.py's `transformers_locally` also serves coref's
    own `load_tokenizer`.
  - CI: no cache (the model caches are near GitHub's 10 GB; owner's decision): golden shard 4 downloads the
    `default_accurate` package, the treebanks' transformer models (their packages' other files from the en-treebanks cache)
    and the transformer's files on every run (`tools/ci-models.sh transformer`: `download --package default_accurate`, ...,
    with retries); the `Transformer*` classes (`SHARD_FOUR`, `.Transformer`) run only there
    (owner's decision, 2026-10-18: in shard 1 they had made its job 82 min), and the other jobs' no-skip checks exempt
    `Transformer`. A pull request runs shard 4 only when it touches the torch=full files, `ModelDownloader.cs`,
    `tools/ci-models.sh`, `tests/golden/**/transformer/` or `Transformer*.cs` tests (`changes`' `shards` output); pushes
    to main and the nightly always run it (owner's decision, 2026-10-18).
