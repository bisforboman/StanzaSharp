# Checkpoint findings

What each ported checkpoint actually uses (Stanza 1.15.0), so only those code paths are ported. Moved from CLAUDE.md on 2026-10-31; add a new model family's findings here.

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
model but MWT is English's architecture with other vocabularies and sizes, read from the checkpoint:
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
- **ner** (`germeval2014_charlm`, also in `default_fast`; since #215, 2026-10-24: Stanford's re-upload of 2026-10-20 replaced
  the old-format `germeval2014` on fasttextwiki): English's format, conll17 [250000, 100] plus a lowercased delta (71,930
  words), one tag set of 21 BIOES tags over PER/LOC/ORG/MISC (`tag_clfs.0`/`crits.0`), input 2148 = 100 + 2048. The old
  format (no `delta` vocab or `delta_emb`, a plain `TagVocab`, heads `tag_clf.*`/`crit._transitions` that Stanza renames on
  load, `NerTagger.HeadPrefix`) is still read for the alternative `conll03` (fasttextwiki [250000, 300], input 2348) and
  English's `i2b2`. The pipeline loads every pretrain `Pipeline.Dependencies` names (`_pretrains`), and each processor gets
  its model's own.
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
