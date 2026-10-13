# German (`de`): research and plan

Research only (2026-10-13): nothing in `src/`, `tests/` or the workflows has changed yet. Stanza 1.15.0's German
models are in `models/stanza/de` (gitignored), from `stanza.download('de', model_dir=r'models\stanza')` and
again with `package='default_fast'`. The configs were dumped with `tools/stanza_convert.py inspect` and diffed
against the English checkpoints. Stanza's code is in `tools/.venv/Lib/site-packages/stanza/`.

**In short:** six of the eight German default models use the same architectures and options as English.
StanzaSharp already reads every difference between them from the checkpoint, apart from vocabulary and tag-set
sizes. Three things are new:

1. **MWT is a seq2seq model**, not the character classifier: `force_exact_pieces` is missing, so
   `MwtExpander` currently throws `NotSupportedException`. It is Stanza's `Seq2SeqModel`, the same class the
   lemmatizer uses, so most of the code exists already in `Lemma`.
2. **NER (`germeval2014`) is an old-format checkpoint**:
   - it has no delta embedding;
   - its single head is named `tag_clf`/`crit`, not `tag_clfs.N`/`crits.N`;
   - its word vectors come from a second pretrain, `fasttextwiki` (300-dimensional), not `conll17`.
3. **Language plumbing**:
   - The pipeline hardcodes `pretrain/conll17` and `*_charlm/1billion`.
   - It infers charlm use from the `_charlm` suffix, which the NER model `germeval2014` lacks.
   - The downloader points at `stanza-en`.

No transformer is needed for `default` or `default_fast`. Rough total: **L**, most of it golden data and tests,
as it was for English.

## Packages and files

From `models/stanza/resources.json` (`de.packages`, `de.default_processors`):

| processor | `default` | `default_fast` | dependencies (resources.json) | size |
|---|---|---|---|---|
| tokenize | `combined_nocharlm` | same | none | 0.66 MB |
| mwt | `combined` | same | none | 0.55 MB |
| pos | `combined_charlm` | `combined_nocharlm` (23.3 MB) | conll17 + newswiki charlms (fast: conll17) | 38.3 MB |
| lemma | `combined_nocharlm` | same | none | 5.6 MB |
| constituency | `spmrl_charlm` | none | conll17 + newswiki charlms | 128.1 MB |
| depparse | `combined_charlm` | `combined_nocharlm` (103.3 MB) | conll17 + newswiki charlms (fast: conll17) | 136.7 MB |
| ner | `germeval2014` | **same (charlm)** | **fasttextwiki** + newswiki charlms | 50.8 MB |
| sentiment | `sb10k_charlm` | same | conll17 + newswiki charlms | 77.6 MB |
| pretrain | `conll17` [250000, 100] | same | | 107.4 MB |
| pretrain | `fasttextwiki` [250000, 300] | same | (ner only) | 307.3 MB |
| charlm | `forward_charlm/newswiki`, `backward_charlm/newswiki` | same (ner and sentiment need them) | | 2 × 20.0 MB |

- `default` comes to 893 MB of `.pt` files; English `default` is 595 MB. `default_fast` comes to 715 MB and still
  needs both charlms and both pretrains, because its NER and sentiment models are the charlm ones.
- The default processors are the same eight as English's. `optional.coref` is `udcoref_xlm-roberta-lora`, a
  transformer; it isn't a default processor, so it is out of scope, as it is for English.
- German has more packages, all out of scope:
  - `default_accurate`: German-NLP-ELECTRA transformers for pos, depparse, sentiment and constituency, plus
    `combined_charlm` tokenize and lemma.
  - `combined`, `gsd`, `hdt`: tokenize to depparse only.
  - NER `conll03` and transformer variants.
- `stanza.download` left `models/stanza/de/default.zip` (809 MB) next to the extracted files. It is gitignored,
  and I didn't delete it.
- The Hugging Face repo is `stanfordnlp/stanza-de`. The MD5s are in resources.json `de.<processor>.<model>.md5`,
  for example tokenize `a52679e759e3f6c29b749cfa8ca506be`, ner `e2d33aecdfc079b18dbdbb06ec6d419a`,
  fasttextwiki `29fe2ec4cf2d8b18c18b6aa9f5d48d78`.

## Per processor

| processor | German model | same as English? | what's new |
|---|---|---|---|
| tokenize | `combined_nocharlm` | Same architecture: emb 32, biLSTM 64, `hierarchical`, `hier_invtemp` 0.5, `use_mwt`, `conv_res` null, no dictionary or lexicon. | `feat_funcs` has 5 entries (`space_before`, `capitalized`, `numeric`, `end_of_para`, `start_of_para`); English has 9 (also `labeled_field`, `phone_id`, `date_pattern`, `currency`). `Tokenizer` already reads `feat_funcs` from the config (`rnn.weight_ih_l0` [256, 37] = 32 + 5). `max_seqlen` is 400 (English 200): windows stay `max(1000, 400)`, and tokens over 400 code points become `<UNK>`. Both come from the config already. Vocab 262 units, umlauts and ß included. Stanza's tokenizer has nothing language-specific for `de` (`lang_replaces_spaces` is zh/ja/ko only; `la_ittb` is a special case). |
| mwt | `combined` | **No.** | A `Seq2SeqModel` (`models/common/seq2seq_model.py`): emb 50, vocab 25, 1-layer biLSTM 2×50, `LSTMCell` decoder 100, soft attention, `copy`, greedy (`beam_size` 1), `max_dec_len` 10. No POS input, no edit classifier. `dict` has 17 entries (im, zum, zur, am, vom, beim, ins, ans, ums, aufs, übers, plus capitalized forms), and `ensemble_dict` is on. See "MWT seq2seq" below. |
| pos | `combined_charlm` | Same; the full config diff shows only paths, `lang` and `shorthand`. | XPOS is STTS (56 entries with specials), and there are 23 feats instead of 21 (adds `ExtPos`, `Gender[psor]`, `Polite`, ...). UPOS 21 (with specials). Word vocab 3381. All of it is read from the vocab and tensor shapes. |
| lemma | `combined_nocharlm` | Same; the config diff shows only paths and `vocab_size` 276. | Only the `dicts` differ. On a 136-word German sample, 5 words went through the seq2seq; in English it is about 10%. |
| constituency | `spmrl_charlm` | Same architecture: `IN_ORDER`, LSTM stacks, `MAX`, ReLU, 2 output layers, `use_lattn` false, `pattn_num_layers` 0, `combined_dummy_embedding`, 2-layer biLSTM 512, input 2268. | `retag_method: 'upos'` (`retagging.py` `RETAG_METHOD['de']`): the parser reads **UPOS**, not XPOS. `ConstituencyParser` already reads `retag_method`. `unary_limit` is 8 (English 4), read from the checkpoint. There are 26 SPMRL constituents (`VROOT`, `PN`, `CNP`, `S`, `NP`, ...), and trees print as `(ROOT (VROOT ...))`. `lattn_d_proj: 64` and the `lora_*` keys are unused, since `use_lattn` and `use_peft` are false. |
| depparse | `combined_charlm` | Same; the config diff shows only paths. | 46 deprels (with specials), including German ones such as `expl:pv`, `obl:agent`, `nsubj:outer`. Read from the vocab. |
| ner | `germeval2014` | **Partly.** | Old-format checkpoint, 49 config keys: no `predict_tagset`, `connect_output_layers`, `bert_model` and so on. See "NER" below. |
| sentiment | `sb10k_charlm` | Same: CNN, filters (3, 4, 5, (5, 5)) × 1000, biLSTM 300, FC (400, 100), `extra_wordvec_method` SUM, labels `0,1,2`. | Only `extra_vocab` (24,235 instead of 31,014). |
| charlm | `newswiki` | Same shape: emb 100, LSTM 1024, `h_init`/`c_init`, `pad` false. | 345 characters. Both `" "` and `"\n"` (depparse's ROOT) are in the vocab. |
| pretrain | `conll17` | Same: [250000, 100]. | Plus `fasttextwiki` [250000, 300], used only by NER. |
| pos/depparse `_nocharlm` | `default_fast` | Same as English `_nocharlm`: `charmodel` unidirectional 400 with attention, `trans_char` 400→125. | Nothing new. |

### MWT seq2seq (new, M)

`stanza/models/mwt/trainer.py` picks `Seq2SeqModel` when `force_exact_pieces` is false. `MWTProcessor.process`
then works like this:

- **Every** MWT candidate goes through the seq2seq; `ensemble_dict` then overrides the result with
  `dict_expansion`. Spying on `trainer.predict` confirms it: `übers` came out of the model as `übers` (no space,
  so the original was kept), and the dictionary turned it into `über das`.
  - On the samples I ran, the tokenizer flagged only dictionary words (`im`, `zum`, `am`, `ins`, `beim`, ...), so
    the seq2seq output was always overridden.
  - Words such as `hinters`, `fürs`, `unterm`, `vorm` were not flagged, and neither were `ZUM` and `AM` in capitals.
  - Model output still decides any flagged word that isn't in the dictionary, so it has to be exact.
- **Input** (`models/mwt/data.py`):
  - `src = [<SOS>] + chars + [<EOS>]`.
  - `DeltaVocab(data, vocab)` (`expand_unk_vocab=True`) gives unknown characters ids past the vocab, so they can
    be copied. It is the same class the lemmatizer uses.
  - Batches of 50 in document order (`batch_size` 50). Each batch is sorted by `sort_all` (longest first).
- **Decoding**:
  - `predict_greedy` with `never_decode_unk=True` (the lemmatizer doesn't set it): `log_probs[:, :, UNK_ID] = -inf`
    after the copy mix.
  - Decoding stops at `max_dec_len` 10 steps.
  - Then:
    - ids < 4 are dropped (`<PAD>/<UNK>/<SOS>/<EOS>`);
    - the ids are unmapped through the DeltaVocab and passed through `prune_decoded_seqs`;
    - an output that is blank or has no space falls back to the original token (`trainer.py` `predict`).
- **Reuse:**
  - `ILemmaNet`/`ILemmaDecoder`, `ManagedLemmaNet` and the TorchSharp `LemmaNet` already implement this model:
    the packed encoder, the `LSTMCell` decoder, soft attention, the copy gate with columns for unknown ids, and
    greedy decoding.
  - The work is to move that into a shared seq2seq (in `StanzaSharp.Nn` / `StanzaSharp.Cuda`), with optional POS
    step, edit head and `never_decode_unk`, used by both `Lemma` and `Mwt`.
  - `MwtExpander` would then branch on `force_exact_pieces`. `MwtExpander.DictExpansion` is unchanged.
- **Test:** `mwt.json` for German, with words through the pipeline and **seq2seq-only** expansions of a word list
  (calling `trainer.predict` directly, as English's classifier-only list does). The list should include non-dict
  forms (`unterm`, `hinters`, `fürs`, `durchs`, `vorm`, `überm`, capitals, unknown characters).

### NER (`germeval2014`, S)

| | English `ontonotes-ww-multi_charlm` | German `germeval2014` |
|---|---|---|
| word input | conll17 (100) + `delta_emb` [48309, 100] | **fasttextwiki (300)** only; there is no `vocab.delta` and `emb_finetune` is false. The checkpoint's `vocab.word` equals fasttextwiki's vocab exactly (checked). |
| input_transform / LSTM | 2148 → 2148, biLSTM 256 | 2348 (300 + 2048) → 2348, biLSTM 256, `h_init`/`c_init` |
| heads | `tag_clfs.0/1`, `crits.0/1`, `predict_tagset` 0 | **`tag_clf.*`, `crit._transitions`** (21 tags). Stanza renames them to `tag_clfs.0`/`crits.0` on load (`ner/trainer.py` `load`). |
| tags | OntoNotes, 18 types | BIOES over `PER`, `LOC`, `ORG`, `MISC` (`scheme: 'bioes'`) |
| lowercase | true | true (`lowercase: True`, `char_lowercase: False`) |

The code changes:

- In `NerTagger` and both `INerNet`s:
  - delta becomes optional; without it the word vector is the pretrain row alone;
  - the old `tag_clf`/`crit` names are accepted;
  - `predict_tagset` defaults to 0.
- `NerTagger`'s `Require`s:
  - `vocab["delta"] != null` must go;
  - `word_emb_dim == pretrain.Dim` must be checked against the NER model's own pretrain.
- The pipeline must load a second `Pretrain`. That adds 300 MB of floats (250k × 300 × 4) to a German pipeline's
  memory, or 1.2 GB with every model loaded.
- Viterbi, `fix_singleton_tags`, `Entity.Build` and the charlm path (including `CharlmCache` reuse from the tagger:
  same newswiki charlms) are unchanged.

## Reusable as is

- Every network except MWT. Each reads its sizes, flags and vocabularies from the checkpoint:
  - `Tokenizer` (`feat_funcs`, `max_seqlen`);
  - `PosTagger`, including `_nocharlm`;
  - `Lemmatizer`;
  - `DependencyParser`, including `_nocharlm`;
  - `ConstituencyParser` (`retag_method`, `unary_limit`);
  - `SentimentClassifier`;
  - the charlm and `CharlmCache`;
  - `Pretrain`, the managed kernels and the `.pt` reader.
- `Conllu`, `Tree`, `Document` and `Entity`. They need no change: CoNLL-U, MISC `ner=`, `# constituency` and
  `# sentiment` look the same in German.
- `PyString` (`Lower`, `IsUpper`/`IsLower`) covers what German needs (ß, ẞ→ß, umlauts).
- The bulk, pretokenized and no-ssplit input modes, cancellation, threads and the concurrency gate. None of them
  depend on the language.

## New work

| # | work | effort |
|---|---|---|
| 1 | Language plumbing: a `Packages` table per language; `SharedModels` from per-model dependencies, which can't be inferred from the model name (`germeval2014` uses charlms, and `pos` and `ner` use different pretrains); the per-language `PretrainPath`/charlm paths in `Pipeline.cs` (`conll17`, `1billion` hardcoded); `ModelDownloader` (`BaseUrl` `stanza-en`, `Files` MD5 table per language); `DownloadCommand.DefaultDir` (`models/stanza/en`) | M |
| 2 | More than one pretrain per pipeline (conll17 for pos/depparse/constituency/sentiment, fasttextwiki for ner), in `BackendModels`, `ManagedModels` and `TorchSharpModels` | S–M |
| 3 | NER variant (above), both backends, plus a `ManagedBackendTests` net-vs-net case | S |
| 4 | MWT seq2seq: move the lemma's seq2seq into a shared net (both backends), add the MWT data path, and add tests | M |
| 5 | Golden data: German texts, `make_golden.py --lang de` into `tests/golden/de/`, golden tests as theories over language | L |
| 6 | Tool: `compare` embeds `stanza_resources_en.json` (needs `de`); `compare.py`, `benchmark.py` and `make_golden.py` hardcode `stanza.Pipeline("en", ...)` (about 20 call sites) | S–M |
| 7 | CI: German models in the model caches (about +1 GB of `.pt` files, more once converted, per OS cache: see risks), German golden tests in the shards, `verify-package` | M |
| 8 | Docs: README/PACKAGE.md (languages), `docs/verification.md`, CLAUDE.md "Checkpoint findings (de)", `changes/added/german.md` | S |
| 9 | Optional: `PyString.Upper` for `MwtExpander.DictExpansion`. It uses `ToUpperInvariant`, and Python's `upper()` maps `ß`→`SS`. None of the 17 German expansions contains ß, so nothing changes today. | S |

## Risks to byte-exactness

- **MWT seq2seq greedy near-ties.** The lemma's log-probs differ by 4.8e-4 between the managed and TorchSharp
  backends, and lemmas stayed identical because margins were large. MWT has a 25-character vocab and decodes at
  most 10 steps, so margins are probably wide. It still needs the near-tie report the lemma has.
  - Its batches must match Stanza's (50 in document order, `sort_all` inside each), as with every packed model.
  - Running only the non-dictionary candidates would change the batches. That changes only the last float bits,
    not the decisions; still, staying with Stanza's batches is the safe choice.
- **NER's 300-dimensional embedding** makes `input_transform` 2348-wide sums. Measure the emission drift against
  1e-4 early. The `upos_clf`/arc double-sum fixes are the precedent if it misses.
- **Constituency**:
  - `unary_limit` 8 and the SPMRL label set allow deeper unary chains and more decisions per sentence.
  - `ConstituencyTests.NearTies_StayClearOfBackendDrift` should run on German too.
  - The UD-EWT-style divergence study (`tools/constituency_divergence.py`) needs a German corpus, such as UD German
    GSD or HDT.
- **Sentiment** is ill-conditioned, as in English. The golden data should include `logits64` from the start,
  following the owner's 2026-10-08 decision.
- **Lemma**: German morphology could push more words into the seq2seq on real text. The sample shows a low rate,
  but check it on a larger corpus (UD German) with `tools/lemma_divergence.py`.
- **Tokenizer**: no new code paths. German quotes `„…“`, `‚…‘` and `»…«` and ß only go through vocab lookups and
  `PyString`. Two cases to cover anyway: decomposed umlauts (a + U+0308), which are two units on both sides; and
  `capitalized` on `Ä/Ö/Ü` (`PyString.IsUpper`).
- **Memory and CI cache.**
  - A German pipeline holds two pretrains: +300 MB working set compared with English.
  - English and German `.pt` files together come to about 1.6 GB per OS cache entry, plus the converted copies in
    golden.
  - With `cross-os` and `linux-arm64` caching per OS, the 10 GB Actions cache budget gets tight.
  - Options: cache German only where German tests run, or run German golden tests in fewer jobs (light: managed
    only).

## Owner decisions (2026-10-13)

1. Language: `PipelineOptions.Language` (string, default `"en"`, Stanza's codes); `Load` still takes the language folder.
2. Downloader: `DownloadAsync(dir, options)` honours `Language`; `stanzasharp download --lang de` defaults to
   `models/stanza/<lang>`. The 0.1.0 overloads `DownloadAsync(dir)` / `(dir, processors)` stay English-only.
3. One `StanzaSharp` package for every language.
4. German `default_fast` is supported.
5. The partial German packages (`gsd`, `hdt`, `combined`: no ner/sentiment/constituency) are supported too, with
   their processors; `default_accurate` (transformers) throws with the supported list.
6. Entity types are documented per language (README/PACKAGE.md table); no API change.

The questions as they were asked:

### API questions

1. **How a user picks the language.**
   - Recommended: `PipelineOptions.Language` (string, default `"en"`, Stanza's language codes). Packages stay
     strings per language (`"default"`, `"default_fast"`). `Pipeline.Load(dir, options)` still takes the language
     folder (`models/stanza/de`), as today. Additive and binary compatible.
   - Alternatives:
     - `Load` takes the Stanza root (`models/stanza`) and appends the language. Closer to Stanza's `dir`, but it
       breaks today's `Load("models/stanza/en")`.
     - Infer the language from the folder name. Fragile: `compare` already copes with folders not named `en`.
2. **The downloader.**
   - Recommended: `DownloadAsync(dir, options)` honours `options.Language` (repo `stanza-<lang>`, that language's
     MD5 table), and `stanzasharp download --lang de` defaults to `models/stanza/<lang>`.
   - Open: should `DownloadAsync(dir)` / `(dir, processors)` stay English-only (the 0.1.0 API)? Recommended: yes.
3. **Package naming.**
   - Recommended: one `StanzaSharp` package for every language. The code differences are small (seq2seq MWT,
     the NER variant), and models are never in the package, only an MD5 table.
   - The alternative, a `StanzaSharp.German` package, would only make sense if language-specific code grew large.
4. **`default_fast` for German.**
   - In Stanza it keeps the charlm NER and sentiment models (and both pretrains), so it saves less than in English
     (715 vs 893 MB).
   - Support it? It is cheap: the `_nocharlm` pos/depparse paths exist.
5. **Unsupported packages** (`default_accurate`: transformers; `gsd`/`hdt`/`combined`: no ner/sentiment/
   constituency). Recommended: throw with the supported list, as `Package` does today.
6. **Entity types.** `Entity.Type` will carry `PER/LOC/ORG/MISC` for German. Document per language, or add
   nothing?

## Proposed order of work

1. **Owner decisions** on the questions above, before any API work.
2. **Golden data first.**
   - Hand-written German texts in `tests/golden/de/`: `corpus.txt` plus `validation_*.txt` covering news,
     dialogue, instructions, social, poetry, umlauts/ß, compounds, all quote styles, contractions (zum, im, am,
     ins, beim, vom, übers, aufs, ans, ums, zur), capitals (ZUM, AM), CRLF whitespace, non-BMP and a long
     paragraph.
   - `make_golden.py --lang de` writes `pipeline.conllu`, `validation*.conllu`, intermediates (charlm, UPOS
     logits, parser scores, NER emissions, sentiment logits incl. `logits64`) and `mwt.json` with seq2seq-only
     cases.
   - Then the per-processor folders (`lemma/`, `depparse/`, `ner/`, `sentiment/`, `fast/`, `bulk/`,
     `pretokenized/`, `no_ssplit/`, `long_token`), and the converted models in `models/converted/de`.
3. **Plumbing** (items 1–2): language option, per-language package and dependency tables, several pretrains,
   downloader. The English golden data must stay byte-identical.
4. **tokenize → mwt**: the tokenizer and dictionary MWT first (most German MWTs). Then the shared seq2seq, with
   the lemma's golden tests as the refactor's regression check.
5. **pos, lemma, depparse**: no code changes expected; the golden data should pass as is.
6. **constituency** (UPOS retag, `unary_limit` 8) and its near-tie check.
7. **ner** (old format, no delta, fasttextwiki).
8. **sentiment**.
9. **`default_fast` German**, bulk, pretokenized, no-ssplit, long-token.
10. **Tool** (`compare`, `benchmark --lang`), **CI** (caches, shards, no-skip), **docs**, `changes/added/`.
