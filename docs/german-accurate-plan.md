# German `default_accurate`: research and plan

Research only (2026-10-18, owner's choice after English `default_accurate`): nothing in `src/`, `tests/` or the workflows
changes. The four German transformer checkpoints were downloaded into `models/stanza/de` (gitignored) from
`huggingface.co/stanfordnlp/stanza-de/resolve/v1.15.0/models/`, the transformer's small files from Hugging Face, and
the tokenizer was run with `tools/requirements-transformer.txt`'s transformers 5.20.1.

**In short: go, effort M.** German `default_accurate` reuses everything English needed (the managed encoder, the
window logic, the layer mixes, LoRA at run time, the per-call cache). What is new is small: a `vocab.txt` tokenizer
without accent stripping, two config defaults, and a LoRA pass for the constituency parser. The transformer is a
*base* model (12 layers, 768 wide, 445 MB), so its three passes together cost about what one English pass does.

## The package (Stanza 1.15.0's resources.json)

| processor | model | transformer | how it reads it |
|---|---|---|---|
| tokenize | `combined_charlm` | – | already an alternative |
| mwt | `combined` | – | as `default` |
| lemma | `combined_charlm` | – | already an alternative |
| pos | `combined_german-nlp-electra` (46 MB) | frozen | legacy mix (`bert_hidden_layers` None), like English pos |
| depparse | `combined_german-nlp-electra` (152 MB) | frozen | `bert_layer_mix` over 4 layers, like English depparse |
| constituency | `spmrl_german-nlp-electra` (179 MB) | **LoRA** | `use_peft` with its own `bert_lora` (120 tensors, a top-level key of the checkpoint, not under `params`), `bert_layer_mix` over 4, `sentence_boundary_vectors` EVERYTHING, `retag_method` upos |
| ner | `germeval2014` | – | as `default` (no transformer) |
| sentiment | `sb10k_german-nlp-electra` (102 MB) | **LoRA** | `bert_lora` (120 tensors) under `params`, like English sentiment |

Each model's dependencies are conll17 + the newswiki charlms (ner: fasttextwiki). Every LoRA has rank 64, alpha 128 and
the targets `query`, `value`, `output.dense`, `intermediate.dense`, which `LoraWeights` already reads (12 layers × 5
modules × 2 = 120). No checkpoint has `bert_model.*` weights of its own.

English constituency was frozen. German's is LoRA-adapted, so the German package runs **three** encoder passes per
sentence: the frozen one (pos and depparse share it through `TransformerCache`), constituency's LoRA pass and
sentiment's LoRA pass.

## The transformer: `german-nlp-group/electra-base-german-uncased`

Revision `16e66fe6c6d52b459b7de7330bc9db2c62f3620d`; files `config.json` (467 B), `tokenizer_config.json` (103 B),
`vocab.txt` (275,501 B), `pytorch_model.bin` (444,916,553 B, sha256 `a13cf267dfad08dd5c889af59f2e7f74120ff28107619d6c81904c3250efe76e`).
There is no `tokenizer.json`.

- **Architecture:** `ElectraForPreTraining`, `model_type` electra, hidden 768, 12 layers, 12 heads, intermediate 3,072,
  `embedding_size` = hidden (no embeddings projection), `max_position_embeddings` 512, `hidden_act` gelu,
  vocabulary 32,767. `ManagedTransformer` and `TransformerNet` read this as they are.
- **Config gap:** `position_embedding_type` is absent (transformers' default is `absolute`); `TransformerConfig.Load`
  requires it today and would throw. Fix: default it to `absolute`.
- **Tokenizer:** transformers builds a `BertTokenizer` from `vocab.txt` with a `BertNormalizer` of
  `{"clean_text": true, "handle_chinese_chars": true, "strip_accents": false, "lowercase": true}`: it lowercases but
  keeps accents and doesn't decompose (`Größe` → `größe`, `naïve` → `na ##ï ##ve`, a decomposed `x` + U+0308 stays two
  pieces). Our `WordPiece` reads `tokenizer.json` and always strips accents (NFD minus Mn, right for English). Needed:
  a `vocab.txt` loader (the vocabulary one token per line, id = line; `[UNK]`, `##`, 100 characters per word, the
  special tokens `[PAD] [UNK] [CLS] [SEP] [MASK]` as added tokens) and a `strip_accents` switch.
- **Window size:** `tokenizer_config.json` has the old `"max_len": 512` and no `model_max_length`;
  transformers 5.20.1 still maps it (`model_max_length` 512), and Stanza's `update_max_length` doesn't list the model.
  `TransformerEmbedder.ReadMaxLength` reads only `model_max_length` and would throw. Fix: read `max_len` too.
- Stanza's `BAD_TOKENIZERS` (which replace unknown characters) doesn't list it either: no special handling.

## Work

1. `TransformerConfig`: `position_embedding_type` defaults to absolute. `ReadMaxLength`: `max_len` as a fallback.
2. `WordPiece.FromVocab(vocab.txt, tokenizer_config.json)`: lowercase on (the model name says uncased; the config has no
   `do_lower_case`, whose default is true), `strip_accents` false. Golden: `make_golden.py --transformer-only` gains the
   German tokenizer on every German golden sentence's words (as English's `tokenizer.json` golden does), so the
   normalizer is checked against transformers case by case, umlauts, ß and decomposed text included.
3. `ModelDownloader.Transformers`: the repository at its revision, its four files and sha256s.
4. `ConstituencyParser`: the sentiment classifier's LoRA path (`LoraWeights.Read` on the checkpoint's top-level
   `bert_lora`, `LayerMix(..., lora: true)`, its own `WithLora` net, no cache). About 40 lines.
5. `Pipeline`: German `Alternatives` (the four models), `Dependencies`, `Transformers`, `ModelDownloader.Files` (their
   MD5s), and `Packages["de"]["default_accurate"]`.
6. Golden: `MODEL_CHOICES` rows for the four models and the package (German `make_golden.py` with `electra_locally` for
   the German transformer), `TransformerModelTests` rows with `lang: "de"`, `tests/golden/de/transformer/` hidden states
   for a few sentences (float32 and float64, like English's).
7. CI: golden shard 4 also downloads the German package (~480 MB of models + 445 MB transformer, uncached like
   English's), `tools/ci-models.sh transformer` gets a German line.

Each step is small; the risk is in step 2 (normalizer edge cases), which the golden tokenizer cases settle before any
model work, as English's PR 1a did.

## Speed and memory (estimates)

The base encoder does about 7.1 M multiply-adds per piece and layer (12·h² + the attention), 12 layers: about 85 M per
piece and pass against English large's 302 M, so three German passes (≈ 255 M) cost about 0.85 of one English pass.
German `default_accurate` should then run near English `default`'s speed plus one English-large pass: roughly 12–14 s
on round 45's text and machine (English `default_accurate` 19.3 s), with about 1.4 GB more than German `default` (the
transformer's float weights 0.44 GB, three passes' buffers, the cache).

## Owner's decisions (2026-10-18)

- **Two PRs**, as English's: (a) the tokenizer, the config fixes and the golden tokenizer cases; then (b) the models, the
  constituency LoRA path (which also unlocks English `ptb3_electra-large`, #171) and the package.
- **CI uncached**, like English's: golden shard 4 downloads the German package and transformer (about 0.9 GB) on every
  run.

## Recommendation

Go, as two PRs. The design questions English answered (shared frozen pass, LoRA at run time, exact output on both
backends) carry over unchanged; nothing found here needs a new kind of code.
