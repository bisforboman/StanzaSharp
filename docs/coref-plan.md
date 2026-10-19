# Coreference (`coref`): research and plan

> **Owner's decision (2026-10-14): not now.** Coref needs a 24-layer transformer; the smaller gaps (JSON output,
> processing pretagged CoNLL-U) come first. This report is kept as the record for a later decision.
>
> **Gap re-check (2026-10-19, owner's request; docs only):** the managed transformer encoder, this plan's biggest item,
> has landed with `default_accurate` and its follow-ups. What is left is smaller: effort **L**, down from XL. See
> [Gap re-check](#gap-re-check-2026-10-19) below; the rest of the report is the 2026-10-14 research, unchanged where
> the re-check doesn't say otherwise.

## Gap re-check (2026-10-19)

Since 2026-10-14, `default_accurate` (docs/transformer-plan.md) and its follow-ups (#168–#172, #200, #201) built most of
what this plan called new. Checked against `FacebookAI/xlm-roberta-large` at the revision below (its `config.json` and
`tokenizer.json`, Hugging Face API, 2026-10-19) and today's code:

| piece (2026-10-14) | status now | what is left | effort |
|---|---|---|---|
| Transformer encoder | **Done.** `Nn.ITransformerNet`: `ManagedTransformer` (row-invariant GEMMs, attention per sentence without padding, double LayerNorm/GELU/softmax, the same bits alone, batched and on any thread count) and the TorchSharp `TransformerNet`; ELECTRA, BERT (#201) and RoBERTa (#198: positions from `pad_token_id` + 1, one token-type row). XLM-R is RoBERTa's architecture (`XLMRobertaModel`); its config (24 layers, hidden 1024, 16 heads, FFN 4096, eps 1e-5, `max_position_embeddings` 514, `pad_token_id` 1, one token type) is ELECTRA-large's shape with RoBERTa's positions. | `TransformerConfig` to accept `model_type` `xlm-roberta` (prefix `roberta.`, offset 2); a 250,002 × 1,024 embedding table (1 GB, a row lookup, not packed). | **S** |
| LoRA | **Done.** `Nn.LoraWeights`, `ITransformerNet.WithLora`: peft's eval forward on `query`, `value`, `attention.output.dense`, `intermediate.dense` and `output.dense`, coref's exact targets (sentiment's, German constituency's). The joint frozen + LoRA pass (#169) doesn't apply: coref is the only reader of XLM-R. | coref's `bert_lora` key layout (a top-level key of the `.pt`, like German constituency's). | **S** |
| Files | **Pattern done.** `ModelDownloader.Transformers` (pinned revision, per-file sha256: Hugging Face's LFS hash for the weights), `DownloadAsync(dir, options)`, `VerifyChecksums`; `pytorch_model.bin` (2,244,861,551 bytes) is in the repository, so `Checkpoint.Load` reads it as it reads ELECTRA's. | the table entry (revision `c23d21b0620b635a76227c604d44e43a9f0ee389`); the coref `.pt` MD5s in `ModelDownloader.Files`. | **S** |
| Several transformers per pipeline | **Done** (#199): every transformer the models read loads once by name. Coref next to `default_accurate` loads ELECTRA-large and XLM-R. | none. | – |
| Unigram tokenizer | **Not started.** `tokenizer.json` is a `Unigram` model (250,002 pieces with scores, `unk_id`), normalizer `Precompiled` (its charsmap is a base64 double-array trie in the file, 316,720 characters, so no `.model` protobuf is needed), pre-tokenizer `WhitespaceSplit` + `Metaspace` (`▁`, `add_prefix_space`), `TemplateProcessing` `<s>` … `</s>`. It fits `ITransformerTokenizer` next to `WordPiece` and `ByteLevelBpe`, and the golden approach is theirs (transformers' ids on every golden word, plus hand-picked cases: NFKC-like folding `Ｆｕｌｌ` → `▁Full`, `n't` → `▁n` `'` `t`, blank words). | the charsmap trie walk (sentencepiece's `normalizer.cc`), Viterbi over piece scores, Metaspace. No package (`Microsoft.ML.Tokenizers` would be the main package's second dependency). | **M** |
| Windowing | **Different from what exists.** `TransformerEmbedder` is extract_bert_embeddings (overlapping 512 windows per sentence); coref's `get_subwords_batches` cuts the whole document into 510-subword windows at sentence boundaries, without overlap, and batches them with a mask. The managed attention needs no mask (per sequence, no padding), so each window is one sequence. | a coref-only windowing function, with golden subword ids per window. | **S** |
| Coref head | Not started; unchanged. `Gemm`, `Act` (ReLU, LeakyReLU) and the pooled-buffer rules cover it; Conv1d (kernel 3) is new but small, like sentiment's convolutions. | word encoder, rough scorer + topk, pairwise, anaphoricity, clusterize, span predictor, zeros predictor. | **M** |
| Document model | Not started; unchanged. | `Document.Coref`, `Word.CorefChains`, empty words (`N.M`) in `Sentence`, `Conllu` and `Json`. | **M** |
| Golden data and tests | The transformer goldens' pattern exists (`make_golden.py --transformer-only`, `hidden*.safetensors`, `tests/golden/**/transformer/`, `Transformer*` tests in golden shard 4); the pinned `transformers`/`peft` live in `tools/requirements-transformer.txt`. | `--coref-only`, chains and mentions JSON, intermediates (subword ids, encoder output for one window, rough/anaphoricity/span scores), a margins test. | **M** |

**Revised total: L**: the tokenizer, the head, the document model and the golden data, each M; the encoder side S. The risks have moved too:

- **Encoder drift** is no longer a guess. The managed encoder matches Stanza's hidden states within 1.2e-5 on both
  backends (roberta-base, #198; `TransformerEncoderTests`), and every ELECTRA-large golden row downstream (24 layers) is
  byte-identical. 1.2e-5 is 40x inside coref's smallest golden margin (4.9e-4); (a) below measures XLM-R itself.
- **Speed** is measured: one ELECTRA-large pass over 6,044 words took 24.5 s on a 4-core Xeon (round 45), about 250
  words/s. XLM-R-large costs the same per piece, plus coref's head (about 30% more, by this report's FLOP count) and the
  LoRA adapters (about 6%). That is the 2026-10-14 estimate, now with a number under it.
- **Memory:** XLM-R's 2.24 GB of float32 weights, with 1 GB of it the embedding table. That is about 3.3–3.6 GB with the
  default package, and about 4.6 GB next to `default_accurate`'s ELECTRA-large (round 44: 4.1 GB peak without coref).
- **Unchanged:** the `topk(sorted=False)` order (Precision, below) and empty words. These are the two decisions only
  the owner can make.

**What the re-check changes in the owner's questions (below):**

- **Q1 (package):** the transformer code now lives in the main package (`Nn`, the facade), and its weights download
  only on request. A separate `StanzaSharp.Coref` package would split code that `default_accurate` already ships. The
  recommendation becomes: in the main package, opt-in as a processor (Q2).
- **Q6 (models):** `roberta-large`'s byte-level BPE exists now (#198). The three English-only coref models would add a
  1.4 GB transformer but no new tokenizer code: S once the default model works.
- **Q7 (CI):** the transformers are no longer cached (owner's decision, 2026-10-18). Golden shard 4 downloads them on
  every run, so coref adds about 2.4 GB of download per shard-4 run (XLM-R and the `.pt`), not cache space.
- **Q3–Q5 and Q8 (API, CoNLL-U, exactness, precision)** are unchanged.

**Recommendation now:** unpark only with Q3–Q5 decided, and only as these PRs:

1. **(a) The Unigram tokenizer and XLM-R in `TransformerConfig`.** Golden tokenizer ids and hidden states for the
   golden sentences, like #198. This PR alone settles the encoder exactness for XLM-R.
2. **(b) The document model.** Coref chains, attachments and empty words in `Document`, `Conllu` and `Json`, with
   CoNLL-U goldens written by Stanza.
3. **(c) The head and the processor** (`Processor.Coref`, opt-in), with golden chains on the corpus and validation
   texts and the margins test.

(a) is useful by itself, as the encoder spike this report recommended: it measures XLM-R's drift before the owner
commits to the rest.

**(a) done (2026-10-19, owner's choice):** `Nn.Unigram` (the Precompiled normalizer per grapheme cluster with the
shortest trie match, WhitespaceSplit + Metaspace, the Unigram best path with fused unknowns; picked by
`TransformerEmbedder.Load` when tokenizer.json's model is Unigram, which xlm-roberta-large's file infers from its vocab),
`TransformerConfig` reading `xlm-roberta`, `ModelDownloader.Transformers["FacebookAI/xlm-roberta-large"]`. Measured:
transformers' ids exact on all 1,095 golden cases on the first run (.NET's grapheme clusters agreed with
unicode-segmentation on every case, Devanagari, Bengali and Tamil conjuncts, Hangul jamo and ZWJ emoji included); hidden
states within 2.8e-5 of Stanza on the golden sentences and 4.3e-5 on the 180-word one, on both backends, so about 11x
inside coref's smallest golden margin (4.9e-4); a sentence's states have the same bits alone, batched and on any thread
count. The encoder side of the plan is settled: (b) and (c) wait on Q3–Q5.

Research only (2026-10-14): nothing in `src/`, `tests/` or the workflows has changed. Stanza 1.15.0's English coref
models are in `models/stanza/en/coref` (gitignored), from
`stanza.download('en', model_dir=r'models\stanza', processors='coref')` and again with each other coref package. The
transformer they need, `FacebookAI/xlm-roberta-large`, is in the Hugging Face cache (`~/.cache/huggingface/hub`).
Stanza's code is in `tools/.venv/Lib/site-packages/stanza/` (`pipeline/coref_processor.py`, `models/coref/`).
`tools/.venv` lacks `transformers` and `peft`, so the Python runs below used a scratch venv that adds
`transformers` 5.19.0, `peft` 0.21.2 and `tokenizers` 0.23.3 on top of it (torch 2.14.1, numpy 2.2.6 and stanza 1.15.0
pinned by a constraints file). `tools/.venv` itself was not touched.

**In short:** coref is a **transformer model**: the 24-layer XLM-RoBERTa-large encoder (560M parameters, 2.24 GB in
float32, downloaded from Hugging Face rather than bundled) with LoRA adapters. On top of it sits a small word-level
head of about 8M parameters: the "word-level coreference" model of Dobrovolskii (2021). Porting it means building
three things StanzaSharp lacks:

- a SentencePiece Unigram tokenizer;
- a managed transformer encoder;
- empty words (zero anaphora) in the document model.

Everything after the encoder is small and easy. Effort **XL**: roughly as much as the managed backend's Phase 2.

Three things block it or need the owner's decision:

1. **Weights.** About 2.3 GB of extra weights would sit in CI's model caches, which already run close to the 10 GB
   budget.
2. **Singleton detection depends on an unspecified order.** Detecting singletons reads the candidate order of torch's
   `topk(sorted=False)`. For documents under 3,200 words that order comes from the C++ standard library's
   `nth_element`, so it depends on the platform. Python Stanza's own output may therefore differ between Windows and
   Linux.
3. **Memory.** The pipeline goes from about 1 GB to about 3.5 GB.

**Recommendation: no-go for now; go later as a separate opt-in package (`StanzaSharp.Coref`) if users ask for it.** The
recommended first step is cheaper: a managed XLM-R/RoBERTa encoder spike, also the prerequisite for `default_accurate`.
Details at the end.

## Packages and files

From `models/stanza/resources.json` (`en.coref`, `en.packages.*.optional`). resources.json lists no
`dependencies` for coref: the transformer is not a Stanza resource, and transformers fetches it from Hugging Face the
first time the model loads.

| model (`coref` package) | transformer (`bert_model`) | training data (`train_data`) | `.pt` size | `use_zeros` | `full_pairwise` |
|---|---|---|---|---|---|
| **`udcoref_xlm-roberta-lora`** (default) | `FacebookAI/xlm-roberta-large` | `gerrom_udcoref` (CorefUD, Germanic + Romance) | 135.8 MB | true | false |
| `gum-nospeakers_roberta-large-lora` | `roberta-large` | `en_gum` | 135.9 MB | false | false |
| `gum-speakers_roberta-large-lora` | `roberta-large` | `en_gum` | 135.9 MB | true | **true** (speaker embedding) |
| `ontonotes-singletons_roberta-large-lora` | `roberta-large` | `en_ontonotes` | 135.9 MB | false | false |

- **Default:** `udcoref_xlm-roberta-lora`. It is listed as `"optional": {"coref": "udcoref_xlm-roberta-lora"}` in every
  English package (`default`, `default_fast`, `default_accurate`). Coref is *not* in `default_processors`, so Stanza
  never runs it unless `processors` names it.
- **German:** `de.coref` has the same `udcoref_xlm-roberta-lora`, with the same MD5 (`952f55dad09d2350249ed1e2a15a34f8`):
  one multilingual model covers both languages.
- **Transformers:**
  - `FacebookAI/xlm-roberta-large` (revision `c23d21b0620b635a76227c604d44e43a9f0ee389`):
    - `model.safetensors` 2,244,817,354 bytes;
    - `tokenizer.json` 9,096,718 bytes, or `sentencepiece.bpe.model` 5,069,051 bytes.
  - `roberta-large` (byte-level BPE, 355M parameters, about 1.4 GB) is needed only by the three English-only models.
- Each `.pt` holds the head's weights, the LoRA adapters (`bert_lora`, 26.2M parameters, about 105 MB of the 136 MB) and
  the config. The 560M base weights come only from Hugging Face.
- `.pt` keys: `we`, `rough_scorer`, `pw`, `a_scorer`, `sp`, `zeros_predictor`, `bert_lora`, `epochs_trained`, `config`.
  `torch.load(weights_only=True)` reads it, so StanzaSharp's `Checkpoint`/`TorchCheckpoint` should read it unchanged
  (to be confirmed with `tools/stanza_convert.py convert`).

## Architecture

The code is `models/coref/model.py` (`CorefModel.run`), with the modules listed below. The config values are those of
the default checkpoint:

- `bert_window_size` 512, `embedding_size` 20, `sp_embedding_size` 64, `a_scoring_batch_size` 128;
- `hidden_size` 1024, `n_hidden_layers` 1, `rough_k` 50;
- `lora` true: `lora_rank` 64, `lora_alpha` 128, `lora_target_modules` `query, value, output.dense, intermediate.dense`;
- `singletons` true, `clusters_starts_are_singletons` true, `use_zeros` true, `full_pairwise` false.

### Input (`pipeline/coref_processor.py`)

- `REQUIRES_DEFAULT = {TOKENIZE}`. It reads every **word** (after MWT) of the whole document as one sequence
  (`cased_words`), plus each word's sentence index (`sent_id`) and `sentence.speaker`, or `_` without one.
  - POS, lemmas and depparse are not model inputs.
  - UPOS is read only to break ties when choosing the representative mention: the one with the most `PROPN` wins.
  - Speakers come only from the tokenizer's `speaker_delim` option, and only `full_pairwise` models (gum-speakers) use
    them.
- The whole document is one model call: there is no batching across sentences or documents. Bulk input would see one
  combined document, as with sentiment.

### Encoder

**Subwords** (`build_doc`): each word goes through the transformer tokenizer separately: `tokenizer.tokenize(word)`, with
`add_prefix_space` (from `tokenizer_kwargs`). The result is `word2subword` ranges.

- XLM-R's tokenizer is SentencePiece **Unigram** (250,002 pieces) with the `Precompiled` normalizer (an NFKC-like
  charsmap) and the Metaspace `▁` prefix. Some examples:

  | word | subwords |
  |---|---|
  | `Obama` | `▁Obama` |
  | `n't` | `▁n` `'` `t` |
  | `.` | `▁` `.` |
  | `Ｆｕｌｌ` | `▁Full` |
  | `naïve` | `▁na` `ï` `ve` |

- A word that tokenizes to nothing, such as U+200B, makes Stanza throw (`WordEncoder`: "Found a blank word").
- `TOKENIZER_MAPS`/`TOKENIZER_FILTERS` apply only to `roberta-large`/albert, not to XLM-R.

**Windows** (`bert.get_subwords_batches`): the subwords are cut into windows of 510, each backed up to a sentence boundary
(a single sentence longer than 510 subwords is cut hard). Each window is wrapped in `<s>`…`</s>` and padded to 512, and
all windows run as one batch with an attention mask. There is no overlap between windows: a word sees only its own
window's context.

**Transformer** (`XLMRobertaModel` through `peft`; transformers' `sdpa` attention):

- 24 layers, hidden 1024, 16 heads of 64, FFN 4096 with exact (erf) GELU, post-LayerNorm with eps 1e-5;
- learned absolute positions (RoBERTa style, starting at `padding_idx + 1`); one token type.
- **LoRA** on `query`, `value`, `attention.output.dense`, `intermediate.dense` and `output.dense` in every layer:
  `y = W·x + b + 2·B·(A·x)` (scaling = alpha/rank = 128/64; dropout is off in eval). That is 240 adapter tensors. The
  pooler (`lora_modules_to_save`) is saved but unused: only `last_hidden_state` is read, at the non-special positions.

### Head (`models/coref/`)

1. **`WordEncoder`** (`word_encoder.py`): each word is an attention-weighted sum of its subwords: `attn` Linear 1024→1,
   then a softmax within the word. Output: one 1024-vector per word.
2. **`RoughScorer`** (`rough_scorer.py`):
   - a bilinear score `(W·m_i + b)·m_j` over all n × n word pairs, masked to earlier words (log of 0/1);
   - then `torch.topk(k=min(50, n), sorted=False)` per word: the 50 candidate antecedents.
3. **`PairwiseEncoder`** (`pairwise_encoder.py`): a distance-bucket embedding (9 buckets, 20 dimensions). It also embeds
   same/different speaker when `full_pairwise`. The genre embedding is unused.
4. **`AnaphoricityScorer`** (`anaphoricity_scorer.py`):
   - for each (word, candidate): `[a, b, a*b, pw]` (3092) → Linear 3092→1024 → LeakyReLU → Linear 1024→1;
   - plus the rough score, plus a dummy column `EPSILON = 1e-7`;
   - with `singletons`, a "start of chain" score is prepended: `start_map` (Linear 50→1, no bias) over the 50 ffnn
     scores, *in topk's output order*. See Precision.
   - It runs in batches of 128 words. The batching is invisible: rows are independent.
5. **`_clusterize`** (`model.py`):
   - antecedent = argmax over [dummy, candidates];
   - links form an undirected graph whose connected components (DFS) are the clusters, each a sorted word list;
   - `is_start` (start score > dummy) adds every word in no cluster as a singleton cluster;
   - the clusters are sorted lexicographically.
6. **`SpanPredictor`** (`span_predictor.py`): for each head word in a cluster, it scores every word of the same sentence
   as span start and span end:
   - input: `[head, candidate, distance_emb(64)]` (2112);
   - layers: FFNN 2112→1024→256→64 (ReLU), then Conv1d 64→4→2 (kernel 3, padding 1) along the sentence;
   - a log mask makes start ≤ head ≤ end; then an argmax each.
   - Mentions are therefore contiguous, within one sentence, and can nest (`their` inside `their years in the White
     House`).
7. **`zeros_predictor`**: Linear 1024→1024 → ReLU → Linear 1024→1 per clustered word. Where the score is > 0,
   `_handle_zero_anaphora` adds an **empty word** (`id` `(word_id, k)`, text `_`) before that word, and the mention
   moves onto it. On the 8,401-word golden text the highest score was −2.28, so no zeros were created. They are meant
   for pro-drop languages, but nothing stops them in English.

### Output (`coref_processor.process`, `coref_chain.py`, `doc.py`)

- `Document.coref`: a list of `CorefChain(index, mentions, representative_text, representative_index)`.
  - `CorefMention(sentence, start_word, end_word)`: 0-based word indices within the sentence, end exclusive. For a zero
    the start and end are the empty word's id tuple.
  - The **representative** is the longest mention. Ties go to the most `PROPN` words, then the earliest. Zeros are
    never representative; a chain of zeros only gets text `_`.
  - `representative_text` is rebuilt from tokens and words with their `SpaceAfter` (`extract_text`), so a fully
    covered MWT prints as its token (`Its members`, not `It s members`).
- `Word.coref_chains`: one `CorefAttachment(chain, is_start, is_end, is_representative)` per mention covering the word
  (`Document._attach_coref_mentions`).
- **CoNLL-U** (`doc.py` `dict_to_conll_text`): MISC gets `coref_chains=` after `start_char`/`end_char`/`ner`. Each item is
  `unit-`/`start-`/`middle-`/`end-`, an optional `repr-`, then `id<chain>`, comma-separated. It is written on word
  lines (MWT words too). Zeros would add empty-word lines `N.M`. Stanza's CoNLL-U *reader* does not parse
  `coref_chains` back: the format is write-only and loses the representative text.
- **JSON** (`to_dict`): each word dict gets `coref_chains` with
  `{index, representative_text, is_start?, is_end?, is_representative?}` (`CorefAttachment.to_json`).

## Python run

These ran in the scratch venv with `processors='tokenize,mwt,pos,lemma,depparse,coref'` on this desktop (Ryzen 7
5800X, torch's 8 threads). Each paragraph took about 1.2 s with all six processors; loading took 10 s.

```
Barack Obama was born in Hawaii. He later became president. Michelle Obama is his wife, and she wrote a memoir about
their years in the White House.
chain 0 'Barack Obama'   [(0,0,2) Barack Obama, (1,0,1) He, (2,3,4) his]
chain 1 'Hawaii'         [(0,5,6) Hawaii]
chain 2 'president'      [(1,3,4) president]
chain 3 'Michelle Obama' [(2,0,2) Michelle Obama, (2,3,5) his wife, (2,7,8) she]
chain 4 'a memoir about their years in the White House' [(2,9,18)]
chain 5 'their'          [(2,12,13) their]
chain 6 'their years in the White House' [(2,12,18)]

The committee rejected the proposal because it was too expensive. Its members said they would consider a cheaper
version next year.
chain 0 'The committee'  [(0,0,2) The committee, (1,0,2) It s]          <- words of the MWT "Its"
chain 1 'the proposal'   [(0,3,5) the proposal, (0,6,7) it]
chain 2 'Its members'    [(1,0,3) It s members, (1,4,5) they]
chain 3 'a cheaper version' [(1,7,10)]
chain 4 'next year'      [(1,10,12)]
```

Most chains are singletons. The model predicts them (`singletons` true): 1,260 of 1,504 chains on the golden texts.

CoNLL-U for "Mary told John that she didn't trust him. He replied that the feeling was mutual, and they haven't spoken
since." (MISC only):

```
1 Mary    start_char=0|end_char=4|coref_chains=unit-repr-id0
3 John    start_char=10|end_char=14|coref_chains=unit-repr-id1
5 she     start_char=20|end_char=23|coref_chains=unit-id0
9 him     SpaceAfter=No|start_char=37|end_char=40|coref_chains=unit-id1
1 He      start_char=42|end_char=44|coref_chains=unit-id1
4 the     start_char=58|end_char=61|coref_chains=start-repr-id2
5 feeling start_char=62|end_char=69|coref_chains=end-repr-id2
10 they   start_char=86|end_char=90|coref_chains=unit-repr-id3
```

### Speed and memory (Python)

All golden texts (`corpus.txt` + `validation*.txt`) ran as one document: 8,401 words, 845 sentences, 11,965 subwords
(25 windows). Coref alone took:

| | 8 threads | 1 thread |
|---|---:|---:|
| total | 39.6 s | 129.1 s |
| transformer (`_bertify`) | 27.1 s | 84.8 s |
| `a_scorer` | 6.3 s | 23.5 s |
| span predictor | 3.9 s | 13.6 s |
| word encoder + rough scorer | 1.8 s | 6.6 s |

- That is about 210 words/s with 8 threads. The whole StanzaSharp pipeline (all eight processors) does 220–250 words/s
  on the 4-thread CI runners. Coref alone would at least double a pipeline's time.
- **Stanza's `CorefProcessor.process` has no `torch.no_grad()`** (every other processor has one). The LoRA parameters
  require grad, so autograd keeps every layer's activations: the peak working set was **40.7 GB** on that text.
  Wrapped in `no_grad` it was 9.7 GB, with identical output. That is worth reporting upstream.
- Stanza with the coref model loaded: about 3.6 GB resident before processing.

## Precision and nondeterminism

The measurements below cover the 8,401-word document.

- **float32 vs float64** (the whole model in double): identical chains.

  | decision | smallest margin | f32 − f64 at that decision |
  |---|---:|---:|
  | span start | 4.9e-4 | 1.6e-5 |
  | antecedent | 1.1e-3 | |
  | is_start | 2.5e-3 | |
  | span end | 5.5e-3 | |

  The ratio of margin to drift is about 30, comparable to constituency's on its golden data.
- **Threads:** 1 and 8 threads give identical output.
- **The `topk(sorted=False)` order is part of the model** (`rough_scorer.py` `_prune`). `AnaphoricityScorer._ffnn`
  applies `start_map.weight[:, :k]` to the 50 candidate scores *in the order topk returns them*, so the start-of-chain
  score, and with it singleton detection, depends on an order PyTorch does not specify.
  - In ATen's CPU topk, rows with `k * 64 <= n` (n ≥ 3,200 words) use `partial_sort`, so the output is sorted.
    Shorter rows use `std::nth_element`, whose output order belongs to the C++ standard library: MSVC on Windows,
    libstdc++ on Linux, libc++ on macOS.
  - Measured on the 8,401-word document: forcing sorted order changes nothing (as predicted). Ordering by index changes
    25 chains; a random order changes 20.
  - Per golden file (each under 3,200 words): sorted vs unsorted differs in 17 of 1,651 chains (0 to 3 per file).
  - So **Python Stanza's coref probably differs between Windows and Linux**. This was not checked on Linux.
  - A byte-exact port has to reproduce one platform's `nth_element`, with ATen's comparator, on that platform's float
    values. The values carry the transformer's float drift, so even exact ties and near-ties in the rough scores
    matter.
  - The alternative is the owner's call: define the order as sorted and accept about 1% of chains differing from Python
    on short documents.
- **Graph clusters:** `GraphNode.links` is a Python `set` (hash order), but every cluster is sorted afterwards, so this
  adds no nondeterminism.
- **Versions:** results depend on the `transformers` version (`sdpa` vs eager attention, tokenizer details) and on
  `peft`, besides torch. Golden data would pin both in `tools/requirements.txt`.
- **Long documents:** `RoughScorer` builds a dense n × n matrix (8,401² floats = 282 MB). The encoder's quality
  depends on how windows fall at sentence boundaries. Neither is a precision risk, but a port must reproduce the
  windowing exactly.

## Porting assessment

### Reusable

- `Core`:
  - `Checkpoint`/`TorchCheckpoint` (the `.pt` is plain `weights_only` tensors);
  - `SafeTensorFile` (reads `model.safetensors` directly, 2.24 GB, with the existing failure contract);
  - the document model (words, MWT parents, `SpaceAfter`), `Conllu` (one more MISC key).
- `Nn/Managed`:
  - `Gemm` with every path, `PackedMatrix`, `ManagedThreads` (row-invariance and panel rules included);
  - `ArrayPool` discipline; `Act` (ReLU and LeakyReLU are trivial).
- The downloader pattern (`ModelDownloader.Files` with checksums), extended to a Hugging Face model repository at a
  pinned revision.
- The golden-data and test approach: theories over the backend, `ConlluDiff`, `compare`/`benchmark` in the tool.

### New

| piece | what | effort |
|---|---|---|
| Unigram tokenizer | SentencePiece Unigram Viterbi over 250k pieces with scores; the `Precompiled` normalizer (a double-array trie charsmap from `tokenizer.json`/`.model`); Metaspace prefix; special tokens. Or check whether `Microsoft.ML.Tokenizers` (managed, MIT) now covers Unigram with the precompiled charsmap; it is a new dependency, acceptable in an opt-in package. | M |
| Transformer encoder | Embeddings (word + position + type, LayerNorm); 24 × (QKV, masked softmax attention per head, output, LayerNorm, FFN with erf-GELU, LayerNorm); LoRA adapters unmerged (merging changes the arithmetic; unmerged costs about 6% more FLOPs). New managed kernels per path, each with a test: LayerNorm, erf-GELU, masked row softmax, batched QKᵀ/AV (or loops over `Gemm`). A TorchSharp twin for the reference/CUDA side. | L |
| Coref head | Word encoder, rough scorer + topk (with the order question above), pairwise, anaphoricity FFNN, clusterize, span predictor (FFNN + two Conv1d), zeros predictor. `a_scorer`'s Linear over `[a, b, a*b, pw]` can be split into per-word and per-pair parts (about 3x fewer FLOPs; a different summation order, the same as any managed GEMM). | M |
| Document model | `Document.Coref` chains, `Word.CorefChains` attachments, **empty words** (`N.M`) in `Sentence` and in `Conllu` read/write; `extract_text`. | M |
| Pipeline plumbing | Processor constant, `Packages` optional entry, `SharedModels`-style transformer dependency, downloader for the Hugging Face files (2.25 GB, SHA-256 from Hugging Face's LFS metadata), `VerifyChecksums`. | S–M |
| Golden data and tests | `make_golden.py --coref-only` (needs `transformers` + `peft` pinned), the corpus and validation files, mention/chain JSON, intermediate tensors (subword ids, encoder output for one window, rough/anaphoricity/span scores), a margins test like `ConstituencyTests.NearTies_*`. | M–L |

**Total: XL.** Roughly:

- 2–3 weeks of agent work for a managed port that reproduces the golden chains;
- most of the risk is in the tokenizer's normalizer, the transformer kernels' accuracy and the topk order.

### Model size and memory

- On disk: 2.24 GB (`model.safetensors`) + 9 MB (tokenizer) + 136 MB (`.pt`). That is 4x everything StanzaSharp loads
  today (595 MB for English `default`).
- In memory: the managed backend keeps float32 weights, about 2.35 GB packed.
  - The embedding matrix (250,002 × 1024, 1 GB) does not need packing: it is a row lookup.
  - Activations per window are small (16 heads × 512² scores = 16 MB, FFN 512 × 4096 = 8 MB) when windows run a few at
    a time instead of all 25 at once.
  - The head needs n × n rough scores (282 MB at 8.4k words; it can be done in row chunks, since topk is per row) and
    the pair matrices in chunks of 128 words.
- Expected: a pipeline with coref at about 3.3–3.6 GB peak, vs about 1.0–1.3 GB today. Stanza needed 9.7 GB with
  `no_grad` and 40.7 GB without.
- Load: reading 2.25 GB plus packing takes about 2–4 s, and the load peak roughly doubles during packing unless the
  weights are packed tensor by tensor.
- float16/bfloat16 weights would halve this but move results: not without the owner's decision.

### Expected speed (managed)

- The encoder costs about 0.65 GFLOP per subword, about 1.0 GFLOP per word including window padding. The head adds about
  0.3–0.4 GFLOP per word (0.1 with the `a_scorer` split). For 8,401 words that is about 10–12 TFLOP.
- The managed GEMM runs at MKL's speed:
  - one thread: 75–95 GFLOP/s on the Ryzen;
  - 8 threads: 290–470 GFLOP/s;
  - CI's 4-thread EPYC/Xeon runners: about 150–250 GFLOP/s.
- Expected throughput:

  | where | words/s | 8,401 words |
  |---|---:|---:|
  | desktop, 8 threads | 300–350 | 25–30 s (Python 34.5 s with `no_grad`) |
  | 4-thread CI runners | 100–200 | |

  A pipeline with coref would take 2–3x as long as without it. Unlike the LSTM processors (3–4x Python), there is no
  big structural win: the encoder is dense GEMM, where libtorch is already efficient.
- A short sentence: the weights alone are 2.24 GB streamed per call, so latency is bounded by memory bandwidth at about
  80–150 ms per call, vs 28 ms for today's whole pipeline (round 7).

### Byte-exact against Python?

- **Discrete output on the golden texts: plausible, with one condition.** The f32/f64 test shows decisions 30x clear of
  float drift. The managed encoder will sit farther from torch than f64 does: 24 layers of float32 LayerNorm, softmax
  and GELU with different summation orders. A rough estimate is 1e-5 to 1e-4 relative on the encoder output, which is
  still inside the smallest golden margin (4.9e-4) but closer to it. The condition is the topk order (above). Without
  reproducing it, about 1% of chains on short documents differ from Python on purpose.
- **Scores within 1e-4: probably not everywhere.** Like the lemmatizer's log-probabilities, the deep encoder will likely
  need the owner to accept testing exact chains plus margins instead of per-score tolerances (the lemma/constituency
  precedent). Measure first: the encoder spike gives the number.
- **Arbitrary text:** like constituency, a near-tie can flip somewhere. A divergence study on a large corpus (CorefUD's
  English GUM, as for UD EWT) would quantify it.

### Questions for the owner

1. **Package:**
   - (Recommended) A separate opt-in package `StanzaSharp.Coref`, carrying the tokenizer, the encoder and the head. The
     main package and its download stay at about 600 MB.
   - The same assemblies in the main package, with weights downloaded only on request.
2. **Pipeline:**
   - (Recommended) Opt-in like Stanza: `Processors = "...,coref"` once the package is referenced, never in the default
     list.
   - A separate `CorefResolver.Process(Document)` outside the pipeline.
3. **API shape:**
   - (Recommended) Stanza's shape: `Document.Coref` (`IReadOnlyList<CorefChain>` with `Index`, `Mentions`,
     `RepresentativeText`, `RepresentativeIndex`); `CorefMention` (`Sentence`, `StartWord`, `EndWord`; or `Word`
     references); `Word.CorefChains` (attachments with `IsStart`, `IsEnd`, `IsRepresentative`); empty words as
     `Sentence.EmptyWords`.
   - A smaller API without zeros (`use_zeros=False`).
4. **CoNLL-U:** write `coref_chains=` like Stanza (byte-identical with its writer; recommended). Also *read* it back,
   which Stanza does not?
5. **Exactness:** reproduce torch's unsorted topk order (and which platform's: MSVC, since golden data is made on
   Windows?), or define it as sorted and document the difference?
6. **Models:**
   - only the default `udcoref_xlm-roberta-lora`, recommended (it covers German too);
   - or also the three `roberta-large` ones, which need a second tokenizer (byte-level BPE) and a 1.4 GB transformer.
7. **CI:** each OS's model cache would grow by 2.4 GB, past the 10 GB budget across five OSes. Options:
   - run coref tests only in golden shards (Linux), with an exemption from the no-skip rule elsewhere;
   - or one shared cache key for the Hugging Face files.
8. **Memory/precision trade-offs:** float32 only (recommended), or optional bfloat16 weights (half the memory, not
   exact)?

### Cheaper alternatives

- **The encoder spike first** (L, about a week):
  - the Unigram tokenizer + a managed XLM-R encoder;
  - compared against transformers' `last_hidden_state` on the golden corpus;
  - its outcome: drift, speed and memory measured, without committing to coref.
  
  The same engine (with a WordPiece tokenizer and ELECTRA's small differences) is most of what `default_accurate`
  needs: pos, depparse, ner, sentiment and constituency over `electra-large`. That package is Stanza's most accurate
  English setting.
- **TorchSharp only:** write XLM-R in TorchSharp (`StanzaSharp.Cuda`), skip the managed encoder. It is faster to build
  and nearer to Python's arithmetic, but against the managed-first direction, and it brings native libtorch back for
  anyone who wants coref.
- **ONNX Runtime:** export the encoder (LoRA merged) and run it with `Microsoft.ML.OnnxRuntime` in the opt-in package.
  - For: fastest to ship, with optional int8.
  - Against: a native dependency (about 50 MB), not exact, and an export step outside Stanza's files.
- **Not at all:** coref is opt-in in Stanza too. Python users who need it already accept a 2.2 GB transformer and a GPU
  or slow CPU runs.

### Recommendation

**No-go on a full coref port now.** It is the biggest single feature left, but it costs XL work and roughly triples
memory. Its exactness depends on reproducing an unspecified C++ library order, and it would strain the CI cache budget.

**Do the encoder spike** when transformer-based features become a priority. It decides coref and `default_accurate`
together, and the smaller gaps below are cheaper wins in the meantime. Separately, the missing `torch.no_grad()` in
Stanza's `CorefProcessor` (40 GB vs 10 GB) and the platform-dependent topk order deserve an upstream issue (the owner's
call).

## Smaller missing features

| feature | Stanza | StanzaSharp today | effort | status (2026-10-19) |
|---|---|---|---|---|
| JSON output | `Document.to_dict()` (list of sentences of word/token dicts), `repr(doc)` = `json.dumps(indent=2, ensure_ascii=False)`, `to_serialized()`/`from_serialized()` = JSON `{text, sentences, comments}` (`models/common/doc.py`) | none | **S** (writer; byte-identical with Python's `json.dumps` is feasible with a golden file); **S–M** with the reader | done (`Json`, 2026-10-14) |
| `process_conllu` / `*_pretagged` | run later processors on a parsed CoNLL-U (`depparse_pretagged`, `lemma_pretagged`, `constituency_pretagged`) | `Conllu.Read` exists, no pipeline entry | **M** | done (`Process(Document)`, `Pretagged`; 2026-10-14, 2026-10-17) |
| English treebank packages | `ewt`, `gum`, `atis`, `childes`, `craft`, `eslspok`, `genia`, `lines`, `mimic`, `partut`, `combined` (`en.packages`) | `default`, `default_fast` (German has `gsd`, `hdt`) | **S–M** each (same architectures; golden data per package) | done (2026-10-15) |
| Other NER models | `ontonotes_charlm`/`_nocharlm`, `conll03_charlm`, biomedical/clinical (`anatem`, `bc5cdr`, `i2b2`, `ncbi_disease`, `radiology`, ...) via `processors={'ner': ...}` | the package's NER only | **S–M** (old-format checkpoints like German's `germeval2014`) | done (`Models`, 2026-10-16) |
| `ner_predict_tagset` | choose the WorldWide head (column 1) of `ontonotes-ww-multi` | head 0 only | **S** | done (`NerTagSet`, 2026-10-18) |
| `{processor}_model_path` | load a custom-trained checkpoint | fixed package models | **M** (config differences need checking per model) | done (`ModelPaths`, 2026-10-19) |
| `mwt_dict_only`, `lemma_dict_only`, `lemma_ensemble_dict` | dictionary-only modes | no | **S** | done for `mwt_dict_only`, `lemma_dict_only` (2026-10-18) |
| `lemma_beam_size` > 1 | beam search in the seq2seq | greedy only (the checkpoint's `beam_size` 1) | **M** | done (`LemmaBeamSize`, 2026-10-19) |
| `{processor}_batch_size`, `pos_batch_maximum_tokens`, `depparse_min_length_to_batch_separately`, `depparse_sort_during_eval`, `sentiment_batch_size`, `constituency_batch_size` | batching knobs; for depparse and sentiment they change results | fixed at Stanza's defaults | **S** (but each changes golden behaviour; low value) | done (typed batch sizes, 2026-10-18) |
| `tokenize_postprocessor` | a callable that edits the tokenizer's output | no | **S** (a delegate) | done (`TokenizePostprocessor`, 2026-10-18) |
| `tokenize_speaker_delim` | marks speakers (`sentence.speaker`) for coref | no | **S** (only matters with coref) | with coref (#176) |
| `tokenize_combined_charlm` | the charlm tokenizer (`default_accurate`'s) | no | **M** | done (`Models`, 2026-10-16) |
| `default_accurate` | electra-large models for pos, depparse, ner, sentiment, constituency | no | **XL** (transformer; shares the coref encoder spike) | done (2026-10-17) |
| langid | `multilingual/langid/ud.pt` (MD5 `84cd508b…`): char biLSTM + Linear (`models/langid/model.py`), `MultilingualPipeline` routing per language | no | **M**; low value with only en/de models | done (`LanguageIdentifier`, `MultilingualPipeline`, 2026-10-19) |
| morphseg | `pipeline/morphseg_processor.py` wraps the external `morphseg` pip package (its own models, not in resources.json); CoNLL-U `Morphemes=` | no | **L**, and outside Stanza's own models: skip | skipped |
| `Pipeline.stream` / `process_many` | generator over many documents | `Process(IEnumerable<string>)` (bulk) | **S** (an `IEnumerable` wrapper), if wanted | done (`Pipeline.Stream`, 2026-10-18) |
| Document helpers | `sentence.dependencies`, `print_dependencies()`, `doc.iter_words()`, `num_tokens`, `num_words` | partly via LINQ | **S** | done (`Document.TokenCount`, `Sentence.Dependencies`, ..., 2026-10-18) |

Not applicable: `use_gpu`/`device` (`PipelineOptions.Backend`), `download_method`/`resources_*`/`proxies` (downloads are
explicit), `foundation_cache`, `logging_level` (`Logger`).

**Top smaller gaps by value per effort:**

1. JSON output (S): the most visible gap for Python users.
2. `process_conllu`/pretagged input (M).
3. The English treebank packages (S–M each; German's took the same route).
4. `ner_predict_tagset` and the other English NER models (S–M).

All four are done as of 2026-10-18, and so is the rest of #174 (closed) as of 2026-10-19: the status column above.
