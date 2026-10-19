# Transformer encoder spike (`default_accurate`): research and plan

> **Owner's decisions (2026-10-17):** spike the managed encoder on **ELECTRA-large first** (it unlocks English
> `default_accurate`; XLM-R and coref reuse the engine later, see [coref-plan.md](coref-plan.md)). Research and a
> measured prototype only: nothing in the pipeline uses it yet.
>
> **Owner's decisions after the spike (2026-10-17):** **go** on `default_accurate`, all five processors, in the four
> PRs below. Hidden states are tested within 1e-4 of Stanza's float32 **or** float64 value (the golden data keeps both),
> the badly conditioned sentences bounded at 1e-2, discrete outputs byte-identical. Sentiment's LoRA adapters are
> applied **at run time**. pos, depparse, ner and constituency **share** one frozen-encoder pass. CI downloads the
> transformer and the five models on every golden run instead of caching them (the caches are near GitHub's 10 GB).

Spike code (all internal, unused by `Pipeline`):

- `src/StanzaSharp.Nn/WordPiece.cs`: the BERT/ELECTRA tokenizer from Hugging Face's `tokenizer.json`, for words already
  split (`is_split_into_words`).
- `src/StanzaSharp.Nn/Managed/ManagedElectra.cs`: the encoder on the managed `Gemm` kernels.
- `samples/StanzaSharp.Benchmark/TransformerSpike.cs`: `StanzaSharp.Benchmark transformer-spike` compares it with the
  Python reference and times it.
- `tools/transformer_spike.py` writes the reference hidden states. `tools/transformer_spike_tags.py` runs Stanza's
  ELECTRA tagger and parser with transformers' features and then with the C# ones, and diffs the output.

**In short:**

- One managed encoder covers all five `default_accurate` transformer models. Four of them (pos, depparse, ner,
  constituency) read the **stock, frozen** `google/electra-large-discriminator`; sentiment adds **LoRA adapters**.
- **Exactness:** fed into Stanza's own ELECTRA tagger and parser, the C# hidden states give **byte-identical UPOS,
  XPOS, feats, lemmas, heads and deprels on all 8,401 golden words**. Hidden-state drift has a per-sentence median of
  9e-6 to 2e-5. 17–23 of 845 sentences exceed 1e-4; they are short ones on which ELECTRA is badly conditioned in
  float32. On those, torch float32 itself differs from float64 by up to 7.6e-3, and from its own unpadded run by 5.4e-3.
- **Speed:** 33–37 s for the 8,401 golden words on 4 cores (Xeon, AVX-512; two runs on a shared cloud machine).
  Stanza's way of running it in Python takes 103.9 s, so 2.8–3.1x faster before any tuning.
- **Memory:** the packed float32 encoder is about 1.35 GB. The process peaks at 2.6 GB with the weights loaded and
  one 128-sentence batch.
- **Tokenizer:** the C# WordPiece gives transformers' ids exactly on all 845 golden sentences (11,958 pieces).
- **Recommendation:** go for `default_accurate`. Port it in four PRs (below), the encoder first. The cost is CI
  download time (+1.9 GB per golden run) and test time, not risk.

## What `default_accurate` needs

From `stanza_resources_en.json` (Stanza 1.15.0), `en.packages.default_accurate`, with each checkpoint's config:

| processor | model | transformer use | own size | status in StanzaSharp |
|---|---|---|---|---|
| tokenize | `combined_charlm` | none | 1.2 MB | done (`Pipeline.Alternatives`) |
| mwt | `combined` | none | – | done |
| lemma | `combined_charlm` | none | – | done (with the contextual classifiers) |
| pos | `combined_electra-large` | frozen; `bert_hidden_layers` None: (h21 + h22 + h23) / 4 (Stanza's "legacy" mix) | 49 MB | needs the encoder |
| depparse | `combined_electra-large` | frozen; `bert_layer_mix` over h21–h24 (Linear 4 → 1): uniform mean + weighted sum; `keep_endpoints`, [CLS] is ROOT | ~100 MB | needs the encoder |
| ner | `ontonotes-ww-multi_electra-large` | frozen; legacy mix | 96 MB | needs the encoder |
| constituency | `ptb3-revised_electra-large` | frozen; `bert_layer_mix` over 4 layers | 129 MB | needs the encoder |
| sentiment | `sstplus_electra-large` | **fine-tuned with LoRA** (`use_peft`, rank 64, alpha 128, on query, value, output.dense, intermediate.dense: 240 tensors, 25.2M parameters in the checkpoint's `bert_lora`) | 170 MB | needs the encoder + LoRA |

Each model also keeps conll17 and both 1billion charlms, as today.

The transformer is not a Stanza resource: transformers downloads `google/electra-large-discriminator` from Hugging
Face. Pinned here as revision `c13c3df7efadc2162f42588bd28eb4e187d602a5`, the files are:

- `pytorch_model.bin`: 1,344,867,008 bytes, sha256 `070894f0…7e63`;
- `tokenizer.json`: 466 KB.

There is no `model.safetensors` in the repository. `Checkpoint.Load` reads the `.bin` unchanged, since it is torch's
zip format.

The architecture is a BERT encoder: 24 layers, hidden 1024, 16 heads, FFN 4096 with exact GELU, post-LayerNorm with
eps 1e-12, absolute positions up to 512, token type 0. The checkpoint carries an `embeddings_project`, but transformers
skips it because `embedding_size` equals `hidden_size`.

How Stanza feeds it (`stanza/models/common/bert_embedding.py`, `extract_base_embeddings`):

- **Batches:** the processor's words of each sentence go to the tokenizer with `is_split_into_words` and
  `padding="longest"`, 128 sentences per forward call, with an attention mask. Each processor sends its own text; pos
  and depparse send the `simplify_punct`ed words.
- **Word vectors:** each word gets the vector of its **last** piece. Without `keep_endpoints` the [CLS] and [SEP]
  vectors are dropped.
- **Long inputs:** over 512 pieces, `build_cloned_features` runs overlapping 512-piece windows (stride 492) and
  stitches them together.
- **Blank words:** if a word yields no piece, `fix_blank_tokens` replaces every such word with `-` and retokenizes
  the whole batch.

## The spike

### Encoder

`ManagedElectra.Forward` takes each sentence's ids and returns the requested hidden states.

- **No padding:** it batches by concatenating the rows of up to 128 sentences. A masked key contributes an exact 0 in
  transformers too, so attention per sentence and head without padding is the same function.
- **Dense layers:** they run on `Gemm.Run`. Q, K and V are packed as one [3072, 1024] matrix.
- **Precision:** attention scores, softmax and sums run in double, and so do the LayerNorm statistics.
- **GELU:** it uses Numerical Recipes' erfc in double, with relative error under 1.2e-7. The first version used a
  double series to 1e-14 and took 58.8 s; this one takes 33–37 s with the same downstream results.

### Exactness

Reference: transformers 5.20.1, torch 2.14.1 (float32, SDPA attention), the same batches as Stanza: 128 sentences
padded to the longest. Measured over the golden sentences' words (`pipeline.conllu` + `validation*.conllu`: 845
sentences, 8,401 words, 11,958 pieces):

| | h21 | h24 |
|---|---|---|
| per-sentence max \|C# − torch f32\|, median | 9.1e-6 | 2.0e-5 |
| … 90th percentile | 1.8e-5 | 3.6e-5 |
| sentences over 1e-4 | 17 | 23 |
| element \|C# − torch f32\|, median / 99% / 99.99% | 6.6e-7 / 6.8e-6 / 2.3e-3 | 5.4e-7 / 6.6e-6 / 2.0e-3 |
| first 8 sentences: torch f32 vs torch f64, max | 1.2e-5 | 1.7e-5 |
| first 8 sentences: C# vs torch f64, max | 2.3e-5 | 3.1e-5 |

The sentences over 1e-4 are short ones, mostly in heavily padded batches: "GO TEAM GO", "So does this one .",
"Proof .". On those, torch disagrees with itself:

| sentence | torch alone vs torch batched | torch f64 alone vs torch f32 batched | C# vs torch f64 alone |
|---|---|---|---|
| "GO TEAM GO" | 5.4e-3 | 7.6e-3 | 2.1e-3 |
| "So does this one ." | 3.0e-3 | 5.2e-3 | 1.3e-3 |
| "Proof ." | 3.9e-4 | 7.3e-4 | 2.0e-3 |

So ELECTRA-large is badly conditioned on such inputs in float32, as the sentiment classifier is
(CLAUDE.md, "sentiment"). Stanza's own float32 error there is of the same size as ours.

**Downstream** (`tools/transformer_spike_tags.py`): Stanza 1.15.0 pretokenized on the golden words, with pos and
depparse `combined_electra-large` and lemma `combined_nocharlm`, run twice:

1. with transformers' ELECTRA, recording the 835 distinct id sequences the transformer saw;
2. with those sequences' hidden states h21–h24 replaced by the C# encoder's.

Out of 8,401 words, **0 differ** in UPOS, XPOS, feats, lemma, head or deprel. That holds with both the exact and the
fast GELU. The 835 sequences cover pos and depparse, whose inputs were identical.

### Speed and memory

All on this cloud machine: 4 cores, Intel Xeon at 2.1 GHz with AVX-512 (`Gemm.Path` Vector512), 15 GB.

| | time for 8,401 words | words/s |
|---|---|---|
| Stanza's way in Python (torch 4 threads, 128 padded sentences per call) | 103.9 s | 81 |
| C# managed, 4 threads, exact-series GELU | 58.8 s | 143 |
| C# managed, 4 threads, fast GELU | 33.2–37.3 s | 226–253 |

- The C# encoder skips the padding Stanza computes. The batch maxima of 34–146 pieces against 14 pieces per sentence
  on average are part of the speed-up.
- For scale, the default package's eight processors handle the same text in a few seconds. Each frozen ELECTRA model
  in `default_accurate` costs a full encoder pass.
- Where the C# time goes: `Gemm.Run` alone reaches 240–330 GFLOP/s here (`managed-spike gemm`), but the whole forward
  pass averages about 195–220. The rest is scalar attention, LayerNorm and GELU, all untuned.
- Memory: the managed weights are 1.35 GB (float32, packed) plus the 125 MB word embeddings. The working set is
  1.6 GB after load and peaks at 2.6 GB during a run; 686 MB is allocated per run over all 845 sentences, all spike
  buffers.

## Plan for `default_accurate`

### Design

- **`Nn.Managed.ManagedTransformer`:** a production version of the encoder, with an `ITransformerNet` seam like the
  other nets.
  - The TorchSharp twin is written with torch ops in StanzaSharp.Cuda, so the GPU package runs it too.
  - Attention, LayerNorm and GELU get a version per `KernelPath` and a test per path (CLAUDE.md, "Rules for the
    managed code").
  - The encoder is row-invariant: `Gemm.Run(rowInvariant: true)`, and attention per sentence. A sentence's hidden
    states then don't depend on its batch.
- **Share one encoder pass.** pos, depparse, ner and constituency all read the same frozen transformer. For each
  sentence and id sequence, the pipeline keeps h21–h24 (4 × 4 KB per piece) in a cache, like `CharlmCache`. Each
  processor then mixes the layers its own way. With a row-invariant encoder, sharing is exact; Stanza instead
  runs the encoder once per processor.
  - pos and depparse sent identical ids on the whole golden set. ner reads tokens and constituency words (without
    `simplify_punct`), so they share whenever their text matches.
  - Sentiment's LoRA encoder is a second, separate pass.
- **LoRA (sentiment).** Two options:
  - **merge at load**, W + (α/r)·B·A: +1.35 GB of weights, the same speed;
  - **apply at run time**, as two rank-64 GEMMs next to each targeted layer: +100 MB of weights, about 8% more
    compute on the sentiment pass.

  Recommended: run time.
- **Tokenizer:** `WordPiece` as in the spike, plus:
  - `fix_blank_tokens`;
  - the 512-piece windows;
  - Python's `str.lower` instead of .NET's, via `PyString`, so the HF tokenizer's Rust lowercasing is matched; this
    needs fuzzing, as `tools/text_fuzz.py` does for the tokenizer.
- **Files:** `ModelDownloader` gets a second source, Hugging Face's `google/electra-large-discriminator` at the pinned
  revision, checked by sha256. `Pipeline.Dependencies` lists it for the five models.

### PRs

PR 1 is split in two (advisor's suggestion, 2026-10-17), so the tokenizer half lands while the nets are reviewed:

| PR | content | effort |
|---|---|---|
| 1a (done) | `WordPiece` (with the added tokens), `TransformerInput` (fix_blank_tokens, convert_to_position_list), `Checkpoint.Load` of a `.bin`, the Hugging Face file table and `ModelDownloader.DownloadTransformerAsync` (internal; the Cli sample's `download-transformer`), `make_golden.py --transformer-only` → `tests/golden/transformer/tokenizer.json`, `TransformerTokenizerTests`; CI downloads `tokenizer.json` (466 KB) | **M** |
| 1b (done) | `ITransformerNet`: the managed encoder (row-invariant, pooled buffers, scalar double attention/LayerNorm/GELU: one version for every CPU) + TorchSharp twin; the 512-piece windows and the layer mixes; golden hidden states for a few sentences (float32 + float64, the badly conditioned ones listed with their bound); bitwise batch- and thread-count invariance tests; CI downloads `pytorch_model.bin` | **L** |
| 2 (done) | pos, depparse, ner (frozen, layer mixes) as `Pipeline.Alternatives` (`--model pos=combined_electra-large`), the shared encoder cache (`TransformerCache`), golden row `pos+depparse+ner=electra-large`; as alternatives they already need the transformer loaded and downloaded, so `Pipeline.Transformers` (apart from `Dependencies`: not a Stanza file), `BackendModels.Transformer` and `DownloadAsync(dir, options)` fetching it came here from PR 4 | **M** |
| 3 (done) | constituency (`bert_layer_mix`, and its sentence boundary vectors: see below), sentiment (LoRA at run time: `LoraWeights`, `ITransformerNet.WithLora`, sharing the base weights); golden rows `constituency=ptb3-revised_electra-large`, `sentiment=sstplus_electra-large` and `electra-large` (all five); `peft` pinned in requirements-transformer.txt | **M–L** |
| 4 (done) | the `default_accurate` package: `Pipeline.Packages`, golden for all eight processors, benchmark rows | **S–M** |

### Testing

Discrete outputs stay byte-identical with Stanza on every golden text, on both backends, as everywhere else. Hidden
states pass within 1e-4 of Stanza's float32 **or** float64 value (the golden data keeps both); the badly conditioned
sentences are an explicit list in the golden data with their bound (1e-2), and their downstream outputs must be exact
(owner's decision, 2026-10-17).

On ordinary sentences every kernel path meets that: the first 40 golden sentences, h24, max \|C# − torch f32\| 4.8e-5
on Vector512, Vector256 and Vector128 (the same bits) and 4.3e-5 on Scalar (`transformer-spike --path`).

### CI

- **Download, no cache** (owner's decision, 2026-10-17): the model caches are already about 9.2 GB of GitHub's 10 GB
  (estimated from their recorded sizes; the session could not list them), so the transformer's files and the five
  Stanza models (~1.9 GB) are downloaded on every golden run (`tools/ci-models.sh transformer FILE...`, with retries), by
  golden shard 4 only (shard 1 until 2026-10-18, when they had made its job 82 min): the `Transformer*` test classes
  run there (`SHARD_FOUR`), and the other jobs' no-skip checks exempt them.
- `transformers` is pinned in its own `tools/requirements-transformer.txt`, not `requirements.txt`, whose hash the
  model cache keys include: CI never needs it (the `.bin` loads without conversion; the golden data is made locally).
- **Test time:** one shared frozen pass plus one LoRA pass over the golden texts takes about 70 s on these 4 cores,
  per backend. TorchSharp would be slower.

## After the spike (2026-10-17)

- **NER:** `ontonotes-ww-multi_electra-large` with the C# features: 0 tag differences on the 8,401 golden words (850 id
  sequences recorded with pos, lemma, depparse and ner), and still 0 in pos, lemma and depparse.
- **Sharing is exact with a row-invariant encoder:** each of 64 sentences alone vs inside one 64-sentence batch
  (`transformer-spike --invariance`): 16 differ in some bit with `Gemm.Run`'s default, 0 with `rowInvariant: true`.
- **One short sentence** (7 words, 9 pieces), median / p90:

  | | 4 threads | 1 thread |
  |---|---|---|
  | C# encoder | 68.4 / 81.6 ms | 175.7 / 191.7 ms |
  | Python transformers | 128.3 / 141.2 ms | 338.4 / 363.8 ms |

  `default_accurate` would cost about two passes (the shared one and sentiment's LoRA one) plus the default package's
  28 ms: about 165 ms per short sentence on 4 cores.
- **The five consumers** read only hidden_states[-4:] (the legacy (h21 + h22 + h23) / 4, or `bert_layer_mix` over
  h21–h24: the uniform mean plus the weighted sum), so the shared cache keeps h21–h24 per id sequence:
  - depparse: `keep_endpoints`, then `x[:-1]` (drops [SEP]; [CLS] is ROOT's vector);
  - constituency: `sentence_boundary_vectors` EVERYTHING, so `keep_endpoints` and learned `word_start`/`word_end`
    embeddings around each sentence, a path the default package's parser doesn't have; words without simplify_punct;
  - sentiment: LoRA rank 64, alpha 128 (scale 2) on query, value, attention.output.dense, intermediate.dense and
    output.dense (240 tensors); the legacy mix; no endpoints.
- **The tokenizer** (PR 1a) gives transformers' ids, word ids and Stanza's positions on all 1,086 golden cases. Two
  rules the spike's version missed: the added tokens' texts ("[CLS]", "[PAD]", ...) are matched in the raw word, and
  unassigned code points (Cn) are kept (they become [UNK]) though tokenizers' `is_other` lists them.
- **The encoder** (PR 1b) against Stanza's own `extract_bert_embeddings` on the golden sentences (managed /
  TorchSharp, the largest distance to the nearer of float32 and float64): ordinary sentences 1.9e-5 / 4.4e-6, the badly
  conditioned ones 9.9e-4 / 1.5e-3 (bound 1e-2), a 180-word sentence in 512-piece windows 3.9e-6 / 2.9e-6; every SIMD
  path within 5.5e-6 on the first three sentences. Stanza windows only when the tokenizer's `model_max_length`
  (tokenizer_config.json) says 512, so that file and `config.json` (the heads, the LayerNorm eps) are in the pinned
  table too.

## RoBERTa (#172, 2026-10-19)

English ner `conll03_electra-large` reads `roberta-base`, not ELECTRA (its `bert_model`; Stanza 1.15.0). The tagger side
is already ported (germeval2014's old checkpoint format, glove + 1billion, the legacy mix over token texts); what it adds:

- **The tokenizer** (`Nn.ByteLevelBpe`, behind `ITransformerTokenizer` with `WordPiece`; `TransformerEmbedder.Load` picks it
  when tokenizer.json's model is BPE, a file that may leave `type` out): GPT-2's byte-level BPE as Stanza's load_tokenizer
  sets it up, `add_prefix_space=True` whatever the file says. Per word, the added tokens first (`<mask>` takes the
  whitespace before it), a space in front of every other piece that doesn't start with U+0020, GPT-2's split pattern
  scanned by code point (no .NET regex: it would split a non-BMP letter's surrogates), bytes to unicode, merges by rank.
- **The encoder:** RoBERTa is BERT's post-LayerNorm architecture; `TransformerConfig` reads `model_type` roberta, its
  weights under `roberta.` (`Prefix`), one token-type row, and positions from `pad_token_id` + 1 = 2
  (`PositionOffset`: create_position_ids_from_input_ids), so `MaxPositions` is 514 − 2 = 512 and both nets slice the
  position table at load; nothing in their forward passes changes.
- **Files:** `ModelDownloader.Transformers["roberta-base"]` (Stanza's alias; Hugging Face serves FacebookAI/roberta-base under
  it), pinned to `e2da8e2f`; golden shard 4 downloads it on every run (`tools/ci-models.sh transformer`).
- **Owner's decisions:** two PRs, (a) the tokenizer, the encoder and the files with goldens
  (`make_golden.py --transformer-only` writes `tokenizer_roberta-base.json` and `hidden_roberta-base.*` next to ELECTRA's),
  then (b) the NER model; a pipeline may load two transformers (ELECTRA-large for `default_accurate` and roberta-base for
  this NER model), keyed by name like the pretrains; roberta-base's ~500 MB is downloaded per CI run like the others.
- **Coref (#176):** the encoder change serves XLM-RoBERTa too (the same architecture, 24 layers); the byte-level BPE only the
  `roberta-large` coref models, since the default `udcoref` model reads SentencePiece Unigram (docs/coref-plan.md).

## Decided

The questions the spike raised were answered on 2026-10-17 (see the top). Still open, for later:

- **Precision options:** bfloat16 or int8 weights would halve or quarter the 1.35 GB but not be exact. Not proposed now.
