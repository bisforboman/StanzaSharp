# Sentence-embedding spike (issue #268)

Can the managed transformer encoder (`ManagedTransformer`) run a sentence-transformers model, with the same vectors as
Python and at a speed near ONNX Runtime's? This page records the first slice of #268 (2026-11-09): a spike in
`samples/StanzaSharp.Benchmark`, nothing public, no new package. Whether to go further is the owner's decision (the
questions on the issue).

## What was built

- `samples/StanzaSharp.Benchmark/EmbeddingSpike.cs` (`StanzaSharp.Benchmark embedding-spike MODEL_DIR REFERENCE`):
  - Loads sentence-transformers/all-MiniLM-L6-v2 at commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (the commit
    samples/StanzaSharp.Rag pins) into `ManagedTransformer` as it is: a `bert` config with an empty weight prefix
    (the checkpoint is a bare `BertModel` state dict, `pytorch_model.bin`). No change to `src/` was needed.
  - Tokenizes each text with the existing `WordPiece` (its `tokenizer.json`), the whole text as one "word", and cuts it
    to `max_seq_length` (256, from `sentence_bert_config.json`) as sentence-transformers does: the last pieces go,
    `[SEP]` stays. (`tokenizer.json` says 128; sentence-transformers overrides it with `max_seq_length`.)
  - Mean pooling over the pieces (`1_Pooling/config.json`; the spike refuses other modes) and L2 normalisation
    (x / max(‖x‖, 1e-12)), both in double.
  - Runs ONNX Runtime 1.31.1 on the model's `onnx/model.onnx` with the same ids, padded per batch with an attention mask
    as samples/StanzaSharp.Rag's `MiniLmEmbeddingGenerator` did then, with as many intra-op threads as the managed pool.
  - Both backends get the same batches, longest text first (sentence-transformers' `encode` order).
- `tools/embedding_spike.py`: the reference, sentence-transformers' `encode` in float32 and in float64, at the pinned
  versions (`tools/requirements*.txt` plus `tools/requirements-embedding.txt`: sentence-transformers 6.2.0,
  transformers 5.20.1, tokenizers 0.23.3, torch 2.14.1).

The texts (935): every `# text =` sentence of tests/golden/pipeline.conllu and validation*.conllu (845), each file's
sentences twelve at a time (77 paragraphs), and each file's whole text (13, past 256 pieces). 25,259 pieces; 14 texts
are truncated to 256.

```
tools/.venv/bin/pip install -r tools/requirements.txt -r tools/requirements-transformer.txt -r tools/requirements-embedding.txt
tools/.venv/bin/python tools/embedding_spike.py models/hf/all-MiniLM-L6-v2 OUT   # --languages [--prefix "query: "] for the later models
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- embedding-spike models/hf/all-MiniLM-L6-v2 OUT --backend managed
```

## Values

| | max \|diff\| vs sentence-transformers float32 | vs float64 |
|---|---|---|
| managed (Vector512) | 5.2e-7 | 4.8e-7 |
| ONNX Runtime | 3.7e-7 | 2.3e-7 |
| managed vs ONNX Runtime | 5.1e-7 | |

The tokenizer's ids equal sentence-transformers' for all 935 texts. Every vector is within 1e-4 on both backends, by
more than two orders of magnitude; sentence-transformers' own float32 and float64 differ by 3.4e-7.

## Speed and memory

A 4-core cloud VM (Intel Xeon @ 2.10 GHz, AVX-512, 1 thread per core), 4 threads on both backends unless said, one
backend per process, median of 5 runs over all 935 texts after a warm-up. The VM is noisy: runs spread by up to 15 %.

| Setting | managed | ONNX Runtime | managed / ORT time | peak working set, managed | ORT |
|---|---|---|---|---|---|
| batch 32, 4 threads | 2.90 s (8,700 pieces/s) | 2.35 s (10,760 pieces/s) | **1.24×** | 409 MB | 806 MB |
| batch 1, 4 threads | 4.08 s | 2.81 s | **1.45×** | 250 MB | 279 MB |
| batch 32, 1 thread | 11.0 s | 6.66 s | **1.65×** | 406 MB | 797 MB |
| batch 128, 4 threads | 2.60 s | 3.92 s | 0.66× | 547 MB | 2,485 MB |

- Load: managed 150–180 ms (working set 177 MB after load), ONNX Runtime 250–330 ms (241 MB).
- Managed allocations per run: about 5 MB on both (the vectors and batch lists; the encoder's buffers are pooled).
- For scale, sentence-transformers on torch 2.14.1 (4 threads, batch 32) took 3.15 s on the same texts.

What the numbers say:

- **Per core, ONNX Runtime is about 1.65× faster** (the 1-thread row): its kernels are better tuned than the managed
  GEMM and attention at MiniLM's small width (hidden 384, 12 heads of 32). With 4 threads the gap narrows to 1.24×
  at batch 32 and 1.45× one text at a time.
- **At batch 128 the managed encoder is faster and much smaller**, because it runs every text unpadded while ONNX
  Runtime pads each batch to its longest text (here a 256-piece paragraph next to short sentences). This favours the
  managed side only when lengths in a batch vary; sorting by length, as both did here, already limits the padding.
- **Memory**: the managed encoder's peak is lower in every setting (half of ONNX Runtime's at batch 32).

Not measured: Native AOT, Arm64, other models (E5, a Swedish model), the Vector256 path, and Windows timings.

## Decisions (the owner, 2026-11-10)

- **Speed gate**: go past the spike if the managed encoder is within 1.5× of ONNX Runtime's time at 4 threads. The
  spike's 1.24× (batch 32) and 1.45× (batch 1) pass. Closing the per-core gap (1.65× on one thread) is later kernel
  tuning.
- **Models**: MiniLM (English) first, then multilingual E5 and a Swedish sentence model.
- **Scope**: a sample only for now. The managed encoder replaces ONNX Runtime in samples/StanzaSharp.Rag, and nothing
  becomes public until the models are settled.

## The Rag sample (2026-11-12)

`samples/StanzaSharp.Rag/MiniLmEmbeddingGenerator.cs` (since replaced, see the last section) now ran the model on `ManagedTransformer`, and the sample no
longer references ONNX Runtime: its output folder holds no native library.

- Files: `config.json`, `pytorch_model.bin` and `vocab.txt` of the same commit, each checked by sha256 (91 MB; the ONNX
  export was 90 MB).
- The ids still come from ML.Tokenizers' `BertTokenizer`, which also sizes the chunks. The spike's `WordPiece` is not
  used there, so one tokenizer both sizes and encodes.
- Each `GenerateAsync` call is one `Encode` call, with no padding; pooling and normalisation are the spike's, in double.
- The encoder is internal, so the sample is in Directory.Build.props's `InternalsVisibleTo`. A user of the package
  cannot copy this file yet; that waits for the decision on what becomes public.
- Check: on the sample's own notes and default query, the printed output (entities, the six hits and their scores to
  three decimals) is the same as on ONNX Runtime. Not measured there: speed and memory (the spike's figures stand).

## Multilingual E5 and a Swedish model (2026-11-14)

The next models of the owner's list, through the same spike (`embedding-spike MODEL_DIR REFERENCE`, now for any
sentence-transformers model with mean pooling):

| Model | Commit | Encoder | Tokenizer | max_seq_length | Normalize | Licence |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | `614241f6` | BertModel, 12 layers, hidden 384, 250,037 pieces | XLM-R Unigram | 512 | yes | MIT |
| intfloat/multilingual-e5-base | `d1287505` | XLMRobertaModel, 12 layers, hidden 768 | XLM-R Unigram | 512 | yes | MIT |
| KBLab/sentence-bert-swedish-cased | `6b5e83cd` | BertModel (KB-BERT), 12 layers, hidden 768 | WordPiece, cased | 384 | no | Apache 2.0 |

The licences are the model cards'; they are not checked further here.

Texts (1,345, `tools/embedding_spike.py --languages`): the 935 above, the sentences of every language's
`tests/golden/languages/*/default.conllu` (285, 106 languages), those joined per language (106) and 19 edge texts
(`EDGE_TEXTS`: an empty text, spaces at either end, tabs, a no-break space, a literal `▁`, the special tokens' texts).
E5 ran with its `query: ` prefix (`--prefix`). 41,291 pieces on E5 (10 texts cut to 512), 41,490 on the Swedish model
(12 cut to 384).

### Values

| | ids equal sentence-transformers' | max \|diff\| vs float32 | vs float64 |
|---|---|---|---|
| multilingual-e5-small, managed | 1,345 of 1,345 | 2.4e-7 | 2.4e-7 |
| multilingual-e5-small, ONNX Runtime | | 1.5e-7 | 8.6e-8 |
| multilingual-e5-base, managed | 1,345 of 1,345 | 5.3e-7 | 3.4e-7 |
| multilingual-e5-base, ONNX Runtime | | 4.9e-7 | 3.3e-7 |
| sentence-bert-swedish-cased, managed | 1,345 of 1,345 | 4.6e-7 | 4.2e-7 |
| all-MiniLM-L6-v2, managed (these texts) | 1,345 of 1,345 | 5.2e-7 | 4.8e-7 |

Two things had to change in `src/` first, both found by the edge texts:

- **E5-small's `tokenizer.json`** has the layout older conversions of the slow tokenizer wrote (`Precompiled` then
  `Replace(" {2,}" → " ")`; `Metaspace` without `WhitespaceSplit`), which `Unigram.Load` refused. transformers 5.20.1's
  `XLMRobertaTokenizer` takes only the vocabulary and the charsmap from the file and builds the usual steps itself, so
  `Unigram.Load` now accepts that layout and runs it as the usual one. Run as written (`tokenizers.Tokenizer.from_file`),
  the file gives an extra `▁` piece for a space at the end of a text or before `<mask>`: 5 of the 19 edge texts.
  `TransformerTokenizerTests.XlmRoberta_OlderConvertedLayoutRunsAsTheUsualOne`.
- **A `<pad>` piece inside a text, on RoBERTa and XLM-RoBERTa** (a bug in both backends, not only in the spike):
  transformers' `create_position_ids_from_input_ids` gives a piece whose id is `pad_token_id` the position row
  `pad_token_id` and doesn't count it; both encoders counted through it. The edge text `<s> and </s> and <pad> and <unk>`
  was off by 7.0e-2 on E5-base (ONNX Runtime was not). Fixed in `TransformerConfig.PositionRow`, used by
  `ManagedTransformer` and the TorchSharp `TransformerNet`; this also changes roberta-base's and xlm-roberta-large's
  states in a pipeline for a sentence with that word, towards Stanza's (it calls the same transformers code). Tests:
  `TransformerTokenizerTests.PositionRows_SkipRobertasPadPiece`,
  `TransformerEncoderTests.Roberta_APadPieceKeepsItsOwnPositionRow_OnBothBackends`. No golden file has such a sentence yet:
  add one to `make_golden.py`'s transformer cases at the next regeneration of the roberta-base goldens.

### Speed and memory

The same VM and method as above (4 threads unless said, one backend per process, median of 3 runs, 5 for E5-small at
batch 32, after a warm-up). E5-base's timings are from before the position fix, which changes one row lookup per piece.

| Model, setting | managed | ONNX Runtime | managed / ORT time | peak working set, managed | ORT |
|---|---|---|---|---|---|
| E5-small, batch 32 | 7.49 s (5,510 pieces/s) | 6.38 s | **1.17×** | 950 MB | 2,720 MB |
| E5-small, batch 1 | 9.83 s | 7.77 s | **1.26×** | 767 MB | 1,033 MB |
| E5-small, batch 32, 1 thread | 25.5 s | 20.8 s | **1.23×** | 932 MB | 2,700 MB |
| E5-small, batch 128 | 9.19 s | 19.3 s | 0.48× | 1,082 MB | 8,567 MB |
| E5-base, batch 32 | 26.0 s (1,590 pieces/s) | 22.9 s | **1.13×** | 2,059 MB | 3,677 MB |
| E5-base, batch 1 | 39.0 s | 28.7 s | **1.36×** | 1,502 MB | 2,055 MB |
| E5-base, batch 32, 1 thread | 92.4 s | 81.2 s | **1.14×** | 2,008 MB | 3,674 MB |
| E5-base, batch 128 | 24.9 s | 47.8 s | 0.52× | 2,368 MB | 10,048 MB |
| Swedish, batch 32 | 24.7 s (1,680 pieces/s) | not run | | 1,296 MB | |

- **Every setting passes the owner's gate** (within 1.5× of ONNX Runtime at 4 threads), with more room than MiniLM had:
  1.13–1.17× at batch 32 and 1.26–1.36× one text at a time. The per-core gap is smaller at these sizes too (1.14–1.23× on
  one thread, against MiniLM's 1.65×).
- **Batch 128** is twice as fast on the managed encoder and takes a quarter of the memory or less: with texts up to 512
  pieces, ONNX Runtime's padding to the batch's longest text costs more than at MiniLM's 256.
- **Memory**: the managed peak is lower in every setting. Most of what stays loaded is the 250,000-row piece table
  (384 MB on E5-small, 768 MB on E5-base); after load the working set is 650 MB (E5-small), 1,380 MB (E5-base) and
  700 MB (Swedish).
- **Load** (files in the page cache): managed 370 ms (E5-small), 1.2 s (E5-base), 0.5 s (Swedish); ONNX Runtime 0.95 s and
  2.4 s.
- **The Swedish model has no ONNX Runtime figure.** It ships no ONNX export, and torch 2.14.1's `torch.onnx.export`
  (the TorchScript exporter) failed on it with transformers 5.20.1. Its encoder has E5-base's shape (12 layers, hidden
  768), so E5-base's ratios are the nearest measure. For scale, sentence-transformers on torch (4 threads, batch 32,
  padded) took 21.1 s on the Swedish model, 7.1 s on E5-small and 23.2 s on E5-base.

Not measured: Native AOT, Arm64, the Vector256 path, Windows, multilingual-e5-large, and retrieval quality (the vectors
equal sentence-transformers', so quality is the models' own).

Open, for the owner: which of these the Rag sample (or a public API) should offer. The sample sizes its chunks with
ML.Tokenizers' `BertTokenizer`; an E5 model there needs the Unigram tokenizer for sizing too, and its `query: ` /
`passage: ` prefixes.

## The Rag sample's `--model` (2026-11-14)

The owner's decision (2026-11-14): after the measurements above, the next step is a model choice in
samples/StanzaSharp.Rag, still a sample and nothing public.

- `--model minilm|e5-small|e5-base|swedish` (`EmbeddingModel.All` in `SentenceEmbeddingGenerator.cs`, which replaces
  `MiniLmEmbeddingGenerator.cs`): each model's commit and the sha256 of its `config.json`, `pytorch_model.bin` and
  `tokenizer.json`, its `max_seq_length`, whether it normalises, and E5's prefixes.
- **One tokenizer sizes and encodes, and it is now StanzaSharp's** (`WordPiece` or `Unigram` from `tokenizer.json`), as
  in the spike. The chunker gets it as a `Microsoft.ML.Tokenizers.Tokenizer` that only counts (`PieceCounter`: the
  pieces and the two special tokens, what the encoder is given).
- **ML.Tokenizers' `BertTokenizer` did not give transformers' ids**, which the sample (and its ONNX Runtime version)
  used until now: on the sample's own notes "Tomas" was `20389, 3022` where transformers and `WordPiece` give `12675`,
  and accented words ("café", "naïve") were `[UNK]`. Its `CountTokens(text, considerNormalization: false)`, which
  `SentenceChunker` calls, also counted without lowercasing. MiniLM's printed scores therefore moved (the filtered hit
  0.149 → 0.141, and the first two unfiltered hits changed places at 0.359 and 0.355); the hits are the same four chunks.
- **E5's prefixes**: `GenerateAsync` (what the vector store calls for a chunk) adds `passage: `; the sample embeds the
  question with `EmbedQueryAsync` (`query: `) and searches by vector. `IEmbeddingGenerator` has no notion of the two
  roles, which a public API would have to settle.
- Checked: all four models run the sample's notes and default query, and the filtered search puts the Nordlys Seafood
  chunk first in each; a first run downloads and verifies the three files. Not checked: Swedish documents (the notes
  and the NER stay English), speed in the sample.

For a public API this leaves: which package holds it, the query/passage roles, whether `PieceCounter` (or a full
`Tokenizer`) becomes public for `SentenceChunker`'s users, and where models download from.
