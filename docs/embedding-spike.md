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
  GEMM and attention at MiniLM's small width (hidden 384, 12 heads of 32). Round 51
  ([performance/round-51-small-width-encoder.md](performance/round-51-small-width-encoder.md), #281) found the GEMMs near
  the core's limit and the gap in attention and GELU (in double); after it the managed encoder takes 1.29× ONNX
  Runtime's time on one thread and 1.19× on four (batch 32), with the same bits. With 4 threads the gap narrows to 1.24×
  at batch 32 and 1.45× one text at a time.
- **At batch 128 the managed encoder is faster and much smaller**, because it runs every text unpadded while ONNX
  Runtime pads each batch to its longest text (here a 256-piece paragraph next to short sentences). This favours the
  managed side only when lengths in a batch vary; sorting by length, as both did here, already limits the padding.
- **Memory**: the managed encoder's peak is lower in every setting (half of ONNX Runtime's at batch 32).

Not measured in this first slice: Native AOT, Arm64, other models, the Vector256 path and Windows timings. The later
sections measure the other models (E5 and the Swedish model, 2026-11-14) and all the rest but Windows (2026-11-16).

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

Not measured here: Native AOT, Arm64, the Vector256 path and multilingual-e5-large (all in "Native AOT, Arm64, Vector256
and multilingual-e5-large" below), Windows, and retrieval quality (the vectors equal sentence-transformers', so quality
is the models' own).

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

## Native AOT, Arm64, Vector256 and multilingual-e5-large (2026-11-16)

The owner's decision (2026-11-16, on #268): measure the remaining gaps before designing a public API. Windows timings
stay out (they need the owner's computer). Nothing here changes `src/` or adds public API.

### What was run

- **The spike**, `embedding-spike` as above, now with `--save FILE` (the managed vectors as raw float32 in text order),
  so that kernel paths, builds and machines can be compared bit for bit (`cmp`, or a sha256).
- **A scratch AOT harness** (not committed, like docs/aot.md's): a console app compiled with the Rag sample's own
  `SentenceEmbeddingGenerator.cs` (assembly named `StanzaSharp.Rag` for the InternalsVisibleTo), referencing
  StanzaSharp.Nn, Core and DataIngestion. It times `SentenceEmbeddingGenerator.LoadAsync` (which checks the three
  files' sha256 before loading; the harness times that hash separately), the first one-text call, and `GenerateAsync`
  over the spike's texts in batches of 32 (longest first), then saves the vectors. Published with
  `dotnet publish -c Release -r linux-x64 -p:PublishAot=true [-p:IlcInstructionSet=x86-64-v3|x86-64-v4]` and run
  next to the same app on the JIT.
- **The Rag sample itself**, `dotnet publish samples/StanzaSharp.Rag -c Release -r linux-x64 -p:PublishAot=true
  -p:IlcInstructionSet=x86-64-v4`, run on its own notes and default query and compared with the JIT build's output.
- **Arm64 in CI**: no Arm64 machine is reachable from the cloud container, so benchmark.yml got a manual mode,
  `embedding_only` (never on pushes or pull requests, never required, like `avx512_only` and `startup_only`; docs/ci.md).
  Its `embedding` jobs run MiniLM through tools/embedding_spike.py and the spike on ubuntu-26.04 (x64),
  ubuntu-26.04-arm and macos-26 (Apple Silicon): batch 32 and batch 1 at the runner's threads and batch 32 on one
  thread, each backend in its own process, median of 5 runs. `embedding-compare` compares the managed vectors' bits
  across the three. Run twice on this branch (runs 70 and 71 of Benchmark); the numbers below are run 71's; run 70's
  agree within 3 % on Linux (macOS: see below).

Local measurements are on the same 4-core VM as the sections above (Intel Xeon @ 2.10 GHz, AVX-512), 4 threads.

### Identical vectors

The managed vectors of MiniLM on the 935 texts, sha256 `78ef002e…`, are **the same bits** in every one of these:

| Build or machine | Kernel path |
|---|---|
| local Xeon, JIT (the spike) | Vector512, Vector256 (forced), Vector128 (forced) |
| local Xeon, JIT (the AOT harness: other batches, through `GenerateAsync`) | Vector512 |
| local Xeon, Native AOT `x86-64-v4` / `x86-64-v3` | Vector512 / Vector256 |
| ubuntu-26.04 (AMD EPYC 9V74, AVX-512), JIT | Vector512 |
| ubuntu-26.04-arm (Neoverse-N2), JIT | Vector128 (NEON) |
| macos-26 (Apple M1, virtual), JIT | Vector128 (NEON) |

multilingual-e5-large's vectors on the 1,345 texts are likewise the same bits on Vector512 and Vector256 (forced), on
the JIT and under Native AOT `x86-64-v4`.

One build differs: **Native AOT for the default x64 target (`x86-64-v2`)**. That target has no FMA3, so `Gemm`'s `Fma`
(`MultiplyAddEstimate`, by design: one fused instruction where the CPU has one, else a·b + c) rounds twice. 344,763 of
359,040 values differ, by at most 3.8e-7; the vectors stay within 4.3e-7 of sentence-transformers' float32, far inside
1e-4. The same would happen on the JIT on an x64 CPU without FMA3 (before Haswell). It also costs speed (below). The
JIT's forced Vector128 path on this machine still fuses, so it keeps the bits. docs/aot.md already recommends
`x86-64-v3`; with it the bits are the JIT's.

### Native AOT

The Rag sample publishes with Native AOT in 47 s into one 12.6 MB file (plus its docs folder). The publish gives 7 trim
and AOT warnings, **all in Microsoft.SemanticKernel.Connectors.InMemory** (IL2026, IL2046, IL3050, IL3051: its
`IQueryable` filters and `GetDynamicCollection`); none in StanzaSharp, StanzaSharp.DataIngestion, Microsoft.ML.Tokenizers,
Microsoft.Extensions.AI or DataIngestion. Run: the download check, the pipeline, chunking, entity enrichment, embedding
and the unfiltered search print exactly what the JIT build prints; the filtered search then throws inside the InMemory
connector (`InMemoryCollection.ConvertFilter`: "Instance property 'Record' is not defined", the code its IL2026 warning
names). So the encoder runs under AOT; the sample as a whole needs another vector store, or no filter, to run AOT.

The harness (the generator alone, 935 MiniLM texts, batch 32; median of 5 after a warm-up):

| Build | Gemm.Path | `LoadAsync` (of it, sha256) | first call | batch 32 | peak working set |
|---|---|---|---|---|---|
| JIT | Vector512 | 354 ms (83 ms) | 60 ms | 2.78 s | 510 MB |
| AOT `x86-64-v4` | Vector512 | 212 ms (68 ms) | 3.5 ms | 2.77 s | 463 MB |
| AOT `x86-64-v3` | Vector256 | 197 ms (78 ms) | 4.0 ms | 3.51 s | 463 MB |
| AOT default (`x86-64-v2`) | Vector128, no FMA | 202 ms (71 ms) | 10 ms | 7.87 s | 462 MB |

- **AOT starts faster**: the model loads in about 130 ms instead of 270 ms (without the hash), the first text takes
  4 ms instead of 60, and process start to the first vector is 222–236 ms instead of 478 ms.
- **Steady speed equals the JIT's on the same path** (2.77 s against 2.78 s on Vector512). Vector256 is 1.26× slower
  than Vector512 on this machine (as in the spike, below); the default target is 2.8× slower than the JIT, twice what
  Vector128 alone costs (the forced JIT Vector128 path takes 4.47 s), because it also has no FMA. As docs/aot.md says,
  an AOT app should set `IlcInstructionSet=x86-64-v3` (or `x86-64-v4` where AVX-512 is certain).
- The harness binary is one 7.5 MB file.

multilingual-e5-large (1,345 texts), AOT `x86-64-v4` against the JIT: `LoadAsync` 4.65 s against 4.82 s (of it 1.8 s
hashing the 2.24 GB of weights, so the load itself is 2.8 s against 3.0 s: reading the weights dominates), first call
62 ms against 112 ms, batch 32 88.2 s against 87.2 s (median of 3), peak working set 3,259 MB against 3,356 MB, and the
same bits. AOT's start-up gain is small next to a large model's load.

### Arm64

MiniLM, 935 texts, the runners' threads (4 on Linux, 3 on macOS), one backend per process, median of 5:

| Runner | Setting | managed | ONNX Runtime | managed / ORT time | peak working set, managed | ORT |
|---|---|---|---|---|---|---|
| ubuntu-26.04-arm (Neoverse-N2) | batch 32 | 4.81 s | 3.82 s | **1.26×** | 435 MB | 832 MB |
| | batch 1 | 5.95 s | 4.13 s | **1.44×** | 258 MB | 301 MB |
| | batch 32, 1 thread | 18.3 s | 14.5 s | **1.27×** | 428 MB | 815 MB |
| ubuntu-26.04 (AMD EPYC 9V74, x64) | batch 32 | 4.19 s | 3.53 s | **1.19×** | 411 MB | 782 MB |
| | batch 1 | 5.05 s | 3.76 s | **1.34×** | 199 MB | 259 MB |
| | batch 32, 1 thread | 8.90 s | 6.99 s | **1.27×** | 401 MB | 789 MB |
| macos-26 (Apple M1, virtual, 3 cores) | batch 32 | 5.41 s | 3.76 s | 1.44× (run 70: 1.62×) | | |
| | batch 1 | 4.58 s | 3.67 s | 1.25× (run 70: 1.33×) | | |
| | batch 32, 1 thread | 10.9 s | 9.16 s | 1.19× (run 70: 1.43×) | | |

- **Linux Arm64 passes the owner's gate** (within 1.5× of ONNX Runtime at the runner's threads): 1.26× at batch 32 in
  both runs, 1.38× and 1.44× one text at a time. Per core (one thread) it is 1.27×, as on the x64 runner, and closer than
  MiniLM's 1.65× on the local Xeon.
- **macOS is too noisy to judge**: on the virtual M1 the runs of one setting spread by up to 70 % in run 70 (managed
  batch 32: 4.67–8.24 s) and 50 % in run 71, and the two runs' ratios differ by up to 0.24; both stay within the gate
  only in run 71. Its peak working set reads 0 on macOS. Its bits are right.
- The managed encoder's memory is lower everywhere, as on x64.
- The values: within 4.4e-7 (Arm64) and 4.7e-7 (macOS) of sentence-transformers' float32, run on the same runner.

Not measured on Arm64: Native AOT (linux-arm64 needs nothing like `IlcInstructionSet`, since NEON is its baseline;
docs/aot.md), Windows on Arm, and the larger models.

### The Vector256 path

Forced with `--path Vector256` on the local AVX-512 machine (JIT, 4 threads, batch 32): the same bits as Vector512
(above), and slower:

| Model | Vector512 | Vector256 | Vector256 / Vector512 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 2.75 s | 3.23 s | 1.18× |
| multilingual-e5-large | 89.8 s | 115.8 s | 1.29× |

(MiniLM's forced Vector128 path: 4.47 s, 1.63×, same bits.) An AVX2 machine without AVX-512 runs this path; its
ratio to ONNX Runtime there is not measured (ONNX Runtime also picks its kernels by CPU), and the cloud runners' CPUs
either have AVX-512 or are drawn at random.

### multilingual-e5-large

intfloat/multilingual-e5-large at commit `3d7cfbda` (MIT): XLMRobertaModel, 24 layers, hidden 1024, 16 heads, the
same XLM-R Unigram `tokenizer.json` as E5-base (the same sha256), `max_seq_length` 512, mean pooling, normalised;
`pytorch_model.bin` is 2.24 GB. It loads into `ManagedTransformer` as E5-base does, with no change. Texts: the 1,345 of
`--languages` with the `query: ` prefix, 41,291 pieces (10 cut to 512).

| | ids equal sentence-transformers' | max \|diff\| vs float32 | vs float64 |
|---|---|---|---|
| managed | 1,345 of 1,345 | 3.5e-7 | 2.5e-7 |
| ONNX Runtime (`onnx/model.onnx` + `model.onnx_data`) | | 5.4e-7 | 4.4e-7 |

sentence-transformers' own float32 and float64 differ by 2.4e-7.

Speed and memory (local VM, 4 threads unless said, one backend per process, median of 3 runs after a warm-up, 1 run on
one thread):

| Setting | managed | ONNX Runtime | managed / ORT time | peak working set, managed | ORT |
|---|---|---|---|---|---|
| batch 32 | 89.8 s (460 pieces/s) | 83.7 s | **1.07×** | 3,356 MB | 4,385 MB |
| batch 1 | 131.2 s | 103.0 s | **1.27×** | 2,790 MB | 1,776 MB |
| batch 32, 1 thread | 338.6 s | 286.3 s | **1.18×** | 3,367 MB | 4,350 MB |
| batch 128 | 86.3 s | not run | | 3,837 MB | |

- **It passes the gate with the most room of any model**: 1.07× at batch 32, 1.27× one text at a time.
- **Load**: managed 2.3–2.8 s (working set after load 2.6 GB), ONNX Runtime 1.7–2.0 s (1.4 GB). One text at a time
  ONNX Runtime's peak is lower here, the only setting so far where it is: the managed encoder keeps its weights as
  float32 arrays (2.24 GB), which ONNX Runtime seems to map from `model.onnx_data` without counting all of it.
- **Batch 128 on ONNX Runtime was not run**: E5-base already peaked at 10 GB there, and large's activations are 4/3
  wider with twice the layers, past this VM's 15 GB.
- At 2.24 GB of weights and about 460 pieces/s on four cores, this model suits batch ingestion more than per-query
  latency; a query of 20 pieces takes about 0.1 s.
