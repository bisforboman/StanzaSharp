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
    as `MiniLmEmbeddingGenerator` does, with as many intra-op threads as the managed pool.
  - Both backends get the same batches, longest text first (sentence-transformers' `encode` order).
- `tools/embedding_spike.py`: the reference, sentence-transformers' `encode` in float32 and in float64, at the pinned
  versions (`tools/requirements*.txt` plus `tools/requirements-embedding.txt`: sentence-transformers 6.2.0,
  transformers 5.20.1, tokenizers 0.23.3, torch 2.14.1).

The texts (935): every `# text =` sentence of tests/golden/pipeline.conllu and validation*.conllu (845), each file's
sentences twelve at a time (77 paragraphs), and each file's whole text (13, past 256 pieces). 25,259 pieces; 14 texts
are truncated to 256.

```
tools/.venv/bin/pip install -r tools/requirements.txt -r tools/requirements-transformer.txt -r tools/requirements-embedding.txt
tools/.venv/bin/python tools/embedding_spike.py models/hf/all-MiniLM-L6-v2 OUT
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

## Open questions (the owner's, on #268)

- Go past the spike only within some factor of ONNX Runtime's speed? Which factor?
- Which models next: MiniLM (English), multilingual E5, a Swedish sentence model?
- Scope: an add-on package here, a project of its own on StanzaSharp.Nn, or a sample only?
