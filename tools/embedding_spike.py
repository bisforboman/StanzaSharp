"""Reference vectors for the sentence-embedding spike (issue #268, docs/embedding-spike.md).

Runs sentence-transformers/all-MiniLM-L6-v2 through sentence-transformers (mean pooling, L2 normalisation, truncation at
the model's max_seq_length of 256) on the spike's texts and writes OUT.safetensors + OUT.json:
  s{i}.ids        int64 [pieces]   input ids (CLS ... SEP, truncated, no padding)
  s{i}.emb        float32 [384]    the sentence embedding (model.encode, float32)
  s{i}.emb.f64    float64 [384]    the same with the model in float64
OUT.json holds the texts, in order, and the package versions.

The texts: every "# text = " sentence of tests/golden/pipeline.conllu and validation*.conllu (845), then each file's
sentences joined twelve at a time with a space (paragraphs), then each file's sentences joined (past 256 pieces, so
truncation is exercised).

  tools/.venv/bin/python tools/embedding_spike.py models/hf/all-MiniLM-L6-v2 OUT [--batch 32] [--threads N]
"""
import argparse
import glob
import json
import time

import numpy as np
import sentence_transformers
import tokenizers
import torch
import transformers
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer

PARAGRAPH = 12


def texts():
    files = ["tests/golden/pipeline.conllu"] + sorted(glob.glob("tests/golden/validation*.conllu"))
    per_file = []
    for path in files:
        per_file.append([line[len("# text = "):].rstrip("\n") for line in open(path, encoding="utf-8", newline="\n")
                         if line.startswith("# text = ")])
    sentences = [s for f in per_file for s in f]
    paragraphs = [" ".join(f[i:i + PARAGRAPH]) for f in per_file for i in range(0, len(f), PARAGRAPH)]
    return sentences + paragraphs + [" ".join(f) for f in per_file]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    data = texts()
    model = SentenceTransformer(args.model, device="cpu")
    model.eval()
    print(f"max_seq_length {model.max_seq_length}, {len(data)} texts, torch threads {torch.get_num_threads()}")

    tensors = {}
    lengths = []
    for i in range(0, len(data), args.batch):
        features = model.preprocess(data[i:i + args.batch])
        for j, (ids, mask) in enumerate(zip(features["input_ids"], features["attention_mask"])):
            n = int(mask.sum())
            lengths.append(n)
            tensors[f"s{i + j}.ids"] = ids[:n].numpy().astype(np.int64)

    model.encode(data[:args.batch], batch_size=args.batch)  # warm-up
    start = time.perf_counter()
    emb = model.encode(data, batch_size=args.batch, convert_to_numpy=True)
    elapsed = time.perf_counter() - start
    print(f"torch float32 encode: {elapsed:.2f} s for {len(data)} texts, {sum(lengths)} pieces ({sum(lengths) / elapsed:.0f} pieces/s)")
    print(f"pieces per text: max {max(lengths)}, {sum(n == model.max_seq_length for n in lengths)} texts truncated to {model.max_seq_length}")

    model.double()
    emb64 = model.encode(data, batch_size=args.batch, convert_to_numpy=True)
    print(f"torch f32 vs f64: max |diff| {np.abs(emb.astype(np.float64) - emb64).max():.2e}")
    for i in range(len(data)):
        tensors[f"s{i}.emb"] = np.ascontiguousarray(emb[i], dtype=np.float32)
        tensors[f"s{i}.emb.f64"] = np.ascontiguousarray(emb64[i], dtype=np.float64)

    save_file(tensors, args.out + ".safetensors")
    versions = {"sentence-transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "torch": torch.__version__}
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump({"texts": data, "versions": versions}, f, ensure_ascii=False, indent=0)
    print(versions)


if __name__ == "__main__":
    main()
