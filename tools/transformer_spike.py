"""Reference outputs for the managed transformer encoder spike (docs/transformer-plan.md).

Runs google/electra-large-discriminator through transformers the way Stanza's extract_base_embeddings does (words of
each sentence, is_split_into_words, padding="longest", 128 sentences per batch, output_hidden_states) on the words of
tests/golden/pipeline.conllu and validation*.conllu (845 sentences) and writes OUT.safetensors + OUT.json:
  s{i}.ids        int64 [pieces]   input ids (CLS ... SEP, no padding)
  s{i}.words      int64 [pieces]   word index per piece (-1 for CLS/SEP)
  s{i}.h{L}       float32 [pieces, 1024]  hidden_states[L] for L in LAYERS (the last four, which Stanza mixes)
  s{i}.h{L}.f64   float64, the same batch run in float64 (for --f64 sentences only)
and prints the forward time per batch.

  tools/.venv/bin/python tools/transformer_spike.py models/hf/electra-large OUT [--sentences N] [--f64 N] [--threads N]
"""
import argparse
import glob
import json
import time

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

LAYERS = (21, 22, 23, 24)


def golden_sentences(path):
    sents, cur = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur)
            cur = []
        elif not line.startswith("#"):
            cols = line.split("\t")
            if "-" not in cols[0] and "." not in cols[0]:
                cur.append(cols[1])
    if cur:
        sents.append(cur)
    return sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--conllu", nargs="+", default=["tests/golden/pipeline.conllu"] + sorted(glob.glob("tests/golden/validation*.conllu")))
    ap.add_argument("--sentences", type=int, default=0, help="save only the first N sentences (default: all)")
    ap.add_argument("--f64", type=int, default=8, help="also run the first N sentences' batch in float64")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    model = AutoModel.from_pretrained(args.model).eval()
    data = [s for path in args.conllu for s in golden_sentences(path)]
    keep = len(data) if args.sentences == 0 else args.sentences
    tensors, meta, words, pieces = {}, {"layers": list(LAYERS), "sentences": keep}, sum(map(len, data)), 0
    total = 0.0
    with torch.no_grad():
        for b0 in range(0, len(data), 128):
            batch = data[b0:b0 + 128]
            tok = tokenizer(batch, padding="longest", is_split_into_words=True, return_attention_mask=True)
            ids = torch.tensor(tok["input_ids"])
            mask = torch.tensor(tok["attention_mask"])
            start = time.perf_counter()
            hs = model(ids, attention_mask=mask, output_hidden_states=True).hidden_states
            total += time.perf_counter() - start
            pieces += int(mask.sum())
            for j in range(len(batch)):
                i = b0 + j
                if i >= keep:
                    break
                n = int(mask[j].sum())
                tensors[f"s{i}.ids"] = ids[j, :n].clone()
                tensors[f"s{i}.words"] = torch.tensor([-1 if w is None else w for w in tok.word_ids(j)[:n]])
                for layer in LAYERS:
                    tensors[f"s{i}.h{layer}"] = hs[layer][j, :n].clone()
            if b0 == 0 and args.f64:
                m64 = model.double()
                hs64 = m64(ids, attention_mask=mask, output_hidden_states=True).hidden_states
                for j in range(min(args.f64, keep)):
                    n = int(mask[j].sum())
                    for layer in LAYERS:
                        tensors[f"s{j}.h{layer}.f64"] = hs64[layer][j, :n].clone()
                model = m64.float()
                meta["f64"] = min(args.f64, keep)
    save_file(tensors, args.out + ".safetensors")
    meta.update(words=words, pieces=pieces, seconds=total, threads=torch.get_num_threads(),
                torch=torch.__version__)
    json.dump(meta, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
