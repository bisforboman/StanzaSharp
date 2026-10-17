"""Does the managed encoder's drift change Stanza's output? (docs/transformer-plan.md)

Runs Stanza's default_accurate tagger and parser (pos and depparse combined_electra-large, lemma combined_nocharlm)
on the golden sentences' words (pretokenized: tests/golden/pipeline.conllu + validation*.conllu) twice: with
transformers' ELECTRA, then with its hidden states replaced by the C# ones (`StanzaSharp.Benchmark transformer-spike
... --ids IDS --dump FILE`, layers 21-24, looked up by each row's ids). Prints how many UPOS/XPOS/feats/lemmas/heads/
deprels differ. Two steps, since the taggers' inputs (simplify_punct and all) are what the transformer must see:

  tools/.venv/bin/python tools/transformer_spike_tags.py models/hf/electra-large record PREFIX   # PREFIX.ids.safetensors, PREFIX.torch.conllu
  StanzaSharp.Benchmark transformer-spike models/hf/electra-large REF --ids PREFIX.ids.safetensors --dump PREFIX.managed.safetensors
  tools/.venv/bin/python tools/transformer_spike_tags.py models/hf/electra-large replay PREFIX
"""
import argparse
import glob
import os
import sys
import types

import torch
from safetensors.torch import load_file, save_file

import stanza
from stanza.models.common import bert_embedding
from stanza.utils.conll import CoNLL

sys.path.insert(0, os.path.dirname(__file__))
from transformer_spike import golden_sentences  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("mode", choices=("record", "replay"))
    ap.add_argument("prefix")
    args = ap.parse_args()

    original = bert_embedding.load_bert
    bert_embedding.load_bert = lambda name, **kw: original(args.model, **kw)  # local files instead of the hub
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nlp = stanza.Pipeline("en", dir=os.path.join(root, "models", "stanza"), package=None,
                          processors={"tokenize": "combined", "pos": "combined_electra-large",
                                      "lemma": "combined_nocharlm", "depparse": "combined_electra-large",
                                      "ner": "ontonotes-ww-multi_electra-large"},
                          tokenize_pretokenized=True, download_method=None,
                          resources_filepath=os.path.join(root, "src", "StanzaSharp.Tool", "stanza_resources_en.json"))
    paths = [os.path.join(root, "tests", "golden", "pipeline.conllu")] + sorted(glob.glob(os.path.join(root, "tests", "golden", "validation*.conllu")))
    data = [s for p in paths for s in golden_sentences(p)]
    bert = nlp.processors["pos"].trainer.model.bert_model
    assert nlp.processors["depparse"].trainer.model.bert_model is bert, "pos and depparse share one transformer"
    assert nlp.processors["ner"].trainers[0].model.bert_model is bert, "ner shares it too"
    if args.mode == "record":
        seen = {}
        call = bert.forward

        def record(self, input_ids, attention_mask=None, **kw):
            for j in range(input_ids.shape[0]):
                n = int(attention_mask[j].sum())
                seen.setdefault(tuple(input_ids[j, :n].tolist()), len(seen))
            return call(input_ids, attention_mask=attention_mask, **kw)

        bert.forward = types.MethodType(record, bert)
        reference = nlp(data)
        save_file({f"s{i}.ids": torch.tensor(ids) for ids, i in seen.items()}, args.prefix + ".ids.safetensors")
        open(args.prefix + ".torch.conllu", "w", encoding="utf-8").write("{:C}\n".format(reference))
        print({"sequences": len(seen)})
        return

    reference = CoNLL.conll2doc(input_file=args.prefix + ".torch.conllu")
    dumped = load_file(args.prefix + ".managed.safetensors")
    layers = {}
    for key, ids in dumped.items():
        if key.endswith(".ids"):
            s = key[:-4]
            layers[tuple(ids.tolist())] = [dumped[f"{s}.h{l}"] for l in (21, 22, 23, 24)]

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, **kw):
        rows = []
        for j in range(input_ids.shape[0]):
            n = int(attention_mask[j].sum())
            rows.append(layers[tuple(input_ids[j, :n].tolist())])
        width = input_ids.shape[1]
        out = []
        for k in range(4):
            t = torch.zeros(input_ids.shape[0], width, 1024)
            for j, r in enumerate(rows):
                t[j, :r[k].shape[0]] = r[k]
            out.append(t)
        return types.SimpleNamespace(hidden_states=[None] * 21 + out)

    bert.forward = types.MethodType(forward, bert)
    managed = nlp(data)

    fields = ("upos", "xpos", "feats", "lemma", "head", "deprel")
    diffs = {f: 0 for f in fields}
    words = 0
    for a, b in zip(reference.sentences, managed.sentences):
        for wa, wb in zip(a.words, b.words):
            words += 1
            for f in fields:
                va, vb = getattr(wa, f), getattr(wb, f)
                if f == "feats":  # the reference went through CoNLL-U, which sorts them
                    va, vb = (None if v is None else sorted(v.split("|"), key=str.lower) for v in (va, vb))
                if va != vb:
                    diffs[f] += 1
                    if sum(diffs.values()) <= 10:
                        print(f"  {f}: {wa.text!r} {getattr(wa, f)} -> {getattr(wb, f)}")
    ner = sum(ta.ner != tb.ner for a, b in zip(reference.sentences, managed.sentences) for ta, tb in zip(a.tokens, b.tokens))
    print({"words": words, "differences": diffs, "ner tags": ner})
    open(args.prefix + ".managed.conllu", "w", encoding="utf-8").write("{:C}\n".format(managed))


if __name__ == "__main__":
    main()
