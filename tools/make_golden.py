#!/usr/bin/env python3
"""
Generate golden test data from Python Stanza for the C# port.

  python tools/make_golden.py [--models models/stanza] [--out tests/golden]

Reads <out>/corpus.txt and writes:
  pipeline.conllu       full pipeline output: token offsets (misc start_char/end_char), MWT ranges,
                        UPOS/XPOS/feats, and a "# constituency = ..." comment per sentence
  intermediates.safetensors + intermediates.json
                        tensors for the first INTERMEDIATE_SENTENCES sentences, each stage run
                        on one sentence at a time (no batching) so they are easy to reproduce:
                          s{i}.tokenize.pred        tokenizer output logits for the sentence's paragraph text
                          s{i}.charlm_forward       forward charlm per-word representation [n_words, 1024]
                          s{i}.charlm_backward      backward charlm per-word representation [n_words, 1024]
                          s{i}.pos.upos_logits      tagger UPOS logits [n_words, n_upos]
                          s{i}.constituency.scores  parser transition logits per step [n_steps, n_transitions]
                        intermediates.json lists the words/text each tensor was computed from.
  tokenize_stress.txt + tokenize_stress.conllu
                        tokenizer-only output on text that exercises long-paragraph windowing
                        and multi-batch padding (see write_tokenize_stress)
  mwt.json              MWT expansions of MWT_WORDS, with and without the dictionary
  validation.conllu     full pipeline output for validation.txt, a varied hand-written corpus
"""
import argparse
import json
import sys
from pathlib import Path

import stanza
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stanza_convert import write_safetensors  # noqa: E402

INTERMEDIATE_SENTENCES = 3
PROCESSORS = "tokenize,mwt,pos,constituency"


class Capture:
    """Records the outputs of a module (or a wrapped method) while active."""

    def __init__(self):
        self.outputs = []

    def hook(self, _module, _inputs, output):
        self.outputs.append(output.detach().cpu().clone())


def capture_module(module):
    cap = Capture()
    handle = module.register_forward_hook(cap.hook)
    return cap, handle


def write_tokenize_stress(models, corpus, out):
    """
    tokenize_stress.txt/.conllu: tokenizer-only output for text that hits the batching paths the
    main corpus doesn't: one paragraph over the 1000-character window, and more paragraphs than
    one batch (32), with differing lengths so padding varies across rows.
    """
    lines = [line for line in corpus.splitlines() if line.strip()]
    long_para = " ".join(lines * 3)
    short_paras = [lines[i % len(lines)] * (1 + i % 3) for i in range(40)]
    text = "\n\n".join([short_paras[0], long_para] + short_paras[1:]) + "\n"
    (out / "tokenize_stress.txt").write_text(text, encoding="utf-8", newline="\n")

    nlp = stanza.Pipeline("en", dir=models, processors="tokenize,mwt", download_method=None,
                          use_gpu=False, logging_level="WARN")
    with torch.no_grad():
        doc = nlp(text)
    (out / "tokenize_stress.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
    print(f"tokenize_stress: {len(long_para)}-char paragraph, {len(short_paras) + 1} paragraphs, "
          f"{len(doc.sentences)} sentences")


# Words to expand with the MWT stage: dictionary hits and misses, case variants, characters
# outside the MWT vocabulary, and words that shouldn't split.
MWT_WORDS = [
    "don't", "Don't", "DON'T", "can't", "Can't", "won't", "it's", "It's", "we'll", "she'd", "I'd've",
    "wouldn't've", "y'all", "gonna", "wanna", "dunno", "cannot", "ain't", "o'clock", "Mary's",
    "children's", "Smiths'", "Kowalczyk's", "Zürich's", "naïve's", "東京's", "rock'n'roll",
    "they’re", "CEO's", "1990's", "U.S.'s", "x", "it'll've", "Jean-Luc's",
]


def write_mwt_golden(models, out):
    """
    mwt.json: for each word in MWT_WORDS, the words the MWT stage expands it to, both with the
    pipeline's dictionary+model ensemble and with the character classifier alone.
    """
    nlp = stanza.Pipeline("en", dir=models, processors="tokenize,mwt", download_method=None,
                          use_gpu=False, logging_level="WARN")
    mwt = nlp.processors["mwt"]

    def expand(ensemble):
        doc = stanza.Document([[{"id": 1, "text": w, "misc": "MWT=Yes"}] for w in MWT_WORDS])
        old = mwt.config["ensemble_dict"]
        mwt.config["ensemble_dict"] = ensemble
        try:
            with torch.no_grad():
                mwt.process(doc)
        finally:
            mwt.config["ensemble_dict"] = old
        return [[w.text for w in s.words] for s in doc.sentences]

    result = [{"word": w, "pipeline": p, "model": m}
              for w, p, m in zip(MWT_WORDS, expand(True), expand(False))]
    with open(out / "mwt.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"mwt.json: {len(result)} words, {sum(r['pipeline'] != r['model'] for r in result)} differ by dictionary")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models/stanza")
    p.add_argument("--out", default="tests/golden")
    args = p.parse_args()
    out = Path(args.out)

    torch.manual_seed(0)
    nlp = stanza.Pipeline("en", dir=args.models, processors=PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    text = (out / "corpus.txt").read_text(encoding="utf-8")

    with torch.no_grad():
        doc = nlp(text)
    (out / "pipeline.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
    print(f"pipeline.conllu: {len(doc.sentences)} sentences")

    validation = (out / "validation.txt").read_text(encoding="utf-8")
    with torch.no_grad():
        vdoc = nlp(validation)
    (out / "validation.conllu").write_text("{:C}\n".format(vdoc), encoding="utf-8", newline="\n")
    print(f"validation.conllu: {len(vdoc.sentences)} sentences, {vdoc.num_words} words")

    tok_model = nlp.processors["tokenize"].trainer.model
    pos_model = nlp.processors["pos"].trainer.model
    con_model = nlp.processors["constituency"]._model

    tensors, index = {}, []
    for i, sent in enumerate(doc.sentences[:INTERMEDIATE_SENTENCES]):
        words = [w.text for w in sent.words]
        entry = {"sentence": i, "text": sent.text, "words": words}

        with torch.no_grad():
            # Tokenizer: run on the sentence text alone so the input is exactly `sent.text`.
            cap, h = capture_module(tok_model)
            nlp.processors["tokenize"].process(sent.text)
            h.remove()
            tensors[f"s{i}.tokenize.pred"] = torch.cat([o[0] for o in cap.outputs]).numpy()

            fwd = pos_model.charmodel_forward.build_char_representation([words])[0]
            bwd = pos_model.charmodel_backward.build_char_representation([words])[0]
            tensors[f"s{i}.charlm_forward"] = fwd.cpu().numpy()
            tensors[f"s{i}.charlm_backward"] = bwd.cpu().numpy()

            # Tagger and parser: feed a one-sentence document built from the golden words.
            single = stanza.Document([[{"id": j + 1, "text": w} for j, w in enumerate(words)]])
            cap, h = capture_module(pos_model.upos_clf)
            nlp.processors["pos"].process(single)
            h.remove()
            tensors[f"s{i}.pos.upos_logits"] = cap.outputs[0].numpy()
            entry["xpos"] = [w.xpos for w in single.sentences[0].words]

            cap, h = capture_module(con_model.output_layers[-1])
            nlp.processors["constituency"].process(single)
            h.remove()
            tensors[f"s{i}.constituency.scores"] = torch.cat(cap.outputs).numpy()
            entry["tree"] = "{}".format(single.sentences[0].constituency)
        index.append(entry)

    write_tokenize_stress(args.models, text, out)
    write_mwt_golden(args.models, out)

    write_safetensors(tensors, out / "intermediates.safetensors", {"stanza": stanza.__version__})
    with open(out / "intermediates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"stanza": stanza.__version__, "sentences": index}, f, indent=1, ensure_ascii=False)
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


if __name__ == "__main__":
    main()
