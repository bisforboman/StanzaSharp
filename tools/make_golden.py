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
  validation*.conllu    full pipeline output for each validation*.txt, varied hand-written corpora
                        (validation_nonbmp has text outside the BMP; C# reports its offsets in
                        UTF-16 units, so its test ignores start_char/end_char)
  depparse/<name>.conllu
                        tokenize,mwt,pos,lemma,depparse output for corpus.txt (as corpus.conllu) and
                        each validation*.txt; `--depparse-only` regenerates just depparse/
  depparse/intermediates.safetensors + .json
                        for the first INTERMEDIATE_SENTENCES sentences, each parsed alone from its
                        golden words/tags/lemmas: s{i}.unlabeled (arc log-probs [n+1, n+1], -inf on
                        the diagonal) and s{i}.deprel (label log-probs [n+1, n+1, n_deprels])
  lemma/<name>.conllu   tokenize,mwt,pos,lemma output for corpus.txt and each validation*.txt
  lemma/words.json      lemmatizer on LEMMA_WORDS: pipeline lemma, raw seq2seq output and edit class
                        (`--lemma-only` regenerates just lemma/)
  sentiment/<name>.conllu + <name>.json
                        tokenize,mwt,sentiment output for corpus.txt and each validation*.txt, and per
                        sentence the label and the 3 class logits (`--sentiment-only`)
  pt/tiny_{legacy,zip}.pt + their stanza_convert.py output (.json/.safetensors)
                        small checkpoints in both torch.save formats for the C# .pt loader tests;
                        `python tools/make_golden.py --pt-only` regenerates just these
"""
import argparse
import json
import sys
from pathlib import Path

import stanza
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stanza_convert import convert_object, drop_skipped, load_checkpoint, write_safetensors  # noqa: E402

INTERMEDIATE_SENTENCES = 3
PROCESSORS = "tokenize,mwt,pos,lemma,depparse,constituency"  # the default Pipeline


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


DEPPARSE_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"


def write_depparse_golden(models, out):
    """
    depparse/: full output with heads and deprels, kept apart from the files above. The C# parser
    gets each file's own words, tags and lemmas, since the lemmatizer is ported separately.
    """
    nlp = stanza.Pipeline("en", dir=models, processors=DEPPARSE_PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    dep = out / "depparse"
    dep.mkdir(exist_ok=True)
    write_mst_golden(out)
    sources = [out / "corpus.txt"] + sorted(out.glob("validation*.txt"))
    corpus_doc = None
    for path in sources:
        with torch.no_grad():
            doc = nlp(path.read_bytes().decode("utf-8"))
        name = "corpus" if path.name == "corpus.txt" else path.stem
        corpus_doc = corpus_doc or doc
        (dep / f"{name}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        print(f"depparse/{name}.conllu: {len(doc.sentences)} sentences, {doc.num_words} words")

    # Intermediates: parse each sentence alone, capturing GraphParser.forward's predictions.
    processor = nlp.processors["depparse"]
    model = processor.trainer.model
    captured = []
    original = model.forward

    def forward(*a, **k):
        loss, preds = original(*a, **k)
        captured.append(preds)
        return loss, preds

    model.forward = forward
    tensors, index = {}, []
    try:
        for i, sent in enumerate(corpus_doc.sentences[:INTERMEDIATE_SENTENCES]):
            words = [{"id": w.id, "text": w.text, "lemma": w.lemma, "upos": w.upos, "xpos": w.xpos,
                      "feats": w.feats} for w in sent.words]
            single = stanza.Document([words])
            captured.clear()
            with torch.no_grad():
                processor.process(single)
            tensors[f"s{i}.unlabeled"] = captured[0][0][0]
            tensors[f"s{i}.deprel"] = captured[0][2][0]
            index.append({"sentence": i, "words": [w["text"] for w in words],
                          "heads": [w.head for w in single.sentences[0].words],
                          "deprels": [w.deprel for w in single.sentences[0].words]})
    finally:
        model.forward = original
    write_safetensors(tensors, dep / "intermediates.safetensors", {"stanza": stanza.__version__})
    with open(dep / "intermediates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"stanza": stanza.__version__, "sentences": index}, f, indent=1, ensure_ascii=False)
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


def mst_scores(n, seed):
    """Integer-valued scores with many ties, cheap to rebuild in C# (DepparseTests.MstScores)."""
    return np.array([[float(((i * 31 + j * 17 + seed * 7) * (i + 2 * j + seed + 1)) % 23 - 11)
                      for j in range(n)] for i in range(n)])


def write_mst_golden(out):
    """depparse/mst.json: chuliu_edmonds_one_root on mst_scores(n, seed); null where Stanza asserts."""
    from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
    cases = []
    for n in (2, 4, 7, 12, 30, 60):
        for seed in range(40):
            try:
                tree = [int(h) for h in chuliu_edmonds_one_root(mst_scores(n, seed))]
            except AssertionError:
                tree = None
            cases.append({"n": n, "seed": seed, "tree": tree})
    with open(out / "depparse" / "mst.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(cases, f, separators=(",", ":"))
# (word, UPOS) pairs for the lemmatizer: dictionary hits by POS and POS-independent, and seq2seq
# words for every edit type, unknown characters (copied through the delta vocab), non-BMP text,
# a word longer than max_dec_len, and odd inputs.
LEMMA_WORDS = [
    ("saw", "VERB"), ("saw", "NOUN"), ("was", "AUX"), ("children", "NOUN"), ("better", "ADJ"),
    ("Running", "VERB"), ("flibbertigibbets", "NOUN"), ("unfriended", "VERB"), ("GOOGLING", "VERB"),
    ("Zoomers", "PROPN"), ("ãntennae", "NOUN"), ("Zürichers", "PROPN"), ("naïvely", "ADV"),
    ("東京", "PROPN"), ("𝒳s", "NOUN"), ("😀😀s", "SYM"), ("blorptastically", "ADV"),
    ("supercalifragilisticexpialidociousnessesqwertyuiopasdfgh", "NOUN"), ("_", "PUNCT"),
    ("<UNK>", "X"), ("quizzeroos", "NOUN"), ("Wugs", "NOUN"), ("ÉCOLES", "NOUN"), ("ﬁnalized", "VERB"),
]


def write_lemma_golden(models, out):
    """
    lemma/: lemmatizer golden data, in its own folder.
      <name>.conllu  tokenize,mwt,pos,lemma output for corpus.txt and every validation*.txt
      words.json     for each LEMMA_WORDS pair: the pipeline lemma (dictionary + seq2seq ensemble),
                     and the seq2seq model alone: raw decoded string, edit class and final lemma
    """
    from stanza.models.lemma.data import DataLoader

    lemma_out = out / "lemma"
    lemma_out.mkdir(exist_ok=True)
    nlp = stanza.Pipeline("en", dir=models, processors="tokenize,mwt,pos,lemma", download_method=None,
                          use_gpu=False, logging_level="WARN")
    for path in [out / "corpus.txt"] + sorted(out.glob("validation*.txt")):
        with torch.no_grad():
            doc = nlp(path.read_bytes().decode("utf-8"))
        conllu = lemma_out / path.with_suffix(".conllu").name
        conllu.write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        print(f"lemma/{conllu.name}: {len(doc.sentences)} sentences, {doc.num_words} words")

    proc = nlp.processors["lemma"]
    trainer = proc.trainer
    doc = stanza.Document([[{"id": 1, "text": w, "upos": u}] for w, u in LEMMA_WORDS])
    with torch.no_grad():
        proc.process(doc)
        batch = DataLoader(doc, proc.config["batch_size"], proc.config, vocab=proc.vocab,
                           evaluation=True, expand_unk_vocab=True)
        raw, edits = [], []
        for b in batch:
            r, e = trainer.predict(b, proc.config["beam_size"], batch.vocab)
            raw += r
            edits += e
    words = [w for w, _ in LEMMA_WORDS]
    model = trainer.postprocess(words, raw, edits=edits)
    result = [{"word": w, "upos": u, "lemma": s.words[0].lemma, "seq2seq": r, "edit": e, "model": m}
              for (w, u), s, r, e, m in zip(LEMMA_WORDS, doc.sentences, raw, edits, model)]
    with open(lemma_out / "words.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"lemma/words.json: {len(result)} words")


SENTIMENT_PROCESSORS = "tokenize,mwt,sentiment"


def write_sentiment_golden(models, out):
    """
    sentiment/: tokenize,mwt,sentiment output, kept apart from the files above.
      <name>.conllu + <name>.json   for corpus.txt, every validation*.txt and sentiment/reviews.txt
                                    (opinionated sentences): the CoNLL-U, and per sentence the label
                                    and the 3 class logits as computed in the pipeline's batches
      all.json                      the same for all of those texts as one document (joined by blank
                                    lines), which takes several 5000-word batches
    """
    from stanza.models.common.utils import sort_with_indices

    sent_out = out / "sentiment"
    sent_out.mkdir(exist_ok=True)
    nlp = stanza.Pipeline("en", dir=models, processors=SENTIMENT_PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    model = nlp.processors["sentiment"]._model
    captured = []
    original = model.forward

    def forward(*a, **k):
        result = original(*a, **k)
        captured.append(result)
        return result

    def run(text, name, conllu):
        captured.clear()
        with torch.no_grad():
            doc = nlp(text)
        if conllu:
            (sent_out / f"{name}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        # label_sentences runs the batches in length-sorted order; put the logits back in document order.
        _, orig_idx = sort_with_indices(model.extract_sentences(doc), key=len, reverse=True)
        sorted_logits = torch.cat(captured).tolist()
        logits = [None] * len(sorted_logits)
        for k, i in enumerate(orig_idx):
            logits[i] = sorted_logits[k]
        result = [{"sentiment": s.sentiment, "logits": l} for s, l in zip(doc.sentences, logits)]
        with open(sent_out / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"stanza": stanza.__version__, "batches": len(captured), "sentences": result}, f,
                      indent=None if name == "all" else 1)
        print(f"sentiment/{name}: {len(doc.sentences)} sentences, {len(captured)} batches, labels "
              f"{[sum(r['sentiment'] == c for r in result) for c in range(3)]}")

    model.forward = forward
    try:
        texts = []
        for path in [out / "corpus.txt"] + sorted(out.glob("validation*.txt")) + [sent_out / "reviews.txt"]:
            text = path.read_bytes().decode("utf-8")
            texts.append(text)
            run(text, "corpus" if path.name == "corpus.txt" else path.stem, True)
        run("\n\n".join(texts), "all", False)
    finally:
        model.forward = original


def write_pt_fixtures(out):
    """
    pt/: a tiny checkpoint saved in the legacy format (as Stanza's models are) and in the zip format
    (torch.save's default), each next to what stanza_convert.py makes of it. It covers what the real
    models don't: non-contiguous views, non-float dtypes, big ints, nan/inf, non-string dict keys and
    colliding tensor keys.
    """
    from collections import OrderedDict

    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    ckpt = {
        "model": OrderedDict(
            weight=base, weight_t=base.t(), rows=base[1:3], col=base[:, 2], step=base[::2, 1::3],
            ids=torch.tensor([3, -1, 2**40]), half=torch.tensor([1.5, -2.0], dtype=torch.float16),
            flag=torch.tensor([True, False]), scalar=torch.tensor(7.0), empty=torch.zeros(0, 3)),
        "config": {"lr": 1e-05, "eps": 0.0001, "big": 1e16, "whole": 2.0, "third": 1 / 3,
                   "nan": float("nan"), "ninf": float("-inf"), "huge": 2**70, "neg": -5, "flag": True,
                   "none": None, "name": "naïve ✓", "shape": (3, (4, 5)), "list": [1, 2.5, "x"]},
        "by_id": {1: "one", 2: torch.ones(2)},
        "a.b": torch.zeros(1),
        "a": {"b": torch.ones(1)},
        "twice": [base[0], base[0]],
        "optimizer": {"state": {}},
    }
    (out / "pt").mkdir(exist_ok=True)
    for name, kwargs in [("tiny_legacy", {"_use_new_zipfile_serialization": False}),
                         ("tiny_zip", {})]:
        path = out / "pt" / f"{name}.pt"
        torch.save(ckpt, path, **kwargs)
        convert_object(drop_skipped(load_checkpoint(path, False), {"optimizer", "scheduler"}),
                       out / "pt" / name, source_name=path.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="models/stanza")
    p.add_argument("--out", default="tests/golden")
    p.add_argument("--pt-only", action="store_true", help="only regenerate the pt/ loader fixtures")
    p.add_argument("--depparse-only", action="store_true", help="only regenerate depparse/")
    p.add_argument("--lemma-only", action="store_true", help="only regenerate the lemma/ golden data")
    p.add_argument("--sentiment-only", action="store_true", help="only regenerate sentiment/")
    args = p.parse_args()
    out = Path(args.out)
    if args.depparse_only:
        torch.manual_seed(0)
        write_depparse_golden(args.models, out)
        return
    if args.lemma_only:
        write_lemma_golden(args.models, out)
        return
    if args.sentiment_only:
        write_sentiment_golden(args.models, out)
        return
    write_pt_fixtures(out)
    if args.pt_only:
        return

    torch.manual_seed(0)
    nlp = stanza.Pipeline("en", dir=args.models, processors=PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    text = (out / "corpus.txt").read_text(encoding="utf-8")

    with torch.no_grad():
        doc = nlp(text)
    (out / "pipeline.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
    print(f"pipeline.conllu: {len(doc.sentences)} sentences")

    # Read as bytes so CR/CRLF reach Stanza unchanged, as File.ReadAllText passes them in C#.
    for path in sorted(out.glob("validation*.txt")):
        with torch.no_grad():
            vdoc = nlp(path.read_bytes().decode("utf-8"))
        conllu = path.with_suffix(".conllu")
        conllu.write_text("{:C}\n".format(vdoc), encoding="utf-8", newline="\n")
        print(f"{conllu.name}: {len(vdoc.sentences)} sentences, {vdoc.num_words} words")

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

    write_depparse_golden(args.models, out)
    write_tokenize_stress(args.models, text, out)
    write_mwt_golden(args.models, out)
    write_lemma_golden(args.models, out)
    write_sentiment_golden(args.models, out)

    write_safetensors(tensors, out / "intermediates.safetensors", {"stanza": stanza.__version__})
    with open(out / "intermediates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"stanza": stanza.__version__, "sentences": index}, f, indent=1, ensure_ascii=False)
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


if __name__ == "__main__":
    main()
