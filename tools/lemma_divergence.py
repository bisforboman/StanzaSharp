#!/usr/bin/env python3
"""
Lemma divergence study (issue #29, docs/backends.md "lemma"): how often do the lemmatizer's backends disagree on real text?

  python tools/lemma_divergence.py text UD_EWT_DIR DOCS.json        # raw text per document from en_ewt-ud-*.conllu
  python tools/lemma_divergence.py stanza DOCS.json OUT.json [--package default|default_fast] [--models models/stanza]
  StanzaSharp.Benchmark lemma-divergence --docs DOCS.json --out OUT.json [--backend torch|managed] [--double-gates] ...
  python tools/lemma_divergence.py compare stanza.json torch.json managed.json managed-double.json

UD English EWT (CC BY-SA 4.0, github.com/UniversalDependencies/UD_English-EWT) is not ours to redistribute: download its
train/dev/test .conllu files outside the repository. Each document (# newdoc) is one Process call, its paragraphs
(# newpar) separated by blank lines, so every run sees the same documents and the same lemmatizer batches.

Every run writes, per document, each sentence's words (text, UPOS, XPOS, feats, lemma) and, per word that went through
the seq2seq model: the decoded string, the edit class and logits, and per decoder step the top-2 ids and log-probs.
The Stanza run also runs each batch through a float64 copy of the model (the *64 fields).
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

EOS = 3


def build_docs(ewt_dir):
    docs, paras = [], None
    for split in ("train", "dev", "test"):
        for line in (Path(ewt_dir) / f"en_ewt-ud-{split}.conllu").read_text(encoding="utf-8").splitlines():
            if line.startswith("# newdoc"):
                paras = [[]]
                docs.append(paras)
            elif line.startswith("# newpar"):
                if paras[-1]:
                    paras.append([])
            elif line.startswith("# text = "):
                paras[-1].append(line[len("# text = "):])
    return ["\n\n".join(" ".join(p) for p in d if p) for d in docs]


def steps_of(log_probs, row):
    """[[top1, lp1, top2, lp2], ...] for one row until its first EOS (inclusive); log_probs is [steps, batch, cols]."""
    import numpy as np
    out = []
    for s in range(log_probs.shape[0]):
        r = log_probs[s, row].astype(np.float64)
        a = int(np.argmax(r))  # the first maximum, like torch's max
        rest = r.copy()
        rest[a] = -np.inf
        b = int(np.argmax(rest))
        out.append([a, float(r[a]), b, float(r[b])])
        if a == EOS:
            break
    return out


def run_stanza(args):
    import stanza
    import torch
    from stanza.models.lemma import trainer as lemma_trainer

    texts = json.loads(Path(args.docs).read_text(encoding="utf-8"))
    if args.threads:
        torch.set_num_threads(args.threads)
    nlp = stanza.Pipeline("en", dir=args.models, package=args.package, processors="tokenize,mwt,pos,lemma",
                          download_method=None, use_gpu=False, logging_level="WARN")
    proc = nlp.processors["lemma"]
    trainer = proc.trainer
    model32 = trainer.model
    model64 = copy.deepcopy(model32).double()
    for m in model64.modules():  # tensors held as plain attributes
        for k, v in list(vars(m).items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                setattr(m, k, v.double())

    captured = {"steps": [], "edit": None}
    for model in (model32, model64):
        def hook(model):
            decode, predict = model.decode, model.predict

            def d(*a, **k):
                log_probs, state = decode(*a, **k)
                captured["steps"].append(log_probs.squeeze(1).detach().cpu().numpy())
                return log_probs, state

            def p(*a, **k):
                preds, edit_logits = predict(*a, **k)
                captured["edit"] = edit_logits.detach().cpu().double().numpy()
                return preds, edit_logits
            model.decode, model.predict = d, p
        hook(model)

    original_predict = trainer.predict
    state = {"records": [], "offset": 0, "f64": 0.0}

    def capture(model):
        captured["steps"].clear()
        trainer.model = model
        return captured

    def predict(batch, beam_size=1, vocab=None):
        import numpy as np
        capture(model32)
        preds, edits = original_predict(batch, beam_size, vocab)
        clock = time.perf_counter()  # the float64 run and the records are not part of the timings
        steps32, edit32 = np.stack(captured["steps"]), captured["edit"]
        torch.set_default_dtype(torch.float64)
        try:
            capture(model64)
            preds64, edits64 = original_predict(batch, beam_size, vocab)
        finally:
            torch.set_default_dtype(torch.float32)
            trainer.model = model32
        steps64, edit64 = np.stack(captured["steps"]), captured["edit"]
        orig_idx, words = batch[6], batch[7]  # rows are sorted; orig_idx[r] is row r's place in the batch
        unsorted = [None] * len(orig_idx)
        for r, i in enumerate(orig_idx):
            unsorted[i] = r
        for i, r in enumerate(unsorted):
            word = words[r]
            lemma = trainer.postprocess([word], [preds[i]], edits=[edits[i]])[0]
            lemma64 = trainer.postprocess([word], [preds64[i]], edits=[edits64[i]])[0]
            state["records"].append({
                "s2s": state["offset"] + i, "lemma": None if lemma in ("", "_") and word != "_" else lemma, "dec": preds[i], "edit": edits[i], "el": edit32[r].tolist(),
                "steps": steps_of(steps32, r),
                "dec64": preds64[i], "edit64": edits64[i], "el64": edit64[r].tolist(), "steps64": steps_of(steps64, r),
                # LemmaProcessor maps "" to "_", which the Word.lemma setter stores as None unless the word is "_".
                "lemma64": None if lemma64 in ("", "_") and word != "_" else lemma64})
        state["offset"] += len(orig_idx)
        state["f64"] += time.perf_counter() - clock
        return preds, edits

    trainer.predict = predict
    if args.no_dict:  # every word through the seq2seq model (its lemma in the records; the document keeps the dictionary's)
        trainer.skip_seq2seq = lambda pairs: [False] * len(pairs)
    process = proc.process
    lemma_time = [0.0]

    def timed(document):
        clock = time.perf_counter()
        result = process(document)
        lemma_time[0] += time.perf_counter() - clock
        return result
    proc.process = timed

    out_docs = []
    nlp("Warm up the pipeline.")
    state["f64"], lemma_time[0] = 0.0, 0.0
    clock = time.perf_counter()
    for n, text in enumerate(texts):
        state["records"], state["offset"] = [], 0
        with torch.no_grad():
            doc = nlp(text)
        words = [w for s in doc.sentences for w in s.words]
        skip = trainer.skip_seq2seq([(w.text, w.upos) for w in words])
        misses = [i for i, s in enumerate(skip) if not s]
        for r in state["records"]:
            r["i"] = misses[r.pop("s2s")]
        out_docs.append({"sents": [[[w.text, w.upos, w.xpos, w.feats, w.lemma] for w in s.words] for s in doc.sentences],
                         "s2s": state["records"]})
        if n % 100 == 0:
            print(f"{n}/{len(texts)} documents", file=sys.stderr)
    total = time.perf_counter() - clock - state["f64"]
    meta = {"backend": "stanza", "package": args.package, "no_dict": args.no_dict, "threads": torch.get_num_threads(),
            "seconds": total, "lemma_seconds": lemma_time[0] - state["f64"], "study_seconds": state["f64"],
            "stanza": stanza.__version__}
    Path(args.out).write_text(json.dumps({"meta": meta, "docs": out_docs}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta))


def compare(args):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.runs]
    names = [r["meta"]["backend"] + (" double-gates" if r["meta"].get("double_gates") else "") for r in runs]
    ref = runs[0]
    import itertools

    for r, name in zip(runs, names):
        m = r["meta"]
        print(f"{name:24} total {m['seconds']:8.2f} s  lemma {m['lemma_seconds']:7.2f} s  threads {m.get('threads')}")

    # Tokens and tags first: a difference there is a tokenizer/mwt/pos difference, not a lemma one.
    bad_docs = set()
    for d in range(len(ref["docs"])):
        shapes = [[[w[:4] for w in s] for s in r["docs"][d]["sents"]] for r in runs]
        if any(s != shapes[0] for s in shapes[1:]):
            bad_docs.add(d)
            for name, s in zip(names[1:], shapes[1:]):
                if s != shapes[0]:
                    a = [w for x in shapes[0] for w in x]
                    b = [w for x in s for w in x]
                    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    print(f"doc {d}: {name} tokens/tags differ from stanza at word {k}: "
                          f"{a[k] if k < len(a) else None} vs {b[k] if k < len(b) else None}")
    total = sum(len(s) for d in ref["docs"] for s in d["sents"])
    print(f"\n{len(ref['docs'])} documents, {total} words; {len(bad_docs)} documents with token/tag differences (excluded below)")

    # Per word: each run's lemma (and Stanza's float64), keyed by (doc, word).
    good = [d for d in range(len(ref["docs"])) if d not in bad_docs]
    words = sum(len(s) for d in good for s in ref["docs"][d]["sents"])
    s2s = [{(d, rec["i"]): rec for d in good for rec in r["docs"][d]["s2s"]} for r in runs]
    keys = sorted(s2s[0])
    print(f"compared: {words} words, {len(keys)} through seq2seq, "
          f"{sum(len(rec['steps']) for rec in s2s[0].values())} decoder steps (stanza float32)")
    for r, name in zip(s2s[1:], names[1:]):
        assert set(r) == set(s2s[0]), f"{name}: different seq2seq words"

    # Margins of every decoder step (top-1 minus top-2), per run, plus Stanza float64.
    def margins(recs, field):
        return [st[1] - st[3] for rec in recs.values() for st in rec[field]]
    print("\ntop-2 margins per decoder step (rows still decoding):")
    print(f"{'':24}{'steps':>8}{'<1e-2':>8}{'<5e-3':>8}{'<1e-3':>8}{'<5e-4':>8}{'<1e-4':>8}  {'min':>9}")
    series = [(n, margins(r, "steps")) for n, r in zip(names, s2s)] + [("stanza float64", margins(s2s[0], "steps64"))]
    for n, ms in series:
        print(f"{n:24}{len(ms):8}" + "".join(f"{sum(m < t for m in ms):8}" for t in (1e-2, 5e-3, 1e-3, 5e-4, 1e-4)) + f"  {min(ms):9.3e}")
    print("edit-class margins (top-1 minus top-2 logit):")
    for n, r in zip(names, s2s):
        em = [sorted(rec["el"])[2] - sorted(rec["el"])[1] for rec in r.values()]
        print(f"{n:24} min {min(em):.3e}, {sum(m < 1e-2 for m in em)} below 1e-2, {sum(m < 1e-3 for m in em)} below 1e-3")

    # The closest calls: the steps with the smallest Stanza float32 margin while every run is still on the same prefix,
    # with each run's margin, and how far each run's margins are from float64's over all such steps.
    chars = None
    if Path(args.vocab).exists():
        chars = json.loads(Path(args.vocab).read_text(encoding="utf-8"))["vocab"]["char"]["_id2unit"]

    def ch(i):
        return repr(chars[i]) if chars and i < len(chars) else f"id {i}"

    def word_at(d, i):
        sents = ref["docs"][d]["sents"]
        sid = 0
        while i >= len(sents[sid]):
            i -= len(sents[sid])
            sid += 1
        return sid, i, sents[sid]

    fields = [(n, r, "steps", "el") for n, r in zip(names, s2s)] + [("stanza float64", s2s[0], "steps64", "el64")]
    aligned, worst = [], {n: 0.0 for n, *_ in fields}
    for k in keys:
        for s in range(len(s2s[0][k]["steps"])):
            sts = [r[k][f][s] if s < len(r[k][f]) else None for _, r, f, _ in fields]
            if any(st is None or st[0] != sts[0][0] or st[2] != sts[0][2] for st in sts):
                break
            for (n, *_), st in zip(fields, sts):
                worst[n] = max(worst[n], abs((st[1] - st[3]) - (sts[-1][1] - sts[-1][3])))
            aligned.append((sts[0][1] - sts[0][3], k, s, sts))
    print("largest |margin - stanza float64 margin| over aligned steps: "
          + ", ".join(f"{n} {v:.2e}" for n, v in worst.items() if n != "stanza float64"))
    print(f"\n{args.closest} closest decoder steps:")
    for m, (d, i), s, sts in sorted(aligned)[:args.closest]:
        sid, k, sent = word_at(d, i)
        print(f"- doc {d} sentence {sid}: {sent[k][0]!r} ({sent[k][1]}), lemma {s2s[0][(d, i)]['lemma']!r}, step {s}: "
              f"top-1 {ch(sts[0][0])}, top-2 {ch(sts[0][2])}; margins "
              + ", ".join(f"{n} {st[1] - st[3]:.4e}" for (n, *_), st in zip(fields, sts)))
        print("  sentence: " + " ".join(x[0] for x in sent))
    print(f"\nclosest edit-class calls:")
    em = sorted((sorted(s2s[0][k]["el"])[2] - sorted(s2s[0][k]["el"])[1], k) for k in keys)
    for m, (d, i) in em[:args.closest // 2]:
        sid, k, sent = word_at(d, i)
        print(f"- doc {d} sentence {sid}: {sent[k][0]!r} ({sent[k][1]}), lemma {s2s[0][(d, i)]['lemma']!r}; edit {s2s[0][(d, i)]['edit']}, "
              "margins " + ", ".join(f"{n} {sorted(r[(d, i)][e])[2] - sorted(r[(d, i)][e])[1]:.4e}" for n, r, _, e in fields))

    # A seq2seq word's lemma is the model's (with --no-dict the document keeps the dictionary's); others the document's.
    lemmas = {n: {} for n in names + ["stanza float64"]}
    for d in good:
        for n, r, recs in zip(names, runs, s2s):
            flat = [w for s in r["docs"][d]["sents"] for w in s]
            for i, w in enumerate(flat):
                rec = recs.get((d, i))
                lemmas[n][(d, i)] = rec["lemma"] if rec else w[4]
                if rec and not r["meta"].get("no_dict"):
                    assert rec["lemma"] == w[4], f"{n}: doc {d} word {i}: record {rec['lemma']!r} vs document {w[4]!r}"
        flat = [w for s in ref["docs"][d]["sents"] for w in s]
        for i, w in enumerate(flat):
            rec = s2s[0].get((d, i))
            lemmas["stanza float64"][(d, i)] = rec["lemma64"] if rec else w[4]
    print("\nlemma flips per pair (words whose lemma differs):")
    allnames = names + ["stanza float64"]
    for a, b in itertools.combinations(allnames, 2):
        n = sum(lemmas[a][k] != lemmas[b][k] for k in lemmas[a])
        print(f"  {a} vs {b}: {n}")
    diff = sorted(k for k in lemmas[names[0]] if len({lemmas[n][k] for n in allnames}) > 1)
    print(f"\n{len(diff)} words whose lemma differs between any two runs")
    for d, i in diff:
        sents = ref["docs"][d]["sents"]
        k, sid = i, 0
        while k >= len(sents[sid]):
            k -= len(sents[sid])
            sid += 1
        w = sents[sid][k]
        print(f"\n- doc {d} sentence {sid} word {k}: {w[0]!r} UPOS {w[1]} XPOS {w[2]}")
        print("  sentence: " + " ".join(x[0] for x in sents[sid]))
        for n in allnames:
            print(f"  {n:24} lemma {lemmas[n][(d, i)]!r}")
        rec0 = s2s[0].get((d, i))
        if rec0 is None:
            print("  dictionary hit (no seq2seq)")
            continue
        recs = [(n, r[(d, i)], "") for n, r in zip(names, s2s)] + [("stanza float64", rec0, "64")]
        step = next((s for s in range(max(len(r["steps" + f]) for _, r, f in recs))
                     if len({tuple(r["steps" + f][s][:1]) if s < len(r["steps" + f]) else None for _, r, f in recs}) > 1), None)
        print(f"  seq2seq; decoder steps diverge at step {step}")
        for n, r, f in recs:
            dec, edit, el = r["dec" + f], r["edit" + f], r["el" + f]
            es = sorted(el)
            line = f"  {n:24} decoded {dec!r} edit {edit} logits [{', '.join(f'{x:.4f}' for x in el)}] margin {es[2] - es[1]:.3e}"
            if step is not None and step < len(r["steps" + f]):
                a, la, b, lb = r["steps" + f][step]
                line += f"; step {step}: top-1 id {a} {la:.5f}, top-2 id {b} {lb:.5f}, margin {la - lb:.3e}"
            print(line)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text")
    t.add_argument("ewt")
    t.add_argument("out")
    s = sub.add_parser("stanza")
    s.add_argument("docs")
    s.add_argument("out")
    s.add_argument("--package", default="default")
    s.add_argument("--models", default=str(Path(__file__).resolve().parent.parent / "models" / "stanza"))
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--no-dict", action="store_true", help="send every word through the seq2seq model")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="the Stanza run first")
    c.add_argument("--closest", type=int, default=20, help="list this many of the closest decoder steps")
    c.add_argument("--vocab", default=str(Path(__file__).resolve().parent.parent / "models" / "converted" / "en" / "lemma" / "combined_nocharlm.json"),
                   help="the lemmatizer's converted checkpoint JSON, for character names")
    args = p.parse_args()
    if args.cmd == "text":
        docs = build_docs(args.ewt)
        Path(args.out).write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        print(f"{len(docs)} documents, {sum(len(d.split()) for d in docs)} whitespace tokens")
    elif args.cmd == "stanza":
        run_stanza(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
