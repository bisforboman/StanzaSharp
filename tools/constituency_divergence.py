#!/usr/bin/env python3
"""
Constituency near-tie study (docs/backends.md "constituency"): how close are the parser's transition decisions on real
text, and do the backends ever choose differently?

  python tools/lemma_divergence.py text UD_EWT_DIR DOCS.json        # raw text per document from en_ewt-ud-*.conllu
  python tools/constituency_divergence.py stanza DOCS.json OUT.json [--models models/stanza] [--threads N]
  StanzaSharp.Benchmark constituency-divergence --docs DOCS.json --out OUT.json [--backend torch|managed] ...
  python tools/constituency_divergence.py compare stanza.json torch.json managed.json [more C# runs ...]

UD English EWT (CC BY-SA 4.0) is not ours to redistribute: download it outside the repository. Each document is one
Process call of tokenize,mwt,pos,constituency. Every run writes, per document, each sentence's words and XPOS, its tree,
and per parser step the decision: [best legal transition, its score, second-best legal transition, its score] (second
-1 when only one transition is legal). The Stanza run also parses each document's tagged words with a float64 copy of
the parser (charlms included), from its own float64 states (the *64 fields).
"""
import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path


def decision(row, legal):
    """[best, score, second, score] over the legal transitions of a score row (numpy), walking it best first."""
    import numpy as np
    out = []
    for j in np.argsort(-row, kind="stable"):
        if legal(int(j)):
            out += [int(j), float(row[j])]
            if len(out) == 4:
                return out
    return out + [-1, 0.0] if out else [-1, 0.0, -1, 0.0]


def hook(model, records):
    """Records each state's decisions under its word queue (records: id -> [words, steps, queue])."""
    queues, predict = model.initial_word_queues, model.predict

    def initial_word_queues(tagged_word_lists):
        result = queues(tagged_word_lists)
        for tagged, queue in zip(tagged_word_lists, result):
            records[id(queue)] = [tuple(w.children[0].label for w in tagged), [], queue]  # the queue keeps the id unique
        return result

    def p(states, is_legal=True):
        predictions, transitions, scores = predict(states, is_legal)
        rows = predictions.detach().cpu().double().numpy()
        for k, state in enumerate(states):
            records[id(state.word_queue)][1].append(
                decision(rows[k], lambda j: model.transitions[j].is_legal(state, model)))
        return predictions, transitions, scores
    model.initial_word_queues, model.predict = initial_word_queues, p


def by_sentence(records, sentences):
    """Each sentence's steps, matching the records by words (equal sentences have equal steps)."""
    pool = {}
    for words, steps, _ in records.values():
        pool.setdefault(words, []).append(steps)
    return [pool[tuple(s)].pop() for s in sentences]


def run_stanza(args):
    import stanza
    import torch

    texts = json.loads(Path(args.docs).read_text(encoding="utf-8"))
    if args.threads:
        torch.set_num_threads(args.threads)
    nlp = stanza.Pipeline("en", dir=args.models, processors="tokenize,mwt,pos,constituency",
                          download_method=None, use_gpu=False, logging_level="WARN")
    proc = nlp.processors["constituency"]
    model32 = proc._model
    model64 = copy.deepcopy(model32).double()
    for m in model64.modules():  # tensors held as plain attributes
        for k, v in list(vars(m).items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                setattr(m, k, v.double())
    rec32, rec64 = {}, {}
    hook(model32, rec32)
    hook(model64, rec64)

    process = proc.process
    stage = [0.0]

    def timed(document):
        clock = time.perf_counter()
        result = process(document)
        stage[0] += time.perf_counter() - clock
        return result
    proc.process = timed

    nlp("Warm up the pipeline.")
    out_docs, seconds, study = [], 0.0, 0.0
    stage[0] = 0.0
    for n, text in enumerate(texts):
        rec32.clear()
        rec64.clear()
        clock = time.perf_counter()
        with torch.no_grad():
            doc = nlp(text)
        seconds += time.perf_counter() - clock
        clock = time.perf_counter()
        sents = [s for s in doc.sentences if s.words]
        tagged = [[(w.text, w.xpos) for w in s.words] for s in sents]
        words = [[w for w, _ in s] for s in tagged]
        steps = by_sentence(rec32, words)
        out_docs.append({"sents": [[[w, t] for w, t in s] for s in tagged],
                         "trees": ["{}".format(s.constituency) for s in sents], "steps": steps})
        if args.no_f64:
            study += time.perf_counter() - clock
            continue
        # float64: the same tagged words, sorted as ConstituencyProcessor does, through the float64 parser.
        order = sorted(range(len(tagged)), key=lambda i: len(tagged[i]), reverse=True)
        torch.set_default_dtype(torch.float64)
        try:
            trees64 = model64.parse_tagged_words([tagged[i] for i in order], proc._batch_size)
        finally:
            torch.set_default_dtype(torch.float32)
        unsorted = [None] * len(order)
        for k, i in enumerate(order):
            unsorted[i] = trees64[k]
        steps64 = by_sentence(rec64, words)
        out_docs[-1].update({"trees64": ["{}".format(t) for t in unsorted], "steps64": steps64})
        study += time.perf_counter() - clock
        if n % 100 == 0:
            print(f"{n}/{len(texts)} documents", file=sys.stderr)
    meta = {"backend": "stanza", "threads": torch.get_num_threads(), "seconds": seconds,
            "constituency_seconds": stage[0], "study_seconds": study, "stanza": stanza.__version__,
            "transitions": [str(t) for t in model32.transitions]}
    Path(args.out).write_text(json.dumps({"meta": meta, "docs": out_docs}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "transitions"}))


def compare(args):
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.runs]
    names = [r["meta"]["backend"] + (f" {r['meta']['variant']}" if r["meta"].get("variant") else "") for r in runs]
    ref = runs[0]
    transitions = ref["meta"]["transitions"]

    def name(j):
        t = transitions[j]
        return t.replace("OpenConstituent(('", "Open(").replace("',))", ")").replace("CloseConstituent", "Close")

    for r, n in zip(runs, names):
        m = r["meta"]
        print(f"{n:28} total {m['seconds']:8.2f} s  constituency {m['constituency_seconds']:7.2f} s  threads {m.get('threads')}")

    # Tokens and tags first: a difference there is a tokenizer/mwt/pos difference, not a parser one.
    bad = set()
    for d in range(len(ref["docs"])):
        for r, n in zip(runs[1:], names[1:]):
            if r["docs"][d]["sents"] != ref["docs"][d]["sents"]:
                bad.add(d)
                a = [w for s in ref["docs"][d]["sents"] for w in s]
                b = [w for s in r["docs"][d]["sents"] for w in s]
                k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"doc {d}: {n} tokens/tags differ from stanza at word {k}: "
                      f"{a[k] if k < len(a) else None} vs {b[k] if k < len(b) else None}")
    good = [d for d in range(len(ref["docs"])) if d not in bad]
    total = sum(len(s) for d in ref["docs"] for s in d["sents"])
    words = sum(len(s) for d in good for s in ref["docs"][d]["sents"])
    sents = sum(len(ref["docs"][d]["sents"]) for d in good)
    print(f"\n{len(ref['docs'])} documents, {total} words; {len(bad)} documents with token/tag differences (excluded below)")
    print(f"compared: {len(good)} documents, {sents} sentences, {words} words")

    # Series: each run's steps, plus Stanza float64.
    series = [(n, r, "steps", "trees") for n, r in zip(names, runs)]
    if "steps64" in ref["docs"][0]:
        series.append(("stanza float64", ref, "steps64", "trees64"))
    allnames = [s[0] for s in series]

    print("\ntree flips per pair (sentences whose tree differs):")
    trees = {n: [t for d in good for t in r["docs"][d][f]] for n, r, _, f in series}
    for a, b in itertools.combinations(allnames, 2):
        print(f"  {a} vs {b}: {sum(x != y for x, y in zip(trees[a], trees[b]))}")

    print("\ndecision margins (best legal minus second legal; steps with one legal transition left out):")
    print(f"{'':28}{'steps':>9}{'decisions':>10}{'<1e-2':>8}{'<1e-3':>8}{'<1e-4':>8}{'<1e-5':>8}  {'min':>9}")
    for n, r, f, _ in series:
        all_steps = [st for d in good for s in r["docs"][d][f] for st in s]
        ms = [st[1] - st[3] for st in all_steps if st[2] >= 0]
        print(f"{n:28}{len(all_steps):9}{len(ms):10}" + "".join(f"{sum(m < t for m in ms):8}" for t in (1e-2, 1e-3, 1e-4, 1e-5))
              + f"  {min(ms):9.3e}")

    # Aligned steps: each sentence's steps while every series has made the same decisions so far.
    aligned, worst, flips = [], {n: 0.0 for n in allnames}, []
    for d in good:
        for i, sent in enumerate(ref["docs"][d]["sents"]):
            ss = [r["docs"][d][f][i] for _, r, f, _ in series]
            for k in range(max(len(s) for s in ss)):
                sts = [s[k] if k < len(s) else None for s in ss]
                if any(st is None or st[0] != sts[0][0] for st in sts):
                    flips.append((d, i, k, sts))
                    break
                if sts[0][2] < 0:
                    continue
                ref64 = sts[-1][1] - sts[-1][3]
                for n, st in zip(allnames, sts):
                    if st[2] == sts[-1][2]:
                        worst[n] = max(worst[n], abs((st[1] - st[3]) - ref64))
                aligned.append((min(st[1] - st[3] for st in sts), d, i, k, sts))
    if allnames[-1] == "stanza float64":
        print("\nlargest |margin - stanza float64 margin| over aligned decisions: "
              + ", ".join(f"{n} {v:.2e}" for n, v in worst.items() if n != "stanza float64"))
    print()
    for a, b in itertools.combinations(range(len(allnames)), 2):
        gap = max(abs((x[4][a][1] - x[4][a][3]) - (x[4][b][1] - x[4][b][3])) for x in aligned if x[4][a][2] == x[4][b][2])
        score = max(max(abs(x[4][a][1] - x[4][b][1]), abs(x[4][a][3] - x[4][b][3])) for x in aligned if x[4][a][2] == x[4][b][2])
        print(f"{allnames[a]} vs {allnames[b]}: largest |margin difference| {gap:.2e}, largest |top-2 score difference| {score:.2e}")

    def text(d, i):
        return " ".join(w for w, _ in ref["docs"][d]["sents"][i])

    print(f"\n{args.closest} closest decisions (aligned):")
    for m, d, i, k, sts in sorted(aligned)[:args.closest]:
        print(f"- doc {d} sentence {i} step {k}: {name(sts[0][0])} over {name(sts[0][2])}; margins "
              + ", ".join(f"{n} {st[1] - st[3]:.3e}" for n, st in zip(allnames, sts)))
        print(f"  {text(d, i)[:200]}")

    print(f"\n{len(flips)} sentences whose decisions diverge between any two series:")
    for d, i, k, sts in flips:
        print(f"- doc {d} sentence {i} step {k}: " + ", ".join(
            f"{n} {name(st[0]) if st else 'done'}" + (f" over {name(st[2])} ({st[1] - st[3]:.3e})" if st and st[2] >= 0 else "")
            for n, st in zip(allnames, sts)))
        print(f"  {text(d, i)[:200]}")
        print("  trees vs stanza: " + "; ".join(f"{n} {'same' if r['docs'][d][tf][i] == ref['docs'][d]['trees'][i] else 'differs'}"
                                                for n, r, _, tf in series[1:]))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stanza")
    s.add_argument("docs")
    s.add_argument("out")
    s.add_argument("--models", default=str(Path(__file__).resolve().parent.parent / "models" / "stanza"))
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--no-f64", action="store_true", help="skip the float64 parse (it doubles the run time)")
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="the Stanza run first")
    c.add_argument("--closest", type=int, default=20, help="list this many of the closest decisions")
    args = p.parse_args()
    if args.cmd == "stanza":
        run_stanza(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
