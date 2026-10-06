#!/usr/bin/env python3
"""
Python Stanza counterpart of samples/StanzaSharp.Benchmark: times each pipeline stage on the same
text (validation.txt, corpus.txt and tokenize_stress.txt from tests/golden, repeated) and prints the same report.

  python tools/benchmark.py [--models models/stanza] [--copies 8] [--runs 3] [--threads N] [--out FILE]
                            [--package default|default_fast] [--documents N]

--documents N times N one-sentence texts instead (the golden validation sentences, cycled): one nlp(text) call
per text vs one nlp.bulk_process(texts) call, as samples/StanzaSharp.Benchmark --documents does.
"""
import argparse
import ctypes
import ctypes.wintypes
import statistics
import sys
import time
from pathlib import Path

import stanza
import torch

STAGES = ["tokenize", "mwt", "pos", "lemma", "constituency", "depparse", "sentiment", "ner"]  # Stanza's order
ROOT = Path(__file__).resolve().parent.parent


def build_text(copies):
    golden = ROOT / "tests" / "golden"
    unit = "\n\n".join((golden / f).read_text(encoding="utf-8") for f in ("validation.txt", "corpus.txt", "tokenize_stress.txt"))
    return "\n\n".join([unit] * copies)


def peak_working_set_mb():
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD)] + \
                   [(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                         "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                         "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    psapi = ctypes.WinDLL("psapi")
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_void_p, ctypes.wintypes.DWORD]
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 1048576


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=str(ROOT / "models" / "stanza"))
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--package", default="default", help="Stanza's English package (default_fast has no constituency)")
    parser.add_argument("--documents", type=int, default=0, help="time N one-sentence texts, one by one vs bulk")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.documents > 0:
        return time_documents(args)

    text = build_text(args.copies)
    start = time.perf_counter()
    stages = STAGES if args.package == "default" else [s for s in STAGES if s != "constituency"]
    nlp = stanza.Pipeline("en", dir=args.models, package=args.package, processors=",".join(stages), download_method=None,
                          use_gpu=False, logging_level="WARN")
    load = time.perf_counter() - start

    assert list(nlp.processors) == stages, list(nlp.processors)
    times = {s: [] for s in stages}
    peaks = {"load": peak_working_set_mb()}  # peak working set after each stage of the warm-up run
    for run in range(args.runs + 1):
        doc = text if run > 0 else build_text(1)  # run 0 warms up on one copy
        for s in stages:
            start = time.perf_counter()
            doc = nlp.processors[s].process(doc)
            if run > 0:
                times[s].append(time.perf_counter() - start)
            if run == 0:
                peaks[s] = peak_working_set_mb()
    if args.out:
        Path(args.out).write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")

    words = doc.num_words
    print(f"Python Stanza {stanza.__version__} ({args.package}), torch threads {torch.get_num_threads()}, {args.copies} copies: "
          f"{len(text)} chars, {len(doc.sentences)} sentences, {words} words, {args.runs} runs")
    print(f"{'load':<14}{load:9.2f} s {'':16} {peaks['load']:8.0f} MB peak")
    total = 0
    for s in stages:
        median = statistics.median(times[s])
        total += median
        print(f"{s:<14}{median:9.2f} s {words / median:10.0f} words/s {peaks[s]:8.0f} MB peak (warm-up)")
    print(f"{'total':<14}{total:9.2f} s {words / total:10.0f} words/s")
    print(f"{'peak memory':<14}{peak_working_set_mb():9.0f} MB (peak working set)")


def time_documents(args):
    golden = ROOT / "tests" / "golden"
    sentences = [line[len("# text = "):] for path in sorted(golden.glob("validation*.conllu"))
                 for line in path.read_text(encoding="utf-8").split("
") if line.startswith("# text = ")]
    texts = [sentences[i % len(sentences)] for i in range(args.documents)]
    nlp = stanza.Pipeline("en", dir=args.models, package=args.package, download_method=None, use_gpu=False, logging_level="WARN")
    with torch.no_grad():
        nlp.bulk_process(texts[:50])  # warm-up
        start = time.perf_counter()
        for text in texts:
            nlp(text)
        alone = time.perf_counter() - start
        start = time.perf_counter()
        nlp.bulk_process(texts)
        bulk = time.perf_counter() - start
    print(f"Python Stanza {stanza.__version__} ({args.package}), torch threads {torch.get_num_threads()}, {len(texts)} documents of one sentence")
    print(f"{'one by one':<14}{alone:9.2f} s {len(texts) / alone:10.0f} docs/s")
    print(f"{'bulk':<14}{bulk:9.2f} s {len(texts) / bulk:10.0f} docs/s")


if __name__ == "__main__":
    main()
