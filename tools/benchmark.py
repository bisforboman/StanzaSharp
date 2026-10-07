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
    return memory_counters().PeakWorkingSetSize / 1048576


def working_set_mb():
    if sys.platform != "win32":
        return int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 1048576
    return memory_counters().WorkingSetSize / 1048576


def memory_counters():
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
    return counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=str(ROOT / "models" / "stanza"))
    parser.add_argument("--copies", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--package", default="default", help="Stanza's English package (default_fast has no constituency)")
    parser.add_argument("--documents", type=int, default=0, help="time N one-sentence texts, one by one vs bulk")
    parser.add_argument("--memory", type=int, default=0, help="load, then one nlp(text) call on about N words; report the peaks")
    parser.add_argument("--processors", help="--memory: the processors to load (default: all of the package's)")
    parser.add_argument("--chunk-words", type=int, default=0, help="--memory: one call per part of about K words")
    parser.add_argument("--bulk", action="store_true", help="--memory with --chunk-words: one bulk_process call on the parts")
    parser.add_argument("--calls", type=int, default=1, help="--memory: repeat the call(s) N times, reporting the memory after each")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.documents > 0:
        return time_documents(args)
    if args.memory > 0:
        return measure_memory(args)

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
                 for line in path.read_text(encoding="utf-8").split("\n") if line.startswith("# text = ")]
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


def word_count(s):
    return len([w for w in s.replace("\t", " ").replace("\r", " ").replace("\n", " ").split(" ") if w])


def build_paragraphs(words):
    """The same text as samples/StanzaSharp.Benchmark --memory: corpus.txt and validation*.txt, repeated, cut after
    the paragraph that reaches the word count."""
    golden = ROOT / "tests" / "golden"
    files = ["corpus.txt"] + sorted(p.name for p in golden.glob("validation*.txt"))
    unit = [p.strip("\n") for f in files for p in (golden / f).read_bytes().decode("utf-8").split("\n\n")]
    unit = [p for p in unit if p.strip()]
    result, count = [], 0
    while count < words:
        for p in unit:
            if count >= words:
                break
            result.append(p)
            count += word_count(p)
    return result


def chunk(paragraphs, words):
    parts, current, count = [], [], 0
    for p in paragraphs:
        current.append(p)
        count += word_count(p)
        if count >= words:
            parts.append("\n\n".join(current))
            current, count = [], 0
    if current:
        parts.append("\n\n".join(current))
    return parts



def measure_memory(args):
    paragraphs = build_paragraphs(args.memory)
    start = time.perf_counter()
    nlp = stanza.Pipeline("en", dir=args.models, package=args.package, processors=args.processors, download_method=None,
                          use_gpu=False, logging_level="WARN")
    load = time.perf_counter() - start
    load_peak, after_load = peak_working_set_mb(), working_set_mb()
    after = []
    for call in range(1, args.calls + 1):
        start = time.perf_counter()
        with torch.no_grad():
            if args.chunk_words <= 0:
                docs = [nlp("\n\n".join(paragraphs))]
            else:
                parts = chunk(paragraphs, args.chunk_words)
                docs = nlp.bulk_process(parts) if args.bulk else [nlp(p) for p in parts]
        elapsed = time.perf_counter() - start
        if call == 1:
            seconds = elapsed
        after.append(f"after call {call}  {elapsed:7.2f} s  peak {peak_working_set_mb():6.0f} MB  working set {working_set_mb():6.0f} MB")
    if args.out:
        Path(args.out).write_text("".join("{:C}\n".format(d) for d in docs), encoding="utf-8", newline="\n")
    words = sum(d.num_words for d in docs)
    mode = "one call" if args.chunk_words <= 0 else f"{len(docs)} parts of ~{args.chunk_words} words, " + ("one bulk call" if args.bulk else "one call each")
    print(f"Python Stanza {stanza.__version__} ({args.package}: {args.processors or 'all'}), torch threads {torch.get_num_threads()}: {words} words, {mode}")
    print(f"load          {load:7.2f} s  load peak {load_peak:6.0f} MB  after load {after_load:6.0f} MB")
    print(f"process       {seconds:7.2f} s  peak      {peak_working_set_mb():6.0f} MB  at the end {working_set_mb():6.0f} MB")
    print("\n".join(after))


if __name__ == "__main__":
    main()
