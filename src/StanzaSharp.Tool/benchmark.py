"""The Python half of "stanzasharp benchmark", written next to stanza_resources_en.json in a temporary folder.

  python benchmark.py MODELS PACKAGE PROCESSORS THREADS VERSION      (THREADS 0: torch's default)

Loads Stanza with the models in MODELS (Stanza's <dir>/en/<processor>/<name>.pt layout, as compare.py does) and
writes one JSON line to stdout: {"load", "processors", "python", "torch", "stanza", "threads", "peak_mb"}. Then it
answers one JSON command per stdin line with one JSON line, until stdin closes, so the tool can alternate runs with
StanzaSharp's in the same order and conditions:

  {"op": "run", "text": T}     -> {"stages": {processor: seconds}, "words", "conllu", "peak_mb"}  (one document)
  {"op": "calls", "texts": [...]} -> {"ms": [per call], "peak_mb"}                                 (nlp(text) each)

Exit code 3: Python or Stanza is not usable (the message says what to install).
"""
import json
import os
import platform
import sys
import time


def peak_mb():
    """The process's peak working set (Windows) or maximum resident set size (Linux: KB, macOS: bytes)."""
    if sys.platform == "win32":
        import ctypes
        import ctypes.wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("rest", ctypes.c_size_t * 7)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.c_void_p, ctypes.wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1048576
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1048576 if sys.platform == "darwin" else rss / 1024


def main():
    models, package, processors, threads, version = sys.argv[1:6]
    # Results go to the real stdout as ASCII JSON lines; anything printed by Stanza or torch goes to stderr.
    out = sys.stdout.buffer
    sys.stdout = sys.stderr

    def reply(obj):
        obj["peak_mb"] = peak_mb()
        out.write((json.dumps(obj) + "\n").encode("ascii"))
        out.flush()

    install = f"{sys.executable} -m pip install stanza=={version}"
    try:
        import stanza
        import torch
    except ImportError as e:
        print(f"Stanza is not installed for {sys.executable} ({e}). Install it with: {install}", file=sys.stderr)
        sys.exit(3)
    if stanza.__version__ != version:
        print(f"{sys.executable} has Stanza {stanza.__version__}; StanzaSharp ports Stanza {version}. "
              f"Install it with: {install} (or pass another Python with --python)", file=sys.stderr)
        sys.exit(3)

    if int(threads) > 0:  # 0: torch's default (the physical cores)
        torch.set_num_threads(int(threads))
    resources = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stanza_resources_en.json")
    start = time.perf_counter()
    nlp = stanza.Pipeline("en", dir=os.path.dirname(os.path.abspath(models)), package=package, processors=processors,
                          download_method=None, resources_filepath=resources, use_gpu=False, logging_level="WARN")
    reply({"load": time.perf_counter() - start, "processors": list(nlp.processors), "python": platform.python_version(),
           "torch": torch.__version__, "stanza": stanza.__version__, "threads": torch.get_num_threads()})

    with torch.no_grad():
        for line in sys.stdin.buffer:
            command = json.loads(line.decode("utf-8"))
            if command["op"] == "run":
                # What nlp(text) does, timed per processor (as tools/benchmark.py does).
                doc, stages = command["text"], {}
                for name, processor in nlp.processors.items():
                    start = time.perf_counter()
                    doc = processor.process(doc)
                    stages[name] = time.perf_counter() - start
                reply({"stages": stages, "words": doc.num_words, "conllu": "{:C}\n".format(doc)})
            elif command["op"] == "calls":
                ms = []
                for text in command["texts"]:
                    start = time.perf_counter()
                    nlp(text)
                    ms.append((time.perf_counter() - start) * 1000)
                reply({"ms": ms})


if __name__ == "__main__":
    main()
