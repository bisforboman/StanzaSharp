"""The Python half of "stanzasharp compare", written next to stanza_resources_en.json in a temporary folder.

  python compare.py TEXT OUT MODELS PACKAGE PROCESSORS VERSION

Runs Stanza on TEXT (UTF-8 bytes, decoded like tools/make_golden.py) with the models in MODELS (Stanza's
<dir>/en/<processor>/<name>.pt layout) and writes JSON to OUT: the CoNLL-U as make_golden.py writes it ("{:C}\\n"), the
processors Stanza ran, the model files it loaded (relative to MODELS) and the load and processing times.
Exit code 3: Python or Stanza is not usable (the message says what to install).
"""
import json
import os
import sys
import time


def model_files(value):
    """Every .pt path in a Stanza pipeline config (strings, the ner lists and the ner dependency dicts)."""
    if isinstance(value, str):
        if value.endswith(".pt"):
            yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from model_files(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from model_files(v)


def main():
    text_path, out_path, models, package, processors, version = sys.argv[1:7]
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

    # Stanza reads <dir>/<lang>/<processor>/<name>.pt; resources.json (here Stanza 1.15.0's English entry, shipped with
    # the tool) maps package and processors to models. download_method=None: nothing is downloaded.
    models = os.path.abspath(models)
    resources = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stanza_resources_en.json")
    start = time.perf_counter()
    nlp = stanza.Pipeline("en", dir=os.path.dirname(models), package=package, processors=processors,
                          download_method=None, resources_filepath=resources, use_gpu=False, logging_level="WARN")
    load = time.perf_counter() - start

    # As bytes, so CR/CRLF reach Stanza unchanged.
    with open(text_path, "rb") as f:
        text = f.read().decode("utf-8")
    start = time.perf_counter()
    with torch.no_grad():
        doc = nlp(text)
    process = time.perf_counter() - start

    files = sorted({os.path.relpath(p, models).replace(os.sep, "/") for p in model_files(nlp.config)})
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"conllu": "{:C}\n".format(doc), "processors": list(nlp.processors), "files": files,
                   "load": load, "process": process}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
