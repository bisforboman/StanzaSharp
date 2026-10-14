"""The Python half of "stanzasharp compare", written next to stanza_resources_<lang>.json in a temporary folder.

  python compare.py TEXT OUT MODELS LANG PACKAGE PROCESSORS VERSION

Runs Stanza's LANG pipeline on TEXT (UTF-8 bytes, decoded like tools/make_golden.py) with the models in MODELS (Stanza's
<processor>/<name>.pt files, in a folder of any name) and writes JSON to OUT: the CoNLL-U as make_golden.py writes it
("{:C}\\n"), the processors Stanza ran, the model files it loaded (<processor>/<name>.pt) and the load and processing
times.
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


def load_stanza(models, lang, package, processors):
    """stanza.Pipeline(lang) on the .pt files in MODELS, a folder of any name, with the language's entry of Stanza's
    resources.json shipped with the tool (stanza_resources_<lang>.json); download_method=None: nothing is downloaded.
    Stanza reads <dir>/<lang>/<processor>/<name>.pt, so a folder not named after the language is linked under that name
    in this script's (temporary) folder while Stanza loads."""
    import stanza
    here = os.path.dirname(os.path.abspath(__file__))
    models = os.path.abspath(models)
    link = None
    if os.path.basename(models) != lang:
        link = os.path.join(here, lang)
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(models, link)  # unlike a symlink, a junction needs no privilege or developer mode
        else:
            os.symlink(models, link)
    try:
        return stanza.Pipeline(lang, dir=os.path.dirname(link or models), package=package, processors=processors,
                               download_method=None, resources_filepath=os.path.join(here, f"stanza_resources_{lang}.json"),
                               use_gpu=False, logging_level="WARN")
    finally:
        if link:
            os.unlink(link)  # the link only: its target stays


def main():
    text_path, out_path, models, lang, package, processors, version = sys.argv[1:8]
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

    start = time.perf_counter()
    nlp = load_stanza(models, lang, package, processors)
    load = time.perf_counter() - start

    # As bytes, so CR/CRLF reach Stanza unchanged.
    with open(text_path, "rb") as f:
        text = f.read().decode("utf-8")
    start = time.perf_counter()
    with torch.no_grad():
        doc = nlp(text)
    process = time.perf_counter() - start

    files = sorted({"/".join(p.replace(os.sep, "/").split("/")[-2:]) for p in model_files(nlp.config)})
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"conllu": "{:C}\n".format(doc), "processors": list(nlp.processors), "files": files,
                   "load": load, "process": process}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
