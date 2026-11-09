"""Adds languages to src/StanzaSharp/stanza_resources.json, the resources.json StanzaSharp embeds (StanzaResources),
and pins their files' SHA-256s in src/StanzaSharp/model_sha256.json.

    python tools/stanza_resources.py add LANG...
    python tools/stanza_resources.py sha256 [LANG...]
    python tools/stanza_resources.py docs

docs writes docs/languages.md, the table of supported languages, from the file and tests/golden/languages/ (each
folder's <package>.conllu files are its packages; <package>+<models>.conllu files are model choices, not packages); run it after adding a language.

Each language's entry comes from stanza-resources' resources_<version>.json as it is today, and the script prints the
commit stanfordnlp/stanza-<lang> is at now, for ModelDownloader.Revisions: the entry and the commit must be taken the
same day, so the MD5s are those of the files at that commit (Stanford re-uploads models under the same version). Pin
the commit printed; never refresh an existing language's entry without moving its commit too.

sha256 (after pinning the commit in ModelDownloader.Revisions) writes each model file's SHA-256 at the language's
pinned commit, from Hugging Face's LFS pointers, into model_sha256.json (all languages, or only those named, and
multilingual's language identifier): what ModelDownloader checks downloads and VerifyChecksums against (#256).
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "StanzaSharp" / "stanza_resources.json"
SHA256 = ROOT / "src" / "StanzaSharp" / "model_sha256.json"
DOWNLOADER = ROOT / "src" / "StanzaSharp" / "ModelDownloader.cs"


def fetch_json(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def revisions():
    """ModelDownloader.Revisions: language → pinned commit."""
    names = {"Pipeline.English": "en", "Multilingual": "multilingual"}
    pairs = re.findall(r'\[(?:"([\w-]+)"|(Pipeline\.English|Multilingual))\] = "([0-9a-f]{40})"', DOWNLOADER.read_text(encoding="utf-8"))
    return {code or names[name]: sha for code, name, sha in pairs}


def tree_sha256(lang, revision):
    """Each .pt file under models/ of stanfordnlp/stanza-<lang> at revision (processor/name → its LFS SHA-256)."""
    url = f"https://huggingface.co/api/models/stanfordnlp/stanza-{lang}/tree/{revision}/models?recursive=true"
    result = {}
    while url:
        with urllib.request.urlopen(url) as response:
            for entry in json.load(response):
                path = entry["path"]
                if entry["type"] != "file" or not path.endswith(".pt"):
                    continue
                if "lfs" not in entry:
                    sys.exit(f"{lang}: {path} is not an LFS file, so Hugging Face lists no SHA-256 for it")
                result[path[len("models/"):-len(".pt")]] = entry["lfs"]["oid"]
            link = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link") or "")
            url = link.group(1) if link else None
    return result


def write_sha256(resources, langs):
    pinned = revisions()
    table = json.loads(SHA256.read_text(encoding="utf-8")) if SHA256.exists() else {}
    for lang in langs:
        files = tree_sha256(lang, pinned[lang])
        if lang == "multilingual":
            wanted = ["langid/ud"]
        else:
            wanted = [f"{processor}/{name}" for processor, models in resources[lang].items() if isinstance(models, dict)
                      for name, model in models.items() if isinstance(model, dict) and "md5" in model]
        missing = [w for w in wanted if w not in files]
        if missing:
            print(f"{lang}: not at {pinned[lang]}: {', '.join(missing)}", file=sys.stderr)
        table[lang] = {w: files[w] for w in sorted(wanted) if w in files}
        print(f"{lang}: {len(table[lang])} files")
    order = [l for l in resources if isinstance(resources[l], dict)] + ["multilingual"]
    table = {l: table[l] for l in order if l in table}
    SHA256.write_text(json.dumps(table, indent=0, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


PROCESSORS = ["tokenize", "mwt", "pos", "lemma", "depparse", "ner", "sentiment", "constituency"]
# English and German have golden data of their own (tests/golden, tests/golden/de) and more packages than the table shows.
MAIN = {"en": "default, default_fast, default_accurate, treebank packages", "de": "default, default_fast, default_accurate, treebank packages"}


def write_docs(resources):
    rows = []
    for lang, entry in resources.items():
        if not isinstance(entry, dict):
            continue
        golden = ROOT / "tests" / "golden" / "languages" / lang
        packages = MAIN.get(lang) or ", ".join(sorted(p.stem for p in golden.glob("*.conllu") if "+" not in p.stem))  # <package>+<models>: a Models choice
        processors = ", ".join(p for p in PROCESSORS if p in entry["default_processors"])
        name = entry.get("lang_name", lang).replace("_", " ")
        ner = entry["default_processors"].get("ner")
        rows.append(f"| {name} | `{lang}` | {packages} | {processors} | {('`' + ner + '`') if ner else ''} |")
    rows.sort(key=lambda r: r.lower())
    text = (
        "# Supported languages\n\n"
        "Generated by `python tools/stanza_resources.py docs` from `src/StanzaSharp/stanza_resources.json` and\n"
        "`tests/golden/languages/`; don't edit by hand.\n\n"
        f"StanzaSharp runs {len(rows)} of Stanza 1.15.0's languages, each byte for byte against Python Stanza on its golden\n"
        "text ([verification.md](verification.md)). Select one with `PipelineOptions.Language` (the code) and download its\n"
        "models with `ModelDownloader.DownloadAsync(dir, options)` or `stanzasharp download --lang CODE`. The processors are\n"
        "those of the language's `default` package, in Stanza's order.\n\n"
        "| Language | Code | Packages | Processors (`default`) | NER model |\n|---|---|---|---|---|\n"
        + "\n".join(rows) + "\n")
    (ROOT / "docs" / "languages.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"docs/languages.md: {len(rows)} languages")


def main(args):
    if args == ["docs"]:
        write_docs(json.loads(TARGET.read_text(encoding="utf-8")))
        return
    if args[:1] == ["sha256"]:
        resources = json.loads(TARGET.read_text(encoding="utf-8"))
        write_sha256(resources, args[1:] or [l for l in resources if isinstance(resources[l], dict)] + ["multilingual"])
        return
    if len(args) < 2 or args[0] != "add":
        sys.exit(__doc__)
    resources = json.loads(TARGET.read_text(encoding="utf-8"))
    version = resources["stanza_version"]
    upstream = fetch_json(f"https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_{version}.json")
    for lang in args[1:]:
        if lang not in upstream or "default_processors" not in upstream[lang]:
            sys.exit(f"resources_{version}.json has no language '{lang}'")
        resources[lang] = upstream[lang]
        sha = fetch_json(f"https://huggingface.co/api/models/stanfordnlp/stanza-{lang}/revision/v{version}")["sha"]
        print(f'        ["{lang}"] = "{sha}",')
    print("Pin these in ModelDownloader.Revisions, then run: python tools/stanza_resources.py sha256 " + " ".join(args[1:]))
    TARGET.write_text(json.dumps(resources, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1:])
