"""Adds languages to src/StanzaSharp/stanza_resources.json, the resources.json StanzaSharp embeds (StanzaResources).

    python tools/stanza_resources.py add LANG...

Each language's entry comes from stanza-resources' resources_<version>.json as it is today, and the script prints the
commit stanfordnlp/stanza-<lang> is at now, for ModelDownloader.Revisions: the entry and the commit must be taken the
same day, so the MD5s are those of the files at that commit (Stanford re-uploads models under the same version). Pin
the commit printed; never refresh an existing language's entry without moving its commit too.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "StanzaSharp" / "stanza_resources.json"


def fetch_json(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def main(args):
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
    TARGET.write_text(json.dumps(resources, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1:])
