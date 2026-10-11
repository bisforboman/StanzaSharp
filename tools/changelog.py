"""Unreleased CHANGELOG entries live one per file in changes/<section>/<slug>.md, so parallel pull requests never
conflict on CHANGELOG.md. This folds them in at release time.

    python tools/changelog.py preview          # print the next version's section
    python tools/changelog.py release 1.1.0    # write it into CHANGELOG.md and delete the files

Standard library only. Entries keep their file's text (Markdown bullets) and are ordered by file name.
"""

import datetime
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGES = ROOT / "changes"
CHANGELOG = ROOT / "CHANGELOG.md"
# Keep a Changelog's order.
SECTIONS = ["added", "changed", "deprecated", "removed", "fixed", "security"]


def section_text() -> str:
    unknown = [d.name for d in CHANGES.iterdir() if d.is_dir() and d.name not in SECTIONS]
    if unknown:
        sys.exit(f"Unknown section folder(s) in changes/: {', '.join(unknown)}; use {', '.join(SECTIONS)}")
    parts = []
    for name in SECTIONS:
        files = sorted((CHANGES / name).glob("*.md")) if (CHANGES / name).is_dir() else []
        if files:
            entries = [f.read_text(encoding="utf-8").strip() for f in files]
            parts.append(f"### {name.capitalize()}\n" + "\n".join(entries) + "\n")
    return "\n".join(parts)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) >= 2 and sys.argv[1] == "preview":
        print(section_text() or "(no unreleased entries)")
    elif len(sys.argv) == 3 and sys.argv[1] == "release":
        text = section_text()
        if not text:
            sys.exit("No entries in changes/")
        version, today = sys.argv[2], datetime.date.today().isoformat()
        log = CHANGELOG.read_text(encoding="utf-8")
        at = log.index("\n## [")  # the newest released version
        log = log[: at + 1] + f"## [{version}] - {today}\n\n{text}\n" + log[at + 1 :]
        CHANGELOG.write_text(log, encoding="utf-8", newline="\n")
        for name in SECTIONS:
            for f in (CHANGES / name).glob("*.md") if (CHANGES / name).is_dir() else []:
                f.unlink()
        print(f"CHANGELOG.md: added {version}; changes/ emptied")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
