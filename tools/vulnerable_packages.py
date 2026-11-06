"""Reads `dotnet list package --vulnerable --include-transitive --format json` reports and lists what they found.

    python tools/vulnerable_packages.py REPORT.json [REPORT.json ...]

Prints a Markdown table (package, version, severity, advisory, direct or transitive, projects) of every vulnerable
package, one row per package, version and advisory. Exits 0 when none is found, 1 when some are, 2 when a report is
unreadable or names a problem (a failed restore, say): the weekly vulnerable-packages.yml then fails either way.
"""

import json
import os
import sys


def main(paths: list[str]) -> int:
    if not paths:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    root = os.getcwd()
    found: dict[tuple[str, str, str, str], dict] = {}
    problems: list[str] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8-sig") as f:
                report = json.load(f)
        except (OSError, ValueError) as e:
            problems.append(f"{path}: {e}")
            continue
        for problem in report.get("problems", []):
            problems.append(f"{path}: {problem.get('level', '')} {problem.get('text', '')}".strip())
        for project in report.get("projects", []):
            name = os.path.relpath(project.get("path", "?"), root).replace(os.sep, "/")
            for framework in project.get("frameworks", []):
                for kind, key in (("direct", "topLevelPackages"), ("transitive", "transitivePackages")):
                    for package in framework.get(key, []):
                        for vulnerability in package.get("vulnerabilities", []):
                            row = found.setdefault(
                                (
                                    package.get("id", "?"),
                                    package.get("resolvedVersion", "?"),
                                    vulnerability.get("severity", "?"),
                                    vulnerability.get("advisoryurl", "?"),
                                ),
                                {"kinds": set(), "projects": set()},
                            )
                            row["kinds"].add(kind)
                            row["projects"].add(f"{name} ({framework.get('framework', '?')})")
    for problem in problems:
        print(f"Problem: {problem}")
    if not found:
        if not problems:
            print("No vulnerable package found.")
        return 2 if problems else 0
    print("| Package | Version | Severity | Advisory | Reference | Projects |")
    print("|---|---|---|---|---|---|")
    for (package, version, severity, url), row in sorted(found.items()):
        kinds = ", ".join(sorted(row["kinds"]))
        projects = "<br>".join(sorted(row["projects"]))
        print(f"| {package} | {version} | {severity} | {url} | {kinds} | {projects} |")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
