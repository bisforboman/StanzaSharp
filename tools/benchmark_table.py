#!/usr/bin/env python3
"""Combines `stanzasharp benchmark --json` results into one Markdown table (.github/workflows/benchmark.yml).

  python tools/benchmark_table.py DIR > benchmarks.md

DIR holds one folder per runner, named benchmark-<runner>, each with benchmark.json and benchmark.md (the block the
command printed). The output is the table, one row per runner, then every runner's block.
"""
import json
import sys
from pathlib import Path


def row(runner, r):
    m, cs, py = r["Machine"], r["StanzaSharp"], r["Stanza"]
    cores = f'{m["Cores"]}/{m["LogicalProcessors"]}' if m["Cores"] else f'?/{m["LogicalProcessors"]}'

    def both(key, fmt):
        return format(cs[key], fmt) + (" / " + format(py[key], fmt) if py else "")

    def words_per_second(side):
        return f'{r["Words"] / side["Total"]:,.0f}' if side else "-"
    speed_up = f'{py["Total"] / cs["Total"]:.2f}x' if py else "-"
    return (f'| {runner} | {m["Cpu"]} | {cores} | {m["Simd"]} | {both("Threads", "d")} | {words_per_second(cs)} '
            f'| {words_per_second(py)} | {speed_up} | {both("CallMedian", ".1f")} | {both("PeakMB", ",.0f")} |')


def main():
    folders = sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_dir() and p.name.startswith("benchmark-"))
    results = [(f.name[len("benchmark-"):], json.loads((f / "benchmark.json").read_text(encoding="utf-8")))
               for f in folders if (f / "benchmark.json").exists()]
    if not results:
        sys.exit("No benchmark.json found")
    first = results[0][1]
    print("## StanzaSharp vs Python Stanza on GitHub-hosted runners\n")
    print(f'{first["Package"]} package ({first["Processors"]}), {first["Words"]:,} words; StanzaSharp {first["Machine"]["StanzaSharp"]}. '
          "Pairs are StanzaSharp / Python Stanza. Shared virtual machines: compare runs of the same day only; macOS runners "
          "are virtualized (Apple Silicon).\n")
    print("| runner | CPU | cores/logical | SIMD | threads | StanzaSharp words/s | Stanza words/s | speed-up "
          "| per call, median ms | peak memory MB |")
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|")
    for runner, r in results:
        print(row(runner, r))
    for f in folders:
        md = f / "benchmark.md"
        if md.exists():
            print(f"\n## {f.name[len('benchmark-'):]}\n")
            print(md.read_text(encoding="utf-8").replace("### StanzaSharp benchmark\n\n", "", 1).rstrip())


if __name__ == "__main__":
    main()
