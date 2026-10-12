"""Startup cost under several settings, each run in a fresh process (docs/performance.md, round 17).

Runs `StanzaSharp.Benchmark startup` (samples/StanzaSharp.Benchmark/Startup.cs) from one or more builds under each
setting, alternating settings within each round, and prints a Markdown table of medians.

    python tools/startup.py --models models/stanza/en --rounds 5 --text one --calls 6 \
        jit=out/jit r2r=out/r2r "jit-tc0=out/jit@DOTNET_TieredCompilation=0"

A setting is NAME=BUILD_DIR[@VAR=VALUE[,VAR=VALUE...]]; BUILD_DIR holds StanzaSharp.Benchmark.dll.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', required=True)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--text', default='one')
    p.add_argument('--calls', type=int, default=6)
    p.add_argument('--threads', type=int)
    p.add_argument('settings', nargs='+')
    a = p.parse_args()

    settings = []
    for s in a.settings:
        name, rest = s.split('=', 1)
        build, _, env = rest.partition('@')
        settings.append((name, build, dict(kv.split('=', 1) for kv in env.split(',') if kv)))

    results = {name: [] for name, _, _ in settings}
    for r in range(a.rounds):
        for name, build, env in settings:
            cmd = ['dotnet', os.path.join(build, 'StanzaSharp.Benchmark.dll'), 'startup', '--models', a.models,
                   '--text', a.text, '--calls', str(a.calls)]
            if a.threads:
                cmd += ['--threads', str(a.threads)]
            out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
            m = re.search(r'load=(\S+) jit=(\d+)/(\S+) cpu=(\S+) calls=(\S+) hash=(\w+)', out.stdout)
            if not m:
                sys.exit(f'{name}: {out.stdout}{out.stderr}')
            calls = [tuple(float(x) for x in c.split('/')) for c in m.group(5).split(',')]
            results[name].append(dict(load=float(m.group(1)), jm=int(m.group(2)), jt=float(m.group(3)), cpu=float(m.group(4)),
                                      calls=calls, hash=m.group(6)))
            print(f'round {r + 1} {name}: {out.stdout.strip()}', file=sys.stderr, flush=True)

    med = statistics.median
    print(f'| setting | load ms (CPU) | JIT in load: methods / ms | call 1 ms (CPU) | call 2 (CPU) | call 3 | calls 4+ '
          f'| JIT in calls: methods / ms | load + {a.calls} calls | CoNLL-U |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    hashes = set()
    for name, rs in results.items():
        def call(i, k=0):
            return med(r['calls'][i][k] for r in rs)
        rest = med([c[0] for r in rs for c in r['calls'][3:]]) if a.calls > 3 else float('nan')
        jm = med(sum(c[1] for c in r['calls']) for r in rs)
        jt = med(sum(c[2] for c in r['calls']) for r in rs)
        total = med(r['load'] + sum(c[0] for c in r['calls']) for r in rs)
        hs = sorted({r['hash'] for r in rs})
        hashes.update(hs)
        print(f"| {name} | {med(r['load'] for r in rs):,.0f} ({med(r['cpu'] for r in rs):,.0f}) "
              f"| {med(r['jm'] for r in rs):,.0f} / {med(r['jt'] for r in rs):,.0f} "
              f"| {call(0):,.0f} ({call(0, 3):,.0f}) | {call(1):,.0f} ({call(1, 3):,.0f}) | {call(2):,.0f} | {rest:,.1f} "
              f"| {jm:,.0f} / {jt:,.0f} | {total:,.0f} | {', '.join(hs)} |")
    print()
    print(f'{a.rounds} processes per setting, alternating; medians. Text: {a.text}. '
          + ('The CoNLL-U was identical in every process.' if len(hashes) == 1 else '**The CoNLL-U differed between settings.**'))
    return 0 if len(hashes) == 1 else 1


if __name__ == '__main__':
    sys.exit(main())
