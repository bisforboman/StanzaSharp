"""Round 34 (docs/performance/round-34-load-packed-cache.md; prototype branch): Pipeline.Load split by stage, under
several settings, each process fresh, the settings alternating within each round.

    python tools/load_profile.py --build out/jit --rounds 5 --config "en default pt=models/stanza/en" \
        --config "en fast pt=models/stanza/en --package default_fast" [--cold "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"]

Settings (fixed): main's packing loop, the faster serial loop, parallel packing, and the packed cache (hit, read into
arrays / mapped / mapped without the GCs). Cache settings are primed with one miss per config first (reported as the
"miss" row). Needs the prototype's instrumentation (`stages=` on stderr of `StanzaSharp.Benchmark startup`).
"""
import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

SETTINGS = [
    ('main loop', {'STANZASHARP_PACK': 'old'}),
    ('fast loop', {}),
    ('parallel', {'STANZASHARP_PACK': 'parallel'}),
    ('dict reader', {'STANZASHARP_DICT': 'reader'}),
    ('prefetch', {'STANZASHARP_PREFETCH': '1'}),
    ('cache read', {'STANZASHARP_PACK_CACHE': '{cache}'}),
    ('cache map', {'STANZASHARP_PACK_CACHE': '{cache}', 'STANZASHARP_PACK_CACHE_MODE': 'map'}),
    ('cache map, no GC', {'STANZASHARP_PACK_CACHE': '{cache}', 'STANZASHARP_PACK_CACHE_MODE': 'map', 'STANZASHARP_HIT_NO_GC': '1'}),
]
KEYS = ['load', 'call1', 'cpu', 'peak', 'jit', 'Model', 'Parse', 'Vocab', 'Tensor', 'Pack', 'Prep', 'Dict', 'Gc', 'CacheRead', 'CacheWrite']


def run(build, models, extra, env, cold):
    if cold:
        subprocess.run(cold, shell=True, check=True)
    cmd = ['dotnet', os.path.join(build, 'StanzaSharp.Benchmark.dll'), 'startup', '--models', models, '--calls', '1'] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
    out = p.stdout + p.stderr
    m = re.search(r'load=(\S+) jit=\d+/(\S+) cpu=(\S+) peak=(\S+) calls=([\d.]+)/', out)
    st = re.search(r'stages=(\S+)', out)
    h = re.search(r'hash=(\w+)', out)
    if not m or not st:
        sys.exit(f'{cmd} {env}:\n{out}')
    r = dict(load=float(m[1]), jit=float(m[2]), cpu=float(m[3]), peak=float(m[4]), call1=float(m[5]), hash=h[1])
    r.update({k: float(v) for k, v in (x.split(':') for x in st[1].split(','))})
    r['models'] = re.findall(r'^model .*$', out, re.M)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', required=True)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--config', action='append', required=True, help='NAME=MODEL_DIR [startup args]')
    p.add_argument('--cold', help='a shell command run before each process (drops the OS file cache)')
    p.add_argument('--settings', help='comma-separated setting names (default: all)')
    a = p.parse_args()
    configs = []
    for c in a.config:
        name, rest = c.split('=', 1)
        models, *extra = rest.split()
        configs.append((name, models, extra))
    settings = [s for s in SETTINGS if not a.settings or s[0] in a.settings.split(',')]
    root = tempfile.mkdtemp(prefix='packcache-')

    def env_of(setting, config):
        cache = os.path.join(root, re.sub(r'\W+', '_', config))
        return {k: v.replace('{cache}', cache) for k, v in setting[1].items()}

    res = {}
    for cname, models, extra in configs:  # prime each cache: one miss per config
        r = run(a.build, models, extra, env_of(next(s for s in SETTINGS if s[0] == 'cache read'), cname), a.cold)
        res.setdefault(('cache miss (writes)', cname), []).append(r)
        print('miss', cname, {k: r[k] for k in KEYS}, file=sys.stderr, flush=True)
    for i in range(a.rounds):
        for s in settings[i % len(settings):] + settings[:i % len(settings)]:  # rotated: no setting always first
            for cname, models, extra in configs:
                r = run(a.build, models, extra, env_of(s, cname), a.cold)
                res.setdefault((s[0], cname), []).append(r)
                print(i, s[0], cname, {k: r[k] for k in KEYS}, file=sys.stderr, flush=True)

    med = statistics.median
    print('| config | setting | load ms | call 1 | load + call 1 | load CPU | peak MB | JIT ms | model thread ms | parse '
          '| vocab | lemma dict | tensor read | pack | prep | GC | cache read | cache write | other | CoNLL-U |')
    print('|---|---|' + '---:|' * 18 + '---|')
    hashes = {}
    for cname, _, _ in configs:
        for sname in ['cache miss (writes)'] + [s[0] for s in settings]:
            rs = res.get((sname, cname))
            if not rs:
                continue
            v = {k: med(r[k] for r in rs) for k in KEYS}
            other = v['Model'] - sum(v[k] for k in ['Parse', 'Vocab', 'Tensor', 'Pack', 'Prep', 'Dict', 'Gc', 'CacheRead', 'CacheWrite'])
            hs = sorted({r['hash'] for r in rs})
            hashes.setdefault(cname, set()).update(hs)
            total = med(r['load'] + r['call1'] for r in rs)
            print(f"| {cname} | {sname} | {v['load']:,.0f} | {v['call1']:,.0f} | {total:,.0f} | {v['cpu']:,.0f} | {v['peak']:,.0f} "
                  f"| {v['jit']:,.0f} | {v['Model']:,.0f} | {v['Parse']:,.0f} | {v['Vocab']:,.0f} | {v['Dict']:,.0f} | {v['Tensor']:,.0f} "
                  f"| {v['Pack']:,.0f} | {v['Prep']:,.0f} | {v['Gc']:,.0f} | {v['CacheRead']:,.0f} | {v['CacheWrite']:,.0f} "
                  f"| {other:,.0f} | {', '.join(hs)} |")
    print()
    print(f'{a.rounds} processes per setting and config, alternating (the order rotated each round); medians; thread ms are summed over the load lanes. '
          + ('Cold: the OS file cache dropped before each process. ' if a.cold else 'Warm OS file cache. ')
          + ('The CoNLL-U was identical within each config.' if all(len(h) == 1 for h in hashes.values())
             else '**The CoNLL-U differed between settings.**'))
    first = res[('main loop' if any(s[0] == 'main loop' for s in settings) else settings[0][0], configs[0][0])][0]
    print()
    print(f'Per model (thread ms), {configs[0][0]}, one process:')
    print()
    for line in first['models']:
        print(f'    {line}')
    sizes = {c: sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
             for c in os.listdir(root) if os.path.isdir(d := os.path.join(root, c))}
    print()
    print('Cache size: ' + ', '.join(f'{c} {s / 1048576:,.0f} MB' for c, s in sizes.items()))
    shutil.rmtree(root, ignore_errors=True)
    return 0 if all(len(h) == 1 for h in hashes.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
