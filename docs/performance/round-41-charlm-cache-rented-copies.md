# Results, round 41: the tagger's CharlmCache copies, rented (dropped)

Part of [performance.md](../performance.md)'s measurement history.

The idea (from [round 38](round-38-long-text-garbage-left.md)): the managed tagger copies each cached sentence's
forward and backward charlm outputs into new arrays, 8 KB per word up to `CharlmCache.MaxWords` (32,768), so about
256 MB of garbage per long call. Renting the copies from `ArrayPool<float>.Shared` and returning them when the call's
cache is disposed would let the next call reuse them.

It works for garbage: a warm long call allocates 268 MB less (7.0 → 5.7 KB/word), and the output is the same bit for
bit. **Dropped by the owner** (2026-10-17) because of what it leaves behind: after the README's advice for an idle
service (an aggressive GC a minute after the last call) the process still holds 226–308 MB more than today per call
that was in flight, and about 1.2 GB more after four concurrent long calls. No chunk size fixed that. The code stays on
the branch `perf/charlm-cache-rented-copies` as the record.

## What was tried

1. **Hand the tagger's batch buffers to the cache.** `PosNet.CharlmFeatures` already rents its whole batch's forward,
   backward and final-cell buffers. The cache would keep them (entries as slices) and return them at `Dispose`, so
   there would be no copy at all. The pool rounds each buffer up to a power of two (a 5,000-word batch uses 20 MB of
   a 32 MB array), so the cache holds about 13 such arrays, and the processors after the tagger find the pool's 32 MB
   bucket empty and allocate their own. Peak +47 MB after call 1 and +103 MB after call 2; the first call allocates
   +150 MB.
2. **Copy into rented chunks.** `CharlmCache.TryAdd` copies each sentence's rows end to end into fixed-size chunks
   rented from the pool (a power of two, so nothing is rounded up; a longer sentence gets its own array), and
   `Dispose` returns them. Entries become `ArraySegment<float>`s. Measured with 4 MB chunks first, then 16, 32 and
   64 MB.

## Measured

Windows, Ryzen 7 5800X, converted models, `--threads 8`, all eight processors, a fresh process per run. The machine
was shared with other work, and times moved ±30% between runs, so no time change is claimed. The 4 MB runs compare
against `40691a5`, the larger chunks against `9c90188` (main after 1.11.0); the two bases agree to within 10 MB.

**Long text, bytes and peak** (`--memory 150000 --verbose --calls 2`, 201,313 words; the warm-call bytes with
`DOTNET_SYSTEM_BUFFERS_ARRAYPOOL_TRIMSHARED=false`, as in round 38):

| | call 1: pos allocated | warm call: pos allocated | warm call: all eight | peak after call 1 / 2 |
|---|---:|---:|---:|---:|
| base | 494 MB | 430 MB | 1,409 MB (7.0 KB/word) | 1,816–1,833 / 1,955–1,959 MB |
| batch buffers | 554 MB | 161 MB | 1,141 MB | 1,872 / 2,060 MB |
| 4 MB chunks | 481 MB | 161 MB | 1,141 MB (5.7 KB/word) | 1,817 / 1,941 MB |
| 16 MB chunks | 433 MB | 161 MB | | 1,821 / 1,956 MB |
| 32 MB chunks | 513 MB | 161 MB | | 1,819 / 2,071 MB |
| 64 MB chunks | 545 MB | 161 MB | | 1,820 / 2,011 MB |

32 MB is the bucket the tagger's 5,000-word batch buffers come from (8M floats), so the cache's chunks and the
tagger's buffers compete for the same arrays and call 2's peak rises by 116 MB. 64 MB still adds 56 MB.

**After one call** (`--memory 40000`, 54,056 words, which fills the cache; `--idle-gc`: an aggressive GC at once, a
full GC 61 s later and an aggressive GC right after it, which is the README's advice; then, from a temporary loop not
kept, an aggressive GC every 15 s):

| working set | base | 4 MB | 16 MB | 32 MB | 64 MB |
|---|---:|---:|---:|---:|---:|
| aggressive GC at once | 1,197–1,203 MB | 1,433 | 1,461 | 1,459 | 1,464 |
| 61 s idle + aggressive GC | **753–760 MB** | **985** (+225) | **1,014** (+261) | **1,012** (+259) | **1,061** (+308) |
| back to base | at 91 s | not by 436 s (722 MB, −8 MB per GC) | at 376 s | at 196 s | at 196 s |

With 201,313 words and 4 MB chunks the same point was 869 → 1,095 MB (+226).

`ArrayPool<T>.Shared` trims only on gen2 GCs. Per bucket and partition, the first trim after an array has been idle
for a minute drops about one array, and the next comes 15 s later. Bigger chunks free more megabytes per trim, so the
drain gets shorter, but the first trim, the one the README's advice triggers, still frees only a chunk or so. An idle
service makes no gen2 GCs on its own, so without more GCs (or memory pressure, not measured here) the memory stays.

**Four concurrent long calls** (`--memory 40000 --concurrent 4 --calls 2 --idle-gc`: four threads each process the
54,056-word text at once, each with its own cache; a benchmark option on the code branch):

| | peak after call 1 / 2 | aggressive GC at once | 61 s idle + aggressive GC |
|---|---:|---:|---:|
| base (`40691a5`) | 4,043 / 4,629 MB | 1,704 MB | 1,261 MB |
| 4 MB chunks | 3,777 / 4,399 MB | 2,591 MB | 2,117 MB (+856) |
| base (`9c90188`) | 4,162 / 4,775 MB | 1,904 MB | 1,125 MB |
| 16 MB chunks | 4,180 / 5,026 MB | 3,034 MB | 2,303 MB (+1,178) |
| 32 MB chunks | 4,153 / 4,781 MB | 3,049 MB | 2,310 MB (+1,185) |

Each call in flight leaves its own ~256 MB of chunks in the pool.

**Short texts** (4 MB chunks only): `--per-call 200`, two pairs: allocation 0.86 → 0.75 MB per call, median 42.8 / 34.2 ms
base vs 40.2 / 32.5 ms (noise). `--per-call 50 --concurrent 8 --with-long`, two pairs: 15.9 / 15.5 calls/s base vs
16.1 / 15.1.

**Not measured** (the owner stopped the runs): the 64 MB four-caller row, short per-call and 8 short + 1 long callers for
16, 32 and 64 MB chunks, and the cost of a chunk's unused tail on short and medium texts (300 and 3,000 words).

## Exactness

The change only moves where the cached floats live. SHA-256 of the all-eight CoNLL-U was the same on every variant:
`CD3C66EC…` (201,313 words) and `B1FB6B1E…` (54,056 words), both round 38's. An early draft of the chunk version read
the sentiment classifier's cached backward rows at the forward rows' offset, which is right only for the batch-buffer
version. The hash caught it before any test run. 151 targeted tests passed on the fixed branch.

## What would still work

The garbage is real, but at most 256 MB per long call, and it costs no time. Anything that keeps it past the call
holds that memory per concurrent caller until the pool lets go. One way remains, not tried: a chunk store of the
cache's own, which drops its chunks after a set idle time. That would be process-wide state, against round 11's
decision to keep nothing after a call.
