# Results, round 41: the tagger's CharlmCache copies, rented (draft, waiting on the owner)

Part of [performance.md](../performance.md)'s measurement history. Status: **not merged**. The memory left after a
call crossed the line the owner set for this round (a large retained set that the README's idle-GC advice doesn't
cover), so the numbers below are for the owner's decision. The idea file stays until then.

The idea ([ideas/charlm-cache-rented-copies.md](ideas/charlm-cache-rented-copies.md), from
[round 38](round-38-long-text-garbage-left.md)): the managed tagger copied each cached sentence's forward and backward
charlm outputs into new arrays, 8 KB per word up to `CharlmCache.MaxWords` (32,768), so about 256 MB of garbage per
long call. Renting the copies from `ArrayPool<float>.Shared` and returning them when the call's cache is disposed lets
the next call reuse them.

## Two ways to rent

1. **Hand over the batch buffers.** `PosNet.CharlmFeatures` already rents its whole batch's forward, backward and
   final-cell buffers. The cache could keep them (entries as slices) and return them at `Dispose`, with no copy at all.
   Measured and dropped: the pool rounds each buffer up to a power of two (a 5,000-word batch uses 20 MB of a 32 MB
   array), the cache then holds about 13 of them, and later processors find the pool's 32 MB bucket empty and
   allocate their own. Peak +47 MB after call 1 and +103 MB after call 2, and the first call allocates +150 MB.
2. **Copy into rented 4 MB chunks** (this branch). `CharlmCache.TryAdd` copies each sentence's rows end to end into
   `float[1 << 20]` chunks rented from the pool. That's a power of two, so nothing is rounded up. A sentence longer
   than a chunk gets its own array. The copy is the same memcpy `ToArray` did. `Dispose` returns the chunks; the
   pipeline's `using` disposes the cache on cancellation too. Entries are `ArraySegment<float>`s, so `TryGetArrays` and
   `TryGetBackward` return segments, and the readers slice them: depparse copies `backward` and `finalC` by their
   counts, the TorchSharp conversion builds tensors from the segment, and sentiment takes each sentence's forward
   and backward segments separately (they sit at different offsets now; an earlier draft read both at the forward
   offset, and the whole-output hash caught it). `Release` drops the entry but frees its chunk space only at `Dispose`.

## Measured

Windows, Ryzen 7 5800X, converted models, `--threads 8`, all eight processors. Each run was a fresh process, base
(`40691a5`) and branch alternating. The machine was shared with other work, so times move ±30% between runs; no time
change is claimed below.

**Bytes per call** (`--memory 150000 --verbose --calls 2`, `DOTNET_SYSTEM_BUFFERS_ARRAYPOOL_TRIMSHARED=false` as in
round 38, 201,313 words; base twice, handover once, chunks once alone plus once beside a stray process, whose
memory numbers agreed):

| | call 1 allocated | warm call: pos | warm call: all eight | peak after call 1 / 2 |
|---|---:|---:|---:|---:|
| base | 2,038 MB | 430 MB | 1,409 MB (7.0 KB/word) | 1,825–1,833 / 1,957–1,959 MB |
| handover | 2,189 MB | 161 MB | 1,141 MB | 1,872 / 2,060 MB |
| **chunks** | **2,033 MB** | **161 MB** | **1,141 MB (5.7 KB/word)** | **1,817 / 1,941 MB** |

So chunks remove the 268 MB per warm call and don't raise the peak. The rest of the warm call is unchanged
(tokenize 148, mwt 52, lemma 89, constituency 448, depparse 119, sentiment 32, ner 92 MB).

**After the call** (pool trimming on, one call, then the benchmark's `--idle-gc`: an aggressive GC at once, a full GC
61 s later, and an aggressive GC right after it, the README's advice):

| working set | base | chunks |
|---|---:|---:|
| 201,313 words: at the end | 1,817 MB | 1,817 MB |
| aggressive GC at once | 1,345 MB | 1,604 MB |
| after 61 s, aggressive GC | **869 MB** | **1,095 MB** (+226) |

The pool doesn't give the chunks up at the first idle gen2 GC. `ArrayPool<T>.Shared` trims only on gen2 GCs: per
bucket and partition, the first trim after a minute drops one or two arrays, and the next comes 15 s later. A
temporary loop (an aggressive GC every 15 s after the idle minute; `--memory 40000`, 54,056 words, which also fills the
cache) shows the drain:

| working set | base | chunks |
|---|---:|---:|
| after 61 s, aggressive GC | 760 MB | 985 MB |
| 91 s | 667 MB (settled) | 915 MB |
| 241 s | 666 MB | 826 MB |
| 436 s | 667 MB | 722 MB (−8 MB per GC: two chunks) |

Getting back to base takes about 30 gen2 GCs over about 9 minutes. An idle service makes no gen2 GCs on its own, so
in practice the memory stays until something triggers them (more allocation, or memory pressure, under which the pool
trims more per GC and sooner; not measured here).

**Concurrent long calls** (`--memory 40000 --concurrent 4 --calls 2 --idle-gc`, a benchmark option added here: four
threads each process the 54,056-word text at once, each with its own cache):

| | allocated, call 1 | peak after call 1 / 2 | aggressive GC at once | after 61 s, aggressive GC |
|---|---:|---:|---:|---:|
| base | 4,545 MB | 4,043 / 4,629 MB | 1,704 MB | 1,261 MB |
| chunks | 4,349 MB | 3,777 / 4,399 MB | 2,591 MB | 2,117 MB (+856) |

The peak is lower (the copies are reused instead of piling up as garbage), but each call in flight leaves its ~256 MB
of chunks in the pool: four callers leave about 1 GB, and it drains at the same two chunks per 15 s.

**Short texts** (`--per-call 200`, one sentence per call, two pairs): allocation 0.86 → 0.75 MB per call (the
sentence's copy goes into a pooled chunk); median 42.8 / 34.2 ms base vs 40.2 / 32.5 ms, within the noise.
`--per-call 50 --concurrent 8 --with-long`, two pairs: 15.9 / 15.5 calls/s base vs 16.1 / 15.1. (A first pair taken
under heavy machine load, 10.2 vs 7.2, didn't repeat.)

## Exactness

Only where the cached floats live changed, not their values, so output can't change, and it didn't: SHA-256 of the
all-eight CoNLL-U `CD3C66EC…` (201,313 words) and `B1FB6B1E…` (54,056 words), both round 38's. Targeted tests on the
branch (151, all pass): `NnTests`, `PipelineTests` (with `OneSentenceDocuments_EqualTheUncachedPipeline`),
`DepparseTests` (the `ContinueTheTaggers` cases), `SentimentTests`, `NerTests`, `ConstituencyTests`, `InputModeTests`,
`ConcurrencyTests` (cancellation leaks nothing: the cache is a `using` in `AnnotateWithModels`) and the
`ManagedBackendTests` cache, POS and sentiment cases.

## For the owner

- **Ship as is**: −268 MB of garbage per long warm call, peak unchanged or lower, but up to ~256 MB per concurrent long
  call stays in the pool after the call, and the README's "one aggressive GC a minute after the last call" then gets
  back only part of it (another ~9 minutes of repeated GCs, or memory pressure). The README would need to say so.
- **Fewer, larger chunks** (say 32 or 64 MB): the pool drains them in a few GCs instead of ~30, at the cost of up to
  one chunk's tail per call. Which partitions the returns land in depends on the threads, so it needs measuring.
- **A chunk store of our own** with its own drop-after-idle: process-wide state, against the round 11 decision to keep
  nothing after a call.
- **Drop it**: this file is the record, and the idea file goes.
