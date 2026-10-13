# Results, round 5: memory after `Process` returns, Linux and Windows

Part of [performance.md](../performance.md)'s measurement history.

Issue #19's reporter runs a long-running service on Linux x64 (glibc, `mcr.microsoft.com/dotnet/aspnet:10.0`), where
what matters is the memory the process keeps between calls, not only its peak. Round 3 measured Windows only and found
the working set stays near the peak after `Process`. `--memory N --calls 2` (both benchmarks) now reports the memory
after each of two calls on the same text; `--verbose` adds the GC heap and, on Linux, glibc's `mallinfo2` and
`/proc/self/smaps_rollup`.

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --memory 5000 --calls 2 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza\en [--verbose] [--no-trim]
tools\.venv\Scripts\python tools\benchmark.py --memory 5000 --calls 2 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza
```

- **Linux:** Docker Desktop (WSL 2) on the same machine. The benchmark is published for `linux-x64` and run on
  `mcr.microsoft.com/dotnet/aspnet:10.0` (Ubuntu 24.04, glibc 2.39); Python on `python:3.12-slim` with
  `tools/requirements.txt` (CPU torch 2.14.1). The `.pt` models are copied into a Docker volume: read through a
  Windows bind mount, loading took minutes (issue #48: one `fstat` per pickle opcode, fixed since). Memory is the
  RSS (`VmRSS`), peaks are `VmHWM`.
- **Windows:** as in round 3 (`.pt` models, workstation GC).
- **0.4.0** is the tag `v0.4.0` with this benchmark; "now" is this branch. `--memory 500`, `5000` and `15000` are
  680, 6,761 and 20,513 words after tokenizing.
- **Times are not comparable:** other jobs kept the host at 100% CPU throughout, and in the container libgomp's
  spinning threads made calls 2–10× slower (some Linux runs use `OMP_WAIT_POLICY=PASSIVE`, which changes no
  memory). Only memory is compared in the tables; the trim's timing was measured later, on a quieter host.

Linux, MB. "Peak" is the peak during call 1 (call 2's in parentheses where it is higher):

| | 0.4.0 | main before | **now** (malloc_trim) | Python |
|---|---:|---:|---:|---:|
| load peak / after load | 1,018 / 1,011 | 568 / 561 | 595 / 588 | 800 / 733 |
| 680 words: peak | 1,124 | 674 (704) | 705 (762) | 841 (857) |
| after call 1 / after call 2 | 952 / 958 | 660 / 651 | 705 / 711 (not trimmed) | 842 / 811 |
| 6,761 words: peak | 2,656 | 1,410 (1,562) | 1,403 (1,515) | 2,225 (2,290) |
| after call 1 / after call 2 | 1,071 / 1,269 | 929 / 947 | **655 / 648** | 889 / 855 |
| 20,513 words: peak | 2,668 | 1,510 (1,597) | 1,490 (1,595) | 2,282 (2,333) |
| after call 1 / after call 2 | 1,163 / 1,193 | 960 / 977 | **642 / 641** | 918 / 961 |

Windows, MB (StanzaSharp's Windows behaviour is unchanged by this round; the last C# column is an environment
setting, see below):

| | 0.4.0 | now | now, `MIMALLOC_PURGE_DELAY=0` | Python |
|---|---:|---:|---:|---:|
| load peak / after load | 968 / 968 | 538 / 537 | 535 / 530 | 733 / 660 |
| 680 words: peak | 1,085 | 702 (712) | 639 (688) | 761 (797) |
| after call 1 / after call 2 | 930 / 922 | 702 / 712 | 594 / 603 | 699 / 703 |
| 6,761 words: peak | 2,628 (2,792) | 1,650 | 1,283 | 2,343 (2,404) |
| after call 1 / after call 2 | 2,082 / 2,773 | 1,620 / 1,601 | **583 / 579** | 713 / 716 |
| 20,513 words: peak | 2,863 | 1,744 (1,781) | 1,358 (1,384) | 2,399 |
| after call 1 / after call 2 | 2,415 / 2,303 | 1,419 / 1,440 | **581 / 596** | 741 / 811 |

Who holds the memory after a call:

- **Not StanzaSharp, and not the GC.** No tensor survives a call (round 3), the second call ends where the first did,
  and the GC heap is 80–125 MB (committed 100–180 MB) before and after. Once the native heap is trimmed, the process
  is back within 60–100 MB of its size after loading.
- **Linux: glibc's malloc.** libtorch allocates CPU tensors with `malloc` (its `c10` allocator does not cache on
  the CPU). glibc serves blocks over its mmap threshold with `mmap` and unmaps them on `free`, but the threshold is
  dynamic: each freed mmapped block raises it to that block's size, up to 32 MB. After the first large tensors are
  freed, nearly every later tensor comes from the arenas, and memory freed there stays in the process unless it is at
  the top of a heap. After a 6,761-word call `mallinfo2` shows 394 MB of arenas of which 299 MB are free, plus
  265 MB mmapped (the large weights). Per-thread arenas are not the problem: `MALLOC_ARENA_MAX=2` changes nothing.
- **Windows: mimalloc inside libtorch.** libtorch 2.10's `c10.dll` has mimalloc 2.2.4 built in for CPU tensors
  (`MIMALLOC_VERBOSE=1` prints its options). It keeps freed pages committed and schedules their purge for later
  (`purge_delay`, 10 ms), but after a call the purge never came: neither waiting up to a second nor allocating and
  freeing a 256 MB tensor or a small one afterwards lowered the working set. Python's torch 2.14.1 has mimalloc
  2.4.1 with the same options and does drop back, so the newer mimalloc (or torch around it) behaves differently.
  `MIMALLOC_PURGE_DELAY=0` purges on every free: the working set after a call drops by 0.8–1 GB and the peak by
  370 MB, as freed pages are no longer counted while new ones are committed.

What changed:

- **Linux: `malloc_trim(0)` after each call of at least 1,000 words** (`NativeHeap`, internal switch
  `PipelineOptions.TrimNativeHeap`, default on; `--no-trim` in the benchmark). It gives the free pages of every
  arena back: 270–320 MB after 6,761 and 20,513 words, leaving the RSS 55–70 MB above its size after loading and
  about 250 MB below Python's. The trim itself takes 23–42 ms after such a call (125–145 ms while the host was at
  full load and the calls took 55–145 s), and the next call faults the pages in again. In three alternated pairs of
  runs (6,761 words, two calls each, calls of about 22 s) the median call took 22.8 s with the trim and 22.4 s
  without, which is within the noise.
  Below 1,000 words a call frees little (15 MB at 680 words) while trimming still walks every arena (2–12 ms under
  load), so it is skipped; one-sentence calls in a loop pay nothing.
- It is found with `NativeLibrary` in `libc.so.6`, so it never runs on Windows, macOS or musl.

Environment settings, Linux, 6,761 words, without the trim:

| setting | peak | after call 1 / 2 |
|---|---:|---:|
| none | 1,410 (1,562) | 929 / 947 |
| `MALLOC_ARENA_MAX=2` | 1,411 (1,571) | 938 / 987 |
| `MALLOC_MMAP_THRESHOLD_=131072` | 1,277 (1,360) | 726 / 785 |
| `MALLOC_TRIM_THRESHOLD_=131072` | 1,341 (1,401) | 764 / 809 |
| `MALLOC_MMAP_THRESHOLD_=131072 MALLOC_ARENA_MAX=2` | 1,342 (1,397) | 764 / 812 |
| the trim (no setting) | 1,414 (1,513) | 646 / 646 |

Setting any `MALLOC_*_THRESHOLD_` turns off the dynamic threshold, so large tensors are mmapped and unmapped again
and the peak is 70–130 MB lower, but tensors under the threshold still fragment the arenas. With the trim no setting
is needed; a fixed mmap threshold could be combined with it to lower the peak, at the cost of an `mmap` per large
tensor (not measured for time).

Windows: `MIMALLOC_PURGE_DELAY=0` costs time. Three alternated runs of three calls each on 6,761 words (host at
97–100% CPU): median 18.5 s per call against 15.2 s without it, about 20% slower. It must be in the environment before
libtorch loads: set it outside the process, or with `Environment.SetEnvironmentVariable` before the first TorchSharp
call (checked: mimalloc then reports `purge_delay: 0`).
