# Results, round 6: memory after `Process` on the managed backend

Part of [performance.md](../performance.md)'s measurement history.

Round 5 measured the TorchSharp backend, whose tensors live in libtorch's native heap. The managed backend (the default
since 0.5) allocates nothing natively: the models and every buffer are .NET arrays. Is the memory kept after a call a
problem there too? Same benchmark mode as round 5 (`--memory N --calls 2 --verbose`, `.pt` models, 8 threads,
workstation GC), with `--backend managed` and `--backend torch`, `tokenize,mwt,pos,constituency` and all eight
processors; `--idle-gc` (new) then runs an aggressive GC at once and another after 61 s idle. 500, 5,000 and 15,000 are
680, 6,761 and 20,513 words after tokenizing. Another job kept the host near 100% CPU, so times are not compared.

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --memory 5000 --calls 2 --backend managed --threads 8 --models models\stanza\en [--processors tokenize,mwt,pos,constituency] [--verbose] [--idle-gc]
```

Windows, MB: load peak / after load, then peak during call 1 (call 2's in parentheses where higher), then the working
set after call 1 / call 2:

| | managed, 4 processors | TorchSharp, 4 processors | managed, all eight | TorchSharp, all eight |
|---|---:|---:|---:|---:|
| load | 561 / 518 | 536 / 536 | 863 / 844 | 870 / 870 |
| 680 words: peak | 561 | 663 (705) | 863 | 1,338 |
| after call 1 / 2 | 551 / 558 | 657 / 667 | 852 / 856 | 1,074 / 1,020 |
| 6,761 words: peak | 711 (755) | 1,053 (1,165) | 2,366 (3,056) | 4,253 (5,319) |
| after call 1 / 2 | 704 / 720 | 1,046 / 1,148 | 2,360 / 3,056 | 4,034 / 4,749 |
| 20,513 words: peak | 933 (1,106) | 1,455 (1,473) | 2,027 (2,197) | 2,844 (3,015) |
| after call 1 / 2 | 927 / 994 | 1,125 / 1,124 | 2,027 / 2,136 | 2,792 / 2,982 |

Linux (Docker Desktop, `aspnet:10.0`, as in round 5), managed, RSS after call 1 / 2 in MB, with the trim and with
`--no-trim`:

| | 4 processors | 4, `--no-trim` | all eight | all eight, `--no-trim` |
|---|---:|---:|---:|---:|
| after load | 544 | 529 | 856 | 856 |
| 680 words | 573 / 584 | 565 / 573 | 863 / 868 | 863 / 868 |
| 6,761 words | 710 / 760 | 726 / 761 | 2,366 / 2,428 | 2,376 / 3,089 |
| 20,513 words | 939 / 1,061 | 947 / 1,038 | 2,018 / 2,345 | 2,025 / 2,070 |

So yes, the managed backend keeps memory too, at about its peak, on both systems. Who holds it:

- **Not the native heap.** On Linux glibc's arenas stay at 13–22 MB in every run, so `malloc_trim` has nothing to give
  back (with and without it the numbers differ only by run-to-run noise). `NativeHeap.Trim` still runs after large
  calls; it returns nothing on this backend and still matters on the TorchSharp backend.
  `MIMALLOC_PURGE_DELAY` is libtorch's, so it does nothing here either.
- **The GC heap: the models, then the call's scratch buffers in `ArrayPool<T>.Shared`.** After loading the GC heap is
  370 MB (4 processors) or 660 MB (all eight): the weights. After a 6,761-word call with all eight it is 2.8 GB, and a
  full GC leaves 2.75 GB: the pool holds the per-batch buffers of the managed kernels (the largest, rounded up to a
  power of two: a charlm LSTM's gate buffer of 386 MB, sentiment's 345 MB input, ...), on purpose, so the next call
  reuses them instead of making the GC find hundreds of MB of garbage. Without the pool (large arrays left to the GC,
  tried) the peak rose from 2.4 to 3.1 GB in call 1 and 3.8 GB in call 2.
- .NET's shared pool gives an array back only from a gen2 GC at least a minute after the array was last used, and the
  GC keeps memory it freed committed until an aggressive GC (or decommits it slowly, over later GCs). So in an idle
  service the memory stays, and no GC right after the call can free it:

| managed, Windows, MB | working set after call 2 | aggressive GC at once | 61 s idle, then a full GC | then an aggressive GC |
|---|---:|---:|---:|---:|
| 4 processors, 6,761 words | 720 | 608 (53 ms) | 604 | **364** (56 ms) |
| 4 processors, 20,513 words | 949 | 686 (134 ms) | 687 | **425** (142 ms) |
| all eight, 6,761 words | 2,417 | 2,229 (50 ms) | 2,226 | **821** (230 ms) |
| all eight, 20,513 words | 2,786 | 1,940 (745 ms) | 1,941 | **663** (196 ms) |

  An aggressive GC a minute after the last call brings the process back to about its size after loading (below it,
  as it also compacts what loading left). That is the guidance in the README: a service that wants the memory back when
  idle calls `GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true)` itself, a minute or more
  after its last call. The library does not: a GC is process-wide, and its cost grows with the application's own heap.
  On the TorchSharp backend it does nothing (all eight, 6,761 words: 4,824 MB after call 2, 4,547 MB after both GCs;
  the GC heap is 40 MB): there the memory is mimalloc's, as round 5 found.
- Splitting the text into ~1,000-word calls does not shrink what stays (all eight, 20,513 words in 15 parts: 1.4–1.8 GB
  after, 0.67 GB after the idle aggressive GC): the parts' batches have different sizes, and the pool keeps arrays of
  every size it was asked for.
- Call 2 sometimes ends higher than call 1 (all eight, 6,761 words: 2.4 → 3.1 GB, GC heap 2.8 → 3.8 GB, in 2 of 4
  runs on Windows and Linux): the shared pool keeps arrays per thread and per core, so it can hold a second set.

Tried and not kept (branch `exp/managed-buffer-pool`, for a decision):
- A private pool (`ArrayPool<T>.Create`) instead of the shared one: one set of buffers, no growth on call 2 or 3
  (2.4–2.5 GB), but it never trims, so the idle aggressive GC no longer helps (2.2 GB after it).
- The private pool dropped after each call of 1,000+ words: an aggressive GC right after the call then brings the
  process to 730 MB (330 ms), but without that GC the next call's buffers are allocated before the old ones are
  collected, and call 2 peaked at 3.2 GB instead of 2.5 GB.
