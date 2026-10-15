# Results, round 27: a CPU profile of the whole managed pipeline (measurement only)

Part of [performance.md](../performance.md)'s measurement history.

Earlier rounds measured stage times, memory and single kernels. This round profiled where the CPU time of a whole
all-eight call goes, at 1, 4 and 8 threads, and why the benchmark's timed runs sometimes varied between 4.9 and 8.1 s
on this desktop. Nothing was changed. The targets at the end are ideas, each with an estimated gain. Every one of them
must keep the output byte-exact and leave every tolerance as it is.

## Setup

- Windows 11, Ryzen 7 5800X (8 cores, 16 logical processors), .NET 10.0.13, Vector256, commit `1d207aa`, Release build,
  the `.pt` models (`models/stanza/en`), the default package with all eight processors.
- The text is `stanzasharp benchmark`'s: 2,009 words in 163 sentences. A scratch harness (not committed) loads the
  pipeline the way the tool does and calls `Process` on that text repeatedly. Per run it prints the per-processor times
  (the pipeline's Debug log, as the tool reads it), the GC counts per generation, `GC.GetTotalPauseDuration` and the
  bytes allocated.
- The profile used `dotnet-trace collect --profile dotnet-sampled-thread-time,gc-collect --format Speedscope` (dotnet-trace
  10.0.745401). This is EventPipe's thread-time sampler: it samples every managed thread at about 100 Hz. On Windows,
  dotnet-trace's `cpu-sampling` profile exists only for `collect-linux`. Each sample says whether the thread was in
  managed code (`CPU_TIME`) or in native code (`UNMANAGED_CODE_TIME`, which here is almost always a blocking wait).
- Attribution: the caller's stack names the stage (the processor's assembly) and the component (`ManagedCharLanguageModel`,
  or one of `ManagedLstm`, `ManagedHighwayLstm`, `PackedLstm`, `StackLstm`, `ManagedCharacterModel`). A pool worker's
  stack only shows `Pool.Work → Region.Drain → task`, so a worker sample takes the stage and component the caller had
  at that moment. With one caller, the workers only ever help that caller's region.
- The kind of each sample comes from its deepest frame:
  - a `Gemm.Kernel*` method;
  - Act, which includes `PackedLstm.Step`'s task itself, because `Act.LstmCell` is `AggressiveInlining` and has no
    frame of its own;
  - ManagedThreads, meaning `Pool.Work`/`Help`/`Run` with no task below them: spinning, `SpinWait`, `PollGC` and the
    caller's join spin;
  - asleep (`SemaphoreSlim.Wait`), counted separately and not as CPU;
  - everything else.
- The first runs of each process (JIT tiering) are dropped. Traced runs take about 10% longer than untraced ones (8
  threads: 1.93 s against 1.75 s).

## Headline numbers on an idle machine

`stanzasharp benchmark --python none` gave a total of 1.73 s (load 0.56 s): pos 0.29, lemma 0.03, constituency 0.28,
depparse 0.38, sentiment 0.55, ner 0.17, tokenize 0.03, and a 27.0 ms median for a one-sentence call. The 5.2 s the
brief quotes (load 1.5 s) was measured on a loaded machine. Twelve spinning threads in another process reproduce it
(see [Run-to-run variance](#run-to-run-variance)).

Untraced harness runs, median of 5 timed runs, in s:

| threads | total | tokenize | pos | lemma | constituency | depparse | sentiment | ner |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.78 | 0.064 | 1.693 | 0.068 | 1.294 | 2.322 | 3.447 | 0.881 |
| 2 | 5.10 | 0.041 | 0.872 | 0.044 | 0.694 | 1.210 | 1.782 | 0.458 |
| 4 | 2.78 | 0.030 | 0.471 | 0.036 | 0.409 | 0.644 | 0.945 | 0.247 |
| 8 | 1.85 | 0.028 | 0.293 | 0.032 | 0.325 | 0.397 | 0.561 | 0.170 |
| 16 | 1.77 | 0.043 | 0.288 | 0.039 | 0.311 | 0.354 | 0.585 | 0.170 |
| speedup at 8 | 5.3x | 2.3x | 5.8x | 2.1x | **4.0x** | 5.8x | 6.1x | 5.2x |

A 10-run set at 8 threads gave a 1.75 s median, with constituency at 0.289 s (4.5x). That set also appears in the
variance section.

## Where the CPU time goes

### By component

Each stage's wall time and its CPU time, both in ms per call, then the share of that CPU time in each component, in %.
"Asleep" is the workers' blocked time, which is not counted as CPU. Act outside the LSTMs never reaches 0.1% (the highway
LSTM's `AddGatedTanh` is counted under other LSTMs), so it has no column.

**8 threads**

| stage | wall | CPU | charlm | other LSTMs | GEMM outside LSTMs | ManagedThreads | everything else | asleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tokenize | 31 | 220 | 0 | 13.7 | 4.2 | **78.1** | 4.0 | 38 |
| pos | 320 | 2,496 | 60.2 | 10.7 | 1.8 | 25.7 | 1.6 | 51 |
| lemma | 36 | 291 | 0 | 0.2 | 4.6 | **78.8** | 16.4 | 4 |
| constituency | 336 | 2,666 | 0.9 | 50.5 | 3.6 | **43.8** | 1.3 | 29 |
| depparse | 417 | 3,285 | 25.4 | 31.3 | 15.3 | 15.4 | 12.5 | 30 |
| sentiment | 606 | 4,796 | 11.0 | 38.3 | 30.5 | 13.8 | 6.4 | 57 |
| ner | 182 | 1,420 | 34.1 | 12.5 | 13.7 | 37.8 | 1.8 | 32 |
| **all** | 1,927 | 15,175 | 22.2 | 30.9 | 15.3 | 25.8 | 5.8 | 244 |

**4 threads**

| stage | wall | CPU | charlm | other LSTMs | GEMM outside LSTMs | ManagedThreads | everything else |
|---|---:|---:|---:|---:|---:|---:|---:|
| pos | 483 | 1,902 | 74.4 | 12.7 | 1.8 | 9.0 | 2.1 |
| constituency | 421 | 1,675 | 1.3 | 73.2 | 5.0 | 18.8 | 1.7 |
| depparse | 659 | 2,615 | 29.0 | 34.5 | 16.7 | 5.6 | 14.2 |
| sentiment | 970 | 3,872 | 12.4 | 41.4 | 33.6 | 6.0 | 6.6 |
| ner | 258 | 1,024 | 48.4 | 16.4 | 13.7 | 19.6 | 2.0 |
| **all** | 2,868 | 11,374 | 27.9 | 36.6 | 17.7 | 11.0 | 6.8 |

**1 thread** (wall time is CPU time)

| stage | CPU | charlm | other LSTMs | GEMM outside LSTMs | everything else |
|---|---:|---:|---:|---:|---:|
| tokenize | 70 | 0 | 80.0 | 8.3 | 11.7 |
| pos | 1,744 | 82.1 | 13.6 | 1.9 | 2.4 |
| lemma | 71 | 0 | 6.1 | 30.8 | 63.1 |
| constituency | 1,338 | 1.8 | 88.4 | 7.6 | 2.2 |
| depparse | 2,401 | 31.9 | 36.1 | 16.8 | 15.1 |
| sentiment | 3,560 | 13.7 | 44.3 | 35.1 | 6.9 |
| ner | 900 | 64.1 | 18.5 | 15.0 | 2.4 |
| **all** | 10,084 | 32.6 | 40.6 | 19.4 | 7.5 |

### By kind

The same CPU time, divided by the deepest frame of each sample, in %. Act includes `PackedLstm.Step`'s task, where
`Act.LstmCell` is inlined. The other column holds memmoves, vocabulary lookups, batching, MST, allocation and GC.

| stage | 1 thread: GEMM / Act / ManagedThreads / other | 4 threads | 8 threads |
|---|---|---|---|
| pos | 88.0 / 8.3 / 0 / 3.7 | 79.9 / 7.9 / 9.0 / 3.2 | 65.5 / 6.0 / 25.7 / 2.8 |
| constituency | 88.7 / 2.7 / 0 / 8.6 | 71.8 / 2.3 / 18.8 / 7.1 | 49.1 / 1.3 / 43.8 / 5.9 |
| depparse | 78.5 / 4.3 / 0 / 17.2 | 73.5 / 4.2 / 5.6 / 16.7 | 66.4 / 3.6 / 15.4 / 14.6 |
| sentiment | 86.7 / 4.2 / 0 / 9.1 | 81.2 / 4.1 / 6.0 / 8.7 | 74.3 / 3.2 / 13.8 / 8.7 |
| ner | 89.8 / 5.8 / 0 / 4.4 | 73.3 / 4.4 / 19.6 / 2.8 | 56.2 / 3.0 / 37.8 / 2.9 |
| **all** | 84.9 / 5.0 / 0 / 10.2 | 75.4 / 4.5 / 11.0 / 9.2 | 62.8 / 3.3 / 25.8 / 8.1 |

Inside the charlm, 90% of the 1-thread time is the `Kernel*Blocked` GEMM and 9% the inlined LSTM cell. Inside the
other LSTMs, 89 to 93% is GEMM.

- **The GEMM kernels are at the hardware's limit.** At 1 thread the charlm's kernels spend about 1.29 s in pos on the
  hidden-state product. Each row step is 2 × 1024 × 4096 = 8.4 MFLOP, and the call has roughly 11k character steps per
  direction (2,009 words of about 4.5 characters plus a space). That is about 140 GFLOP/s, close to the 150 GFLOP/s of
  Zen 3's two 256-bit FMA pipes at single-core boost (an estimate). At 8 threads the kernels need 9.5 s of CPU for the
  8.6 s they need at 1 thread, about 11% more, from the lower all-core clock. Faster kernels can't buy much. Whatever is
  left comes from computing less, or from the time the cores spend outside the kernels.
- **ManagedThreads at 8 threads is 26% of all CPU (3.9 s per call), almost all of it workers spinning between regions.**
  On an idle machine that costs no wall time, since those cores have nothing else to do. It does matter when the machine
  is shared (see [Run-to-run variance](#run-to-run-variance)). Workers are asleep for only 244 ms of their 13.5 s of
  thread time per call.
- **GC is negligible.** The traces show 2 to 5 gen0/gen1 GCs per call, with a gen2 in about one call of three, for a
  total pause of 4 to 14 ms per call (under 1%). A warm call allocates 51 MB; the first call allocates 286 MB.

### The caller's critical path at 8 threads

What the calling thread does during a call, in ms per call (traced). A join wait is the caller spinning in `Pool.Run`
after its own share, until the last task of the region is done. Serial means the caller works outside any region.

| stage | wall | GEMM (its share of the regions) | Act | join wait | serial and other |
|---|---:|---:|---:|---:|---:|
| tokenize | 31 | 5 | 3 | 8 | 15 |
| pos | 320 | 222 | 18 | 52 | 28 |
| lemma | 36 | 11 | 0 | 3 | 22 |
| constituency | 336 | 188 | 7 | 48 | 92 |
| depparse | 417 | 289 | 17 | 37 | 74 |
| sentiment | 606 | 461 | 19 | 37 | 88 |
| ner | 182 | 114 | 8 | 38 | 21 |
| **all** | 1,927 | 1,290 | 72 | 222 | 340 |

With perfect scaling, 8 threads would take 10,084 / 8 = 1,260 ms. The 667 ms between that and the measured 1,927 ms is
mostly the 222 ms of join waits and the 340 ms of serial work. The charlm's share of the join waits is 122 ms: pos 48,
ner 33, depparse 25, sentiment 16.

## Constituency

**Thread start and join is not the largest cost.** Per call at 8 threads (traced 336 ms):

- 188 ms is the caller's share of the GEMMs. That is 16% of the 1,183 ms single-thread GEMM time, against an even
  share of 12.5%: workers take a little while to join each small region.
- 92 ms is serial: `StackLstm.Push` runs `Act.LstmCell` over every row after each layer's GEMM (64 ms with its
  `[x | h]` copies and memmoves), plus the transition system and the `Score`/`Compose` bookkeeping.
- 48 ms (14%) is join waits.

At 4 threads (421 ms) the split is 312 GEMM, 83 serial, 22 join. At 1 thread (1,338 ms) the serial part is about
150 ms, 11%. That alone limits 8 threads to about 4.7x (Amdahl), and 4.0 to 4.5x was measured. The workers' spinning
is 44% of constituency's CPU at 8 threads and 19% at 4. That is burnt CPU, but not wall time. The scaling table above
gives 1.29 s at 1 thread, 0.41 s at 4 and 0.29 to 0.33 s at 8.

## Run-to-run variance

**Idle machine, 8 threads, 10 timed runs in one process:**

- The total was 1.69 to 1.86 s (median 1.75 s, CV 2.6%). Sentiment ranged 0.543 to 0.616 s, depparse 0.363 to 0.404 s
  and constituency 0.266 to 0.316 s.
- GC pauses were 5.6 to 13.5 ms per call, with no link to the slow runs. The slowest run (1.859 s) had 9.6 ms of pause;
  the two runs with a gen2 took 1.77 s and 1.69 s.

**The cause of 4.9 to 8.1 s is other processes using the cores.** The machine is shared with other sessions, which
build and run tests. A second process ran N threads that spin at normal priority while the harness ran (8 threads):

| other busy threads | runs | total, range (s) | median (s) |
|---:|---:|---:|---:|
| 0 | 10 | 1.69 to 1.86 | 1.75 |
| 4 | 10 | 2.10 to 2.20 | 2.15 |
| 8 | 10 | 2.52 to 2.77 | 2.59 |
| 10 | 5 | 3.89 to 5.77 | 4.44 |
| 12 | 5 | **5.72 to 8.55** | **7.29** |
| 14 | 5 | 11.7 to 15.5 | 13.9 |
| 16 | 10 | 16.8 to 23.9 | 19.3 |

Twelve busy threads reproduce the brief's numbers, both the range and the even spread over the stages. Once our 8
threads plus the others outnumber the 16 logical processors, each region waits for whichever item is held by a
preempted worker, for a scheduler quantum. A trace at 16 busy threads shows the caller spending 19.9 of every 24.1 s
(83%) in `Pool.Run`'s join spin, and workers spending 87% of their CPU samples in the pool's spin loops. GC stayed
under 64 ms per call.

Fewer threads do better under load (total per call, range of 4 runs in s; at 8 threads, the runs above):

| our threads | 12 other busy threads | 16 other busy threads |
|---:|---:|---:|
| 1 | | 14.1 to 17.7 |
| 2 | 6.84 to 7.07 | 17.7 to 18.3 |
| 4 | **4.27 to 4.72** | 20.7 to 21.5 |
| 6 | 5.21 to 6.82 | |
| 8 | 5.72 to 8.55 | 16.8 to 23.9 |

So the variance isn't GC, and it isn't workers falling asleep: on an idle machine they are asleep 2% of the time, and
waking them is not where the time goes. It is the pool's barrier per region meeting CPU contention from outside the
process. Round 12's gate counts only this process's threads. Benchmarks on this desktop are comparable only when
nothing else runs.

## Targets, ranked

The estimates are for this text on this machine. The 8-thread figures are against a 1.75 s idle call.

1. **Sentiment: compute the padding rows' first-layer input projection once per batch**
   ([idea](ideas/sentiment-padding-rows.md)). The benchmark text is one sentiment batch, 163 sentences padded to 71
   tokens, so 1,911 real rows out of 11,573 (83.5% padding). Every padding row has the same input, so its 2148 → 2400
   projection, the largest GEMM in the stage, is the same too. Estimate: about 0.7 s of sentiment's 3.45 s at 1 thread,
   about 110 ms at 8 (6% of the call). The gain is large for short and mid-length documents, small for long ones (whose
   batches are sorted by length), and zero for one-sentence calls.
2. **The pool under CPU contention** ([idea](ideas/managed-threads-under-contention.md)). Nothing changes on an idle
   machine. With 12 other busy threads: 5.7 to 8.6 s at 8 threads against 4.3 to 4.7 s at 4 threads, so a pool that
   sheds workers when its joins stall would be about 35% faster and steady. The benchmark could also report other
   processes' CPU use during its runs, so loaded runs stand out.
3. **Run both charlms in one region per character step** ([idea](ideas/charlm-one-region-per-step.md)). This halves the
   charlm's regions. Estimate: about 60 ms of its 122 ms of join waits at 8 threads (3%); nothing at 1 thread.
4. **Constituency: run `StackLstm.Push`'s LSTM cell inside the GEMM's tasks**
   ([idea](ideas/constituency-push-cell-in-tasks.md)). This parallelizes 64 ms of serial work per call at 8 threads.
   Estimate: 40 to 50 ms (13 to 15% of constituency, 2.5% of the call); nothing at 1 thread.
5. **Depparse: score each row's padding columns once, and the remaining dot products with SIMD**
   ([round 29](round-29-depparse-padding-columns.md)). 72% of the 156,384 arc pairs are padding columns, whose score is
   the same within a row. Estimate: about 200 ms of depparse's 2.32 s at 1 thread (9%), and 25 to 30 ms at 8 (1.5%).
6. **Sentiment: vectorize the strided (5, 5) filter across its 8 channels**
   ([idea](ideas/sentiment-strided-filter-simd.md)). Its scalar loops take 157 ms at 1 thread (4.5% of sentiment).
   Estimate: about 130 ms at 1 thread and 15 to 20 ms at 8 (1%).

Left out:
- Tokenize and lemma: 78% of their CPU at 8 threads is pool overhead, but together they take 67 ms of wall time.
- GC: under 1% of a call.
- The GEMM kernels: near peak (above).
