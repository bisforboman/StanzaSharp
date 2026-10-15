# Results, round 32: the managed pool under CPU contention

Part of [performance.md](../performance.md)'s measurement history.

[Round 27](round-27-whole-pipeline-profile.md#run-to-run-variance) found that when other processes keep the cores busy,
an 8-thread call goes 3–10x slower while 4 threads stay steady. Every parallel region waits for its last item, and a
worker preempted while it holds one stalls the caller for a scheduler quantum. Round 12's gate (`_busy`/`_limit`) counts
only this process's threads. This round makes `ManagedThreads`' pool notice those stalls and put workers to sleep.

## What changed

`ManagedThreads.Pool.Run` already spins until the region's last item is done (the join). Now it times that join:

- **A stall** is a join wait over 0.5 ms and over 8 times the caller's own mean item in that region. A preempted worker
  costs a scheduler quantum (milliseconds). An imbalance costs about one item, and items are microseconds.
- **Every 100 ms** (a window, ended by whichever caller's region crosses it), with k threads per region (the caller plus
  the awake workers):
  - if stalls took more than 1/k of the window, one more worker stays asleep (`_shed`, up to `Count` − 1);
  - if they took less than 1/(4k), one comes back.
- The 1/k rule is the conservative one: k threads that lose a fraction f of their time to stalls do k(1 − f) work, so
  dropping to k − 1 only pays when f > 1/k, even if the stalls then vanish entirely.
- A worker keeps spinning only while at least `_shed` workers are asleep, and a caller wakes at most
  `sleepers − _shed` of them. The rest of the pool is as before.

Nothing about how a region is split changed: items are still claimed one at a time from the same counter. Shedding only
changes which thread runs an item. Cost on an idle machine: one `Stopwatch.GetTimestamp` per region, two more when the
caller has to wait at the join, and a few volatile reads.

The first version shed a worker on every single stall and brought one back after 100 ms without stalls. Under heavy
load it went down to 1 thread most of the time (59% of samples at 1 thread, 16 s per call), while a fixed 4 threads
took 5.6 s. That version was dropped in favour of the stall fraction.

## Setup

Ryzen 7 5800X (8 cores, 16 logical processors), Windows 11, .NET 10.0.13, Vector256, converted models, all eight
processors, the benchmark text (2,009 words). A scratch harness (not committed) loads the pipeline at a fixed `Threads`,
makes two warm-up calls, then 5 timed calls. It prints median, min and max, the process's CPU time per call, the CoNLL-U's
SHA-256, and how often a 10 ms sampler saw each `_shed` level. Main (`1fe32df`) and the branch ran alternately.

Four other agents were running builds and tests on this machine the whole time, so there was never an idle machine.
Their load varied from about 50% to 100% CPU (at worst three test hosts using about 4.5 cores each). The deliberate load
is a second process with 12 threads that spin at normal priority, as in round 27. Idle speed was measured on the
Benchmark workflow's free runners.

## Under load

**Ambient load plus 12 spinning threads** (about 28 other busy threads on 16 logical processors), seconds per call:

| round | main, 8 threads | branch, 8 threads | main, 4 threads |
|---:|---|---|---|
| 1 | 25.0 (23.0–28.3) | **20.8** (19.9–22.3) | 18.4 (17.0–24.6) |
| 2 | 23.9 (22.1–28.8) | **17.1** (16.7–18.1) | 20.9 (11.0–22.1) |
| 3 | 23.1 (19.2–25.0) | **8.4** (7.6–14.3) | 15.7 (13.5–18.6) |
| CPU per call | 72–74 s | **15.5–17.9 s** | 33–35 s |

The branch ran mostly at 1–2 threads in rounds 1–2 (shed 6 or 7 in 89–91% of samples) and at 3–4 threads in round 3,
when the other agents' load dropped. It was faster than main at 8 threads in every round, and faster than a fixed
4 threads too. Main's spinning workers burned 72 s of CPU per call. That is CPU time taken from the other processes.

**Ambient load only**, alternating, 4 rounds of 5 calls (machine 40–90% busy):

| | main, 8 threads | branch, 8 threads |
|---|---|---|
| median per round (s) | 2.27, 2.28, 2.24, 2.28 | 2.23, 2.26, 2.27, 2.24 |
| CPU per call | 17.7–18.2 s | 17.6–18.1 s |

At this load the branch never shed a worker (`_shed` was 0 in every sample), and both sides took the same time.

## Concurrent callers

Round 12's method (`StanzaSharp.Benchmark --per-call 60 --concurrent T [--with-long]`, 8 threads), two alternating
sweeps under the ambient load:

| callers | main calls/s | branch calls/s | main p90 / max ms | branch p90 / max ms |
|---|---|---|---|---|
| 2 | 14.0, 14.7 | **19.0, 23.9** | 293 / 498, 259 / 585 | 171 / 279, 132 / 179 |
| 8 | 14.5, 18.1 | **18.2, 21.5** | 1,115 / 3,375, 901 / 4,210 | 676 / 1,458, 555 / 989 |
| 1 + long | 7.1, 7.0 | **9.1, 9.3** | 222 / 356, 275 / 527 | 173 / 298, 197 / 270 |
| 8 + long | 14.4, 16.4 | **18.8, 18.0** | 1,054 / 4,282, 930 / 2,228 | 619 / 1,413, 706 / 1,433 |

Throughput was higher in every case and the tail latencies lower. With 8 short callers the median call took longer
(316–376 ms against 154–221 ms) because the calls became more even: main's fastest caller had a 122–147 ms median and
its slowest 194–273 ms, the branch's 257–322 ms and 365–435 ms. With 8 short callers plus the long one, the long call
got less help. Its first run took 23–24 s on the branch against 12–14 s on main, so 2 long calls finished in the window
instead of 3–4, while the short callers made 18 calls/s instead of 14–16. Round 12's rule still holds: no caller stopped
making progress. `ConcurrencyTests` passed (all 7 cases, both backends), as did the `ManagedThreadsOptionTests`.

## Idle speed (free runners)

The Benchmark workflow (`stanzasharp benchmark`), branch run 38709412313 against main run 38709415811, total seconds
for all eight processors and the one-sentence median:

| runner | main | branch |
|---|---|---|
| ubuntu-26.04 (EPYC 9V74, both) | 8.00 s, 81.0 ms | 8.11 s, 81.7 ms |
| ubuntu-26.04-arm | 7.92 s, 91.4 ms | 7.94 s, 90.3 ms |
| windows-11-arm | 7.95 s, 90.6 ms | 7.83 s, 96.9 ms |
| macos-26, run and rerun | 9.89 s, 124 ms; 11.00 s, 152 ms | 10.92 s, 138 ms; 9.23 s, 117 ms |
| windows-2025 | 6.04 s (Xeon 8573C, Vector512) | 7.42 s (EPYC 7763, Vector256): not comparable |

Within noise on every runner that drew the same CPU. On macOS the order flipped between the run and its rerun (the
virtualized runner's usual spread).

## Exactness

The CoNLL-U (all eight processors) and the SHA-256 of three score sets on the benchmark text: every UPOS logit (pos,
`combined_charlm`), every sentiment logit, and the backward charlm's per-word vectors, each computed 3 times per process.
Main and the branch ran at 1 and 8 threads, idle (ambient load only) and with 12 spinning threads: 8 processes, 24
score sets in all.

| | SHA-256 (first 16 hex digits) |
|---|---|
| CoNLL-U | `93D9524692F2D690` in all 8 processes, and in every timed call above (each compared with the first) |
| UPOS logits | `6E17E2D3EF88EBCE` in all 24 |
| sentiment logits | `3237876DB4C70D2D` in all 24 |
| charlm vectors | `19FECB3074201698` in all 24 |

The 8-thread branch runs with 12 spinning threads ran at 4 down to 1 thread during the call (`_shed` 4–7), and the
bits stayed the same. A new test, `ParallelFor_ShedsWorkersWhileJoinsStallAndBringsThemBack`, makes workers sleep 3 ms per
item. It checks that the pool sheds workers, that every item still runs exactly once per region, and that the workers
come back once the stalls stop.

## Not done

- **Sizing from free cores** (`GetSystemTimes`, `/proc/stat`): this needs code for each platform and is coarser. The
  stall rule needs neither.
- **A public option** (for example `PipelineOptions.AdaptiveThreads`, on by default): everything is internal. Turning
  it off would only help someone who wants to grab cores from other processes. That is the owner's decision.
- The constants (0.5 ms, 8 items, 100 ms, 1/k and 1/(4k)) were set once and not tuned. Under the heaviest load the
  pool ran at 1–2 threads, where a fixed 4 threads might have been better (the 4-thread runs were themselves 16–21 s).
