# Results, round 12: concurrent calls and the managed thread pool

Part of [performance.md](../performance.md)'s measurement history.

How `ManagedThreads`, the managed backend's pool (used by every GEMM and LSTM step), behaves when several `Process` calls
run at once on one pipeline, and why one-sentence calls were slow on the macOS runner.

**Method.** `--per-call 60 --concurrent T` (T threads, 60 one-sentence calls each, on one shared pipeline; default
package); new: the slowest and fastest caller's median, and `--with-long`, where one more thread processes one copy of
the golden texts (14,534 characters, 3,283 words; alone 4.5 s on 8 threads, 19.2 s on 1) in a loop meanwhile. Ryzen 7
5800X (8 cores, 16 logical processors), Windows, converted models. Other sessions' test runs loaded the machine
throughout (20-100% CPU), so single numbers moved by up to 2× between runs: compare within a table, where before and
after were alternated.

**Callers and threads, before this round** (two sweeps, calls/s, then median / max ms per call):

| callers | 8 threads | 1 thread |
|---:|---|---|
| 1 | 22.9, 23.8; 42 / 101 | 8.2, 6.9; 115 / 298 |
| 2 | 26.3, 27.6; 71 / 182 | 12.8, 10.4; 142 / 415 |
| 4 | 21.8, 21.3; 155 / 562 | 18.9, 15.4; 176 / 649 |
| 8 | 18.9, 18.9; 319 / 2,442 | 18.7, 19.2; 350 / 1,355 |

- **Serialize or oversubscribe?** Neither, while `Threads` is at most the number of logical processors minus the
  callers: each caller works its own regions, the pool's 7 workers help whichever region has work, so 8 callers use
  15 threads. Latency grows with the callers because throughput stays flat, not because calls queue.
- **Is N callers' throughput at least one caller's?** Up to 2 callers yes (+15%), beyond that no (8 callers −20%).
  The pool isn't the cause: on 1 thread, with no pool at all, 8 callers reach the same ~19 calls/s. That is the
  memory bandwidth and L3 limit of round 7, and `MaxConcurrentCalls = 2` remains the advice.
- **Does a long call starve short ones?** No, they share the pool. One short caller next to the long call (8 threads)
  took 103-117 ms per call instead of ~40, and the long call 6.9-11.7 s instead of 4.5. Among 8 short callers the
  long call slowed most (6-25 s); the short callers' medians spread 1.8-2.2× (1.3-1.4× without it), and no caller
  stopped making progress.
- **But `Threads` = all logical processors (the default) oversubscribed.** 15 workers + N callers exceed 16 logical
  processors, and every waiting thread spins. A region ends with its slowest item, so a worker descheduled while it
  holds one stalls its caller for a scheduler quantum: with 8 callers on 16 threads throughput fell to 3.4-10 calls/s,
  single calls took up to 19.6 s, and the slowest caller's median was 2-10× the fastest's.

**What changed** (`ManagedThreads.Pool`): workers no longer wake or keep spinning while the busy threads (callers
inside a region plus awake workers) already number the logical processors (or `Threads`, if larger). One caller still
gets its full team; N callers still use at most `Threads` − 1 + N threads, but now also no more than the logical
processors (or N, when there are more callers). Items are still claimed one at a time from the same split, so results can't change. Before / after, alternated,
three runs each (calls/s; p90 / max ms):

| threads, callers | before | after |
|---|---|---|
| 16, 2 | 8.4, 7.7, 8.4; 455-570 / 1,041-1,707 | 11.2, 6.7, 13.0; 271-599 / 483-1,182 |
| 16, 8 | 10.1, 9.2, 10.2; 1,660-1,917 / 5,221-6,111 | **16.6, 13.1, 14.7**; 808-875 / 1,927-4,940 |
| 16, 4 + a long call | 10.2, 6.3, 8.3; 704-1,399 / 2,731-3,804 | **14.4, 8.9, 13.4**; 439-659 / 973-1,502 |
| 8, 1 | 21.2, 13.0, 18.0 | 23.5, 20.9, 18.1 |
| 8, 8 | 16.9, 15.3, 13.7 | 17.9, 17.0, 14.7 |

(16 threads, 1 caller: unchanged within noise.) With 8 callers on 16 threads the slowest and fastest callers' medians
are now 1.2-1.4× apart instead of 2-3×; the median call rose (164-204 → 349-449 ms) because no caller runs ahead of
the others any more. The long call took 7-16 s instead of 13-21 s.

Tried and not kept:
- **Threads as one budget across calls** (workers stay out while callers + workers exceed `Threads`): on 16 threads
  as good as the change above, but with 8 threads and 4-8 callers 10-15% less throughput, since callers then leave
  the other 8 logical processors idle.
- **Blocking instead of spinning** while the caller waits for a region's last items (an event after 50 spins): no
  better on the macOS runner (below), within noise.

Exactness: `Pipeline` CoNLL-U for one copy of the golden texts, both packages, on 1, 3, 8 and 16 threads after the
change and on 16 threads before, SHA-256 identical (default `8B626630…`, default_fast `F3311B8E…`); ConcurrencyTests,
ManagedBackendTests, OptionsTests and PipelineTests pass.

**macOS runner, one-sentence calls** (run 38150941695: 766 ms median, p90 1,197, against 69-86 ms on the other runners
and 338 for Python Stanza on the same Mac). Reproduced on macos-15 (Apple M1 virtual, 3 vCPUs), `--per-call`, per-call
median ms:

| | total | pos | constituency | depparse |
|---|---:|---:|---:|---:|
| benchmark command right after `pip install` (Spotlight indexing torch's files) | 845 | 264 | 215 | 187 |
| the same, Spotlight on, at the indexing peak | 717 | | | |
| the same, Spotlight on, between peaks | 117-152 | 55 | 27 | 40 |
| the same, Spotlight off (`mdutil -a -i off`), two runs | 94, 95 | | | |
| no Python process, after the first minutes | 104 | 38 | 19 | 25 |
| `--threads 1` (with Python; without) | 266; 207 | 92; 69 | 58; 48 | 81; 57 |

- The cause is the runner, not the pool's dispatch: `top` showed `mds`, `mds_stores` and six `mdworker_shared` at
  15-30% each (load average 16-33 on 3 vCPUs). Every processor slowed alike; with Spotlight off the median was 94 ms.
- Why StanzaSharp suffers more than Python: a call opens about 850 parallel regions (one per LSTM step), each ending
  with its slowest item, so a helper the host preempts holds up the whole call; torch runs far fewer, larger
  operations. Pool variants, alternated four times in one job (the caller blocking instead of spinning, workers
  spinning 10× shorter before they sleep; in another job 100× longer or not at all, which was slower), moved the median by less than the run-to-run noise
  (136-234 ms on 3 threads; 192-217 on 1, 181-273 on 2).
- `benchmark.yml` now turns Spotlight off on macOS first; "Run it yourself" says so.
- On **macos-26** (the runner CI has used since; macOS 26.6.2, the same Apple M1 virtual with 3 vCPUs) the slow calls don't
  show: the same benchmark command right after `pip install`, two runs each, 149 and 104 ms with Spotlight on (the
  first while `mds_stores` used 57% of a vCPU), 132 and 133 ms with it off; Python Stanza 303-704 ms. The `mdutil` step
  stays, as it costs nothing and removes a source of noise.

**Default thread count: physical cores or logical processors?** Loaded, one caller on 16 threads (the default, all
logical processors) took 98-180 ms per call against 42-65 on 8 (the physical cores). Measured again on an idle machine
(under 5% CPU before each run) and on the free runners, alternated, three rounds each (macos-26 and the Arm runners
have no SMT, so physical = logical there):

| | Ryzen 7 5800X, idle: 8 / 16 threads | ubuntu-26.04 (EPYC, 2 / 4) | windows-2025 (EPYC, 2 / 4) |
|---|---|---|---|
| default, words/s | 1,085, 1,068, 1,078 / 1,076, 1,137, 1,124 | 227, 231, 226 / 240, 233, 241 | 174, 214, 213 / 119, 120, 147 |
| default, per call median ms | 42.6, 45.9, 45.3 / 52.5, 57.7, 58.7 | 80.6, 86.5, 84.8 / 73.9, 76.0, 72.2 | 85.6, 92.5, 91.4 / 77.9, 72.7, 78.3 |
| default_fast, words/s | 2,379, 2,395, 2,403 / 2,452, 2,574, 2,603 | 378, 375, 366 / 402, 394, 395 | 352, 349, 355 / 382, 389, 391 |
| default_fast, per call median ms | 22.8, 23.4, 22.6 / 27.9, 29.9, 31.0 | 40.1, 41.0, 40.8 / 40.0, 50.3, 39.2 | 43.0, 44.0, 43.6 / 39.5, 41.2, 41.8 |
| 2 callers, calls/s | 22.6, 22.4, 21.2 / 19.1, 19.5, 15.3 | 9.7, 9.3, 9.4 / 10.3, 10.5, 10.4 | 10.6, 10.4, 9.9 / 12.1, 11.5, 11.5 |

(Ryzen: the big call is the benchmark text, 26,264 words, one run; runners: `stanzasharp benchmark --python none`, about
2,000 words and 50 one-sentence calls; 2 callers: `--per-call 60 --concurrent 2`.) Neither wins everywhere. On the
idle Ryzen the physical cores give 18-27% shorter calls and 15-39% more calls/s with 2 callers, but up to 8% fewer
words/s on the big text. On the 2-core runners the logical processors give 3-8% more words/s (except below), 7-17% more
calls/s with 2 callers and 8-21% shorter `default` calls (`default_fast` calls: even on ubuntu, 3-10% shorter on
windows), and lose only windows-2025's big `default` text (31-44% fewer words/s). So the default stayed all logical processors; round 13 changes it.
