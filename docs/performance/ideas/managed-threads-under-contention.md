# The managed pool under CPU contention

An open idea ([list](../../performance.md#remaining-ideas)), from [round 27](../round-27-whole-pipeline-profile.md).

Every region waits for its last task. When other processes keep the cores busy, a worker that has claimed a task can be
preempted for a scheduler quantum, and the caller spins until it is back. On the 16-logical-processor desktop, with other
busy threads, the 8-thread all-eight call took:

| other busy threads | 0 | 8 | 10 | 12 | 14 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| 8-thread call (median, s) | 1.75 | 2.59 | 4.44 | 7.29 | 13.9 | 19.3 |

With 12 other busy threads, 4 threads took 4.3 to 4.7 s, steadily, while 8 threads took 5.7 to 8.6 s. With 16, even
1 thread (14 to 18 s) was no slower than 8. Round 12's gate (`_busy`/`_limit`) counts only this process's threads.

Possible fixes. All of them are exact, since a region's split never depends on how many threads help:

- **Shed workers when joins stall.** `Pool.Run` already knows how long the caller spins after its own share. If that
  wait is many times a task's length for several regions in a row (preemption, not imbalance), fewer workers could go
  awake, and the count could grow back slowly. A worker that only spins because the caller is in serial code is not a
  stall, so measure the join, not the worker's idle time.
- **Size from free cores.** At the start of a call, read the system's busy fraction (`GetSystemTimes`, `/proc/stat`) and
  cap the awake workers at the cores that are idle. This is cheaper but coarser.
- **Benchmark hygiene, cheap and independent.** `stanzasharp benchmark` (and the sample) could compare the system's
  CPU time with the process's own during the timed runs and print a warning when other processes used more than about
  one core. Rounds 24 and 27 both lost time to loaded runs.

**Estimate.** Nothing on an idle machine. Under the load that other sessions put on this desktop (about 12 busy
threads), roughly 35% faster and steady, which is the 4-thread result above. Spinning workers also burn 3.9 s of CPU
per 1.75 s call (26% of the call's CPU), which a machine shared with other work feels too.
