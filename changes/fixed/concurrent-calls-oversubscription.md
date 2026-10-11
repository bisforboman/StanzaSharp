- Managed backend: concurrent `Process` calls with `Threads` near the number of logical processors (the default) no
  longer oversubscribe the CPU. The pool's workers stay asleep while callers and workers already fill every logical
  processor; before, 8 callers on 16 threads got 3-10 calls/s and single calls stalled for up to 20 s, now 13-17
  calls/s (docs/performance.md, round 12). Output is unchanged.
