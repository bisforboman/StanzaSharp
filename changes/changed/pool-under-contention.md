- Managed backend: when other processes keep the cores busy, the thread pool puts some workers to sleep instead of
  letting every parallel step wait for a preempted one, and wakes them again once the stalls stop. On a loaded machine
  a call took less wall time and about a third of the CPU time. Concurrent calls share the pool as before, and results
  are unchanged (bitwise).
