- Managed backend: when other processes keep the cores busy, the thread pool puts workers to sleep instead of letting
  every parallel step wait for a preempted one, and wakes them again once the stalls stop. Results are unchanged
  (bitwise); on a loaded machine an 8-thread call took less wall time and a quarter of the CPU time.
