- Managed backend, startup: `Pipeline.Load` loads two models at a time (21–37% less time on GitHub's free runners, a
  load peak 25–89 MB higher), and the sigmoid/tanh exponential and the thread pool's per-step code are compiled
  optimized from the first call instead of starting at JIT tier 0. A fresh process that loads and makes six
  one-sentence calls finishes 18–29% sooner; the first call is 5–11% faster, the next few 17–52%. Warmed speed is
  unchanged, and so is the output, bit for bit. README has advice for short-lived processes (tiering, ReadyToRun).
