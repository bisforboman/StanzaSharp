# Results, round 13: physical cores by default from 4 up

Part of [performance.md](../performance.md)'s measurement history.

The owner's rule after round 12 (2026-10-12): with `Threads` null the managed pool now starts at the physical cores when
there are at least 4 of them, otherwise at all logical processors, and never above `Environment.ProcessorCount` (a
container's CPU quota). `CpuCores.Physical()` finds the cores without starting a process (Linux: distinct sysfs
`thread_siblings_list`; Windows: `GetLogicalProcessorInformationEx`; macOS: `sysctlbyname("hw.physicalcpu")`) and falls
back to the logical processors if that fails. The benchmark's core count uses it too. TorchSharp is unchanged (libtorch
already defaults to the physical cores).

**Ryzen 7 5800X** (8 cores / 16 logical): 8 threads (the new default) against 16 (the old one), `samples/StanzaSharp.Benchmark`,
three rounds, 8 and 16 back to back with the order flipped each round. Other sessions shared the machine, so each run
waited for two idle readings (< 5% CPU) and was repeated if the machine was busy (≥ 10%) right after it (two runs were):

| | 8 threads | 16 threads | 8 vs 16 (medians) |
|---|---|---|---:|
| default, 26,264 words, words/s | 1,197, 1,230, 1,196 | 1,315, 1,247, 1,313 | -9% |
| default_fast, 26,264 words, words/s | 2,682, 2,721, 2,633 | 2,921, 2,858, 2,758 | -6% |
| default, one sentence, median ms | 42.1, 32.2, 37.5 | 34.4, 33.3, 40.2 | +9% |
| default, one sentence, p90 ms | 60.1, 49.8, 59.7 | 59.1, 71.4, 72.9 | -16% |
| default_fast, one sentence, median ms | 17.3, 15.9, 16.1 | 17.3, 17.4, 22.9 | -7% |
| default_fast, one sentence, p90 ms | 26.0, 24.8, 23.9 | 38.7, 54.7, 38.7 | -36% |
| 2 callers (default), calls/s | 30.3, 33.6, 36.1 | 26.4, 30.3, 30.2 | +11% |
| 2 callers, slowest call ms | 135, 119, 107 | 386, 246, 279 | -57% |

(Big text: `--runs 1`, 8 copies; one sentence: `--per-call 50`; 2 callers: `--per-call 60 --concurrent 2`.) The
one-sentence `default` median moved back and forth, so six more back-to-back rounds of it alone followed: over all
nine, 8 threads had the shorter median in 5 (overall medians 36.8 vs 35.8 ms) and a shorter p90 in 8 (57.0 vs 72.5 ms).
So on a busy desktop 8 threads mostly remove the slow calls rather than shorten the typical one, which round 12 (idle)
found 18-27% shorter. The big texts lose 1-9% (medians: 9% and 6%), as round 12 found; the owner accepted that with the rule. CoNLL-U of the
big runs is byte-identical on 8 and 16 threads (MD5 db91abcc… default, a3a3916d… default_fast).

**Free runners** ([38232917125](https://github.com/bisforboman/StanzaSharp/actions/runs/38232917125), this branch): the
detection found 2 cores on ubuntu-26.04 (AMD EPYC 7763) and windows-2025 (Intel Xeon Platinum 8573C), 4 on both Arm
runners and 3 on macos-26, so the default stays 4, 4, 4, 4 and 3 threads there, as before; CoNLL-U identical to Python
Stanza's on every runner that ran it.
