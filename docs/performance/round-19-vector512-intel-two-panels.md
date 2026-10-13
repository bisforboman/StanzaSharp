# Results, round 19: Vector512 on Intel, and two panels per LSTM step

Part of [performance.md](../performance.md)'s measurement history.

Round 14 measured Vector512 only on AMD. The Benchmark workflow's `avx512` jobs (`avx512_only`) now also try three
windows-2025 runners (Git Bash's `/proc/cpuinfo` has the flags too) and print the kernels' GEMM rates per path
(`managed-spike gemm`). In 8 dispatches (69 jobs) the CPUs drawn were mostly the EPYC 7763 (no AVX-512) and the EPYC 9V74
(about a third of those with AVX-512 masked), then the EPYC 9V45. Intel came up 8 times: a Xeon Platinum 8573C (Emerald
Rapids) 5 times (3 on ubuntu-26.04, 2 on windows-2025), a Xeon 6973P-C (Granite Rapids) twice, both on ubuntu-26.04, and a
Xeon Platinum 8370C (Ice Lake) once. Same method as round 14 (2,009 words, `default` package, Vector256 and Vector512
alternated on one machine, two rounds each; totals of both rounds).

**Before the change** (main, 1.3.0):

| CPU (2 cores / 4 logical) | threads | Vector256 total | Vector512 total | gain | per call, median |
|---|---:|---:|---:|---:|---:|
| Xeon 6973P-C (Granite Rapids) | 4 | 8.21–8.39 s | 6.82–6.97 s | +18–23% | 133–136 → 119–124 ms |
| Xeon 6973P-C (Granite Rapids) | 1 | 18.27–18.91 s | 15.15–15.97 s | +14–25% | 278–293 → 237–253 ms |
| Xeon Platinum 8573C (Emerald Rapids), Windows | 4 | 10.87–11.01 s | 8.50–8.65 s | +26–30% | 159–169 → 141–150 ms |
| Xeon Platinum 8573C (Emerald Rapids), Windows | 1 | 24.49–24.79 s | 17.99–18.14 s | +35–38% | 323–328 → 268–271 ms |

Runs [38308654625](https://github.com/bisforboman/StanzaSharp/actions/runs/38308654625) and
[38311028336](https://github.com/bisforboman/StanzaSharp/actions/runs/38311028336), 2026-10-12. A second Granite Rapids
draw ([38311308180](https://github.com/bisforboman/StanzaSharp/actions/runs/38311308180)) was a noisy machine (4 threads
barely faster than 1) but agreed: +14% at 4 threads, +20–22% at 1.

So Vector512 was faster on Intel too, but by less than on Zen 5, and `Speed_IsReported` showed why: the charlm, which is
the LSTM recurrence, gained only 8% on Granite Rapids (263 → 243 ms), while the highway gained 32%. `PackedLstm.Step`
called the blocked kernel one panel at a time. On Vector512 that is 6 accumulators, and per k it does 1 panel load and 6
broadcasts for 6 FMAs, while the GEMM's `KernelPair512` does 2 panel loads and 6 broadcasts for 12 FMAs.

**The change.** `PackedLstm.Step` hands Vector512 two adjacent panels of its task per call
(`Gemm.KernelBlockedPair512`: 12 accumulators for 6 rows, 6 for 3), then runs the LSTM cell on each panel. Each column
gets the single-panel kernel's FMAs in the same order and the same flushes, so the results are still Vector256's bit for
bit (`Vector512_GivesVector256sBits`: its charlm and highway cases go through paired and single panels). Measured with
main's tool (Vector512) built and alternated in the same job (a temporary step on the branch); totals are each round's, gains are each job's mean of
its two rounds (the range over jobs). The JIT's code for the pair is what you would write: 2 `vmovups`, 6
`vbroadcastss` from memory and 12 `vfmadd231ps` per k.

| CPU | threads | Vector256 | Vector512 before | Vector512 after | after vs before | after vs Vector256 |
|---|---:|---:|---:|---:|---:|---:|
| Xeon Platinum 8573C (Emerald Rapids), 3 jobs | 4 | 9.10–10.17 s | 6.92–7.78 s | 6.69–7.25 s | +5–7% | +37–39% |
| Xeon Platinum 8573C (Emerald Rapids), 3 jobs | 1 | 20.96–22.94 s | 14.81–16.28 s | 13.27–15.13 s | +7–9% | +51–54% |
| Xeon Platinum 8370C (Ice Lake) | 4 | 9.94–9.95 s | 10.54–10.65 s | 9.51–9.59 s | +11% | +4% |
| Xeon Platinum 8370C (Ice Lake) | 1 | 21.86–22.06 s | 23.12–23.20 s | 20.75 s | +12% | +6% |
| AMD EPYC 9V45 (Zen 5), 6 jobs | 4 | 5.10–5.66 s | 3.36–3.78 s | 2.96–3.27 s | +11–15% | +65–76% |
| AMD EPYC 9V45 (Zen 5), 6 jobs | 1 | 10.26–11.81 s | 6.56–7.71 s | 5.71–7.07 s | +11–15% | +73–82% |
| AMD EPYC 9V74 (Zen 4), 4 jobs | 4 | 6.95–7.60 s | 6.15–6.45 s | 6.21–6.63 s | −3 to +1% | +12–16% |
| AMD EPYC 9V74 (Zen 4), 4 jobs | 1 | 13.53–15.11 s | 12.83–13.96 s | 12.60–13.31 s | +3–5% | +8–16% |

Runs [38312471435](https://github.com/bisforboman/StanzaSharp/actions/runs/38312471435),
[38314372520](https://github.com/bisforboman/StanzaSharp/actions/runs/38314372520),
[38317679896](https://github.com/bisforboman/StanzaSharp/actions/runs/38317679896) and
[38318677685](https://github.com/bisforboman/StanzaSharp/actions/runs/38318677685), 2026-10-13. A fourth 8573C job, on
windows-2025, agrees at 1 thread (+9% over before, +50% over Vector256); its 4-thread rounds were disturbed (one
Vector256 round had a 592 ms median call). Per stage on the Emerald Rapids at 4 threads (run 38312471435), Vector256 → before → after: pos 1.24 → 1.10 → 0.96 s, depparse 2.20 → 1.83 → 1.65 s, sentiment
2.49 → 1.92 → 1.83 s, constituency 2.63 → 1.74 → 1.73 s, ner 0.71 → 0.63 → 0.60 s. `Speed_IsReported`'s charlm, Vector256
→ Vector512: 250 → 184 ms there (0.74x; Granite Rapids before the change: 0.92x), 147 → 87 ms on the 9V45.

- **Ice Lake** was the only CPU where Vector512 lost before the change: 5–7% slower than Vector256 in total (pos, depparse
  and ner slower, constituency faster). After it, it is 4–6% faster in total; pos and depparse at 1 thread are still about
  3–6% slower than on Vector256, constituency 20% faster. One draw only. Its GEMM rates show the AVX-512 frequency drop
  of that generation: at 1 thread Vector512 beats Vector256 by 7–16% (80.9 vs 75.7 GFLOP/s on the charlm's shape), at 4
  threads it loses by 4% (150 vs 157). Not enough to make a CPU rule worth its risk: Vector512 is ahead overall there now.
- **No throttling on the newer Intel parts**: on the Emerald Rapids at 1 thread Vector512 reached 134–156 GFLOP/s against
  Vector256's 74–81 (torch's AVX-512 kernels 170–196), so the 512-bit FMAs run at nearly twice the 256-bit rate.
- **No vendor rule.** Vector512 is faster than Vector256 on every Intel and AMD CPU drawn, so it stays the default
  wherever .NET accelerates it. (.NET itself leaves `Vector512` unaccelerated on Skylake-X and Cascade Lake, which
  throttle hardest.)
- **Where Intel loses to Zen 5 is not the kernels.** The 9V45 runs the pipeline 2.3–3.5x faster than these Xeons with the
  same code, most of all constituency (0.41 s against 1.65–2.1 s at 4 threads). The single-row GEMM, which streams the 16 MB
  charlm-sized weight matrix, ran at 13–16 GFLOP/s on the Xeons against 43–48 on the 9V45 (and so did torch's): about a
  third of the memory bandwidth per core. The constituency parser's many small steps (a few rows against large weights)
  are bound by that, so wider FMAs can't help them; only reading the weights fewer times would.
- Not tried: more accumulators. The pair already has 12 independent chains, more than the 8 that
  two FMA ports with a 4-cycle latency need, and its broadcasts come from memory (load ports, not the shuffle port that
  Intel's second FMA unit shares).
