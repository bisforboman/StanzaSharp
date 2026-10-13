# Results, round 14: AVX-512 kernels (Vector512)

Part of [performance.md](../performance.md)'s measurement history.

The managed kernels stopped at AVX2 (`Vector256`), while on the AVX-512 runner Python Stanza ran about 40% faster than on
an EPYC 7763 (above). A fourth kernel path, `Vector512`, is now the detected one wherever .NET accelerates `Vector512`
(AVX-512 that doesn't throttle; `DOTNET_PreferredVectorBitWidth=256` turns it off):

- The panels stay 16 floats wide, so one Vector512 register holds a panel row: the 6-, 3- and 1-row kernels and the
  LSTM's blocked kernel have one accumulator per row instead of two. `Gemm.Run` gives each kernel call two adjacent
  panels (12 accumulators for 6 rows, enough to cover the FMA latency). The LSTM cell takes one exp over its 16 gate
  values; the highway's gated tanh runs 16 lanes at a time.
- Every output gets the same fused multiply-adds in the same order as on Vector256, and the activations are the same
  lane by lane, so the results are Vector256's **bit for bit** (`ManagedBackendTests.Vector512_GivesVector256sBits`:
  GEMMs of 1–7 and 97 rows with paired and single panels, the blocked kernel, the charlm, the highway). Every
  tolerance and golden file holds unchanged, and which CPU a CI run draws can't change a result.

**Method.** The Benchmark workflow's new `avx512` jobs (`avx512_only` runs only them): six ubuntu-26.04 jobs; each prints
its CPU and stops if `/proc/cpuinfo` has no `avx512f`. One that has it runs `stanzasharp benchmark` without Python
(`default` package, 2,009 words, 3 timed runs, 50 one-sentence calls) eight times, alternating Vector256
(`DOTNET_PreferredVectorBitWidth=256`) and Vector512, two rounds each at the default threads (4) and at 1, then the
managed kernel and pipeline tests. Medians of the two rounds (each itself a median of 3 runs):

| CPU (2 cores / 4 logical) | threads | Vector256 total | Vector512 total | words/s | per call, median |
|---|---:|---:|---:|---:|---:|
| AMD EPYC 9V45 (Zen 5) | 4 | 5.48–5.60 s | 3.69–3.77 s | 359–366 → 533–545 (+48%) | 53.5 → 43.8 ms |
| AMD EPYC 9V45 (Zen 5) | 1 | 10.75–11.06 s | 7.08–7.15 s | 182–187 → 281–284 (+53%) | 117 → 92 ms |
| AMD EPYC 9V74 (Zen 4), run 1 | 4 | 7.15 s, 7.15 s | 6.50–6.63 s | 281 → 303–309 (+9%) | 65 → 59 ms |
| AMD EPYC 9V74 (Zen 4), run 1 | 1 | 13.65–13.66 s | 13.13–13.23 s | 147 → 152–153 (+4%) | 138 → 117 ms |
| AMD EPYC 9V74 (Zen 4), run 2 | 4 | 7.16–7.28 s | 6.48–6.66 s | 276–281 → 302–310 (+10%) | 67 → 59 ms |
| AMD EPYC 9V74 (Zen 4), run 2 | 1 | 13.80–14.45 s | 13.19–13.77 s | 139–146 → 146–152 (+3%) | 142 → 120 ms |

Runs [38276588610](https://github.com/bisforboman/StanzaSharp/actions/runs/38276588610) (EPYC 9V74) and
[38277973171](https://github.com/bisforboman/StanzaSharp/actions/runs/38277973171) (EPYC 9V45 and 9V74), 2026-10-12. Per
stage on the 9V45 at 4 threads: pos 0.92 → 0.68 s, constituency 0.72 → 0.51 s, depparse 1.44 → 1.00 s, sentiment 1.69 →
1.04 s, ner 0.45 → 0.32 s. `Speed_IsReported` (320 words, 2 threads) on the 9V45: charlm 144 → 109 ms, highway 18.7 →
10.3 ms (TorchSharp 321 / 23.2 ms).

- Zen 5 executes 512-bit FMAs at full width, so the kernels gain about half; Zen 4 splits them into two 256-bit halves,
  and the gain is fewer instructions and loads (+3–10% throughput, 10–15% shorter one-sentence calls). Nothing got slower.
- Of 6 + 6 jobs, the ubuntu-26.04 runners were an EPYC 7763 (no AVX-512) 9 times, an EPYC 9V74 with AVX-512 twice and
  once without it (masked), and an EPYC 9V45 once; none drew the Intel Xeon Platinum 8573C of the round 12 benchmark
  run. On that Xeon (two 512-bit FMA ports) the gain should be nearer the 9V45's; not measured.
- So Vector512 ships as the default where .NET accelerates it. 105 tests (ManagedBackendTests on every path, PipelineTests
  on Vector512) passed on both AVX-512 CPUs.
