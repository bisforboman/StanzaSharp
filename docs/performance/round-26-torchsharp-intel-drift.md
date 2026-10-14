# Results, round 26: the TorchSharp backend's drift on Intel AVX-512 runners

Part of [performance.md](../performance.md)'s measurement history.

main's CI run 38504825407 (the 1.7.0 release commit) failed one test in golden-shard 2, on an Intel Xeon Platinum
8370C runner:

```
FastPackageTests.Intermediates_MatchGolden(lang: "de")
s0.pos: logit 146 is -285.12122, expected -285.12134 (float32) or -285.12133562518386 (float64)
```

That is 1.2e-4 from both references against the 1e-4 tolerance, on the **TorchSharp** backend (German `default_fast`,
the nocharlm tagger). Every tag was right. The managed backend passed on the same runner.

## Method

A temporary test (`TorchDriftDiagnostics`, on the throwaway branch `diag/torch-intel-drift`, not merged) rebuilds the
TorchSharp tagger's forward pass op by op from its modules (checked bit for bit against `PosTagger.Predict`). It does
this for German and English, `default` and `default_fast`, on the golden intermediates' 3 sentences. Each stage is
written to a file. With a reference dump, each op also runs on the reference's inputs ("isolated"), so an op's own
difference shows apart from what it inherits. A temporary workflow ran it on free `ubuntu-26.04` runners (10 jobs, then
16), under several libtorch/MKL environment settings. With it ran every golden-intermediates test of both backends
(tokenizer, POS, constituency, depparse, NER, `default_fast`), sentiment's golden logits and
`Vector512_GivesVector256sBits`. Runners drawn: AMD EPYC 7763 (Zen 3, AVX2), EPYC 9V74 (Zen 4) and 9V45 (Zen 5), both
with AVX-512, and Intel Xeon Platinum 8370C (Ice Lake) and 8573C (Emerald Rapids). Locally: Ryzen 7 5800X, Windows.

## Numbers

Worst "accepted" UPOS logit difference (the nearer of Stanza's float32 and float64 values; limit 1e-4), TorchSharp
backend, 3 sentences each:

| CPU | de `default_fast` | de `default` | en `default_fast` | en `default` |
|---|---|---|---|---|
| Ryzen 5800X (local, Windows) | 3.05e-5 | 3.24e-5 | 1.35e-5 | 1.39e-5 |
| EPYC 7763 (×8 jobs, identical bits) | 6.10e-5 | 4.48e-5 | 1.53e-5 | 1.52e-5 |
| EPYC 9V74 / 9V45 | 6.10e-5 | 4.48e-5 | 1.35e-5–1.53e-5 | 1.52e-5 |
| Xeon 8370C / 8573C (×5 jobs, identical bits) | **1.20e-4** | 6.10e-5 | 1.53e-5 | 2.04e-5 |

The German logits reach |285|–|380|, where one float32 ulp is 3.05e-5, so 1e-4 is about 3.3 ulps; the English ones
stay under |140| (ulp 7.6e-6–1.5e-5). On Intel the German `default_fast` logit is 4 ulps from Stanza's float32 value.
German `default` (charlm tagger, `PosTests.UposLogits_MatchGoldenIntermediates`) uses 61% of the tolerance on Intel.

Every other TorchSharp golden-intermediates test kept its margin on Intel: tokenizer logits ≤ 1.1e-5, constituency
scores ≤ 3.8e-5 (tolerance 1e-3), depparse arcs ≤ 2.3e-5 and labels ≤ 3.4e-5, NER emissions ≤ 1.3e-5, sentiment
logits ≤ 3.6e-5 (en) and ≤ 5.5e-6 (de). All passed on every runner; only German `default_fast` UPOS failed, on all 5
Intel draws and on none of the 11 AMD ones.

**Managed backend:** identical on every runner (same printed differences to the last digit on the 7763 and the Intel
jobs: German `default_fast` 2.77e-5 / 3.54e-5 / 4.92e-5), and `Vector512_GivesVector256sBits` passed on every
AVX-512 runner. It passes everywhere, as designed.

## Source

libtorch picks its float32 kernels by CPU, in two places, and both change bits:

- **MKL's sgemm** (libtorch 2.10's oneMKL 2025.3; `MKL_VERBOSE` shows every Linear and LSTM GEMM going there, `CNR:OFF`,
  and no oneDNN call at all): its kernel and blocking depend on the ISA it detects (AVX-512 on Intel).
- **ATen's vectorized kernels** (the LSTM cell's sigmoid/tanh, `sum`): compiled for `default`, `avx2` and `avx512`, the
  capability chosen at startup. The Intel runners and some EPYC 9V74/9V45 draws pick avx512, the 7763 avx2.

Op by op, Intel 8370C against the EPYC 7763 on the 7763's own inputs (German `default_fast`; max |difference|):

| op | isolated | end to end | max \|value\| |
|---|---|---|---|
| char LSTM | 8.9e-7 | 8.9e-7 | 1.0 |
| char attention (Linear) | 1.9e-6 | 1.9e-6 | 8.5 |
| `trans_char` (Linear) | 3.6e-7 | 4.8e-7 | 1.6 |
| highway layer 0: LSTM / gate / highway | 1.0e-6 / 1.1e-5 / 1.1e-5 | 1.0e-6 / 1.2e-5 / 1.1e-5 | 1.0 / 42 / 37 |
| highway layer 1: LSTM / gate / highway | 6.9e-7 / 1.9e-6 / 2.9e-6 | 7.5e-7 / 2.9e-6 / 3.8e-6 | 1.0 / 14 / 14 |
| `upos_hid` (Linear, ReLU) | 1.4e-6 | 2.4e-6 | 8.4 |
| `upos_clf` (Linear 400 → 18) | 3.05e-5 (1 ulp) | 9.2e-5 | 329 |

Every GEMM differs by itself (1–2 ulps of its output); the elementwise ops add almost nothing (the highway combine
7e-9). No single op is the culprit: the last layer turns the accumulated relative error of ~3e-7 into ~3 ulps of a
logit of 285. On Intel, switching only MKL to its AVX2 path (`MKL_ENABLE_INSTRUCTIONS=AVX2`) moves `upos_clf` by 6.1e-5
on identical inputs; switching only ATen to avx2 moves the LSTMs by ~2e-7.

## Settings tried (Intel 8370C / 8573C; EPYC 7763)

| Setting | de `default_fast` worst, Intel | 7763 | Intel = AMD bit for bit? |
|---|---|---|---|
| none (today) | 1.20e-4 (fails) | 6.10e-5 | no |
| `ATEN_CPU_CAPABILITY=avx2` | 8.93e-5 | 6.10e-5 | no |
| `MKL_ENABLE_INSTRUCTIONS=AVX2` or `MKL_CBWR=AVX2` | 3.05e-5 | 6.10e-5 / 4.58e-5 | no |
| `ONEDNN_MAX_CPU_ISA=AVX2` | 1.20e-4 (no effect: no oneDNN) | 6.10e-5 | no |
| `ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=AVX2[,STRICT]` (also 1 thread) | 2.16e-5 / 6.10e-5 | 4.58e-5 | no |
| `MKL_CBWR=COMPATIBLE` | 3.05e-5 | 4.92e-5 | only with EPYCs that pick ATen's avx512 |
| **`ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE,STRICT`** | **3.05e-5** | **3.05e-5** | **yes, every stage, all 16 jobs** |

MKL's AVX2 reproducibility mode still differs between Intel and AMD here (with ATen pinned too), so only MKL's
`COMPATIBLE` code path plus ATen's non-vectorized kernels give one result on all five CPU models. With them, all four
taggers are bit-identical on every runner, and the worst German `default_fast` difference is 3.05e-5, one ulp: de
`default` 4.03e-5, en 1.29e-5 / 2.04e-5.

Cost: the TorchSharp tests run slower without the vectorized kernels. The diagnostic set (both backends) took +10% on
the 7763 (1.87 → 2.06 min), +33% on Intel (1.71 → 2.27); sentiment's English golden test on TorchSharp 46 → 59 s (AMD),
39 → 66 s (Intel). Expect the TorchSharp half of golden and cross-os (Windows) to take about a third longer. Managed
tests don't load libtorch and are unaffected.

Validation (temporary workflow, run 38521814068): the whole suite in golden's three shards with these two settings, on
9 `ubuntu-26.04` jobs (.pt models, English and German) and 3 `windows-2025` jobs (English; German skips there as in
cross-os). No test failed. Drawn: EPYC 7763 ×8, 9V74 ×2, 9V45, and Xeon 8573C once, on shard 2, which holds
`FastPackageTests`: `Intermediates_MatchGolden(lang: "de")` passed there. The only skips were the 6 `Run_PythonStanza_*`
tests (no Python in that job) and the German cases on Windows.

## Options

1. **Deterministic libtorch kernels in CI** (prepared on branch `fix/torch-intel-drift`, no PR): ci.yml's two Test
   steps (golden-shard, cross-os-shard) export `ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE,STRICT` on x64 runners
   (`RUNNER_ARCH`). No tolerance, test or library code changes; users keep libtorch's defaults. Costs: ~+30% on
   the TorchSharp tests' time; CI no longer tests the TorchSharp backend under its default (vectorized) kernels, and
   an Intel user of `StanzaSharp.Cuda` on CPU still sees up to 1.2e-4 on German UPOS logits (tags unchanged). arm64
   (macOS, windows-11-arm) is untouched: no MKL there, and those runners never drifted.
2. **Double sums in the TorchSharp tagger's head**, as the managed `upos_clf` does (library change, owner's decision):
   `upos_clf` in float64, rounded once. Measured with the diagnostic (`CLF64`): Intel German `default_fast` 1.20e-4 →
   5.88e-5, AMD 6.10e-5 → 1.53e-5; German `default` on Intel 6.10e-5 → 3.54e-5. Cost: negligible (18 outputs × 400 per
   word). It removes the last layer's rounding but not the upstream drift (Intel's `upos_hid` is 2.4e-6 off), so the
   margin on Intel would be 1.7×, not a guarantee. Extending double sums to `upos_hid` and the highway gates would cost
   more and still leave the LSTMs.
3. **A relative tolerance for large logits** (owner's decision): e.g. 1e-4 × max(1, |logit| / 100), or "within 4 float32
   ulps". At |285| float32 can't resolve better than 3.05e-5, so an absolute 1e-4 is only 3.3 ulps there; the
   English logits (|≤ 140|) are 7–13 ulps. This tests what the tolerance means (a few roundings) on every CPU, keeps CI
   on libtorch's default kernels, and needs no code change. It loosens the German check in absolute terms.
4. Others, not recommended: pinning tests to AMD runners (not possible on free hosted runners); retrying a failed
   shard (hides real drift); testing German TorchSharp UPOS logits only on the managed backend (a skipped case, against
   the owner's rule).

## Recommendation

Option 1 now: it makes golden independent of the runner's CPU without touching any tolerance or the library, and the
validation run shows the whole suite passing with it on Intel and AMD. Option 3 deserves the owner's consideration as
the long-term rule, since the failure is a float32 resolution limit (1e-4 is 3.3 ulps at |285|) rather than a
port error. Option 2 is cheap, mirrors the managed backend and halves the TorchSharp drift; worth taking with either.
