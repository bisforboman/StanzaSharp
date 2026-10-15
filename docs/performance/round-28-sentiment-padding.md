# Results, round 28: sentiment's padding rows and its strided filter

Part of [performance.md](../performance.md)'s measurement history.

[Round 27](round-27-whole-pipeline-profile.md) named two sentiment ideas: the padding rows' first-layer projection, and
the strided (5, 5) filter in SIMD lanes. Both are shipped. The logits are bit for bit the same on every kernel path, and
sentiment is about 22% faster on the runners whose CPU was the same in both runs.

## The changes

**Padding rows** (`ManagedLstm.ForwardUnpacked(..., lengths:)`, called by `ManagedSentimentNet.Forward`). The
sentiment biLSTM runs unpacked: every sentence of a batch runs the batch's full width, and the padding goes through it.
On the benchmark text the one batch is 163 sentences padded to 71 tokens, so 83.5% of the rows are padding. Every
padding row has the same input: PAD's pretrain row plus delta PAD, then zero charlms. Its first-layer input projection
(2148 → 2 × 1200 gates) is therefore the same for every padding row.

- With `lengths`, `ForwardUnpacked` asks the filler for one padding row only. Per direction, it projects that row twice
  at the start: once row-invariant (the arithmetic every row gets in the chunk GEMMs), and once through the single-row
  kernel.
- Each chunk fills and multiplies only its real rows, packed together. A padding row's `Init` pointer goes to the shared
  projection.
- `Gemm.RunRows` sends only global row m − 1 (m = n · width) through the single-row kernel, when m % 6 is 1. In
  time-major order that row is the last sentence's last step. When it is real, it is the last packed row of its chunk and
  runs alone through the single-row kernel. When it is padding, it points at the single-row copy. So every row keeps the
  bits the whole GEMM gave it.
- Layer 2 is unchanged: its input carries each sentence's forward state through the padding.

The optional half of the idea is not done: the shared backward recurrence over the padded tail. It was worth about
50 ms at 1 thread, and it needs per-row start states in the step loop. Batches sorted by length leave little padding in
long documents, and one-sentence calls have none.

**Strided filter** (`ManagedSentimentNet.Strided`, now `internal static` for its test). The 8 channels go in the lanes of
a `Vector256<float>`; Vector512 uses the same code. The weights are transposed at load to `[a·5 + b, channel]`
(`Conv.Transposed`). Each lane starts at its bias and adds `w · broadcast(z)` as a separate multiply and add, never an
FMA, in the scalar loop's order (a, then b). The max into the pooled output is per lane with `Math.Max`, as before.
Vector128 takes the channels four at a time, and Scalar keeps the old loops.

## Tests

- `ManagedBackendTests.ManagedLstm_UnpackedGivesThePackedBits` (every path) gains four cases with `lengths`, each
  padding row set to one shared input. Two of them have n · width % 6 == 1, one with the last row padding and one with
  it real. The output must equal `ForwardPacked` bit for bit, and the filler must be asked for exactly one padding row.
- `ManagedBackendTests.SentimentStrided_GivesTheScalarLoopsBits` (every path): 8, 4 and 3 channels against the scalar
  loop written out in the test, bit for bit.
- `SentimentTests`, the sentiment/LSTM `ManagedBackendTests` and `Vector512_GivesVector256sBits`: 30 passed locally.

## Exactness

A temporary harness (not committed) hashed the managed classifier's logits with SHA-256. The charlm outputs were
computed once (Vector256, 8 threads) and served from a `CharlmCache`, so every path ran the net on the same inputs. It
covered three cases:

- **files:** each golden `sentiment/*.conllu` as its own call (14 files in each language; 886 English sentences);
- **one:** all of them as one call (two 5,000-token batches in English);
- **alone:** 60 one-sentence calls.

It ran on Vector256, Vector128 and Scalar, at 1 and 8 threads, built from main (1fe32df) and from this branch (Ryzen 7
5800X, Windows). The first 16 hex digits are shown:

| lang | path | files | one | alone |
|---|---|---|---|---|
| en | Vector256, 1 and 8 threads | `F7696646DC55CC03` | `06E5C560C690CBF2` | `241F7BB4A5035C07` |
| en | Vector128, 1 and 8 threads | `A10B0E4A89FC1E81` | `C4D02174EBDA5B25` | `0EB30C57C0178140` |
| en | Scalar, 1 and 8 threads | `402FB535755ADC59` | `98831FDD4AD50D5E` | `265DC59992C1FF24` |
| de | Vector256, 1 and 8 threads | `3C9ACC9855031F84` | `7566BD2CDCD5342F` | `32F55F79B9CE5700` |
| de | Vector128, 1 and 8 threads | `E71BA62828118073` | `7566BD2CDCD5342F` | `D07D5D6B5A4EB738` |
| de | Scalar, 1 and 8 threads | `F58A95391066DBBA` | `D6B156B16E39DD08` | `CBEA6B45777F0C61` |

Every hash is the same on main and on the branch. This machine has no AVX-512. Vector512 runs the same strided code as
Vector256, and its GEMM and LSTM kernels give Vector256's bits by test (`Vector512_GivesVector256sBits`).

## Speed and memory

**Free runners.** These are `stanzasharp benchmark` runs: all eight processors, 2,009 words, default threads, and medians
of 3 runs. Branch run [38721863322](https://github.com/bisforboman/StanzaSharp/actions/runs/38721863322) and main run
[38722008833](https://github.com/bisforboman/StanzaSharp/actions/runs/38722008833) were started a minute apart. Main
run [38704860183](https://github.com/bisforboman/StanzaSharp/actions/runs/38704860183) is from earlier that day.

| runner | CPU (branch / main) | sentiment, main | sentiment, branch | change | total, main → branch |
|---|---|---:|---:|---:|---:|
| ubuntu-26.04-arm | Neoverse-N2 / same | 2.55 s (2.54) | 1.99 s | −22% | 7.73 → 7.15 s |
| windows-11-arm | Cobalt 100 / same | 2.54 s (2.58) | 1.99 s | −22% | 7.81 → 7.27 s |
| ubuntu-26.04 | Xeon 8573C / EPYC 9V74 (earlier run: Xeon 8573C) | 2.06 s (1.91) | 1.35 s | −29% vs the Xeon run | 5.75 (6.88) → 5.97 s |
| windows-2025 | Xeon 8573C / same | 1.72 s (1.66) | 1.57 s | −9% | 6.13 → 7.27 s |
| macos-26 | Apple M1 (virtual) / same | 3.35 s (2.55) | 1.81 s | −29 to −46% | 11.14 → 7.13 s |

The earlier main run is in parentheses. The two Arm runners drew the same CPU in all three runs and moved least between
main runs, so −22% is the cleanest number: about 0.55 s of the call, 7% of the total. On windows-2025 every other stage
of the branch run was slower too (total +19%), so its −9% is a noisy draw. macOS varies a lot between runs. The
one-sentence median is unchanged within noise: a one-sentence batch is padded only up to 5 tokens. The peak working set
is unchanged within the runners' noise (−80 to +170 MB in both directions).

**Locally** (Ryzen 7 5800X, `--threads 1`, StanzaSharp only, main and branch alternated, three pairs). Round 32's agent
was running at the same time, so these are noisy. Sentiment took 7.31, 6.25 and 13.95 s on main, and 4.40, 5.20 and
4.57 s on the branch. The branch was faster in every pair, and its median is 1.7 s lower (−27%). That is in line with
round 27's estimate of about 0.7 s of 3.45 s on an idle machine, plus about 0.13 s for the strided filter. Peak memory
was 981 to 984 MB on main and 986 to 1,042 MB on the branch. The new buffers are small (one input row, four
projections, a chunk's row map), and the padding rows are no longer filled.
