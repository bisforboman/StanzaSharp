# Results, round 51: the managed encoder at small widths (MiniLM)

Part of [performance.md](../performance.md)'s measurement history.

Issue #281: the embedding spike ([embedding-spike.md](../embedding-spike.md)) measured the managed encoder at 1.65×
ONNX Runtime's time per core on sentence-transformers/all-MiniLM-L6-v2 (6 layers, hidden 384, 12 heads of 32,
intermediate 1536). Where does the time go, and what can go without changing a bit?

**Machine:** a cloud session's 4-core Intel Xeon @ 2.10 GHz (Sapphire Rapids: AVX-512 with two FMA units, 1 thread per
core), Linux, .NET 10, 2026-11-16. Before = main at d06afbc, after = this change, run alternately (runs vary by up to
10 %). ONNX Runtime 1.31.1 on the model's `onnx/model.onnx`.

**Method:** `StanzaSharp.Benchmark embedding-profile` (new, `samples/StanzaSharp.Benchmark/EmbeddingProfile.cs`), which
needs no Python reference: the spike's 935 texts (the golden files' sentences, those twelve at a time and each file whole;
25,259 pieces, cut at 256), longest first in batches of 32, median of 3 runs after a warm-up.

- `run MODEL_DIR [--onnx] [--save FILE]`: the whole encoder, and a SHA-256 of every output state's bits (to check a
  change keeps them); `--onnx` also times ONNX Runtime on the same batches (padded, with a mask).
- `stages MODEL_DIR`: each step of a layer on its own over the same batches, on the real kernels with the model's shapes
  (random weights), times the layer count.
- `attention N D`: one head of one N-piece sentence, ns per query–key pair. `gemm M K N`: one GEMM's GFLOP/s.

## Where the time goes (1 thread, Vector512)

| stage | before | after |
|---|---:|---:|
| QKV dense (384 → 1152) | 1.06 s | 1.11 s |
| attention (12 heads of 32) | 3.66 s | 3.14 s |
| attention output dense (384 → 384) | 0.34 s | 0.39 s |
| LayerNorm + residual (×2) | 0.14 s | 0.16 s |
| intermediate dense (384 → 1536) | 1.54 s | 1.50 s |
| GELU (with a copy of its input) | 1.83 s | 0.86 s |
| output dense (1536 → 384) | 1.51 s | 1.40 s |
| sum | 10.08 s | 8.56 s |

- **The GEMMs are not the gap.** The four dense layers (537 GFLOP over the corpus) run at 120–137 GFLOP/s, about 85–90 %
  of what this core's two 512-bit FMA units give at its AVX-512 clock (`gemm 864 384 1536`: 128 GFLOP/s;
  `864 1536 384`: 122). The 6×32 kernel's loop is 12 FMAs, 2 panel loads and 6 broadcasts per k, with nothing spilled.
- **Attention is 36 %, GELU 18 %.** Attention runs in double (round 45's exactness rule), 195 million query–key pairs over
  the corpus (2.7 million per head and layer). Per pair (n = 256): about 7.6 ns is `Math.Exp` (a call into the C
  runtime for every score), the rest the scores, the weighted sum and the per-row bookkeeping. GELU is exact GELU in
  double: two polynomials (degree 12 and 8) and a division per value, 233 million values.
- LayerNorm (1.4 %) and the embeddings are negligible.

## What changed (bit-identical)

Every output state keeps its bits: the hash of all 9.7 million states is `5DB9D5712A129655` before and after, on
Vector512, Vector256, Vector128 and Scalar alike.

- **GELU: four lane groups at a time.** One vector at a time, its chains (two Horner polynomials of 12 and 8 steps, a
  division) were too long for the CPU to overlap: about 15 cycles a value. `DoubleLanes.cs` holds four vectors that every
  operation runs on in turn (`Lanes512`: four `Vector512<double>`; `LanesVector`: four `Vector<double>`), and
  `GeluLanes<T>`, `ErfPoly<T>` and `Exp<T>` are written once over them (a static-abstract interface, so the JIT
  specialises each). The Horner loop over `ExpCoefficients` is written out. Each lane runs the scalar operations
  (multiplies and adds never fused, an IEEE division), so every width gives the scalar code's bits. Each width's loop is
  a method of its own: inlined into one method, the two lane types' code ran out of the JIT's tracked locals and the
  `Vector<double>` loop got 2.7× slower.
  - Vector512: 1.83 → 0.80 s. Vector256 (`Vector<double>`): 2.14 → 1.23 s.
- **Attention on Vector512: two keys per register.** The keys are stored a pair at a time (four doubles of one key, then
  the same four of the next) and each query twice, so one 512-bit fused multiply-add serves the four accumulator lanes
  of two keys (`ScoreBlock512`, four keys and four queries per pass). The four lanes' sum `(s0 + s1) + (s2 + s3)` is
  done for four keys at once with unpacks, one permute and one add (`Sums`), not lane by lane; Vector256 does the same
  with `vhaddpd` and two 128-bit permutes. The keys are padded with zero rows to a multiple of four (their scores are
  never read). The row maximum is a vector maximum (it propagates NaN and orders −0 below +0 like `Math.Max`; any
  order gives the same maximum). The weighted sum runs sixteen elements of four rows per pass in eight 512-bit
  registers (`WeightedBlock512`; eight where the head width is not a multiple of 16), a multiply then an add per lane
  as before.
  - `attention N 32`: n = 256: 18.2 → 14.9 ns per pair; n = 64: 18.6 → 15.4; n = 16: 25.1 → 27.1 (the padding and the
    doubled queries cost more than they save on very short texts). Over the corpus: 3.66 → 3.14 s.
- Tests: `Gelu_EveryVectorWidthGivesTheScalarsBits_AndStaysWithinAFloatUlpOfTheReference` is now a theory over the
  paths, and `BlockedAttention_GivesTheRowLoopsBits` runs head widths 32 (MiniLM) and 40 (the eight-wide weighted sum)
  besides 64. Both compare float outputs: a one-ulp change in a double accumulator almost never reaches the float
  result, so they would not catch, for example, a fused multiply-add in the weighted sum (tried: they pass); the whole-
  encoder hash above is the stronger check.

## Results

MiniLM, `embedding-profile run`, two alternating rounds each:

| setting | before | after | ONNX Runtime | managed / ORT, before → after |
|---|---:|---:|---:|---:|
| batch 32, 1 thread | 10.06–10.14 s | 8.66–8.70 s | 6.73 s | 1.50× → **1.29×** |
| batch 32, 1 thread, Vector256 | 13.23–13.31 s | 12.51–12.58 s | 6.69 s | 1.98× → 1.87× |
| batch 32, 4 threads | 2.86 s | 2.39–2.44 s | 2.04 s | 1.40× → **1.19×** |
| batch 1, 4 threads | 3.82–3.83 s | 3.36–3.42 s | 2.50 s | 1.53× → 1.36× |

(The spike's 1.65× was 11.0 s against 6.66 s on a busier VM; on this one the same code took 10.1 s.)

- **About 14 % less time on Vector512, 5 % on Vector256**, where the GEMMs (no AVX-512) are a larger share and attention
  keeps its 256-bit kernels.
- **The processors:** the GEMM is unchanged, so the `default` package's processors (all GEMM, LSTM and charlm work) run
  as before (`StanzaSharp.Benchmark --package default --copies 1`, 4 threads: 7.33–7.35 s against 7.46–7.50 s, within the
  noise). The transformer packages run the same encoder, whose attention and GELU changed, but ELECTRA-large's time is in
  its GEMMs (hidden 1024, 24 layers): `stanzasharp benchmark --package default_accurate`, 4 threads, the transformer
  stage took 7.45 and 7.26 s before, 6.82 and 7.21 s after (totals 13.42 / 13.29 s against 12.60 / 13.35 s), within the
  noise.

## Tried, not kept

- **A 6×64 GEMM kernel** (four panels, 24 accumulators: 10 loads for 24 FMAs instead of 8 for 12; the same per-element
  FMA chain, so the same bits). No measurable change at MiniLM's shapes (864 × 384 → 1152: 124–131 GFLOP/s against
  124–134) nor at the processors' small ones (6 × 384 → 1536: 86–93 against 86; 30 × 400 → 1600: 118–123 against
  92–121): the kernel is already at the FMA units' limit, and runs vary by more than the difference.
- **A vectorized exp in attention** (`Exp<T>` above, the polynomial GELU already uses, within 1e-15 relative of
  `Math.Exp`): `attention 256 32` 14.9 → 7.8 ns per pair, the whole encoder 8.66 → 8.0 s (1 thread; about 1.19× ONNX
  Runtime). It changes the arithmetic: on random arguments 16 % of its results are one double ulp from `Math.Exp`'s. On
  this corpus no output state changed (the hash stays `5DB9D5712A129655`), but nothing guarantees that for other inputs,
  so it is the owner's decision (draft PR, question on #281). It would also make attention independent of the C runtime's
  `exp`, which the encoder now calls through `Math.Exp` and which differs between libm versions.

## What is left

On one thread the managed encoder is now 1.29× ONNX Runtime's time. The rest is mostly attention in double
(round 45 chose double for exactness across paths; ONNX Runtime runs it in float) and `Math.Exp`. A float attention,
or the vectorized exp above, would change output bits.
