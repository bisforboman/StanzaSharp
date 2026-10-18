# Results, round 45: the transformer encoder's scalar parts

Part of [performance.md](../performance.md)'s measurement history.

Round 43 left `default_accurate` at 5.5x the default package's time, nearly all of it the two encoder passes. Timing the
managed encoder's stages (temporary stopwatches, not committed; `tokenize,mwt,pos` with `pos=combined_electra-large` on
all validation*.txt, 6,044 words, one pass) on a cloud session's 4-core Intel Xeon @ 2.10 GHz (Vector512), 2026-10-18:

| stage | before | after |
|---|---:|---:|
| QKV GEMM | 5,125 ms | 5,125 ms |
| attention | 4,728 ms | 1,725 ms |
| attention output GEMM | 1,696 ms | 1,680 ms |
| LayerNorm (both) | 1,809 ms | 266 ms |
| intermediate GEMM | 6,928 ms | 6,950 ms |
| GELU | 11,437 ms | 1,742 ms |
| output GEMM | 6,941 ms | 7,006 ms |
| **pass** | 38.7 s | 24.5 s (1.58x) |

The GEMMs run at about 230 GFLOPS, near this CPU's AVX-512 rate, so this round left them alone. The three scalar double
loops were 46% of the pass:

- **GELU** (`GeluBlock`): `Vector<double>` lanes, with an exp of its own (`Exp`: n = round(x·log₂e) by adding and
  subtracting 1.5·2⁵², a degree-12 Taylor polynomial on |r| ≤ ln 2 / 2, times 2ⁿ; within 1e-15 of `Math.Exp`) instead of
  `Math.Exp`. Only multiplies and adds, never fused, and the scalar tail runs the same operations, so every vector width
  gives the same bits. On 4,103 test values (`Gelu_EveryVectorWidthGivesTheScalarsBits_...`) no float result is more
  than one ulp from the former `Math.Exp` version.
- **Attention's weighted sum** (`WeightedAdd`): acc += p·v over d = 64 in `Vector<double>` lanes, a multiply then an add,
  the scalar loop's exact bits.
- **LayerNorm** (`LayerNormRow`): mean and variance with `Act.Dot`'s four accumulators (lane = index mod 4, the tail into
  s0, then (s0 + s1) + (s2 + s3)) per `Gemm.Path`, and the element-wise steps in lanes; every path gives the scalar path's
  bits (`LayerNormRow_EveryPathGivesTheScalarsBits`). The sums' order changed from left to right, so LayerNorm's last
  bits can move.

All 39 `Transformer*` tests pass (every golden row byte for byte on both backends; hidden states within their tolerance;
every invariance, now with Vector512 = Vector256 for the frozen encoder too). The default package doesn't read the
encoder and is unchanged.

`stanzasharp benchmark --package default_accurate` (StanzaSharp only, as in round 43, same machine):

| | round 43 | round 45 |
|---|---:|---:|
| pos (the frozen pass) | 11.02 s | 6.55 s |
| sentiment (the LoRA pass) | 11.52 s | 6.92 s |
| **total** (1,994 words) | 29.08 s | 19.33 s (1.50x) |
| per call, median / p90 | 342.0 / 532.5 ms | 264.2 / 395.5 ms |
| peak memory | 4,158 MB | 3,860 MB |

Against round 44's Python Stanza (147.78 s on the same text and machine), that is about 7.6x.

Not tried: fusing GELU into the intermediate GEMM's epilogue (round 30's `epilogue`; a locality gain, not arithmetic), the
softmax's exps (part of attention's remaining 1.7 s).
