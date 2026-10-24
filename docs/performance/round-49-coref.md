# Results, round 49: coref on the managed backend

Part of [performance.md](../performance.md)'s measurement history.

Issue #211. Coref (`Processor.Coref`, Stanza's `udcoref_xlm-roberta-lora` on `FacebookAI/xlm-roberta-large`) runs the
whole document through XLM-R large in windows of 510 subwords, with LoRA adapters on query, value and both dense
outputs (rank 64, alpha 128). Then the coref head runs: word vectors, the bilinear rough scorer and its top 50, the
anaphoricity pairs, the span predictor and the zeros predictor. #207 measured the English golden texts at 19 min managed
against 1.2 min on TorchSharp, on a 4-vCPU cloud VM. This round profiles where the time goes, keeps what measured faster
without changing a bit, and puts the result next to Stanza on the runners.

## Where the time went

Measured on a cloud session's 4-core Intel Xeon @ 2.80 GHz with 4 threads and Vector256. .NET doesn't accelerate
`Vector512` on this CPU by default, although it has AVX-512. The text was `tokenize,mwt,pos,coref` on
`tests/golden/validation_long.txt`: 1,151 subwords in 3 windows. Stopwatch marks around every encoder stage and every
coref-head call (a throwaway patch, not committed); ms for one call, the second of a warm process.

| stage | before | after |
|---|---:|---:|
| qkv GEMM | 782 | 779 |
| LoRA on query and value | 131 | 114 |
| **attention** | **4,468** | **1,361** |
| attention output GEMM | 258 | 242 |
| LoRA on the attention output | 63 | 61 |
| LayerNorm 1 | 21 | 18 |
| intermediate GEMM | 1,058 | 1,056 |
| LoRA on the intermediate | 195 | 177 |
| GELU | 276 | 265 |
| output GEMM | 1,090 | 1,023 |
| LoRA on the output | 116 | 109 |
| LayerNorm 2 | 20 | 19 |
| **encoder** | **8,481** | **5,228** |
| word vectors, rough scores | 112 | 40 |
| anaphoricity | 452 | 492 |
| span predictor | 86 | 82 |
| zeros predictor | 4 | 6 |
| **whole call** | **9.85 s** | **6.57 s** |

- **Attention was over half the encoder.** It ran per (query, key) pair: one `Act.Dot` of 64 floats widened to double,
  then one `WeightedAdd` of the 64 values into a buffer in memory. On 510-piece windows that was 4.2M pairs per layer and
  window. ELECTRA-large's sentences (round 45) are a few dozen pieces, so this never showed there.
- **The dense GEMMs are near the machine's peak.** The qkv GEMM computes 1,151 × 1,024 × 3,072 × 2 × 24 = 174 GFLOP in
  0.78 s, about 222 GFLOP/s on 4 cores; the intermediate and output GEMMs are the same. #207's "10 GFLOP/s" was the
  whole encoder, attention included, on a slower VM.
- **The LoRA adapters cost 0.5 s, 6% of the encoder.** The B GEMMs (K = 64) run at about 110 GFLOP/s and the A GEMMs at
  about 220. The scaled add was a scalar loop.
- **The coref head is about 8%** of the whole call. Anaphoricity is a GEMM over 50 pairs per word, 1,024 → 1,024, close
  to peak. The 50 includes masked pairs, whose scores feed the start-of-chain score, so none can be skipped.

## The change

- **Blocked attention, same bits** (`ManagedTransformer.AttentionBlocked`):
  - Each (sentence, head) task widens its keys and values to double once.
  - Query rows then go four at a time. The scores come from a 4-row × 2-key register-blocked kernel, then the maximum and
    the exponentials. The weighted sums of the values come from a 4-row × 8-element kernel (Vector128: 2 rows × 8).
  - Every element gets the row loop's operations in its order. A dot product keeps `Act.Dot`'s four lane accumulators
    (q mod 4), then (s0 + s1) + (s2 + s3). A widened float times a widened float is exact in double, so a fused
    multiply-add rounds just as Dot's add does. The weighted sum is WeightedAdd's multiply-then-add, never fused, in key
    order, then divided by the sum.
  - The row loop stays for the Scalar path (and a head width that isn't a multiple of 8).
    `TransformerKernelTests.BlockedAttention_GivesTheRowLoopsBits` compares both, bit for bit, on every native path for
    1 to 131 rows.
  - `Math.Exp` stays. It is now about a quarter of attention (27 ms of 110 ms per window and layer on 1 thread), but a
    vectorized exp would change bits.
- **LoRA's scaled add vectorized** (`AddScaled`): a multiply and then an add per lane, so the scalar loop's bits.
  `AddScaled_GivesTheScalarLoopsBits`.
- **A latent stride fix.** The B GEMM read A's products with stride `A.N`, but they were written with `A.PaddedN`. These
  are the same for any rank that is a multiple of 16, as all of Stanza's are (64), so nothing changes for them.
- **benchmark.yml has a `processors` input** (`tokenize,coref`). With coref it downloads the model and XLM-R on every run
  and installs the transformer requirements for Python Stanza. The avx512 jobs now print their blocks into the log too.

Every coref golden is byte-identical on both backends: `TransformerCorefTests` (English, locally; golden shard 4 runs
German and the ELECTRA/RoBERTa kernel tests).

## Tried and dropped

Each was tried in alternating processes, three rounds, against the change above:

- **GELU in the intermediate GEMM's epilogue** (and in the adapter's epilogue when the intermediate has one). It measured
  515–565 ms against 244 + 279 ms, within noise. GELU is compute-bound (the erfc polynomial in double), so saving a pass
  over memory saves nothing.
- **The query and value adapters' A matrices stacked into one GEMM** (each padded to whole panels, so the bits are the
  same). The difference was under 3%, within noise.
- **The scaled add in the B GEMM's epilogue.** It was slower than the separate pass: the adapters on query, value,
  attention output and output took 162 / 82 / 137 ms with it, against 113 / 60 / 112 without. Dropped; only the
  vectorized add was kept.
- **A 4-row × 1-key score kernel** (the first version). The 4 × 2 kernel is 3–5% faster on attention alone.

## Results

### Locally (same machine, both builds alternating)

`stanzasharp benchmark --processors tokenize,coref` (the default text: 2,009 words in 163 sentences, 3 timed runs), on
the 4-core Xeon above. The first pair was run with Python Stanza (torch 2.14.1, MKL with AVX-512), the second without.

| | before (15fc52c) | after (ce4e5d0) | Python Stanza |
|---|---:|---:|---:|
| coref, run with Stanza | 21.67 s | 14.41 s | 11.11–12.25 s |
| coref, StanzaSharp alone | 24.28 s | 16.29 s | |
| coref, after, `DOTNET_PreferredVectorBitWidth=512` | | 13.12 s | |
| one-sentence call, median | 128.8 ms | 128.6 ms | 1,265–1,330 ms |

About a third faster (−33% and −34%), with CoNLL-U identical to Stanza's. One-sentence calls don't change: they are bound
by reading the 2.24 GB of weights.

### Runners

benchmark.yml runs [40085622197](https://github.com/bisforboman/StanzaSharp/actions/runs/40085622197) (before) and
[40086051460](https://github.com/bisforboman/StanzaSharp/actions/runs/40086051460) (after), `--processors
tokenize,coref`, the same text. Each value is the median of 3 timed runs. Each runner draws its CPU per job, so only
rows on the same CPU are a before/after.

| runner (CPU, SIMD) | before | after | Python Stanza | after vs Stanza |
|---|---:|---:|---:|---:|
| ubuntu-26.04-arm (Neoverse-N2, Vector128) | 20.56 s | 16.41 s | 16.87–16.90 s | 1.03x |
| windows-11-arm (Cobalt 100, Vector128) | 20.67 s | 16.55 s | 61.58–61.66 s | 3.72x |
| macos-26 (Apple M1, virtual, 3 cores, Vector128) | 23.89 s | 20.54 s | 14.05–15.25 s | 0.68x |
| ubuntu-26.04, before (EPYC 9V45, Vector512) | 12.13 s | | 17.57 s | (1.45x before) |
| ubuntu-26.04, after (EPYC 7763, Vector256) | | 19.27 s | 20.19 s | 1.05x |
| windows-2025, before (EPYC 9V74, Vector512) | 22.80 s | | 21.30 s | (0.93x before) |
| windows-2025, after (EPYC 9V45, Vector512) | | 7.68 s | 15.45 s | 2.01x |

- **Arm64 gains 20%** on both Linux and Windows, with steady runs (every timed run within 4%). That is less than the
  third on x64; the runs don't show which stage takes the difference.
- **macOS gains 14%,** but its runs spread from 19 to 25 s.
- **x64:** no runner drew the same CPU twice. On the EPYC 9V45 (Zen 5, Vector512) StanzaSharp is 2x Stanza after the
  change, against 1.45x before (on Linux).
- **The Windows and macOS outputs differ from Stanza's** (143 and 130 of 163 sentences), before and after alike. CLAUDE.md
  explains why: StanzaSharp reproduces torch.topk's order as libstdc++ gives it. PyTorch on Windows (MSVC) and on macOS
  (libc++) picks near-tied candidates in another order, so Stanza's own chains differ there, and a chain index that
  differs shifts every later sentence's `coref_chains=`. On Linux both sides are identical.

The avx512 jobs (Vector512 against Vector256 on Intel AVX-512 runners) are in run
[40087862482](https://github.com/bisforboman/StanzaSharp/actions/runs/40087862482).

## Remaining ideas

- `Math.Exp` in attention, now about a quarter of its time. A vectorized exp would need the owner's go-ahead, since it
  changes bits (the goldens would likely stay the same, but not bit for bit).
- The LoRA B GEMMs (K = 64) run at half the big GEMMs' rate. A kernel for small K could save about 0.15 s of 5.2.
- Vector512 for attention, with two keys' four lane accumulators in one register, would keep the bits. Attention is now
  a quarter of the encoder, so the gain is bounded.
