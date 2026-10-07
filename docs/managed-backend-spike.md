# Managed backend spike (issue #29, Phase 0)

Can pure managed C# run Stanza's LSTM-heavy paths close enough to libtorch in speed, and within tolerance in
values? This spike ports the two hottest paths, measures them against TorchSharp on the tagger's real batches,
and extrapolates to the full pipeline. Nothing in the shipped pipeline uses this code.

## What was built

All of it is `internal`, in `src/StanzaSharp.Nn/Managed/`. It uses only what .NET 10 ships: `System.Runtime.Intrinsics`
(`Vector256`, `Fma`, `Vector256.Exp`) and `System.Threading`. It has no package dependency; `System.Numerics.Tensors`
is a NuGet package, not in-box, and isn't needed. `StanzaSharp.Nn` gained `AllowUnsafeBlocks`.

- **`Gemm.cs`**
  - `PackedMatrix`: a PyTorch-layout weight [N, K], packed once at load into 16-column panels ([K][16]). A row
    permutation is optional.
  - `Gemm.Run`: C = A·Wᵀ + bias, with a 6×16 AVX2/FMA register-blocked micro-kernel (12 accumulators). There are
    1-row and 3-row kernels for small batches. Blocking is K 256, M 96, N 128, and tasks are spread over threads.
  - `ManagedThreads`: the thread count (the twin of `PipelineOptions.Threads`). It runs a persistent thread team
    that meets at a `Barrier`, which spins briefly before blocking.
- **`PackedLstm.cs`**: a (bi)LSTM with PyTorch semantics (gates i, f, g, o; b_ih + b_hh; given initial states)
  over packed batches, matching `pack_padded_sequence` → `nn.LSTM`.
  - The input projection is one GEMM over all rows.
  - Each time step is a GEMM over h_{t-1}·W_hhᵀ. W_hh's rows are permuted so each 16-column panel holds the four
    gates of four hidden units, so the cell update (σ, tanh, c, h) is fused into the GEMM epilogue. A step is
    one parallel region, with no second pass.
  - The two directions of a biLSTM run in the same region.
- **`ManagedModels.cs`**
  - `ManagedCharLanguageModel.BuildCharRepresentation`: the same ids, packing and outputs as
    `CharLanguageModel`. The input is an embedding, so emb·W_ihᵀ + b_ih + b_hh is a table computed once at load
    ([948, 4096], 15.5 MB per direction). That saves the input GEMM, about 9% of the charlm's work.
  - `ManagedHighwayLstm`: the tagger's 2-layer highway biLSTM (packed path). One GEMM per layer computes both
    directions' input projections plus the gate and highway layers (N = 2400), because they all read the same
    input.
- **Harness**: `StanzaSharp.Benchmark managed-spike check|speed|gemm`
  (`samples/StanzaSharp.Benchmark/ManagedSpike.cs`) runs both implementations on the tagger's batches, built like
  `PosTagger.Process` (250 sentences or 5000 words, `simplify_punct`, packed order).
- **Tests**: `tests/StanzaSharp.Tests/ManagedBackendTests.cs`
  - GEMM vs `linear` (3 shapes and thread counts)
  - a random packed biLSTM vs `nn.LSTM`
  - charlm vs the golden tensors
  - highway vs `HighwayLstm` on the POS checkpoint

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- managed-spike check --models models\converted\en
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- managed-spike speed --models models\converted\en --threads 8 [--copies 8 --runs 3 --impl both|torch|managed]
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- managed-spike gemm --threads 8
```

## Correctness

`check` covers corpus.txt and every validation*.txt: 13 files, 845 sentences, 8,401 words, in the tagger's
batches. All values are max |diff|.

| comparison | max abs diff |
|---|---:|
| charlm vs golden (Python, 3 sentences): managed / TorchSharp | 2.2e-6 / 2.5e-6 (forward), 2.0e-6 / 2.0e-6 (backward) |
| charlm forward, managed vs TorchSharp (tagger batches) | 1.05e-5 |
| charlm backward, managed vs TorchSharp | 1.43e-5 |
| *baseline:* TorchSharp charlm, one sentence alone vs batched | 4.5e-6 |
| highway biLSTM output, same input | 2.0e-6 |
| highway output, managed charlm + managed highway vs all TorchSharp | 6.2e-6 |
| UPOS logits, managed charlm + highway → TorchSharp heads, vs `PosTagger.Predict` | 1.4e-4 |
| **words whose UPOS/XPOS/feats differ from `PosTagger.Predict`** | **0 of 8,401** |

- The managed charlm is as close to Python as TorchSharp is.
- Against TorchSharp, the managed charlm differs about as much as TorchSharp differs from itself when the batch
  changes (the charlm is not batch-invariant).
- The UPOS logits drift 1.4e-4, which is above today's 1e-4 test tolerance for logits vs golden. That is the
  charlm's 1e-5 amplified through the tagger. It is the same order as the `CharlmCache` drift already accepted
  for sentiment (≈1e-4, tested at 1e-3).
- Every tag is identical. A TorchSharp replica of the heads, run on TorchSharp's highway output, also gives 0
  differences, which checks the harness itself.

## Speed

- **Input:** the benchmark's text (8 copies: 1,992 sentences, 26,264 words, 8 tagger batches; 124,288 charlm
  steps per direction).
- **Measured:** "charlm" is the forward and backward charlm over every batch, exactly what the tagger runs.
  "highway" is the highway biLSTM over every batch's real input [words, 2248].
- **Runs:** a warm-up, then 3 runs with the order of managed and TorchSharp alternated. The tables show medians.
- **Machine:** AMD Ryzen 7 5800X (8 cores / 16 threads, AVX2 + FMA, no AVX-512), Windows 11, .NET 10,
  TorchSharp 0.107 / libtorch 2.10.
- **Load:** other agents' test runs kept the machine at 70–100% CPU throughout. These are not quiet-machine
  numbers. The docs note that libtorch's spinning OpenMP threads suffer most under load, and this spike's
  barrier spins too.

| threads | charlm, TorchSharp | charlm, managed | ratio | highway, TorchSharp | highway, managed | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.14 s | 27.90 s | **0.71** | 4.34 s | 4.90 s | **1.13** |
| 8 | 13.42 s | 5.77 s | **0.43** | 1.63 s | 0.95 s | **0.58** |
| 16 | 12.89 s | 5.43 s | **0.42** | 1.41 s | 0.70 s | **0.50** |

Runs (seconds): 1 thread, charlm TorchSharp 39.14/40.41/37.88 vs managed 30.21/27.42/27.90, highway
4.72/4.34/4.01 vs 4.90/5.15/4.71. 8 threads: 13.07/14.96/13.42 vs 6.06/5.65/5.77, highway 2.35/1.63/1.44 vs
0.93/0.95/0.97. 16 threads: 12.89/14.63/12.33 vs 5.43/5.03/5.48, highway 1.87/1.41/1.31 vs 0.67/0.72/0.70.

GEMM alone (`gemm` mode, GFLOP/s, same load; single runs, so ±15%):

| shape [M,K]×[K,N] | 1 thread managed / torch | 8 threads managed / torch |
|---|---:|---:|
| 250×1024×4096 (charlm step, full batch) | 74–81 / 80–89 | 288 / 230 |
| 32×1024×4096 | 69–76 / 48–57 | 157 / 109 |
| 1×1024×4096 (charlm tail; memory-bound) | 13 / 13–19 | 39 / 16 |
| 3000×2248×2400 (highway input) | 87–95 / 92–103 | 466 / 402 |
| 250×200×800 (highway step) | 82 / 88–95 | 262 / 427 |

- At one thread the micro-kernel is on par with MKL/oneDNN: within about 10%, behind on large square-ish shapes
  and ahead on small-M ones. The peak on this CPU is about 144 GFLOP/s per core, so both reach about 55–65%.
- The managed LSTM wins mostly through structure:
  - one region per time step, with the cell fused in;
  - the charlm's input-projection table;
  - no per-step tensor allocations or dispatch, where libtorch's packed LSTM slices, allocates and launches
    several ops per step.
- The highway layer at 1 thread is the one place managed is slower (13%). Its time is the [words, 2248]×2400
  GEMM, where MKL's tuned blocking is a little better.

Peak working set (one implementation per process, 8 threads; both processes hold both sets of models):

| | TorchSharp | managed |
|---|---:|---:|
| peak | 1,695 MB | 1,960 MB |

The managed peak is about 265 MB higher. The spike allocates fresh arrays per call (each highway layer's
[words, 2400] projection is 48 MB at 5000 words, plus the outputs), and the GC returns large-object-heap arrays
lazily. Pooled buffers (`ArrayPool`) would remove most of it. Not done.

### What was tried

| change | effect (8 threads unless noted) |
|---|---|
| 6×16 AVX2/FMA micro-kernel, weights packed once at load | GEMM on par with libtorch at 1 thread |
| fused cell update in the GEMM epilogue (permuted W_hh) | one parallel region per step, no gate buffer |
| charlm input projection as a per-character table | −9% FLOPs, no input GEMM |
| fused LSTM + gate + highway input GEMM (N = 2400) | one big GEMM per highway layer |
| 1-row and 3-row kernels for the small batches of the sequences' tails | M = 1: 10 → 13 GFLOP/s at 1 thread (memory-bound); charlm 0.84 → 0.63 s on one copy |
| persistent `Barrier` team instead of `Parallel.For` per step | within noise under load (ratio 0.49 → 0.42 on one copy) |
| not done: K-blocking the recurrent GEMM, AVX-512, running both charlm directions in one region | the last one would put 32 MB of weights through a 32 MB L3 in the memory-bound tails |

Returns flattened after the small-row kernels, which is where the spike stopped.

## Extrapolation to the full pipeline

- **Base:** the 8-processor benchmark on this branch under the same load, 8 threads: load 1.83, tokenize 4.45,
  mwt 0.02, pos 20.05, lemma 1.89, constituency 21.34, depparse 19.31, sentiment 9.67, ner 22.81; **total 99.55 s**.
  For comparison, docs/performance.md measured 54.5 s on a quiet machine.
- **pos:** charlm 13.4 s + highway 1.6 s of its 20.05 s, as measured above.
- **depparse:** assumed to spend as much in the charlm as pos does (≈13.4 s): it runs both charlms again over
  the same words plus ROOT, uncached.
- **Rest:** everything else (constituency, ner, sentiment, lemma, tokenize, the rest of pos and depparse) is 69.3 s.
  Its managed speed is not measured. It is a mix of LSTMs and GEMMs (like the highway layer) and many small ops
  (constituency's per-step stack LSTMs, biaffine, CRF, conv), where removing TorchSharp's per-op overhead may help
  or a naive port may hurt.

| scenario, 8 threads | rest × | estimated total | vs TorchSharp |
|---|---:|---:|---:|
| only the measured kernels change | 1.0 | 1.8 + 11.5 + 0.95 + 69.3 = **83.6 s** | 0.84× |
| rest ported as well as the highway | 0.58 | **54.5 s** | 0.55× |
| rest ported badly | 1.5 | **118 s** | 1.19× |

At 1 thread the ratios are 0.71 (charlm) and 1.13 (GEMM-bound layers), so the pipeline would land at roughly
0.8–1.05× of today's time. The charlm in pos and depparse is the largest share of the single-threaded pipeline
(39 s per pass here), and the GEMM-heavy stages cost up to 13% more. This is not measured on the whole pipeline.

Those are under-load numbers. On a quiet machine libtorch loses less to spinning, so the 8-thread gap will
shrink. The 1-thread ratios are the most load-independent evidence. Rerun `managed-spike speed` on an idle
machine before deciding.

## Recommendation: **go** (with conditions)

- **Speed:** managed C# reaches libtorch's GEMM throughput on AVX2 and beats it on the LSTM recurrences that
  dominate this pipeline.
- **Values:** every value is within tolerance, at the same distance from Python as TorchSharp, and every POS tag
  is identical on 8,401 words.
- The speed risk the issue named is retired for x64/AVX2.

What is left is engineering, plus these risks:

1. **Other architectures.**
   - `Vector256` is not accelerated on Arm64. linux-arm64 / Apple Silicon (a main gain of #29) need a
     `Vector128`/NEON kernel (e.g. 8×8 with 32 registers). Its speed is unmeasured.
   - AVX-512 machines would want a `Vector512` kernel. It is untested here (no AVX-512).
   - One portable `Vector<T>` kernel plus tuned x64/Arm64 variants is the likely shape.
2. **Near-tie exactness.**
   - Last-bit differences are certain. The 1.4e-4 UPOS logit drift is above the 1e-4 logits-vs-golden
     tolerance, so that tolerance would go to 1e-3 (as sentiment already has).
   - Discrete outputs flip only at near-ties: argmax, Viterbi, Chu-Liu/Edmonds, and the parser's transition
     argmax over thousands of steps.
   - Each processor's port must pass the byte-identical golden tests. A flip there would need investigation,
     not a tolerance.
   - Most exposed: sentiment's unpacked padded biLSTM + convolutions (173 of 886 labels already change with
     batching in Stanza itself), and depparse's padded log-softmax.
3. **Sentiment's Conv2d.** The (k, 600) filters over a [len, 600] batch are GEMMs whose A rows are overlapping
   windows of contiguous memory (row pointer = x + t·600, K = k·600). The kernel's per-row pointers already
   handle that, with no im2col. The (5, 5) stride-(1, 5) filter has 8 channels × K = 25: small direct code.
   Low risk.
4. **Lemma: LSTMCell decoder + attention + copy gate.** Greedy decoding over batches of 50, tiny matrices. It is
   dominated today by per-op overhead, which managed code doesn't have, and exposed to near-ties in the argmax
   of each decoded character.
5. **Constituency's per-step stack LSTMs.** Thousands of small steps (50 states in flight), many small ops per
   step. The parser is the most code to port and the most exposed to near-ties (one flipped transition changes
   a tree).
6. **Threading model.**
   - The spike has one process-wide team behind a lock, so concurrent `Process` calls would take turns.
   - libtorch gives each calling thread its own OpenMP team. That is today's thread-safety contract (8 callers
     × 8 threads measured).
   - A real backend needs per-caller teams or a work-stealing design, plus `CancellationToken` checks at the
     same places as today.
7. **Memory.** Allocate per call from pools (see the peak above). The charlm tables add 31 MB of weights.
8. **GPU** goes away unless TorchSharp stays as an optional backend.

## Questions for the owner (issue #29, 1–3), with these numbers

1. **GPU.** Is CPU-only acceptable? On this CPU the managed backend is as fast as or faster than libtorch, so
   the CPU path loses nothing. GPU users would need the TorchSharp backend as an optional package. That means
   keeping the backend seam permanently, which roughly doubles the Nn surface to test.
2. **Speed budget.** The data says "no regression at 8 threads, ±15% at 1 thread" is achievable on x64.
   - Is a slower Arm64 path (unmeasured) acceptable at first, given that today Arm64 Linux doesn't run at all?
   - Is relaxing logit tolerances from 1e-4 to 1e-3, with discrete outputs still byte-identical, acceptable?
3. **Version.** Dropping `Device`/`DisableTf32` and the `StanzaSharp.Cpu.*` platform packages is breaking.
   With per-processor migration (Phase 2) behind an internal seam, the break happens only in Phase 3, so 0.5
   could ship the managed backend as default with TorchSharp optional, and 1.0 could remove it. Which does the
   owner want?
