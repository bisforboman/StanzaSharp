# GPU

How StanzaSharp runs on an NVIDIA GPU, how close its output stays to the CPU (and so to Python Stanza),
and how fast it is.

## Using it

```csharp
using TorchSharp;

torch.backends.cudnn.allow_tf32 = false;       // for output identical to the CPU; see below
torch.backends.cuda.matmul.allow_tf32 = false;
using var nlp = Pipeline.Load("models/stanza/en", device: torch.CUDA);
```

The application references a CUDA libtorch package (`TorchSharp-cuda-windows` or `TorchSharp-cuda-linux`,
0.107.0) instead of `TorchSharp-cpu`. The `StanzaSharp` package itself only depends on the managed
`TorchSharp` package, on CPU and GPU alike; the GPU support is a few lines of code, not a dependency.

### How it works

- `Weights.On(device, load)` sets the device for the duration of a load. It is flow-local
  (`AsyncLocal`), so concurrent loads on other threads, and other TorchSharp code in the process, are not
  affected. `torch.set_default_device` was not used because it is process-global.
- `Weights.ToTensor` and `Weights.LoadFrom` build every weight on the CPU, as before, and then move it to
  that device. On the CPU nothing moves, so the CPU path is unchanged.
- Each model records `Weights.Device` when it is constructed and creates its input tensors there
  (`torch.tensor(ids, ..., device: _device)`). Results are read back with `Weights.ToArray<T>()`, which
  copies to the CPU only for a GPU tensor. `pack_padded_sequence` lengths stay on the CPU, as torch
  requires.
- The pretrain matrix (250,000 × 100 floats, 100 MB) lives on the GPU with the rest, so lookups are a
  GPU `index_select` on GPU indices. Keeping it on the CPU would save 100 MB of GPU memory but add a copy
  per batch; at this size it isn't worth the special case.

## Exactness

Measured on an RTX 3080 (driver 610.88, libtorch 2.10 for CUDA 12.8 from TorchSharp-cuda-windows 0.107.0)
against the golden files from Python Stanza 1.15.0 on CPU. `GpuTests` runs both comparisons.

**With TF32 off, the GPU output was identical to the golden data on every file.**

With TF32 on (libtorch's default), 3 of 845 sentences differed, one decision each:

| file (all six processors)       | sentences | identical | word lines differing | trees differing |
|---------------------------------|----------:|----------:|---------------------:|----------------:|
| validation_social.conllu        |        74 |        72 |      1 (a head)      |               1 |
| validation_tech.conllu          |        65 |        64 |      1 (a deprel)    |               0 |
| 11 other files                  |       706 |       706 |                    0 |               0 |
| **total**                       |   **845** |   **842** |  **2 of 8,663**      |       **1**     |

- validation_social, sentence 32: "a" attaches to word 3 instead of 4 (depparse).
- validation_social, sentence 43: "brown, small, very friendly." becomes
  `(FRAG (ADJP ... (ADJP (RB very) (JJ friendly))) (. .))` instead of a flat ADJP (constituency).
- validation_tech, sentence 21: deprel `compound` instead of `dep` (depparse).

On the benchmark text (1,992 sentences, 27,688 word lines; the golden texts it is built from do not include
those two files), the GPU output equals the CPU output exactly, with TF32 on and off.

### Where the drift comes from

It is TF32, not cuDNN nondeterminism:

| max \|diff\| vs golden (3 sentences) | CPU        | GPU, TF32 off | GPU, TF32 on |
|---------------------------------------|-----------:|--------------:|-------------:|
| tokenizer logits                      |    ≤ 4e-6  |       ≤ 8e-6  |      ≤ 6e-6  |
| charlm hidden states (forward)        |  ≤ 2.4e-6  |     ≤ 2.4e-6  |    ≤ 1.5e-3  |
| charlm hidden states (backward)       |      —     |     ≤ 3.2e-6  |    ≤ 7.3e-4  |
| UPOS logits                           |      —     |     ≤ 4.6e-5  |    ≤ 1.5e-2  |
| parser transition scores              |      —     |     ≤ 2.3e-5  |    ≤ 1.1e-2  |

- On Ampere and newer GPUs, libtorch lets cuDNN use TF32 (10-bit mantissa) by default
  (`torch.backends.cudnn.allow_tf32 = true`). The drift enters at the first large LSTM, the charlm
  (1024 hidden units over long character sequences): 1e-3 instead of 1e-6. Everything downstream
  inherits it. The tokenizer's small LSTM is unaffected.
- With TF32 off, the GPU is as close to Python as the C# CPU path is (about 1e-6 on hidden states,
  1e-5 on logits): ordinary float32 summation-order differences.
- The parser decision that flips in validation_social had a CPU margin of 3.7e-4 between the two best
  transitions, which the TF32 error (about 1e-2 on scores) easily crosses and float32 rounding does not.
- GPU results were the same from run to run (TF32 on or off); nothing nondeterministic showed up.
  `torch.use_deterministic_algorithms` is not available in TorchSharp 0.107 (it throws "not implemented"),
  and turned out not to be needed.
- Cost of turning TF32 off: none measurable (see below; differences were within run-to-run noise).

StanzaSharp does not change the TF32 settings itself, because they are process-wide and would affect any
other torch code in the application. Without them, expect rare flips on near-ties: about 1 in 300
sentences on the golden files, none on the 1,992-sentence benchmark text.

## Speed and memory

`samples/StanzaSharp.Benchmark` (all six processors) on the same text as docs/performance.md: 26,264
words, medians of 3 runs after a warm-up, Release build. Ryzen 7 5800X, RTX 3080 10 GB.

Seconds per run (lower is better). GPU runs use `--no-tf32` unless noted.

| stage            | CPU, 8 threads | GPU, 8 threads | CPU, 1 thread | GPU, 1 thread | GPU, TF32 on |
|------------------|---------------:|---------------:|--------------:|--------------:|-------------:|
| load             |           1.57 |           1.78 |          1.25 |          1.35 |         1.42 |
| tokenize         |           3.32 |           0.79 |          2.40 |          0.74 |         0.79 |
| mwt              |           0.02 |           0.01 |          0.02 |          0.01 |         0.01 |
| pos              |          26.87 |           1.45 |         72.37 |          1.37 |         1.44 |
| lemma            |           1.44 |           1.43 |          1.82 |          1.36 |         1.42 |
| depparse         |          18.47 |           1.01 |         72.10 |          1.01 |         1.00 |
| constituency     |          17.10 |           7.22 |         39.37 |          6.50 |         7.20 |
| **total**        |      **67.22** |      **11.91** |    **188.09** |     **10.99** |    **11.86** |
| words/s          |            391 |          2,205 |           140 |         2,390 |        2,215 |
| peak working set |       6,256 MB |       2,100 MB |      6,214 MB |      2,099 MB |     2,105 MB |

- **Caveat:** other agents ran CPU benchmarks on the same machine throughout (CPU load 55–100%). The CPU
  columns are therefore slower than an idle machine; docs/performance.md has idle numbers for the four
  original stages, for example pos 17.3 s and constituency 10.5 s at 8 threads, where this table has
  26.9 s and 17.1 s. A fair 8-thread comparison is roughly 4× in the GPU's favour, not the 5.6× in the
  table. The GPU columns suffer less, but the parser's CPU-side work slows down under load too (a
  less loaded GPU run gave 5.3 s for constituency, 9.3 s in total).
- The GPU barely needs CPU threads: 1 and 8 threads give the same times.
- pos and depparse, which run large batched LSTMs and biaffine scorers, gain 15–70×. The constituency
  parser gains least: it runs one small transition step at a time for up to 50 sentences, and every step
  reads its scores back to the CPU to choose the next transitions. Kernel launches and that
  synchronization dominate there, not arithmetic.
- lemma decodes one character per step with a read-back each step, so it runs at the same speed on both.
- Load time is about the same; on the GPU it includes copying the weights over once. The first CUDA
  call in a process also initializes the CUDA context (a second or so), which the first benchmark
  run counted (3.0 s load) and later runs, with warm driver caches, mostly did not.
- TF32 on or off made no measurable difference (11.9 s vs 11.9 s at 8 threads; earlier pairs differed by
  less than run-to-run noise).
- The peak working set (host memory) drops on the GPU because the large activations live in GPU memory.

GPU memory (whole device, `nvidia-smi`, idle desktop baseline about 960 MiB subtracted):

| processors (one run on the benchmark text) | after load | after the run |
|--------------------------------------------|-----------:|--------------:|
| tokenize, mwt, pos                         |    0.4 GB  |       1.2 GB  |
| + lemma                                    |    0.4 GB  |       1.2 GB  |
| + lemma, depparse                          |    0.5 GB  |       3.2 GB  |
| tokenize, mwt, pos, constituency           |    0.6 GB  |       1.6 GB  |
| all six                                    |    0.7 GB  |       3.6 GB  |

- "After load" is mostly the CUDA context; the weights themselves are about 0.3 GB.
- "After the run" is what torch's caching allocator keeps, which is the peak of the largest batch. The
  depparse biaffine scorers (`[batch, words, hidden, labels]` intermediates) dominate. Across the
  benchmark's warm-up plus 3 runs the device peaked at 5.7 GB above idle.

## Running it locally

```powershell
$env:STANZASHARP_CUDA = '1'   # Cli, Benchmark and Tests reference TorchSharp-cuda-windows (several GB download)
$env:STANZASHARP_MODELS = 'C:\path\to\models\converted\en'
dotnet test tests/StanzaSharp.Tests --filter FullyQualifiedName~GpuTests --logger "console;verbosity=detailed"
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --device cuda --no-tf32
```

- `GpuTests.cs` is compiled only in that build, so the CPU suite has nothing to skip and CI is unchanged.
  Without a CUDA device the GPU tests skip.
- Each GPU test runs with TF32 on and off. Intermediates must stay within the CPU tests' tolerances with
  TF32 off (50× those with it on). The pipeline test prints the per-file table above and fails only if
  more than 0.5% of word lines or 5% of trees differ.
- The CPU and CUDA builds share `obj/`; switching restores the other package. Don't run both at once in
  one checkout.
- Not measured: Python Stanza on the GPU (it would need a CUDA torch in a separate venv).
