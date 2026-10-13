# Results, round 17: startup (load and the first calls)

Part of [performance.md](../performance.md)'s measurement history.

A short-lived process (a CLI run, a serverless function, a one-off job) pays for `Pipeline.Load` and for its first
calls, which run while the JIT is still at tier 0. Output unchanged, bit for bit (below).

**Method.** `StanzaSharp.Benchmark startup` loads the default package (all eight processors, default threads) in a
fresh process and makes a few `Process` calls, each with the methods the JIT compiled meanwhile (`JitInfo`) and the
process's CPU time; `tools/startup.py` runs it once per process under each setting, the settings alternating within
each round, and tabulates medians and the CoNLL-U hash. The Benchmark workflow's `startup` jobs (`gh workflow run
benchmark.yml -f startup_only=true [-f startup_base=origin/main] [-f startup_settings=...]`) do that on the five free
runners, with Stanza's `.pt` files, for two texts: one 13-word sentence per call (5 processes, 6 calls) and the
benchmark text, 2,009 words per call (3 processes, 4 calls). Settings: a plain build (JIT), a ReadyToRun publish of
the same app (`-r <rid> -p:PublishReadyToRun=true`), `DOTNET_*` knobs, and with `startup_base` the library at another
ref under the same harness. Load peaks are the working set's peak after `Load` (not measured on macOS).

## Where the time went

Locally (Ryzen 7 5800X, Windows; a machine busy with other work, so only shares are reliable), `Pipeline.Load` took
2.2–3.0 s with converted models and 3.5–4.2 s with `.pt` files:
- **Checkpoint reads:** parsing (`Checkpoint.Load`: the JSON, or the pickle mirrored into JSON) 0.6–0.9 s converted,
  1.6–1.9 s `.pt`; tensor bytes 0.2–0.5 s (630 MB). Almost all of the parsing is the two 250k-word vocabularies:
  the pretrain's and NER's `word` vocab (conll17's again). Warm, in a loop, the same reads take a fraction of that
  (pretrain `.pt` 0.29 s + 0.04 s for its vocabulary, converted 0.04 + 0.07 s; NER 0.33 + 0.06 / 0.05 + 0.12 s): the
  rest of the cold cost is the parsers' code at tier 0.
- **JIT:** 700–1,200 methods, 0.1–0.35 s of JIT time on the runners (`JitInfo`, all threads).
- **Packing** (repacking weights into GEMM panels, `BiaffinePerUpos`, gate reordering) and a GC after each model: the
  rest, about half. Each model's load is one thread's work, and the models loaded one after another.

The first one-sentence call compiled another ~770 methods. A JIT summary (`DOTNET_JitDisasmSummary=1`) of the first
three calls showed what ran unoptimized:
- **`VectorMath.ExpSingle`**, inside every sigmoid and tanh: the runtime's `Vector128/256/512.Exp` is too large to
  inline, and its ReadyToRun code is rejected (it depends on the machine's vector width), so it started at tier 0 even
  inside our fully optimized kernels (the LSTM step, the GEMM epilogues), with its `Widen`/`Narrow` helpers as calls.
- **The thread pool's region machinery** (`ManagedThreads.For`, `Pool.Run`, `Region.Drain`, `PackedLstm.Step`), which
  an LSTM enters once per time step, thousands of times per call, stayed at instrumented tier 0 for the first calls.
- `Act.LstmCell` and `Act.AddGatedTanh` (inlined into optimized callers) were compiled at tier 0 for the callers still
  at tier 0 (the constituency parser's stack LSTM, the lemmatizer's decoder, the highway LSTM).

So on the runners the 4th to 6th one-sentence call of a fresh process took 88–218 ms, while a warmed process
(`stanzasharp benchmark`) takes 63–136 ms.

## What changed

- **`VectorExp`** (`Nn/Managed/VectorExp.cs`): .NET 10's `ExpSingle`, copied operation for operation (the same double
  vector width, the same conversions, `MultiplyAddEstimate` in the same order), `AggressiveInlining |
  AggressiveOptimization`. `Act.Sigmoid` uses it. It gives the runtime's bits for all 2^32 float inputs (checked on
  Vector256, with AVX off and with AVX2 off, and on AVX-512 on a runner); `ManagedBackendTests.VectorExp_GivesTheRuntimesBits`
  checks every 61st pattern on each machine CI runs.
- **`AggressiveOptimization`** on `ManagedThreads.For`, the pool's `Run`, `Help`, `Work` and `Region.Drain`,
  `PackedLstm.Step`, `Act.LstmCell` and `Act.AddGatedTanh`: optimized from the first call. Optimized callers still
  inline the last two (checked with `DOTNET_JitDisasm`), and none of them gained from PGO: the warmed calls 4+ on the
  benchmark text were unchanged on every runner (table below).
- **Two models load at a time** (managed backend; `BackendModels.LoadInParallel`, `Pipeline.LoadLanes`): the pretrain,
  the charlms, tokenize, mwt and lemma first, the pretrain's readers once it is loaded, at most two at a time on the
  thread pool, each still followed by a GC. TorchSharp loads as before (its device and dispose scopes are per thread).
  A load failure still throws the first failing model's own exception, after the other loads finish.

Every model at once, each followed by a GC, took 28–46% less time on the runners and peaked 95–152 MB higher (without
the GCs 250–310 MB); two at a time keeps most of the gain for a quarter to a half of the extra memory: 23–40% less
time, 22–72 MB more
(run [38315161541](https://github.com/bisforboman/StanzaSharp/actions/runs/38315161541)):

| runner | load, one at a time | two at a time | all at once (GC after each) | load peak MB: one / two / all |
|---|---:|---:|---:|---|
| macos-26 | 1,558 ms | 1,177 | 1,001 | - |
| ubuntu-26.04 | 2,613 | 1,565 | 1,409 | 847 / 869 / 969 |
| ubuntu-26.04-arm | 1,787 | 1,260 | 1,215 | 937 / 971 / 1,089 |
| windows-11-arm | 1,601 | 1,223 | 1,017 | 904 / 956 / 1,054 |
| windows-2025 | 1,253 | 969 | 906 | 839 / 911 / 934 |

**Before and after** (run [38317542272](https://github.com/bisforboman/StanzaSharp/actions/runs/38317542272): main
and this branch under the same harness, alternating; ms; one-sentence text, then the benchmark text):

| runner | load | load peak MB | call 1 | calls 4–6 | load + 6 calls | benchmark text: load + 4 calls | calls 4+ (warm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| macos-26 | 1,124 → **890** | - | 192 → 183 | 115 → **95** | 1,889 → **1,551** (-18%) | 28.0 → 27.2 s | 6,518 → 6,098 |
| ubuntu-26.04 | 1,773 → **1,251** | 849 → 938 | 216 → 192 | 88 → **71** | 2,547 → **1,839** (-28%) | 28.4 → 26.9 s | 6,269 → 6,105 |
| ubuntu-26.04-arm | 1,845 → **1,340** | 936 → 1,000 | 262 → 239 | 135 → **95** | 2,806 → **2,219** (-21%) | 37.2 → 36.2 s | 8,466 → 8,467 |
| windows-11-arm | 1,673 → **1,291** | 904 → 965 | 293 → 264 | 218 → **104** | 2,934 → **2,365** (-19%) | 36.8 → 36.0 s | 8,400 → 8,326 |
| windows-2025 | 2,301 → **1,439** | 839 → 864 | 329 → 304 | 190 → **118** | 3,587 → **2,540** (-29%) | 36.5 → 35.2 s | 7,912 → 7,896 |

The load takes 21–37% less, its peak is 25–89 MB higher (+3–11%; a call on a longer text peaks higher anyway, about
1.0–1.1 GB on the benchmark text), the first call 5–11% less and the next ones 17–52% less. `stanzasharp benchmark`
on the branch (run [38317573839](https://github.com/bisforboman/StanzaSharp/actions/runs/38317573839)) loaded in
1.2–1.8 s on four runners (3.2 s on macos-26) against 1.7–4.3 s in the latest run above, with identical CoNLL-U to
Python Stanza on every runner; its words/s were within the run-to-run spread.

Tried and not kept: the NER checkpoint read started before the pretrain was loaded (the two 250k-word vocabularies
are the longest chain: pretrain, then NER): with every model at once another 5–10% off the load locally, but 100 MB
more peak with converted models.

## ReadyToRun

`PublishReadyToRun` doesn't apply to a library's own package; a library ships ReadyToRun code by compiling its
assemblies with Crossgen per RID and packing them as `runtimes/<rid>/lib/net10.0/` assets. Measured with a ReadyToRun
publish of the app instead (`-r <rid> -p:PublishReadyToRun=true`), which compiles every assembly the app ships, ours
included, so it is what such a package would give:
- **Size:** the package's assemblies are 470 KB of IL and 1.2 MB with win-x64 ReadyToRun code (600 KB zipped). One
  set per RID (linux-x64, linux-musl-x64, linux-arm64, win-x64, win-arm64, osx-arm64) would take the package from
  277 KB to about 3.9 MB.
- **What it covers:** with ReadyToRun the first three one-sentence calls compiled about 185 of our methods instead of 876,
  but not the hot ones: the GEMM kernels are `AggressiveOptimization` (never precompiled), and every method touching
  `Vector256`/`Vector512` (`Act.*`, the runtime's own `ExpSingle`) is rejected at run time and JIT-compiled anyway,
  also when compiled with `--instruction-set:x86-64-v3` (they reference `Vector512`, whose support isn't known at
  compile time). Precompiled code is replaced at tier 1 when hot, as usual.
- **Effect** (same run, this branch): call 1 23–38% faster (183–304 → 126–188 ms), the load the same (±5%), load + 6
  one-sentence calls −1 to −5% on four runners (macos-26: +32%, noisy), the benchmark text the same.

So it saves about 50–120 ms on the first call for 14 times the package size; not shipped. An app that wants it gets
it from its own `PublishReadyToRun`.

## Settings a user can choose

Same run, this branch, process totals (load + 6 one-sentence calls; load + 4 benchmark-text calls):

| setting | one sentence | benchmark text | notes |
|---|---:|---:|---|
| default | 1,551–2,540 ms | 26.9–36.2 s | |
| `DOTNET_TieredPGO=0` | -3 to +5% | -2 to -4% | |
| ReadyToRun publish | -5 to +32% | -1 to +1% | call 1 -23 to -38% |
| ReadyToRun + `TieredPGO=0` | -12 to +2% | -4 to +1% | |
| `DOTNET_TieredCompilation=0` | **0 to -16%** | -3 to -6% | call 1 +40 to +52%, later calls the fastest |
| ReadyToRun + `TieredCompilation=0` | +1 to -24% | **+19 to +58% on Arm64**, -3 to -5% on x64 | precompiled code is never re-optimized |

`DOTNET_TieredCompilation=0` (or `<TieredCompilation>false</TieredCompilation>` in the app's project) suits a
short-lived process: everything is compiled optimized once, so the first call is slower and every later one is as
fast as in a warm service. Before this round it made the benchmark text 28–43% slower on the Arm64 runners (the runtime's
`ExpSingle` stayed a call without PGO's inlining); with `VectorExp` it is as fast as the default there. Don't combine
it with ReadyToRun on Arm64 for long texts. `DOTNET_TC_QuickJitForLoops=0` (run
[38312449894](https://github.com/bisforboman/StanzaSharp/actions/runs/38312449894), before this round's changes) was
mixed (-17% to +23% on the one-sentence process).

## Exactness

The CoNLL-U of every `tests/golden/*.txt` (SHA-256, each text processed alone), main against this branch, at 1 and 4
threads: identical (`4FD0F4EE…`), also with `DOTNET_EnableAVX2=0` (the Vector128 path), with tiering off and from
the `.pt` files (the Scalar path is covered by ManagedBackendTests). Every startup process on every runner, every
setting (ReadyToRun, the knobs, main), gave the same CoNLL-U for both texts; `stanzasharp benchmark` matched Python
Stanza on all five runners. `VectorExp` equals the runtime's `Exp` on all 2^32 inputs (above).
PipelineTests, OptionsTests and ManagedBackendTests pass (93 tests, converted models).
