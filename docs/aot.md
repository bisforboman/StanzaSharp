# Native AOT and trimming

**Status (owner's decisions, 2026-10-16, after this study): supported on the managed backend.** The Lemmatizer fix
below shipped, `IsAotCompatible=true` is set on the eleven `StanzaSharp` package projects (Directory.Build.props; not
`StanzaSharp.Cuda`, since TorchSharp is neither trim- nor AOT-safe, nor the Tool), README and PACKAGE.md document it, and
CI runs `tools/verify-aot.sh` (see "CI check"). The rest of this page is the study that led there.

The question: can an app that uses only the `StanzaSharp` package (the default managed backend) be published with
`PublishAot=true` or `PublishTrimmed=true`, and is its output still byte-identical?

**Verdict:** yes, after one small fix in `src/StanzaSharp.Lemma`. With it, both publish modes give zero IL2xxx/IL3xxx
warnings and byte-identical CoNLL-U for English and German `default`. Without it, both publish fine (two warnings) but
`Pipeline.Load` throws as soon as it loads the lemmatizer. AOT also needs a choice of instruction set, or the managed
kernels drop from Vector256 to Vector128 and run about 2x slower.

## How it was checked

A scratch console app (not committed), referencing the src projects like `tests/StanzaSharp.ManagedCheck`, with this
`Program.cs` logic: load `PipelineOptions { Language, Package = "default" }`, one sentence (the time from process start
to that first result is "startup"), corpus.txt three times (the CoNLL-U compared with `tests/golden/pipeline.conllu` /
`tests/golden/de/pipeline.conllu`, and written to disk and `cmp`ed too), 30 one-sentence calls (median), then all
`validation*.txt` of the language as one text three times (median), and the peak working set. Its assembly was named
`StanzaSharp.Benchmark` only to read the internal `Gemm.Path` through the existing `InternalsVisibleTo`.

Publishes (Windows 11, Ryzen 7 5800X: AVX2, no AVX-512; .NET SDK 10.0.402, ILCompiler 10.0.13):

| build | command |
|---|---|
| JIT | `dotnet publish -c Release -r win-x64 --self-contained false` |
| trimmed | `dotnet publish -c Release -r win-x64 --self-contained -p:PublishTrimmed=true` |
| AOT | `dotnet publish -c Release -r win-x64 -p:PublishAot=true` |
| AOT v3 | the same with `-p:IlcInstructionSet=x86-64-v3` |

A quicker check for anyone: `dotnet publish tests/StanzaSharp.ManagedCheck -r win-x64 -p:PublishAot=true` (or
`-p:PublishTrimmed=true --self-contained`) and run it as `ManagedCheckTests` does; it compares both English packages with
the golden files. (Not run in this study; it needs fix 1 like any app.)

### Local toolchain caveat (AOT on Windows)

Native AOT links with MSVC. This machine has only the VS 2017 Build Tools (MSVC 14.16) with a C++ toolset; VS 2022
Community is installed without the "Desktop development with C++" workload. ILC compiled the app fine (so the warning
list below is complete), but the 2017 linker failed on the .NET 10 runtime's static libraries, which are built with a
newer MSVC:

```
Runtime.WorkstationGC.lib(GCHelpers.cpp.obj) : error LNK2001: unresolved external symbol __tls_guard
Runtime.WorkstationGC.lib(GCHelpers.cpp.obj) : error LNK2019: unresolved external symbol __dyn_tls_on_demand_init
```

These are the newer CRT's on-demand thread-local initialization. For the measurements only, a two-line C stub
(`__declspec(thread) char __tls_guard = 1; void __dyn_tls_on_demand_init(void) {}`, compiled with the 2017 `cl` and
passed as `<NativeLibrary Include="tlsstub.obj" />`) lets the old linker finish: the old CRT already runs dynamic TLS
initializers eagerly when a thread starts, so the guard can start as "done". Users and CI need no such thing: a
VS 2022 C++ workload (as on GitHub's `windows-2025` runner) links directly. The AOT numbers here should be confirmed on
such a toolchain once (see Decisions).

linux-x64 AOT needs clang (or gcc) and zlib on Linux; this machine has no Linux toolchain (WSL has only Docker Desktop's
distribution, and Docker was stopped), so linux-x64 was not tried. Nothing found below is Windows-specific.

## Warnings

Every IL2xxx/IL3xxx warning of the whole app (all eleven src assemblies, `Microsoft.Extensions.Logging.Abstractions`,
System.Text.Json, the BCL), with `TrimmerSingleWarn=false` so each is listed by its call site:

| assembly | warning | cause |
|---|---|---|
| StanzaSharp.Lemma | IL2026 (trim and AOT), IL3050 (AOT) | `Lemmatizer.cs:53`, `JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string?>>>(gz)`: reflection-based System.Text.Json |

That is all. With the fix below: **zero** warnings in both modes. What was checked and needs nothing:

- **System.Text.Json elsewhere:** checkpoints are read with `JsonNode`; `TorchCheckpoint`'s large containers are a
  `JsonValue` over a hand-made `JsonTypeInfo` (`JsonMetadataServices.CreateValueInfo<PyContainer>` with its own
  converter), which is trim-safe. `Json.Write`/`Read` use `Utf8JsonWriter`/`JsonNode`.
- **Microsoft.Extensions.Logging.Abstractions** (10.0.0): no warnings; the library calls `LogInformation`/`LogDebug`
  with message templates, which are annotated safe.
- **ArrayPool, `GetAlternateLookup<ReadOnlySpan<char>>`, `Vector128/256/512`, `AdvSimd`:** no warnings (see the SIMD
  section for what AOT does with the vector paths).
- **P/Invoke:** `CpuCores`' `[DllImport("kernel32")]` / `[DllImport("libc")]` (blittable signatures; NativeAOT
  generates the stubs) and `NativeHeap`'s `NativeLibrary.TryLoad` + `Marshal.GetDelegateForFunctionPointer` (Linux
  glibc only; no warning on win-x64; `CanTrim` is false on Windows so it never runs there). Converting them to
  `[LibraryImport]` and a function pointer (`delegate* unmanaged<nuint, int>`) would be the modern form but is not
  needed. The Linux path is untested here.
- **Pickle reader** (`Core/Pickle.cs`, `TorchCheckpoint.cs`): allowlisted globals, nothing instantiated by name, no
  reflection; the `.pt` files load. (The study loaded the converted models; the `.pt` path is the same code with no
  warnings.)
- **Regexes:** `Tokenizer.PyRegex` and the masks are all `[GeneratedRegex]` (source-generated), no
  `RegexOptions.Compiled`, so nothing is interpreted at run time and no warning.
- **Embedded resources:** none in the library (only `StanzaSharp.Tool` has them, and it is not an AOT target).
- **`GetType().Name`** in exception messages: fine (names are kept).

## Blocker and its fix

| # | blocker | location | effect | smallest fix | cost |
|---|---|---|---|---|---|
| 1 | reflection-based `JsonSerializer.Deserialize<T>` | `src/StanzaSharp.Lemma/Lemmatizer.cs:53` | trimmed: `InvalidOperationException: Reflection-based serialization has been disabled` at `Pipeline.Load` (PublishTrimmed and PublishAot set `JsonSerializerIsReflectionEnabledByDefault=false`); AOT with reflection turned back on: `NotSupportedException` (no constructor kept for the dictionary type) | a source-generated context, below | 3 lines; no change in behaviour or output; the AOT binary gets 1.2 MB smaller (7.4 to 6.2 MB) (reflection serializer no longer rooted) |

```csharp
_posDict = JsonSerializer.Deserialize(gz, LemmaJson.Default.DictionaryStringDictionaryStringString)!;
...
[JsonSerializable(typeof(Dictionary<string, Dictionary<string, string?>>))]
internal sealed partial class LemmaJson : JsonSerializerContext;
```

Tested locally first (every result below is with it), then shipped. Before it, a trimmed app could set `<JsonSerializerIsReflectionEnabledByDefault>true</JsonSerializerIsReflectionEnabledByDefault>`
(verified: identical output, two warnings); an AOT app had no workaround.

## SIMD paths under AOT

Native AOT has no JIT, so `Vector256.IsHardwareAccelerated` and `Vector512.IsHardwareAccelerated` are fixed at compile
time by the target instruction set, not detected on the user's CPU. The default x64 target (x86-64-v2) has neither:

| build | `Vector512` | `Vector256` | `Gemm.Detect()` |
|---|---|---|---|
| JIT, trimmed | false (no AVX-512 here) | true | Vector256 |
| AOT (default) | false | **false** | **Vector128** |
| AOT, `IlcInstructionSet=x86-64-v3` | false | true | Vector256 |

Output stays byte-identical on Vector128 (the golden tests already pass on every path), but the kernels are about 2x
slower (table below). An app chooses:

- `x86-64-v3` (AVX2/FMA, every x64 CPU since ~2015): Vector256, the JIT's speed on AVX2 machines; the binary refuses to
  start on older CPUs.
- `x86-64-v4` (AVX-512): Vector512 (bitwise Vector256's results), but only runs on AVX-512 CPUs.
- `native`: the build machine's CPU, not portable.
- One binary cannot dispatch Vector256/Vector512 at run time the way the JIT does: `Gemm` would need explicit
  `Avx2.IsSupported`/`Avx512F.IsSupported` checks around separately written kernels (NativeAOT compiles those checks as
  real run-time tests). Not worth it for a study; the instruction-set property is enough.

`[MethodImpl(MethodImplOptions.AggressiveOptimization)]` (round 17) does nothing under AOT: every method is compiled
once, fully optimized, so the tier-0 problem it solves doesn't exist. It is harmless. (In the trimmed build, which still
JITs, it works as before.)

## Measurements

Five runs per build and language, alternated (two before the long text was added, three with it), best value of each
column. The machine was shared with other agents at 100% CPU for most of the runs, so the full-text column swung up to
5x between rounds (German JIT 8.9-22.9 s, plus one 128 s outlier for AOT); trust its ratios loosely and the start-up
columns (process start, load, first call) more. All 40 outputs were byte-identical to the golden files (string compare
and `cmp`).

English `default` (validation*.txt: 8,261 words):

| build | startup | load | first call | one sentence (median) | validation*.txt | peak WS | binary |
|---|---|---|---|---|---|---|---|
| JIT | 1.71 s | 1.36 s | 293 ms | 45.9 ms | 24.3 s | 1,401 MB | 0.8 MB + the shared runtime (74 MB) |
| trimmed | 2.57 s | 2.00 s | 380 ms | 42.9 ms | 25.1 s | 1,393 MB | 21.4 MB, 62 files |
| AOT (Vector128) | 1.07 s | 0.80 s | 89 ms | 66.3 ms | 52.1 s | 1,330 MB | 6.2 MB, 1 file |
| AOT v3 (Vector256) | 1.11 s | 0.82 s | 102 ms | 40.7 ms | 24.6 s | 1,311 MB | 6.2 MB, 1 file |

German `default` (validation*.txt: 2,091 words):

| build | startup | load | first call | one sentence (median) | validation*.txt | peak WS |
|---|---|---|---|---|---|---|
| JIT | 2.40 s | 1.97 s | 354 ms | 63.6 ms | 8.9 s | 1,614 MB |
| trimmed | 2.43 s | 1.95 s | 350 ms | 51.8 ms | 12.7 s | 1,545 MB |
| AOT (Vector128) | 1.40 s | 1.25 s | 128 ms | 66.5 ms | 31.8 s | 1,538 MB |
| AOT v3 (Vector256) | 1.74 s | 1.39 s | 107 ms | 36.7 ms | 11.5 s | 1,550 MB |

What holds across every round:

- **AOT starts faster:** load about 0.6 s faster (English 0.8 vs 1.4 s) and the first call 3x faster (~100 vs ~300 ms),
  since nothing is JIT-compiled; process start to first result ~1.1 s vs ~1.7 s.
- **Steady state:** AOT v3 runs at the JIT's speed (same Vector256 kernels, both fully optimized); default AOT
  (Vector128) about 2x slower on long texts.
- **Trimmed** is the JIT (it still JITs), slightly slower to load (no ReadyToRun framework, so more is JIT-compiled at
  start); it only saves disk (21 MB self-contained instead of the whole ~74 MB runtime). `PublishReadyToRun=true` would likely win the load time back (not measured).
- **Memory:** within noise (60-90 MB less peak for AOT: no JIT, no IL); the models dominate.

## Is `IsAotCompatible=true` reachable on the src projects?

Yes, for the eleven assemblies in the `StanzaSharp` package (facade, Core, Nn, the eight processors). Checked:
`dotnet build src/StanzaSharp -c Release -p:IsAotCompatible=true` (the global property reaches every referenced
project) gives exactly blocker 1's IL2026 + IL3050 and nothing else; with the fix, zero warnings. Setting
`IsAotCompatible` turns on the trim, AOT and single-file analyzers per library, which also check code an app doesn't
reach, and marks the assemblies trimmable for consumers; CI's warnings-as-errors build then keeps it that way. Not for `StanzaSharp.Cuda` (TorchSharp) or `StanzaSharp.Tool`
(`JsonSerializer` with `object`, `DllImport("libc")` getrusage, a .NET tool anyway).

## Decisions (owner, 2026-10-16)

All as recommended:

1. The Lemmatizer reads its dictionary through a source-generated `LemmaJson` context.
2. `IsAotCompatible=true` on the eleven package projects; new code there must stay AOT-safe (CLAUDE.md, "Native AOT and
   trimming").
3. README and PACKAGE.md say trimming and Native AOT are supported on the managed backend, with a csproj snippet, and
   recommend `IlcInstructionSet=x86-64-v3`: the default x64 target runs the 128-bit kernels (~2x slower), v3 needs
   AVX2/FMA (Intel Haswell, AMD Excavator and later), v4 gives the AVX-512 kernels on AVX-512 CPUs only. Arm64 needs
   nothing (NEON is the Arm64 baseline, so `Vector128.IsHardwareAccelerated` and `AdvSimd.Arm64.IsSupported` are true
   at compile time); linux-arm64/win-arm64 AOT is not tested yet.
4. CI check, below. Not done: run-time AVX2/AVX-512 dispatch inside `Gemm`.

## CI check

`tools/verify-aot.sh MODEL_DIR RID` publishes `tests/StanzaSharp.ManagedCheck` with `-p:PublishAot=true
-p:IlcInstructionSet=x86-64-v3` and with `--self-contained -p:PublishTrimmed=true`, fails on any `warning IL2xxx/IL3xxx`
in the publish output or any torch/c10/libgomp file in the published folder, and runs each binary: ManagedCheck loads
`default` and `default_fast` (with checksums for `.pt` models), compares corpus.txt's CoNLL-U with `pipeline.conllu` and
`fast/corpus.conllu` byte for byte, runs the other input modes, and fails if any torch module loaded.

- golden shard 1 (ubuntu-26.04, linux-x64): on full runs, and on light runs with torch=full (pull requests touching CI,
  Nn or the backend seam). Installs `clang` and `zlib1g-dev` first if the image lacks them.
- `cross-os (windows-2025)` shard 3 (win-x64): full runs only, like the package checks there. The runner's VS 2022 has
  the C++ linker, so no stub.

Locally the Windows half took 107 s per publish and 35 s (AOT) / 71 s (trimmed) to run, on a machine at 100% CPU.
