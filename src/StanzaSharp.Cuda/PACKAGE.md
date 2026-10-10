# StanzaSharp.Cuda

The TorchSharp (libtorch) backend for [StanzaSharp](https://www.nuget.org/packages/StanzaSharp), Stanza's English NLP
pipeline in .NET: it runs the same models on an NVIDIA GPU, or on libtorch's CPU kernels. StanzaSharp alone runs them on
its managed backend, with no native dependencies; add this package only for a GPU (or to compare with libtorch).

## Install

```
dotnet add package StanzaSharp.Cuda
dotnet add package TorchSharp-cuda-windows   # or TorchSharp-cuda-linux; TorchSharp-cpu for the CPU
```

The native package's version must match the `TorchSharp` version this package depends on (0.107.0); otherwise the
build warns with `STANZA001` (`<NoWarn>STANZA001</NoWarn>` silences it). On macOS (Apple Silicon, CPU only), also run
`brew install libomp`.

## Use

```csharp
using StanzaSharp;

using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions { Backend = CudaBackend.Create(disableTf32: true) });
var doc = nlp.Process("Barack Obama was born in Hawaii.");
```

- `CudaBackend.Create(deviceIndex = 0, disableTf32 = false)`: the GPU. `disableTf32: true` gives output identical to
  the CPU (and so to Python Stanza) by turning TF32 off process-wide; with TF32 on, libtorch's default, a few near-tie
  decisions can differ.
- `CudaBackend.Cpu`: libtorch on the CPU, with `TorchSharp-cpu` (or one platform's `libtorch-cpu-<rid>` 2.10.0).
- `<StanzaSharpTrimNative>true</StanzaSharpTrimNative>` leaves out the CPU libtorch files StanzaSharp never loads (35 MB
  on Linux x64, 29 MB on macOS).

Everything else (downloading the models, the document model, CoNLL-U, options) is StanzaSharp's; see its readme and
the repository's docs/gpu.md for GPU measurements.

## License

Apache 2.0, like Stanza, whose code StanzaSharp ports; see NOTICE. The models have their own licenses, which vary
with their training data.
