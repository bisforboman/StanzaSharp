- Native AOT and trimming on the managed backend: the StanzaSharp assemblies are `IsAotCompatible` and publish with
  `PublishAot` or `PublishTrimmed` without a warning, with byte-identical output. The lemmatizer read its dictionary
  through reflection-based System.Text.Json, so such apps failed in `Pipeline.Load`; it now uses a source-generated
  context. Publish AOT apps with `<IlcInstructionSet>x86-64-v3</IlcInstructionSet>`: the default x64 target runs the
  128-bit kernels, about 2x slower. Not `StanzaSharp.Cuda` (TorchSharp isn't AOT-safe).
