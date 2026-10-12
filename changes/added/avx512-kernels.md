- Managed backend: AVX-512 kernels (`Vector512`), used automatically on CPUs where .NET accelerates `Vector512`, with
  results identical bit for bit to the AVX2 kernels. On GitHub's AVX-512 runners the full pipeline ran 48–53% faster on
  an AMD EPYC 9V45 (Zen 5) and 3–10% faster on an EPYC 9V74 (Zen 4). `DOTNET_PreferredVectorBitWidth=256` turns them off.
