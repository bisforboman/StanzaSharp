- Managed backend: the constituency parser's stack LSTMs run their cell updates and row copies inside the GEMM's
  parallel tasks instead of on the calling thread, so constituency is 2 to 5% faster at 4 threads on the free x64 and
  Arm runners (more on macOS). Output is bit-for-bit unchanged.
