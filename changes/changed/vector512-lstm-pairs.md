- Managed backend: on AVX-512 CPUs each LSTM time step now computes two 16-column panels per kernel call, as the other
  matrix products already did. On GitHub's AVX-512 runners the full pipeline got 11–15% faster on an AMD EPYC 9V45
  (Zen 5), 5–9% on an Intel Xeon Platinum 8573C (Emerald Rapids) and 11–12% on a Xeon Platinum 8370C (Ice Lake), where
  the AVX-512 kernels had been 5–7% slower than the AVX2 ones and are now 4–6% faster. The output is unchanged, bit for
  bit.
