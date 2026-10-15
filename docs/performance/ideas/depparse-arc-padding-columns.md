# Depparse: arc scores for padding columns once per row

An open idea ([list](../../performance.md#remaining-ideas)), from [round 27](../round-27-whole-pipeline-profile.md).

`ManagedDepparseNet.Arcs` scores every real row against every column of the padded width. For a padding column j ≥ n,
`in2` is `scorer.Pad2`, so `Dot(tRow, Pad2)` is the same for every padding column of a row. Only the
linearization/distance terms differ, through `offsetJ`. On the benchmark text (one batch, width 72), 72% of the 156,384
pairs are padding columns. At 1 thread, `Dot` takes 228 ms of depparse's 2.32 s.

Two changes:

- **Padding columns.** Compute `Dot(tRow, Pad2) + constant` once per row and scorer, and reuse it for every j ≥ n. The
  per-column terms stay as they are.
- **SIMD.** `Dot` keeps four double accumulators, s0..s3, over q mod 4. A `Vector256<double>` whose lanes are those four
  accumulators widens 4 floats, multiplies and adds. That is the same product and sum per lane in the same order, then
  `(s0 + s1) + (s2 + s3)` as today. Use a separate multiply and add, not an FMA: the scalar code isn't contracted. The
  tail loop stays scalar. Vector128 and Scalar keep today's code (or the same lanes in two halves).

**Exactness.** Same inputs and same operation order per value, so the same bits. `DepparseTests`'s arc log-probs and
the golden CoNLL-U check it.

**Estimate.** About 200 ms of depparse at 1 thread (9%). At 8 threads `Arcs` is already parallel, so 25 to 30 ms
(about 7% of depparse, 1.5% of the call). Documents whose batches have mixed lengths gain the most.
