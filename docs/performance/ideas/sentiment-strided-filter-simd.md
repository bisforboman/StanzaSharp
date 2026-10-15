# Sentiment: the strided (5, 5) filter across its channels

An open idea ([list](../../performance.md#remaining-ideas)), from [round 27](../round-27-whole-pipeline-profile.md).

`ManagedSentimentNet.Strided` runs the (5, 5) filter with stride (1, 5) as scalar loops: 8 channels × 120 columns × 25
products per window. Its ponytail comment says "a GEMM if it ever shows". It shows: 157 ms at 1 thread, 4.5% of
sentiment on the benchmark text (whose batch is 83.5% padding, see
[sentiment-padding-rows](sentiment-padding-rows.md)).

A GEMM would sum in another order. Instead, put the 8 channels in the lanes of one `Vector256<float>`:

- weights transposed at load to `[a, b, channel]`;
- the bias vector as the start value;
- for a = 0..4 and b = 0..4, `s = s + w[a, b, ·] * broadcast(src[b])`, with a separate multiply and add, no FMA, as
  in the scalar `s += wr[b] * src[b]`;
- then the max into the pooled outputs per lane.

Each lane does the scalar code's operations in the scalar code's order. Vector128 is two halves, and Scalar keeps
today's loops.

**Exactness.** Same order per channel, so the same bits. `SentimentTests` (logits) and the golden CoNLL-U check it, on
every kernel path.

**Estimate.** About 130 ms at 1 thread, and 15 to 20 ms at 8 threads (it is already parallel over sentences), about 1%
of the call.
