# Sentiment: the padding rows' first-layer projection once per batch

An open idea ([list](../../performance.md#remaining-ideas)), from [round 27](../round-27-whole-pipeline-profile.md).

The sentiment LSTM runs unpacked: every sentence in a batch runs the batch's full width, and the padding has to go
through it, because the convolutions and the max-pool read it. A batch is up to 5,000 real tokens, so a document under
that size is one batch padded to its longest sentence. On the benchmark text that is 163 sentences padded to 71 tokens:
1,911 real rows out of 11,573, so 83.5% of the rows are padding.

Every padding row has the same input: PAD's pretrain row (id 0) plus delta PAD (id 0), then zeros for both charlms
(`ManagedSentimentNet.Forward`'s `Fill`). So its first-layer input projection (x·W_ihᵀ + b, K = 2148 into 2 × 1200
gates) is the same for every padding row. That projection is about 64% of the LSTM's arithmetic per row, and the LSTM
is about 45% of sentiment. `ManagedLstm.ForwardUnpacked` could:

- compute that projection once per direction (one row through the row-invariant path) and point `StepRows.Init` of
  every padding row at it, with no fill and no GEMM rows for them;
- skip a padding row's input row in `Fill`.

**Exactness.** `Gemm.RunRows` gives every row the same arithmetic, except the single-row kernel's row (global row
m − 1 when m % 6 == 1). In time-major order that row is the last sentence's last step, which is usually padding. That
row must keep its own value, computed through the single-row kernel, as round 24's packed-input change did. The test
should hash the logits on every path at 1 and 8 threads, as round 24 did.

**Optional, smaller.** The first layer's backward direction over the padded tail is the same for every row too. It
starts from h0 at width − 1 and reads the same inputs, and a step gives a row the same bits in any batch
(`KernelBlocked`). A row could join the backward recurrence at its own last real token, from the state of one shared
all-padding row. That is worth about 50 ms at 1 thread on the benchmark text. The second layer's inputs depend on the
sentence (the forward state carries through the padding), so nothing there can be shared.

**Estimate.** About 0.7 s of sentiment's 3.45 s at 1 thread, and about 110 ms of its 0.55 s at 8 threads (6% of the
all-eight call on the benchmark text). The gain shrinks as batches get more uniform: long documents (5,000-token batches,
sorted by length) gain little, and one-sentence calls gain nothing.
