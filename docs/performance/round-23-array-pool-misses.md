# Results, round 23: the `ArrayPool` misses on long texts are its first fill, not garbage

Part of [performance.md](../performance.md)'s measurement history.

Round 22 left one large site on a long call (201,313 words, all eight processors): `ArrayPool<float>.Shared` misses
(`BufferAllocated` with reason `PoolExhausted`) under `ManagedLstm.ForwardPacked`/`ForwardPadded`, for the tokenizer and
the constituency parser's word LSTM, about 0.4 GB. This round finds out why the pool misses there. The answer is that
these allocations are not garbage, so no code changed.

## Method

`--memory 150000 --verbose --threads 8` (converted models, Windows, Ryzen 5800X, .NET 10.0.0, one process per run). A
temporary in-process `EventListener` on `System.Buffers.ArrayPoolEventSource` (not committed) logged every
`BufferRented`, `BufferAllocated`, `BufferReturned`, `BufferTrimmed` and `BufferDropped` event of 1M elements or more
during the call: buffer id, size, OS thread id and current processor. For every miss of at least 64K elements it also
recorded the renting StanzaSharp frames, from a `StackTrace` taken in the callback (managed `EventSource` events reach a
listener synchronously on the thread that wrote them). A replay of the log then tracked, per size, which buffers were
out and which were back in the pool when each miss happened. The pool's own limits, read by reflection on .NET 10: 16
partitions (`SharedArrayPoolStatics.s_partitionCount`, one per core up to 32) of 32 arrays per size
(`s_maxArraysPerPartition`), plus one array per size in each thread's TLS slot, for sizes up to 2^30 elements.

## Why the pool misses

Each suspect, checked against the events:

- **Rented on one thread, returned on another:** no. All 13,183 large-buffer events came from the calling thread. The
  workers of `ManagedThreads` rent nothing this size.
- **Per-core buckets:** no. The misses happened on whichever core the caller was on (0, 5, 6, 7, 12, 14), but a rent
  checks its TLS slot first and then searches every partition, starting at its own core's. In 27 of the 29 large misses,
  no buffer of that size was free anywhere.
- **More arrays in flight than the pool keeps:** no. At most 6 buffers of one size were out at a time (the tokenizer's
  16 MB `[n, 128]` buffers). The pool keeps 1 + 16 × 32 per size. `BufferDropped` never fired.
- **Sizes above the pool's maximum:** no. The largest rent was 2^25 floats (128 MB), far under 2^30. No miss had the
  reason `OverMaximumSize`.
- **Trimming:** no. No large buffer was trimmed during the call (`BufferTrimmed` fired 66 times, all for small arrays).

**The cause:** these are cold misses, the pool's first fill. The first time a call needs *k* buffers of one
power-of-two size at once, the pool has fewer than *k* of that size, so it allocates the rest. `PoolExhausted` is
simply the reason code for any miss under the size limit, and does not mean the pool hit a limit. Each new buffer goes
back into the pool when it is returned, stays live, and serves every later rent of that size. A large miss happens only
at a new high-water mark: the tokenizer's first batches (20 misses in the first 0.3 s), the first chunk of the
constituency parser (its longest sentences, so its largest buffers: 32, 64 and 128 MB), the tagger's and sentiment's
largest batches, and two depparse batches that needed one more buffer than any before them.

The two other misses (4 MB each, during depparse) happened while the replay counted 5–6 free buffers of that size. That
is 8 MB, which doesn't matter, and it's unexplained: possibly a buffer-id hash collision in the replay.

Missed MB by the innermost StanzaSharp frame (≥ 64K elements). The reason code was `PoolExhausted` for every miss:

| site | misses | MB |
|---|---:|---:|
| `ManagedLstm.ForwardPacked` ← `ForwardPadded` ← `ManagedConstituencyNet.EncodeWords` | 2 | 160 |
| `ManagedLstm.ForwardPacked` ← `ForwardPadded` ← `ManagedTokenizerNet.Forward` | 16 | 160 |
| `ManagedTokenizerNet.Forward` (its own `x`, `inp`, `inp2`, `hid2`, `level1/2`) | 20 | 91 |
| `ManagedLstm.ForwardPadded` ← `ManagedConstituencyNet.EncodeWords` | 1 | 64 |
| `ManagedLstm.ForwardUnpacked` ← `ManagedSentimentNet.Forward` | 2 | 48 |
| `ManagedLstm.ForwardPadded` ← `ManagedTokenizerNet.Forward` | 8 | 36 |
| `ManagedHighwayLstm.Forward`, `ManagedPosNet.Forward` | 2 | 64 |
| depparse `Arcs`/`Labels`, lemma, tokenizer whitespace | 14 | 18 |
| **all** | **65** | **641** |

## Confirmed: a second call doesn't pay it

The same command with `--calls 2` (one process, two `Process` calls on the same text, untraced), allocated MB per
processor:

| | tokenize | mwt | pos | lemma | constituency | depparse | sentiment | ner | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| call 1 | 438 | 52 | 494 | 84 | 781 | 214 | 79 | 92 | **2,234 (11.1 KB/word)** |
| call 2 | 148 | 52 | 430 | 83 | 556 | 213 | 31 | 92 | **1,605 (8.0 KB/word)** |

The 629 MB difference matches the 641 MB of cold misses above, and falls exactly on the processors that had them. Call 1
here also matches round 22's 2,210 MB. Peak 1,837 MB after call 1 (round 22: 1,801 / 1,836), 23 gen2 GCs over both calls.
The traced single-call run had 20 gen2 GCs and a 1,864 MB peak; the listener's stack traces add allocation, so its other
numbers aren't compared.

So a fresh process's long call allocates about 0.6 GB of scratch buffers once. They stay live in the pool and are part
of the peak (the largest batch's working set), not garbage. A service pays this once, on its first long call, and
then again only after the pool trims buffers left idle for a minute (round 6).

## Why no code ships

- **A per-call `ArrayPool.Create`** would allocate the whole working set again on every call, and the GC would collect
  it afterwards: the opposite of what we want. Holding buffers per batch, or renting and returning on the same thread,
  is what the code already does: every rent here is on the calling thread, and returned in `finally` before the next
  batch.
- **Fewer, smaller buffers in flight** is the only lever left, and it lowers the first-call allocation and the peak by
  the same amount. The largest candidate is `ManagedConstituencyNet.EncodeWords` filling the packed word-LSTM input
  directly instead of a padded `[n, width, 2268]` copy that `ForwardPadded` then packs. It's exact by construction, but
  by the trace it would save one 64 MB buffer on this text (estimated, not built) (the padded copy reuses the tokenizer's 64 MB buffer; only the packed one was
  new), which is within the peak's ±50 MB run-to-run noise. The gate buffer (`rows × 4096` floats, 128 MB in the first
  chunk) is read whole by the recurrence, so splitting it isn't cheap. The tokenizer's 16 MB buffers are one batch of
  32 paragraphs × a 1,000-unit window each, and they are what that batch needs.
- **Exact-size arrays** (avoiding the power-of-two rounding) raised the peak in round 9.

## Correction to round 22

Round 22's "11 KB/word" includes this first fill. A long call allocates about 8 KB per word in a process that has already
processed a long text. What is left is in
[round 38](round-38-long-text-garbage-left.md).
