# Results, round 18: depparse's forward charlm start; garbage on long texts

Part of [performance.md](../performance.md)'s measurement history.

Two small items from "Remaining ideas". One shipped (the charlm's character ids), one didn't (the stored start state).

## The `ArrayPool`'s churn wasn't the garbage; the charlm's character strings were

Round 11 guessed that what is left of the garbage on a long call is mostly `ArrayPool<T>.Shared` trimming idle buffers at a
gen2 GC and renting fresh ones later. Measured with `--memory 150000 --verbose` (201,313 words, all eight processors,
8 threads, Windows, one process each; other sessions' jobs loaded the machine, so times are not compared):

| | allocated by the call | peak | gen2 GCs |
|---|---:|---:|---:|
| main | 3,673 MB (18.2 KB/word) | 1,946 MB | 20 |
| main, `DOTNET_SYSTEM_BUFFERS_ARRAYPOOL_TRIMSHARED=false` (the pool never trims) | 3,674 MB | 1,948 MB | 20 |
| this round | **3,064 MB (15.2 KB/word)** | 1,958 MB | 19 |

With trimming off, every processor allocated the same to the megabyte: the pool's churn costs nothing here, so a per-call
pool or holding the call's largest buffers can't help (not tried). An in-process `EventListener` (temporary) on the
GC's allocation ticks and `ArrayPoolEventSource` showed where it goes instead: pool misses (a rent that finds no array)
add ~60M elements between 20,513 and 201,313 words (219 buffers trimmed, none needed again), while per-character
objects grow with the text: `String` +290 MB, `Int32[]` +380 MB, boxed `StringRuneEnumerator` +120 MB and a LINQ
`Select` iterator over it +100 MB. They come from `ManagedCharLanguageModel`'s input: every character became a
one-rune string (`EnumerateRunes().Select(r => r.ToString())`) for the vocabulary lookup, and the ids grew in `List<int>`s.
The charlms run over every word several times (pos, depparse, and sentiment, NER and constituency where the cache
misses), so this was ~3 KB per word.

**What changed:** the ids are looked up from each rune's UTF-16 in a stack buffer through the vocabulary's
`GetAlternateLookup<ReadOnlySpan<char>>()`, into an array of the exact length (code points counted first), and the
backward charlm fills it from the end instead of reversing enumerables. The same ids, so the same bits. Per processor
at 201,313 words, allocated MB before → after: pos 649 → 516, constituency 1,024 → 914, depparse 646 → 505, sentiment
199 → 86, ner 439 → 326 (tokenize, mwt and lemma unchanged): −610 MB, −17%. The peak is unchanged within its
±50 MB run-to-run noise: it is set by live data (the models, the document, the cache and the largest batch's buffers),
not by these short-lived objects.

What is left (the same trace): `Int32[]` and `String` outside the charlm (tokenizer inputs, the document), constituency's
`StackState`s (~125 MB), `bool[]`, `object[]` and `string[]` of ~110 MB each; each under 1 KB per word.

## Depparse's forward charlm start: exact, but under the noise

The parser feeds the forward charlm `"\n"` (start), `"\n"` (ROOT) and `" "` before every sentence's words. A row's
arithmetic in an LSTM step doesn't depend on the other rows: since every step sums h·W_hhᵀ through `Gemm.KernelBlocked`
(2026-10-10, docs/backends.md), which has a 3-row and a 6-row kernel with the same per-row FMA order and no single-row
one, so the state after those three steps is the same in any batch. A version that computed it once per kernel path (lazily, read-only after) and ran every
sentence on from it gave the same bits: the depparse arc log-probs, label scores and labels of all 845 golden sentences
alone and in batches of 40, the parser's CoNLL-U on every golden depparse file, the `tokenize,mwt,pos,lemma,depparse`
CoNLL-U of every golden text and of all of them as one document, and 60 one-sentence calls, SHA-256 on Vector256,
Vector128 and Scalar at 1 and 8 threads: identical to main (Vector256 alone `B1272F36…`, batched `E09AECE1…`; CoNLL-U
`84599FFE…` and pipeline `EF8F01BD…` on every path).

It saves 3 of the forward charlm's ~90 steps per sentence. Per-call A/B (same binary, an environment switch, alternated,
`--per-call 300 --processors tokenize,mwt,pos,lemma,depparse`, 8 threads, four undisturbed rounds): depparse median
11.79 ms before, 11.71 after; the rounds themselves spread ±0.6 ms. Timed alone (200 golden sentences of 5–25 words, one at a time, three
processes): the forward charlm took 2.82 / 2.88 / 3.09 ms per sentence (48 steps on average) and 2.62 / 2.75 / 2.96
ms from the stored state, 0.13–0.2 ms saved, under 1% of a 23 ms five-processor call and about 0.5% of a 40 ms
all-eight one. Under the noise of a call, so not shipped.
