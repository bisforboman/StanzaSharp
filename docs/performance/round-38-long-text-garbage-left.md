# Results, round 38: what was left of the garbage on long texts, again

Part of [performance.md](../performance.md)'s measurement history.

After [round 23](round-23-array-pool-misses.md) a long call allocated about 8 KB per word once the `ArrayPool` was warm.
The idea file listed the rest as sites under 0.3 KB/word each. Measuring them per site showed two larger ones: the
dependency parser's Chu-Liu/Edmonds decoding and the constituency parser's per-step buffers. Both now allocate less.
The output is the same bit for bit. Time and peak memory don't move, and they weren't expected to: these were small,
short-lived objects, not the large-object-heap arrays whose removal lowered round 22's peak.

## Where it went

Method as in rounds 22 and 23: `--memory 150000 --verbose --threads 8 --calls 2` (converted models, Windows, Ryzen 7
5800X, 201,313 words, all eight processors). The second call's bytes are what a warm process pays. A temporary
`GC.GetAllocatedBytesForCurrentThread()` count around each part of `ConstituencyParser.Parse` and
`DependencyParser.Parse` split those bytes by site (not committed; the parse runs on the calling thread, so the count
is exact). The second call on main:

| processor | MB | where |
|---|---:|---|
| constituency | 562 | stack-state fill 137, `StackNode`s/`Constituent`s/`Owned` 79, `Apply`'s lists and arrays ~110, `Score` output 64, word queues 94, `Compose` 39, transition pushes 17 |
| pos | 430 | `CharlmCache`'s copies (256 MB once a text has 32,768 words), by design |
| depparse | 213 | Chu-Liu/Edmonds' per-contraction arrays (more than half), the batch inputs |
| tokenize, mwt, lemma, sentiment, ner | 148, 52, 89, 32, 92 | mostly the document itself (510 bytes per word) |

**The stack-state fill** is the largest constituency site, and it stays. The `Recycler` (round 11) reuses every
state a finished sentence gives back, but the first ~50 sentences in flight (the longest, sorted first) need about
16,000 states at 8 KB before any comes back. The recycler drops its lists when the parse ends, so that nothing outlives
the call, and the next call fills them again. Keeping them across calls would be a memory decision (the owner's
decision in round 11 was to keep nothing after a call).

## What changed

- **Chu-Liu/Edmonds** (`ChuLiuEdmonds.Solve`): each contraction's arrays (the contracted tree, the cycle and
  non-cycle node lists, the metanode heads and dependents, the cycle scores) are rented and given back after the
  expansion. `Tarjan` writes only the cycle that gets contracted (the last one found) into a rented buffer and returns
  how many it found, instead of a new `bool[n]` per cycle. `Expand` writes in place into the outer tree, which keeps its
  cycle nodes' heads anyway, and `PairwiseSum` uses `stackalloc`. Only the returned tree is new.
- **Constituency:** `IConstituencyNet.Score` writes into a span the caller gives (like `INerNet.Forward` since round
  22). `Parse` rents one `[50, transitions]` buffer per call. `Apply`'s lists, the chosen transitions and the push inputs
  are one set per `Parse` call (`StepBuffers`), cleared each step. They stay local to the call, and no processor gained
  a field.

Not done: presizing each `ParserState.Owned` list. Its growth costs ~0.2 KB/word, but no size fits every sentence, and a
wrong guess costs more than the growth does.

## Exactness

Neither change touches arithmetic. Chu-Liu/Edmonds compares the same doubles in the same order. Only where its index
arrays live changed, and `Expand` in place writes every entry the old copy wrote (the non-cycle nodes from the
contracted tree, the cycle nodes kept, one cycle node's head replaced). The parser's scores are the same GEMM outputs,
copied into another buffer. So this doesn't depend on the SIMD path, and the golden tests plus whole-output hashes
check it:

- SHA-256 of the all-eight CoNLL-U, main vs this round: `CD3C66EC…` for the 201,313-word text (also equal to main's
  run in a separate process) and `B1FB6B1E…` for a 54,056-word one (`--memory 40000`).
- `DepparseTests` (`mst.json`'s tie-heavy matrices, every golden depparse file, the German rows), `ConstituencyTests` (golden trees and per-step scores, near-ties) and
  `ManagedBackendTests.ConstituencyParser_ManagedMatchesTorchSharpAndGolden` on every kernel path (33 tests, all pass). The biomedical packages, whose head-constraint repair also calls
  Chu-Liu/Edmonds, are checked by `TreebankPackageTests` in CI.

## Measured

MB allocated by each call, 201,313 words. The machine was busy with other work (times doubled between runs), and under
that load the pool's own trimming of arrays idle for a minute made the second call's bytes vary by ±15 MB. So the
constituency pair was rerun with `DOTNET_SYSTEM_BUFFERS_ARRAYPOOL_TRIMSHARED=false` (`tokenize,mwt,pos,constituency`,
main and this round alternated, two runs each, the same bytes to the megabyte in both):

| | constituency, call 1 / call 2 | depparse, call 1 / call 2 |
|---|---:|---:|
| main | 787 / 563 | 214 / 213 |
| this round | **673 / 449** | **119 / 118** |

That is −209 MB per warm call: all eight 1,618 → 1,410 MB, **8.0 → 7.0 KB/word** (a fresh process's first call 2,246
→ 2,038 MB). Peak working set: 1,458–1,461 MB in both (`tokenize,mwt,pos,constituency`, call 1), 1,579–1,583 MB after
call 2; all eight about 1,830 MB after call 1 in both. Times are within the load's noise, as expected.

## What is left

- The constituency stack-state fill (above), kept by design.
- The tagger's `CharlmCache` copies, 256 MB per long call: renting them would also keep them past the call, so it is a
  new idea; measured and dropped in [round 41](round-41-charlm-cache-rented-copies.md).
- The document itself (510 bytes per word) and many sites under 0.2 KB/word each.
