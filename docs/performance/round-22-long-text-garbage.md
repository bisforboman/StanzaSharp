# Results, round 22: what was left of the garbage on long texts

Part of [performance.md](../performance.md)'s measurement history.

After rounds 11 and 18 a long call (201,313 words, all eight processors) still allocated about 15 KB per word. Round 18
named the rest: per-word objects outside the charlm. This round ranks them by bytes and cuts the top ones. All exact.

## Where it went

Method as in round 18: `--memory 150000 --verbose --threads 8` (converted models, Windows, one process per run), plus a
temporary in-process `EventListener` on the GC's allocation ticks (by type, per processor) and on
`ArrayPoolEventSource.BufferAllocated` (pool misses, with the renting frames), and one run under `dotnet-trace` with the
same ticks and their stacks, grouped by the innermost StanzaSharp frame. The largest call-time sites on main, MB sampled:

| site | MB | |
|---|---:|---|
| `ManagedLstm.ForwardPacked` / `ForwardPadded` (`ArrayPool` misses, tokenizer and the parser's word LSTM) | 420 | not changed |
| `ManagedPosNet.CharlmFeatures` (the copies `CharlmCache` keeps) | 256 | by design |
| `ManagedNerNet.Forward`: the [batch, width, 77] emission scores, new per batch (large-object heap) | 200 | **rented** |
| constituency: `StackState`s, `Apply`'s lists, `Choose`'s closure, per-step `Select`/`ToList` | ~400 | **cut** |
| `ChuLiuEdmonds.Tarjan`: a `bool[n]` per strongly connected component, i.e. per node (O(n²) per sentence), plus 5 `int[]`, a `Stack` and a `List` per call | ~300 | **cut** |
| `PyString.Lower`: a `StringBuilder`, its `char[]` and a string per word, even when nothing changes | ~144 | **cut** |
| tokenizer inputs: per unit a string, start, end, id and 9 features, a `Join` per paragraph (twice) | ~130 | **cut** |
| `ManagedConstituencyNet.Score`'s output rows | 57 | not changed |

The ticks are a sample, and a biased one for small objects: they credit about 100 KB to whichever object starts a new
allocation context. `StackState` showed 160 MB, but the parse takes about 5.4 states per word (counted: 36,338 for
6,761 words), so at most ~45 MB at 40 bytes each. The benchmark's per-processor bytes below are exact.

## What changed

- **NER:** `INerNet.Forward` writes into a span the caller gives; `NerTagger.Predict` rents it per batch.
- **Depparse decoding:** Tarjan allocates a cycle's array only for a component of more than one node, takes its
  scratch (indices, lowlinks, the stack, the dependents' counting sort, the call stack) from the pool, and returns an
  empty array when there is no cycle; `Solve` makes its contraction stack only on a first cycle.
- **`PyString.Lower`:** an ASCII string without a capital is returned as is; other ASCII strings are lowered into one
  new string (`Ascii.ToLower`, what `Rune.ToLowerInvariant` does there). Everything else takes the old path.
- **Constituency:** the `Recycler` (round 11) now keeps the `StackState` and `Composed` objects with their arrays, so a
  parse makes only the states of the sentences in flight. The push inputs are built as arrays in one loop instead of
  `Select(...).ToList()` and `Cast<StackState>().ToList()`, `Choose` allocates its closure only on the rare illegal
  best transition, `Apply`'s lists are sized up front, and the score inputs reuse one list per call.
- **Tokenizer:** a paragraph is one normalized string plus each unit's offset into it (and into the input), not a
  string per unit with starts and ends; the units' ids and features are built per batch into rented arrays (with every
  window of a long paragraph reading them) and given back after it, and the structural regexes run on that string with
  `EnumerateMatches`, a binary search mapping a match back to its units. The vocabulary is read by span.

Nothing outlives the call: rented arrays go back in `finally`, the recycler drops its lists when the parse is over, and
no processor gained a field.

## Exactness

A temporary command hashed (SHA-256) every output this round touches on every golden text (`tests/golden/*.txt`,
each alone and all of them as one document), with each of Vector256, Vector128 and Scalar: the all-eight CoNLL-U, the
tokenizer logits of every paragraph (475,430 floats), the NER emissions of every 32-sentence batch and every
constituency step's scores. Main and this round give the same hashes on every path: CoNLL-U `1F0DCD923D879A66` on all
three; tokenizer `3FCC4CAA…`, `8E9D7AAC…`, `50D4FE39…`; NER `0E0D0FBE…`, `9CBF8F5E…`, `F6907293…`; constituency
`F228272B…`, `588B75A0…`, `E744FF5A…` (Vector256, Vector128, Scalar).

## Measured

`--memory 150000 --verbose --threads 8`, 201,313 words, alternated before/after, two runs each (the bytes are the same
to the megabyte in both runs; other sessions' test suites loaded the machine, so times are not compared):

| | allocated by the call | peak | gen2 GCs |
|---|---:|---:|---:|
| main | 3,062 MB (15.2 KB/word) | 1,950 / 1,954 MB | 20 / 19 |
| this round | **2,210 MB (11.0 KB/word)** | **1,801 / 1,836 MB** | 19 / 19 |

Per processor, MB allocated: tokenize 578 → 438, mwt 52 → 52, pos 516 → 481, lemma 84 → 84, constituency 914 → 781,
depparse 505 → 202, sentiment 86 → 79, ner 326 → 92: −852 MB, −28%. The peak drops ~120–150 MB, more than its ±50 MB
noise: NER's emission arrays were large-object-heap garbage at the end of the call, where the peak is.

## Not done

- **The `ArrayPool`'s misses** in `ManagedLstm.ForwardPacked`/`ForwardPadded` (`PoolExhausted`, ~40M floats in each of
  the tokenizer's LSTMs and the parser's word LSTM): the biggest site left, but the pool's own behaviour, not a cheap
  exact fix in our code.
- **`CharlmCache`'s copies** in the tagger (256 MB): kept by design, and its code is another round's.
- **Constituency's `Score` output** (57 MB), each `ParserState`'s `Owned` list growth (~30 MB), the persistent stack
  nodes and constituents (needed), and Chu-Liu/Edmonds' per-cycle arrays (~40 MB): each under 0.3 KB per word.
- **The tokenizer's per-batch net input** (`Run`'s ids and features, the predictions): `ITokenizerNet` takes exact
  arrays; under 0.2 KB per word.
