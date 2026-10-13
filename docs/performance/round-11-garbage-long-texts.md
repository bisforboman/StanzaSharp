# Results, round 11: garbage on long texts (managed backend)

Part of [performance.md](../performance.md)'s measurement history.

Round 8 found that the peak on a long text is 0.6–1.1 GB above the live heap: arrays the `ArrayPool` doesn't hold, which the
GC lets accumulate. This round finds where they come from and rents or reuses them.

**Where it goes** (an allocation trace, `dotnet-trace --profile gc-verbose`, grouped by the allocating method; and the
benchmark's per-processor allocation at 201,313 words, all eight processors): constituency 30.9 KB per word, depparse 9.4,
tokenize 3.5, pos 3.2 (1.3 of it the charlm cache, kept by design), ner 3.0, sentiment 2.9, lemma 0.4, mwt 0.3: 54 KB per
word, 10.8 GB in all. Constituency's is its parser states: every transition made a stack LSTM state (two [2, 512] arrays,
8 KB, for the constituent stack) and every close a 2 KB vector, each live until its sentence was done, long enough to reach
gen2. Depparse's is Chu-Liu/Edmonds (an n × n matrix per sentence plus one per contracted cycle, and LINQ in Tarjan) and
the [batch, width, width] arc and label outputs (large-object heap).

**What changed** (all exact; see below):
- Constituency: the stack states, the word vectors and the Compose outputs come from a `Recycler`, per-thread free lists
  of float arrays (thread-static, so per `Process` call: the parse is synchronous on the caller's thread). The parser
  already disposes what the net made for a sentence when the sentence is done, and that now gives the arrays back; when
  every array is back (the parse is over) the lists are dropped, so nothing outlives the call. A call allocates the states
  of the sentences in flight and reuses them. The transition inputs are embedding slices instead of copies, and the
  per-step score rows are no longer copied.
- Depparse: Chu-Liu/Edmonds works on flat n × n buffers, two rented ones taking turns as cycles are contracted (no
  [c, n] head-score matrix: each score is recomputed, the same double), and Tarjan's dependents come from one counting
  sort instead of a LINQ query per node. `DependencyParser.Parse` rents the arc log-probs and labels (`DepparseBatch.RentScores`)
  and returns them after decoding; `Scores` (tests) still gets exact arrays.
- Tokenizer: the two head GEMMs write into rented buffers. Sentiment: the pooled features and the fully connected
  layers' outputs are rented. NER: Viterbi's trellis and backpointers are rented, and `Process` no longer copies each
  sentence's emissions (only tests ask for them).
- Not changed: the outputs a net returns through its interface (pos scores and tags, NER logits, the tokenizer's
  predictions), since the TorchSharp nets return plain arrays and the caller can't tell which to give back; the
  tokenizer's per-paragraph inputs; the charlm cache; the `Document`. Each is under 1 KB per word.
- Proof: every managed output on all golden texts (corpus.txt, every validation*.txt, tokenize_stress.txt and all of them
  as one document): the CoNLL-U of both packages, every constituency step's transition scores, every sentiment logit
  (without and with the tagger's charlm cache) and the tokenizer logits of every paragraph (1,885,714 floats), SHA-256 per
  kernel path, on both builds (a temporary test): Vector256 `89AAC382…`, Vector128 `2E6181FF…`, Scalar `3A018614…`
  (CoNLL-U `AC2D88C0…` on all three), identical before and after. Chu-Liu/Edmonds also passes its golden tie-heavy
  matrices (`mst.json`).

**Measured** (`--memory N --verbose`, converted models, 8 threads, Windows; before = 8a849eb (PR #69's branch), after =
this round, alternated; other sessions' test suites shared the machine, so times are only indicative):

| | 20,513 words | 201,313 words |
|---|---:|---:|
| peak, before | 1,299 / 1,302 / 1,297 MB | 2,310 / 2,209 MB |
| peak, after | **1,271 / 1,270 / 1,269 MB** | **1,916 / 1,944 MB** |
| allocated, before | 1,486 MB (72 KB/word) | 10.8 / 10.9 GB (54 KB/word) |
| allocated, after | 831 MB (41 KB/word) | 3.7 GB (18 KB/word) |
| process time, before → after (same load) | 25.8 / 25.2 → 26.0 / 24.5 s | 333 → 307 s |

Per processor at 201,313 words (second pair), allocation per word and the peak after each processor, before → after:

| | allocated, KB/word | peak, MB |
|---|---:|---:|
| tokenize | 3.5 → 3.0 | 1,181 → 1,091 |
| mwt | 0.26 → 0.26 | 1,211 → 1,123 |
| pos | 3.2 → 3.2 | 1,442 → 1,437 |
| lemma | 0.42 → 0.42 | 1,443 → 1,452 |
| constituency | 30.9 → 5.1 | 1,862 → 1,791 |
| depparse | 10.0 → 3.2 | 1,998 → 1,805 |
| sentiment | 2.9 → 1.0 | 2,092 → 1,841 |
| ner | 3.0 → 2.2 | 2,209 → 1,944 |

- The peak on 201,313 words drops ~0.3–0.4 GB (−13 to −17%) with no heap limit. On 20,513 words it drops ~30 MB: there
  the garbage was ~0.2 GB and the pool's buffers (~0.4 GB) set most of the peak.
- What is left: the per-word outputs listed above (under 1 KB each) and, likely the larger part on a call this long, the
  `ArrayPool`'s churn: at a gen2 GC it trims buffers left idle for a minute (an earlier processor's), which become
  garbage, and later rents allocate fresh ones.
- Constituency also got faster (104 → 75 s, and 4.9 → 4.8 s at 20k): fewer gen0/gen1 collections promoting live states.
