# Results, round 20: depparse's backward charlm from the tagger's, in batches

Part of [performance.md](../performance.md)'s measurement history.

The dependency parser feeds the charlms `"\n"` (ROOT) + the words. Forward, ROOT comes first, so nothing of the
tagger's applies. Backward, ROOT comes last: the tagger's backward pass over a sentence ("\n", then the words reversed)
is the parser's prefix, followed by ROOT's two characters (`"\n"`, `" "`). Until now only a one-sentence document used
that (`CharlmCache.TryGetAlone`), since a batched row's kernel was thought to depend on its batch. Since
`Gemm.KernelBlocked` (2026-10-10) an LSTM step gives a row the same FMAs in the same order in the 3-row and the 6-row
kernel (there is no 1-row one), so a row's state depends on its own sequence only, whatever batch it runs in. Round 18
checked that for the forward charlm.

**What changed** (managed backend only; `default_fast` has no charlm in depparse, so it is unchanged):

- The tagger keeps each cached sentence's backward final cell state with its entry (`CharlmCache.TryAdd`'s
  `backwardFinalC`): 1,024 floats, 4 KB per sentence, beside 8 KB per word. It is not counted in
  `CharlmCacheOptions.MaxWords` (about 3% more at 15 words per sentence; documented on `CharlmCacheOptions`).
- The parser gets each sentence's `Sentence` (`DepparseBatch.Keys`) and asks `CharlmCache.TryGetBackward`: the entry
  (kept only when simplify_punct left the words unchanged, so they are the parser's texts), else the single-sentence
  table by texts. For every hit it copies the tagger's backward rows for the words and runs ROOT on from the tagger's
  final (h, c): one `ManagedCharLanguageModel.Continue` call over all of the batch's hits, two steps, which now takes a
  start state per sentence. The other sentences run in full, as one batch, as before.
- `samples/StanzaSharp.Benchmark` now passes the cache to depparse, as `Pipeline` does; it never did, so its depparse
  stage hadn't shown even the one-sentence reuse.

## Exactness

A temporary harness (not committed) hashed, with SHA-256: the depparse arc log-probs and label scores of every sentence
of the 13 golden depparse files alone (tagger with a cache, then the parser with it), and in the parser's own batches
of each whole file with the tagger's cache and without one; the `tokenize,mwt,pos,lemma,depparse` CoNLL-U of corpus.txt
and every validation*.txt, each alone and all of them as one document; 60 one-sentence calls; and the all-eight CoNLL-U
of every text (Vector256 only). On Vector256, Vector128 and Scalar (Vector512 gives Vector256's bits by test), at 1 and 8
threads, main and this branch (Ryzen 7 5800X, Windows):

| path | alone | batches, with cache | batches, without | five processors | all eight | one-sentence calls |
|---|---|---|---|---|---|---|
| Vector256, 1 and 8 threads | `C38DBDF35025` | `56263E33185D` | `56263E33185D` | `56E9A5ED6CDC` | `D649479D8BE5` | `B5B50DC22D03` |
| Vector128, 1 and 8 threads | `6E533FC7B21A` | `763267F586F7` | `763267F586F7` | `56E9A5ED6CDC` | | `B5B50DC22D03` |
| Scalar, 1 and 8 threads | `5731995A53AD` | `855B8288813B` | `855B8288813B` | `56E9A5ED6CDC` | | `B5B50DC22D03` |

Every hash is the same on main and on the branch, and with the cache equals without it. The CoNLL-U is the same on
every path. `DepparseTests.Batches_ContinueTheTaggersBackwardCharlm_BitForBit` keeps the batched check (three golden
files: scores with the cache equal scores without, and over 3/4 of the sentences reused).

## Speed and memory

Same binary, an environment switch giving main's behaviour (reuse only for a one-sentence batch, from the single-sentence
table), alternated. All eight processors, `default` package.

**Free runners** (`stanzasharp benchmark`, 2,009 words, StanzaSharp only, three pairs per job; medians of the three):

| runner | CPU | depparse main | depparse new | | total main | total new | | peak MB main / new |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ubuntu-26.04 | EPYC 7763, 4 threads | 2.16 s | 1.62 s | −25% | 7.88 s | 7.35 s | −7% | 1,055 / 1,070 |
| ubuntu-26.04-arm | Neoverse-N2, 4 | 2.42 s | 1.72 s | −29% | 8.61 s | 7.89 s | −8% | 1,124 / 1,109 |
| windows-2025 | EPYC 9V74, 4 | 1.72 s | 1.34 s | −22% | 6.32 s | 5.91 s | −6% | 1,043 / 1,038 |
| windows-11-arm | Cobalt 100, 4 | 2.36 s | 1.67 s | −29% | 8.40 s | 7.64 s | −9% | 982 / 997 |
| macos-26 | M1 (virtual), 3 | 3.16 s | 1.70 s | | 11.90 s | 7.49 s | | 1,052 / 1,031 |

The macOS job was noisy on main (depparse 3.57, 3.16 and 2.22 s; the branch 1.70, 1.77, 1.50), so only its direction
counts. The one-sentence median per call didn't move (the one-sentence case already reused the tagger's state), and the
peaks are within their run-to-run spread.

**Local** (Ryzen 7 5800X, 8 threads, `samples/StanzaSharp.Benchmark --copies 2 --runs 3`, 6,566 words, four pairs; other
sessions' jobs loaded the machine): depparse median 2.04 → 1.56 s (−24%), total 7.14 → 6.72 s (−6%), peak working set
1,259–1,266 MB either way. `--memory 20000` (the pipeline, one call): peak 1,359 / 1,342 MB on main, 1,343 / 1,342 MB
on the branch; the extra 4 KB per sentence (about 5 MB here) is under the noise.

The saving is the backward charlm's pass over the words, about a quarter of the parser's time; ROOT's two steps per
sentence remain. Shipped.
