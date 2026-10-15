# Results, round 29: depparse's padding columns once per row, SIMD arc dot products

Part of [performance.md](../performance.md)'s measurement history. The idea came from
[round 27](round-27-whole-pipeline-profile.md)'s profile.

`ManagedDepparseNet.Arcs` scores every real row against every column of the padded batch width, since the arc
log-softmax counts the padding columns. A padding column's in2 is `Pad2` (ReLU of W2's bias) for every column, so
within a row `T·in2` is the same for all of them; only the linearization and distance terms differ, through the offset.
On the benchmark text (one batch, width 72), 72% of the arc pairs are padding columns.

**What changed** (managed backend only; both packages):

- Each row computes `Dot(tRow, Pad2) + constant` once per scorer (unlabeled, linearization, distance) when the
  sentence is shorter than the width, and reuses it for every padding column. The per-column terms are unchanged.
- The dot product moved to `Act.Dot` (Nn), with a version per path. Its four double accumulators s0..s3 (over q mod 4)
  become SIMD lanes: Vector256 (also used on Vector512) holds them in one `Vector256<double>`, Vector128 in two
  `Vector128<double>`, each fed 4 floats widened to double, multiplied and added; Scalar keeps the old loop. The tail and
  `(s0 + s1) + (s2 + s3)` are as before. A float × float product is exact in double, so a lane adds exactly the
  scalar code's values in the same order (an FMA would give the same bits too). Vector512 runs the Vector256 code, so it
  is Vector256's bits by construction.
- `ManagedBackendTests.Dot_GivesTheScalarSumsBits` checks every native path against the scalar sum, bit for bit, for
  lengths 0 to 9 and 400 to 403 (tails of 0 to 3).

## Exactness

A temporary harness (not committed) hashed, with SHA-256 (first 12 hex digits), for both packages: the depparse arc
log-probs, label argmaxes and label scores of every sentence of the 13 golden depparse files alone (gold tags and
lemmas); the same in the parser's own batches of each whole file; and the `tokenize,mwt,pos,lemma,depparse` CoNLL-U of
corpus.txt and every validation*.txt through `Pipeline`. On Vector256, Vector128 and Scalar, at 1 and 8 threads, main
(`1fe32df`) and this branch (Ryzen 7 5800X, Windows):

| path | package | alone | batches | five processors |
|---|---|---|---|---|
| Vector256, 1 and 8 threads | default | `632291E22613` | `B7675D8FC122` | `9CC67A26F2F4` |
| | default_fast | `448E331533AF` | `DD55ED808297` | `D0D6D4DA8E26` |
| Vector128, 1 and 8 threads | default | `8F57BF8F550F` | `F9D90E52A1E9` | `9CC67A26F2F4` |
| | default_fast | `91E0058C4A24` | `C73DF86C9D48` | `D0D6D4DA8E26` |
| Scalar, 1 and 8 threads | default | `F71A4A5E9F38` | `9763F953B726` | `9CC67A26F2F4` |
| | default_fast | `3791D86888F5` | `CD72D24A7485` | `D0D6D4DA8E26` |

Every hash is the same on main and on the branch. `DepparseTests` (golden CoNLL-U and arc/label log-probs, both
backends) and the new `Dot` test pass.

## Speed

**Local** (Ryzen 7 5800X; four other agents' jobs were running, so pairs alternated; `samples/StanzaSharp.Benchmark
--processors tokenize,mwt,pos,lemma,depparse --copies 1 --runs 3`, 3,283 words, four pairs):

| threads | depparse main | depparse branch | |
|---|---:|---:|---:|
| 1 (the two quiet pairs) | 4.05, 3.99 s | 3.70, 3.68 s | −8% |
| 8 (all four pairs) | 0.74–0.75 s | 0.67–0.71 s | −8% |

At 1 thread the two other pairs were disturbed (main 5.52 and 5.55 s, branch 5.39 and 3.71 s). Round 27 estimated 9% at
1 thread and about 7% at 8.

**Free runners** (`gh workflow run benchmark.yml`, one run on main and one on the branch, the default package, 2,009
words, 4 threads, medians of 3 runs). x64 jobs drew different CPUs on the two runs, so only the Arm jobs compare:

| runner | CPU | depparse main | depparse branch |
|---|---|---:|---:|
| ubuntu-26.04-arm | Neoverse-N2 | 1.68 s | 1.67 s |
| windows-11-arm | Cobalt 100 | 1.70 s | 1.71 s |
| ubuntu-26.04 / windows-2025 | EPYC 7763 | 1.76 s (Windows) | 1.55 s (Ubuntu) |

On the Arm runners the change is under the run-to-run noise (the expected gain at 4 threads is a few percent of
depparse). The EPYC 7763 pair crosses operating systems, so it only shows the direction. Shipped: exact, and 8% of depparse on the desktop.
