# Results, round 30: constituency's stack LSTM cell inside the GEMM's tasks

Part of [performance.md](../performance.md)'s measurement history. The idea came from
[round 27](round-27-whole-pipeline-profile.md) (target 4).

`ManagedConstituencyNet.StackLstm.Push` steps the transition and constituent stack LSTMs (2 layers each; hidden 20 and
512 in English) for every state in flight, at most 50 rows. Per layer it used to:

1. copy each row's `[x | h]` and the parent's c (serially, on the caller);
2. run `Gemm.Run(..., rowInvariant: true)` on the pool;
3. loop `Act.LstmCell` over every row and every four units (serially, on the caller).

Round 27 measured steps 1 and 3 at about 64 ms of constituency's 336 ms at 8 threads, with the workers spinning.

**What changed** (managed backend only):

- `Gemm.Run` takes an optional `epilogue(m0, m1, p0, p1)`, called by the task that computed rows [m0, m1) × panels
  [p0, p1), once their K blocks are all done (every task already runs the whole K for its block). The tasks are the
  same as before: `mBlocks × nBlocks`, independent of the thread count.
- `Push` passes an epilogue per layer. A 16-column panel holds four whole units ([i×4, f×4, g×4, o×4],
  `PackedLstm.GateOrder`), so for its rows and units the task copies the parent's c, runs `Act.LstmCell` (unchanged,
  one version per kernel path) and, below the last layer, copies the new h and the parent's next-layer h into the next
  layer's `[x | h]` rows (a second buffer, since other tasks still read this layer's rows).
- Only layer 0's `[x | h]` rows are still copied on the caller (they come from the caller's `inputs` delegate).

No new kernel or element-wise step: the cell is the same `Act.LstmCell` call per four units, only run by another
thread. `ManagedThreads.cs` is unchanged.

## Exactness

The cell is element-wise and reads only its own gates and c, so where it runs changes no bit; the GEMM keeps its split
and `rowInvariant`. A temporary harness (not committed) hashed with SHA-256 (first 12 hex digits below):

- **batch**: every transition score row and tree of all golden sentences (English: pipeline.conllu + validation*.conllu,
  845 sentences; German: 181), Stanza's words and tags, parsed as one batch (50 in flight);
- **alone**: the first 60 of them, each parsed alone;
- **pipeline**: the `tokenize,mwt,pos,constituency` CoNLL-U of corpus.txt (`Pipeline.Load` with `Threads`).

On Vector256, Vector128 and Scalar, at 1 and 8 threads, main (`1fe32df`) and this branch (Ryzen 7 5800X, Windows):

| language | path | batch | alone | pipeline |
|---|---|---|---|---|
| en | Vector256, 1 and 8 threads | `801475C7A835` | `C8C69491DFE2` | `9E611BDE2783` |
| en | Vector128, 1 and 8 threads | `55E55DD8B72B` | `ED810423F181` | `9E611BDE2783` |
| en | Scalar, 1 and 8 threads | `B8218CDB3CED` | `5B4EFC5C856E` | `9E611BDE2783` |
| de | Vector256, 1 and 8 threads | `B62AC0A924DC` | `660E166015E0` | `62BF9DD7913E` |
| de | Vector128, 1 and 8 threads | `B62AC0A924DC` | `A50FF0C0C22A` | `62BF9DD7913E` |
| de | Scalar, 1 and 8 threads | `6A8296FEDF90` | `CA3AAA704EAD` | `62BF9DD7913E` |

Every hash is the same on main and on the branch, at both thread counts. Vector512 isn't available locally; it runs the
same `Act.LstmCell` and `Gemm.Run` code as before, and `Vector512_GivesVector256sBits` keeps it equal to Vector256 on CI
runners that have AVX-512. `ConstituencyTests` and `ManagedBackendTests` pass (88 tests).

## Speed

**Free runners** (`gh workflow run benchmark.yml`, all eight processors, 2,009 words, 4 threads, 3 on macOS; constituency
stage, median of 3 timed runs per job). Main is `1fe32df` (four runs that day, two of them started by other agents'
rounds), the branch `96ad9db` (two runs). The runner matrix draws different x64 CPUs per run, so only runs on the same
CPU are compared:

| CPU | main runs | branch runs | change (medians) |
|---|---|---|---:|
| Neoverse-N2 (ubuntu-26.04-arm) | 0.99, 1.01, 1.02, 1.04 s | 0.98, 1.00 s | −2% |
| Cobalt 100 (windows-11-arm) | 1.01, 1.01, 1.04, 1.05 s | 0.98, 1.01 s | −3% |
| EPYC 7763 (ubuntu-26.04 / windows-2025) | 1.06 / 1.07, 1.21, 1.46 s | 1.03 / 1.02 s | −3% / −5% or more |
| Apple M1, virtual (macos-26, 3 threads) | 1.26, 1.38, 1.60, 1.69 s | 1.09, 1.09 s | about −25% |

The macOS runner is noisy (main ranged 1.26 to 1.69 s), but both branch runs were below every main run. Other CPUs came
up once each, too few to compare. Totals moved by less than the runners' spread (7.7 to 8.0 s on the Arm runners).

**Local** (Ryzen 7 5800X, `stanzasharp benchmark --processors tokenize,mwt,pos,constituency --threads 1`; four other
sessions were running performance work, so only 1 thread was steady): constituency 1.82 and 1.85 s on main, 1.81 s on
the branch (one more branch run hit by the load, 2.90 s). As expected, nothing changes at 1 thread. At 8 threads the
machine was too loaded for a comparison (8 threads ran slower than 1).

The gain is smaller than round 27's estimate of 40 to 50 ms at 8 threads would suggest for 4 threads, as that estimate
expected: at 4 threads the caller's serial part is a smaller share, and the copies and cells are cheap next to the GEMM
(the layer-0 copies and the transition system remain serial). Shipped: exact, a small change, and never slower.
