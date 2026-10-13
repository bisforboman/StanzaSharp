# Results, round 15: the tokenizer's long paragraphs

Part of [performance.md](../performance.md)'s measurement history.

In the latest benchmark run tokenize was the one stage where StanzaSharp did not lead everywhere (macos-26 0.70x,
ubuntu-26.04 0.97x). The benchmark text has one paragraph of 1,883 characters (from tokenize_stress.txt) among 48.

**Profile** (`dotnet-trace`, `stanzasharp benchmark --processors tokenize,mwt`, Ryzen 7 5800X; share of the tokenize
stage, before this round on 1 thread and after it on 4):

| | before, 1 thread | after, 4 threads |
|---|---:|---:|
| the biLSTMs' recurrence (`PackedLstm.Recur`) | 55% | 56% (31% of it spinning at each step's region) |
| the biLSTMs' input GEMMs | 24% | 9% |
| heads, gating, log-sigmoids (rest of `ManagedTokenizerNet`) | 6% | 8% |
| units, ids, features and their regexes (`AddParagraph`) | 4% | 8% |
| `FixLabels` (the e-mail/URL regex) | < 2% | 6% |
| decoding into tokens and sentences, `MarkWhitespace` | 2% | 2% |
| of all the above, the long paragraph's batch (`PredictWindowed`) | **60%** | 38% |

Allocation is small (round 11: 3 KB per word). The time was in `PredictWindowed`: a paragraph over the window (1,000
units) runs in windows, and every window after the first ran **all** rows of its batch at the full window width, as
Stanza's `advance_old_batch` pads them. The other 31 paragraphs of the batch are finished by then, so the network ran
31 rows of pure padding for each of them: about 30,000 row-steps for 900 useful ones.

**What changed:**
- Windows after the first run only the unfinished rows (`Tokenizer.LiveRows`). Rows are independent in every layer, and
  each row keeps its full-width run, so a row's arithmetic is unchanged, with one exception: `Gemm.Run` gives row m − 1
  of a call the single-row kernel when m % 6 is 1 (m = rows × width, in both LSTMs' input GEMMs and the two head
  GEMMs). That row must use it exactly when it did with all rows, so finished rows (all padding, outputs unused) go
  before the last live row until m % 6 matches. Both directions of that rule occur in the test texts below.
- The six regexes are source-generated (`[GeneratedRegex]`) instead of interpreted: the same patterns (checked
  string-equal to the old ones) and options. About 2 ms of a 38 ms tokenize once warm; no difference while the JIT is
  still tiering up, since the interpreter is precompiled and the generated code starts at tier 0.

**Exactness** (a temporary test, as in rounds 9–11): SHA-256 of every tokenizer logit the processor reads (each row
with any non-padding unit, per network call) and of the `tokenize,mwt` CoNLL-U, over every `tests/golden/*.txt`
(corpus, validation*, tokenize_stress, long_token) and all of them as one document, per kernel path, on 1 and 4
threads; plus the logits of 300 synthetic texts (random cuts of the golden texts, 1-40 paragraphs, many over 1,000
characters, and 100 batches of four long paragraphs whose shortest row outlives the others), which take the
row-count adjustment in both directions (33 and 3 windows). Before and after: identical on every path (golden
logits Vector256 `A3F25BD4…`, Vector128 `CACA5738…`, Scalar `AD1E3B8F…`; CoNLL-U `F1EACF94…` on all three; synthetic
`4ECEF22B…`, `C1EDECCA…`, `75AAC477…`). Without the adjustment the synthetic hashes change on Vector256 and Vector128,
so the test sees the single-row kernel.

**Measured locally** (Ryzen 7 5800X, Windows; `stanzasharp benchmark --processors tokenize,mwt`, 2,009 words; medians
of 5 runs, before and after alternating):

| tokenize | before | after |
|---|---:|---:|
| 8 threads (the default) | 144 ms | **60 ms** (2.4x) |
| 4 threads | 189 ms | **85 ms** (2.2x) |
| 1 thread | 257 ms | **145 ms** (1.8x) |
| 4 threads, `DOTNET_TieredCompilation=0` | 101 ms | **38 ms** (2.7x) |

Python Stanza on this machine: 330–400 ms (8 threads). The one-sentence calls are unchanged (with tiering off,
0.29 → 0.27 ms for `tokenize,mwt`).

**Free runners**, the `default` package, sequential runs of the same commit pair ([38284092033](https://github.com/bisforboman/StanzaSharp/actions/runs/38284092033)
main, [38283443584](https://github.com/bisforboman/StanzaSharp/actions/runs/38283443584) this branch; CoNLL-U identical to
Python's in both). Tokenize, StanzaSharp before → after, and Python Stanza in the after run:

| runner | before | after | Python Stanza | after vs Python |
|---|---:|---:|---:|---:|
| macos-26 | 0.55 s | **0.33 s** | 0.72 s | 2.19x |
| ubuntu-26.04 (AMD EPYC 7763) | 0.23 s | **0.13 s** | 0.32 s | 2.54x |
| ubuntu-26.04-arm | 0.20 s | **0.10 s** | 0.47 s | 4.62x |
| windows-2025 (AMD EPYC 7763) | 0.30 s | **0.18 s** | 0.46 s | 2.54x |
| windows-11-arm | 0.24 s | **0.09 s** | - | - |

What is left, and why the times above still vary so much between runs:
- **JIT warm-up.** The benchmark warms up on a tenth of the text, which has no long paragraph, so the windowed path is
  first compiled in the timed runs, and the per-step methods (`PackedLstm.Step`, `ManagedThreads.For`, `Pool.Run`,
  `Region.Drain`) stay at instrumented tier 0 through all three of them (`DOTNET_JitDisasmSummary`). Locally,
  tokenizing the benchmark text 40 times in one process takes 147, 62, 67, 70, 89 ... ms and settles at 27–33 ms after
  about ten calls; with tiering off it is 26 ms from the second call. So a warmed service tokenizes about twice as
  fast as the benchmark shows.
- **One parallel region per LSTM step.** The tokenizer's LSTMs are tiny (hidden 64) and long (the long paragraph runs
  ~2,000 sequential steps per level), so a step's work is a few µs, about what waking the pool's workers costs. On
  macos-26 (3 vCPUs, virtual) an experiment ([38283174666](https://github.com/bisforboman/StanzaSharp/actions/runs/38283174666),
  `tokenize,mwt` only, three rounds) measured the default 3 threads at 190–1,260 ms, `--threads 1` at 108–126 ms, and every
  step on the calling thread (GEMMs still parallel) at 79–109 ms; on ubuntu-26.04-arm the same serial steps were
  slower than the default (119–131 vs 73–79 ms). Serializing only steps of at most 4 or 8 rows was inconsistent on
  macOS. A fix belongs in `PackedLstm.Step` (fewer tasks when M × panels is small) or the pool, both outside this round.
- The e-mail/URL regex is still ~5% (its first character class matches almost any ASCII character); a literal
  prefilter (`@`, `http`, `www`, `.gov` ...) could skip it for most paragraphs.
