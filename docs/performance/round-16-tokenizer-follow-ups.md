# Results, round 16: tokenizer follow-ups (LSTM step split, warm-up, URL prefilter)

Part of [performance.md](../performance.md)'s measurement history.

Round 15's three leftovers. Output unchanged, bit for bit (below).

## Four tasks per thread in an LSTM step

`PackedLstm.Step` split each time step into one task of whole 16-column panels per thread; a step is a parallel region
that waits for its slowest task. The tokenizer's LSTMs (hidden 64: 32 panels over both directions, a few µs per step)
run thousands of such steps on the benchmark text, and on macos-26 (3 vCPUs) the default 3 threads were slower than 1.

What was tried, on the free runners (experiment runs [38288718064](https://github.com/bisforboman/StanzaSharp/actions/runs/38288718064),
[38291143656](https://github.com/bisforboman/StanzaSharp/actions/runs/38291143656) and [38292254170](https://github.com/bisforboman/StanzaSharp/actions/runs/38292254170);
each setting in its own process, 3–5 rounds, with this round's warm-up):
- **A work threshold** (fewer tasks when rows × panels × 16 × K multiply-adds are few): no threshold helped everywhere.
  16k multiply-adds per task (one row of the tokenizer → 2 tasks) made macos-26 ~40% faster but ubuntu-26.04-arm
  10–35% slower; there even one-row steps (32k multiply-adds, ~8k per task) are fastest on all 4 threads.
- **Measured per step size**: time each step, per power-of-two size class, under each task count (4, 2, 1), then run the
  fastest. Better on macos-26 (90 → 66 ms) but 25–50% worse on both Arm runners: a parallel candidate timed while
  workers were asleep (a class had just run serially) looked slow and was then rarely retried. Dropped.
- **The caller's wait**: spinning instead of `SpinWait.SpinOnce` (which yields) when waiting for the last task: mixed,
  within noise except on macOS (~20% faster).
- **More tasks than threads** (2, 4, 8 per thread, or one per panel): 2 or 4 per thread was the fastest tokenizer on
  every runner, and 4 also sped up the big LSTMs (below). A thread that runs
  late (a vCPU the host is busy with, a slower core, an SMT sibling) then holds a quarter of its share, while the
  others claim the rest; with one task per thread the step waits for its whole share.

**What changed:** `Step` runs min(4 × threads, panels) tasks (`TasksPerThread`). A panel's arithmetic (its rows in
blocks of `Gemm.MR`, `KernelBlocked`, the cell update) doesn't depend on which task runs it, so the split can't change a
result; the pool already hands out a region's items one at a time.

Same-VM A/B (run 38292254170: each runner ran both settings alternately; tokenize alone is the median of 5 processes,
the full pipeline of 2; ms):

| runner | tokenize alone, 1 → 4 tasks/thread | all eight: total | constituency | depparse | per call |
|---|---:|---:|---:|---:|---:|
| macos-26 | 84 → **60** | noisy (7,878 → 9,057, one run 10.2 s) | 1,071 → 1,104 | 2,021 → 2,615 | 117.1 → 107.5 |
| ubuntu-26.04 (EPYC 9V74) | 89 → 86 | 6,383 → **6,248** (-2%) | 853 → 822 | 1,759 → 1,717 | 60.1 → 60.9 |
| ubuntu-26.04-arm | 73 → **63** | 8,776 → **8,441** (-4%) | 1,112 → **997** | 2,540 → 2,383 | 94.1 → 90.7 |
| windows-11-arm | 72 → 75 | 8,659 → **8,396** (-3%) | 1,107 → **1,014** | 2,498 → 2,350 | 85.4 → 84.9 |
| windows-2025 (EPYC 7763) | 104 → **94** | 8,174 → **7,966** (-2.5%) | 1,165 → 1,113 | 2,239 → 2,161 | 76.6 → 75.4 |

Locally (Ryzen 7 5800X, 8 threads, alternating, 5 and 3 processes): tokenize alone 52.9 → 45.1 ms; all eight: pos -3%,
constituency -4%, depparse -7%, ner -3%, total -3%, one-sentence calls 39.9 → 39.0 ms. The one cost: a one-sentence
`tokenize,mwt` call, all tiny steps, takes ~0.07 ms longer (0.38 → 0.45 ms here; +0.1 ms on the Arm runners, less
elsewhere), more claims for the same work; with all eight processors it is lost in the noise of a ~40–100 ms call.

## Benchmark warm-up

`stanzasharp benchmark` warmed up on a tenth of the text, which has no long paragraph, so the tokenizer's windowed path
was first compiled in the timed runs, and its per-step methods stayed at JIT tier 0 through all three. A warmed service
is faster than the numbers showed. Now each side runs the whole text until it settles: no stage got more than 10% and
20 ms faster than in its previous warm-up run (at least 2 runs, at most 5; `--quick`: 1). The first criterion tried,
10% alone, never settled on the runners (some 50 ms stage always moved 10%), so the noise floor of 20 ms was added.
Python Stanza goes through the same loop; its stages settle after 2–3 runs. The block names both counts ("after 4
warm-up runs (Stanza 2)").

**Numbers from before this round aren't directly comparable.** How much they move, on the same runner type (runs
[38288777481](https://github.com/bisforboman/StanzaSharp/actions/runs/38288777481) main and
[38288755731](https://github.com/bisforboman/StanzaSharp/actions/runs/38288755731) the warm-up change alone; only the
two Arm runners drew the same CPU in both):

| | StanzaSharp total | tokenize | lemma | other stages | Python Stanza total |
|---|---:|---:|---:|---:|---:|
| ubuntu-26.04-arm | 9.03 → 8.60 s (-5%) | 110 → 62 ms | 86 → 53 ms | -3 to -5% | -1% |
| windows-11-arm | 9.12 → 8.65 s (-5%) | 110 → 65 ms | 84 → 60 ms | -4 to -7% | -1% |
| Ryzen 7 5800X, 4 threads | 5.2 → 5.2 s | ~70 → ~47 ms | ~62 → ~44 ms | within noise | - |

So StanzaSharp's words/s rise by up to ~5% and its advantage over Python Stanza by as much; tokenize and lemma, the
stages with the most small per-step calls, get about 30–45% faster. The x64 runners drew different CPUs in the two runs, so
they say nothing here. The one-sentence calls are unchanged (they had 5 warm-up calls before and still do). A run now
takes 4–8 minutes per runner instead of 2–5.

## E-mail/URL regex prefilter

The tokenizer runs Stanza's e-mail/URL pattern over every paragraph to keep matches whole (`FixLabels`). Every match
contains one of `@`, `://`, `www.`, `.gov`, `.org`, `.edu`, `.net`, `.co`, read off the pattern's alternatives (the
pattern is case-sensitive): the e-mail alternative has a literal `@`; the four URL forms start `https?:\/\/` or
`www\.`; the last one has `\.(?:gov|org|edu|net|com|co)`, and `.com` contains `.co`. `AddParagraph` checks for them
once with `SearchValues<string>` (vectorized) while it builds the features, and `FixLabels` skips the regex, and the
joined string it would need, when none occurs. 44 of the benchmark text's 48 paragraphs have none.

**Exactness:** `MaskLiterals_EveryAlternativeNeedsOne`: a match for each alternative of the pattern that contains only
the one literal it needs (so the list misses none of these, and each literal is needed); and
`MaskLiterals_PresentWhereverTheRegexMatches`: a literal is present in every paragraph of every golden text where the
regex matches (14) and in every one of 200,000 random strings built from the pattern's own pieces (`http`, `://`,
`www`, `@`, `.`, the TLDs, quotes, brackets, digits, `WWW.` ...) that it matches (3,559).

**Measured:** less than round 15's profile suggested. On the benchmark text the regex costs 0.23 ms per tokenize call
once warm and 0.07 ms with the prefilter (about 0.3% of tokenize); the "~5%" in round 15's profile was probably
the source-generated regex still at JIT tier 0 (its first call here: 2.2 ms). The prefilter keeps that cold cost off the
paragraphs without an address too.

## Exactness

As in rounds 9–15, a temporary test hashed (SHA-256) every tokenizer logit the processor reads (per network call), the
`tokenize` CoNLL-U of every `tests/golden/*.txt` and all of them as one document, both charlms' outputs over the
corpus, validation and stress texts' sentences (batches of 40 and one alone), the all-eight CoNLL-U of those texts as
one document, and 20 one-sentence calls, on Vector256, Vector128 and Scalar at 1, 4 and 16 threads (no local
AVX-512). Before (main) and after this round: identical everywhere (tokenizer logits Vector256 `66E9A9BC…`, Vector128
`A9C8C202…`, Scalar `258F2053…`; charlm `33A8BFDA…` on Vector256/128, `26F21566…` Scalar; pipeline `C904102E…` and
one-sentence `68AB6199…` on all three). The measured-step variant above gave the same hashes too. `ManagedBackendTests.LstmStep_EveryThreadCountGivesTheSameBits`
keeps the property: 4 to 32 tasks over 32 panels, bit for bit.
