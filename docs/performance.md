# Performance

How fast StanzaSharp runs compared with Python Stanza, what was optimized, and what is left.
Every change kept the output byte-identical: all golden tests pass, and on the benchmark text the
C# CoNLL-U equals Python's line for line, before and after.

## Method

- **Benchmark:** `samples/StanzaSharp.Benchmark` (C#) and `tools/benchmark.py` (Python) build the
  same text and print the same report:

  ```powershell
  dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --models models\converted\en --threads 8 --out cs.conllu
  tools\.venv\Scripts\python tools\benchmark.py --models models\stanza --threads 8 --out py.conllu
  ```

- **Input:** `tests/golden/validation.txt`, `corpus.txt` and `tokenize_stress.txt`, joined by blank
  lines and repeated 8 times. That is 116k characters, 1,992 sentences and 26,264 words after MWT.
  - The text repeats, but batches cut across the copies differently, so each copy is batched
    differently.
  - `tokenize_stress.txt` adds long paragraphs and long run-on sentences.
- **Runs:** one warm-up run on a single copy, then 3 timed runs. The tables show medians.
  - Each stage is timed separately: tokenize, mwt, pos, constituency. Words/s is computed over all
    26,264 words.
  - Load time covers all six models: C# reads the converted `.safetensors` + `.json` files, Python
    the `.pt` files.
  - Peak memory is the process's peak working set.
- **Correctness check:** both benchmarks write their CoNLL-U output with `--out`. Every C# run below
  matched the Python output exactly.
- **Machine:** AMD Ryzen 7 5800X (8 cores, 16 threads), 64 GB RAM, Windows 11. Software: .NET 10,
  TorchSharp 0.107 (CPU), Python 3.10, torch 2.14.1, stanza 1.15.0.
- **Threads:** results are given for 8 and for 1 torch intra-op threads, set with `--threads` on
  both sides. 8 is torch's default on this machine (physical cores).
- **Caveats:** other jobs shared the machine overnight.
  - The final runs below were taken when it was mostly idle (5–10% CPU load before and after).
  - Earlier development runs, made under heavy load, were 1.5–3× slower. Compare numbers from the
    same table only.
  - Under load, the default thread count suffers most, because libtorch's OpenMP threads spin
    waiting for each other.

## Results

"Before" is `main` at b4c1464, the merge base of this work. "After" is `agent/perf`. Times are
seconds per run on 26,264 words.

**8 threads**

| stage            | C# before | C# after | Python |
|------------------|----------:|---------:|-------:|
| load             |      0.80 |     0.80 |   3.71 |
| tokenize         |      1.52 |     1.49 |   2.21 |
| mwt              |      0.02 |     0.01 |   0.38 |
| pos              |     43.85 |    17.30 |  23.37 |
| constituency     |     37.49 |    10.45 |  26.04 |
| **total**        | **82.89** | **29.25**| **52.01** |
| words/s          |       317 |      898 |    505 |
| peak memory (MB) |     2,544 |    2,159 |  1,753 |

**1 thread**

| stage            | C# before | C# after | Python |
|------------------|----------:|---------:|-------:|
| load             |      0.72 |     0.69 |   3.56 |
| tokenize         |      1.42 |     1.47 |   2.13 |
| mwt              |      0.02 |     0.01 |   0.35 |
| pos              |    159.37 |    44.27 |  65.10 |
| constituency     |    120.15 |    22.93 |  59.77 |
| **total**        | **280.95**| **68.67**| **127.35** |
| words/s          |        93 |      382 |    206 |
| peak memory (MB) |     2,528 |    2,142 |  1,730 |

Summary:
- C# went from 1.6× slower than Python to 1.8× faster at 8 threads.
- At 1 thread, it went from 2.2× slower to 1.9× faster.
- Pos and constituency together account for over 90% of the time. Tokenize and mwt are minor.

## Where the time went

Measured with stopwatch instrumentation on the stages and inside the parser.

1. **The charlm ran over padding.** Each charlm is a 1-layer LSTM with 1024 hidden units. It costs
   about 9 MFLOP per character, per direction.
   - `BuildCharRepresentation` ran it over the batch padded to its longest sentence, in characters.
   - The tagger's batches are 250 sentences in document order. One long sentence therefore made
     most of the work padding.
   - This was most of the pos time and a large part of the constituency time.
2. **The charlm ran twice per sentence.** The tagger and the parser feed the same words through
   the same two charlms. Python Stanza does this too.
3. **The parser ran in lockstep batches.** It took 50 sentences in document order and stepped them
   until the slowest one finished. Stanza instead keeps 50 states in flight, sorted longest first,
   and refills the batch as states finish.
   - The lockstep version stepped half-empty batches for a long time.
   - All tensors of a batch stayed in one `DisposeScope` until the batch ended.
4. **Peak memory.** Padding the packed charlm output back to [batch, longest sentence, 1024], only
   to read a few rows, created a tensor of about 1 GB for a 250-sentence batch with one very long
   sentence.

Not worth changing:
- JSON/safetensors loading: 0.7–0.8 s, against Python's 3.6 s.
- Tokenizer array building: the tokenizer is already faster than Python's.
- The parser's per-step overhead. After the changes it is about 2.5 s of the 10.5 s parser time
  at 8 threads, and much of that is the constituent-stack LSTM's own compute.

## What changed

Each change has its own commit, with numbers measured at the time.

| commit | change | why it keeps output identical |
|---|---|---|
| Pack the charlm input | `BuildCharRepresentation` packs sequences (`Rnn.RunPacked`), as Stanza does, so the LSTM never runs over padding | the LSTM is one-directional and padding follows each sentence, so the positions read are unchanged |
| Parser scheduling | `ConstituencyParser.Parse` schedules like Stanza's `parse_sentences`. Sentences are sorted longest first, 50 states are in flight, and finished states are refilled from the next 50 sentences. Each step gets its own `DisposeScope`. A state's kept tensors are owned by `ParserState` and freed when that state finishes. | each state's computation is independent. The batch makeup now matches Python's. |
| Charlm reuse | `Nn.CharlmCache`: `PosTagger.Process(doc, cache)` keeps each sentence's charlm outputs, and `ConstituencyParser.Process(doc, cache)` reuses them. The cache is keyed by the `Sentence` object, so it never matches across repeated text. The tagger keeps only sentences whose words `simplify_punct` left unchanged. The parser computes just the missing ones. | see the note below |
| `Rnn.RunPackedAt` | the charlm reads its word-end rows directly from the packed output instead of padding it first | pure indexing; values are unchanged |

Note on charlm reuse: the charlm is **not bitwise batch-invariant**.
- In a direct test, 149 of 208 sentences differed in their last float bits depending on which
  sentences shared the batch.
- So the parser's input from the cache can differ from a fresh computation, by about 1e-7.
- That is far below the existing C#-vs-Python drift (parser scores ≈ 2e-5).
- Outputs stayed identical on all golden files and on the 26k-word benchmark text, at 1 and at 8
  threads.
- If an output ever differed because of this, the fix is to drop the cache:
  `Pipeline.Process` would pass `null`.

## Remaining ideas

Rough payoff estimates at 8 threads, measured against the current 29 s total.

- **Parser step loop: per-state tensor ops.** Each step, every state does a handful of tiny
  TorchSharp ops: `WordHx` row views, `select` on the LSTM state, and `stack`.
  - A shared "bank" tensor per stack (state rows addressed by index, `index_select` /
    `index_copy_`) would make the op count per step constant.
  - Payoff: about 1–1.5 s (≈5%) at 8 threads, little at 1 thread, where the stack LSTM's compute
    dominates. Moderate refactor; exact, since it is data movement only.
- **Load-time memory.** Loading peaks at about 1.0 GB, against Python's 0.73 GB. Over 300 MB of it
  is managed heap from `File.ReadAllBytes` of whole safetensors files and the `JsonNode` tree of
  the 9.5 MB pretrain vocabulary.
  - Fixes: stream tensor reads, and read vocabularies with `Utf8JsonReader`.
  - Payoff: maybe 200–300 MB off the load peak. The overall peak is set during pos, so it falls by
    less.
- **CharlmCache memory.** The cache holds 8 KB per word until the parser finishes, about 210 MB
  for this text. For very large documents, free entries as the parser consumes them, or process
  the document in parts.
- **POS batch memory.** Stanza also cuts POS batches at 5,000 words (`batch_maximum_tokens`). The
  port cuts only at 250 sentences.
  - On normal text this never triggers. On text with very long sentences, adding the cut would
    match Python's batching and bound memory.
  - It only changes floats in the last bits, so check it against golden data the same way.
- **Thread count.** On a loaded machine, fewer threads were often faster than the default 8,
  because spinning OpenMP threads compete. A `--threads` option or a recommendation in the
  README may help users on shared machines.
