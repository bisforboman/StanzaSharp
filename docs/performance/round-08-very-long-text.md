# Results, round 8: one call on a very long text

Part of [performance.md](../performance.md)'s measurement history.

What happens to memory when a whole book goes into one `Process(text)` call? Round 3 measured up to 15,000 words; this
round goes to 670,000, on the managed backend with all eight processors, and asks what grows with the input apart from
the output itself.

**Method.** `--memory N --verbose` as in round 3 (converted models, 8 threads, workstation GC, Windows), one fresh
process per run. New in the benchmark: `--verbose` ends with the documents' own size (the GC heap with the returned
documents minus without them, after full GCs), each processor's log line shows what it allocated, the `process` line
the call's total allocation, and `--live` adds a full GC and the live heap after each processor. 15,000, 50,000,
150,000 and 500,000 whitespace-separated words are 20,513, 67,120, 201,313 and 670,972 words after tokenizing.
"Before" is main at 629634f. Other jobs (three test suites of parallel sessions) kept the machine at 100% CPU for every
"after" run and the 670,000-word "before" run, so their times are 3–15× the idle ones and are not compared;
alternated under the same load, three before and three after runs on 20,513 words took a median 233 s and 224 s
(peaks 1,995–2,092 MB and 2,008–2,077 MB): the time is unchanged within noise.

Peak working set, MB, and the returned documents on the GC heap:

| words after tokenizing | 20,513 | 67,120 | 201,313 | 670,972 |
|---|---:|---:|---:|---:|
| peak, before | 2,006 | 2,010 | 2,389 | 3,636 |
| peak, after | 2,029 | 1,935 | 2,259 | 3,024 |
| documents, before | 12 MB | 38 MB | 114 MB | 380 MB (595 B/word) |
| documents, after | 10 MB | 31 MB | 92 MB | 308 MB (482 B/word) |
| allocated by the call (after) | 2.5 GB | 4.2 GB | 11.2 GB | 34.5 GB |
| time, before (idle machine) | 26 s | 86 s | 288 s | |

Loading peaks at 840 MB and leaves 827 MB (a 638 MB GC heap: the models) in every run. Single runs: the peak moves by
±50 MB between identical runs at 20k words, and more on the long ones, where it depends on when the GC runs.

**What the peak is made of.** The live heap after each processor (`--live`, 670,972 words, after) separates what is
kept from garbage: 983 MB after tokenize, 1,406 after pos, 1,203 after constituency, 1,520 after depparse, 1,784 after
sentiment, and 961 at the end, against 627 MB after loading. At the end that is exactly the models plus the documents
(627 + 308 MB): nothing else survives the call. During it:

- **Independent of the length:** the models (640 MB on the GC heap), the scratch buffers of the largest batches, which
  `ArrayPool<T>.Shared` keeps (0.3–0.8 GB; every processor's batches are capped: pos, depparse and sentiment at 5,000
  words, the tokenizer at 32 paragraphs of at most 1,000 characters, the parser at 50 sentences, NER at 32), and the
  `CharlmCache` (8 KB per word up to its 32,768-word cap, so 256 MB from about 30,000 words on).
- **The output**, the `Document`: 482 bytes per word after this round. It is the only thing that grows. For all eight
  processors (measured per processor set at 6,500 words): tokens and words with their texts, whitespace and offsets
  310 B/word, the constituency tree 150, entities and NER tags 21; tags, features, lemmas, heads and relations about 3
  (fields of `Word` pointing at shared strings; the feats strings took 46 B/word before they were shared).
  `Document.Text` and each `Sentence.Text` add 2 bytes per character each.
- **Per-processor lists over the whole document** (the round-2 worry in "Remaining ideas"): each processor's lists of
  sentences, words or (word, tag) pairs and its sort order are references and indices, 10–40 bytes per word, and are
  dropped when the processor returns. The largest transient one is the tokenizer's: every paragraph's code points,
  offsets, vocabulary ids and features are built before the first batch, about 85 bytes per character (about 0.5 KB per
  word), during tokenizing only, which never sets the peak (1.7 GB at 670,972 words against 3.0 GB later).
- **Garbage the GC hasn't collected yet**, the rest: the peak was 0.6–1.1 GB above the live heap. The call allocates
  56–125 KB per word (per-batch arrays the pool doesn't hold: GEMM results, LSTM outputs, logits), and the GC lets more
  of it accumulate as its heap grows (its gen2 and large-object budgets scale with what survived). It is not needed
  memory: with the GC heap capped by `DOTNET_GCHeapHardLimit` (a container's memory limit sets 75% of it by default),
  201,313 words ran in a 1,792 MB heap (`0x70000000`) and peaked at 1,782 MB instead of 2,259 MB, with 28 gen2 GCs
  instead of 23 and no measurable slowdown under the same load; 670,972 words ran in a 2,304 MB heap
  (`0x90000000`), peaking at 2,230 MB instead of 3,024 MB (41 gen2 GCs instead of 37; 4,593 s against 4,207 s, within this load's noise).

**What changed** (no output changes; all golden tests pass):

- The `Document` shrank from 594 to 482 bytes per word (−19%; 72 MB at 670,972 words): `Token.Words` starts with room
  for one word (an empty `List` grows to four on its first `Add`), `SpaceAfter` shares one `" "` instead of a new
  substring per token, the tagger shares equal feats strings within a call, and constituency trees keep their children
  in arrays instead of lists.
- NER, which runs last, releases each batch's `CharlmCache` entries once it has read them, so the cache (256 MB on long
  texts) is gone by the end of NER instead of held to the end of the call.
- The peak fell by about 0.6 GB at 670,972 words (3.6 → 3.0 GB) and within noise below that; the documents account for
  72 MB of it, the cache release and GC timing for the rest.

**Splitting the input.** Not done inside the library: it would change output. Stanza (and this port) batch the
sentiment classifier and the dependency parser over the sentences of the whole call, sorted by length; a sentence's
sentiment label depends on its batch's padding, and depparse's arc log-softmax runs over its batch's padded width.
Calling `Process` per part of about 1,000 words on 20,513 words (all eight processors) changed 508 of 2,091 sentiment
labels (24%) and nothing else: every token, sentence boundary, lemma, tag, feature, head, relation, tree and NER tag
was the same. Sentence ids restart in each part, offsets are each part's own, and the whitespace at each cut is split
between the parts (round 3). For the memory: 201,313 words in 148 parts of about 1,000 words, keeping every part's
document, peaked at 2,056 MB against 2,259 MB for one call (smaller batches, so smaller scratch buffers and less garbage) and
allocated 15.1 GB instead of 11.2 GB. A caller that drops each part's `Document` after use also saves the
documents' 0.5 KB per word.

Guidance (README, "Very long texts"): one call is fine for a book; the memory beyond the models and batches is the
`Document` (about 0.5 KB per word) plus collectable garbage. In a memory-limited container the GC adapts to the limit.
To bound memory further, split at blank lines and drop each part's `Document` after use, accepting the sentiment and
sentence-id differences.
