# Status and port history

How each processor was ported, what it reproduces and the measurements and decisions along the way (dated in place). Moved from CLAUDE.md on 2026-10-31.

All 7 steps of the build order are done. The bullets below describe the original TorchSharp port; its torch classes
(`Weights`, `Rnn`, `CharLanguageModel`, `HighwayLstm`, `Biaffine`, `CharacterModel`, the nets) live in `StanzaSharp.Cuda`
since 1.0, and every processor also has a managed net (see "Managed backend"), the default.
- Models are converted, golden data is in `tests/golden/`, and the checkpoint configs are summarized
  under "Checkpoint findings".
- `Core` has the document model, `Conllu` (round-trips the golden file byte for byte), `Tree`,
  `SafeTensorFile` and `Checkpoint`.
- `Nn.Weights.LoadFrom` fills any TorchSharp module from a checkpoint by parameter-name prefix.
- `Tokenize.Tokenizer` matches Stanza's tokens, offsets, spacing and sentences on the golden corpus
  and the stress set.
- `Mwt.MwtExpander` uses the dictionary, then the character classifier. It matches the golden
  words, word offsets, and `mwt.json`, which covers dictionary and classifier-only output.
- `Nn.Rnn.RunPacked` wraps pack/LSTM/unpack for models that pack by real length (MWT, POS,
  and the parser's word LSTM).
- `Nn.CharLanguageModel.BuildCharRepresentation` gives one vector per word, taken at the space
  after it. It matches the golden charlm tensors within 1e-4. The tagger and the parser both use it.
- `Nn.Pretrain` provides `UnitToId` (spaces stored as U+00A0) and `Embeddings`. Callers decide
  on lowercasing: POS lowercases every word, while the parser tries the word as written, then
  lowercased.
- `Pos.PosTagger` takes the shared `Pretrain` and both charlms, which the caller owns. It matches the
  golden UPOS/XPOS/feats exactly after tokenize → mwt → pos. `Nn` gained `HighwayLstm` and
  `Biaffine`. Like Stanza's `simplify_punct`, the tagger sees runs such as `??` or `!?!` as `?`
  or `!`. Only the tagger does this, not the parser.
- `Constituency.ConstituencyParser` also takes the shared `Pretrain` and charlms. It reproduces the
  golden trees and the per-step transition scores. It schedules like Stanza's `parse_sentences`:
  sentences sorted longest first, 50 states in flight, each finished state replaced by the next.
  Each step is one forward pass plus batched stack pushes. Stack LSTMs are stepped without
  padding, so batching does not change results. On TorchSharp (round 25) the net's handles are rows of per-thread
  "bank" tensors (vectors, each stack's h and c; a free list, freed when the last row comes back), so a step costs a
  constant number of ops (`index_select`/`index_copy_`), not a few per state.
- `Nn.CharlmCache`: the pipeline passes one to `PosTagger.Process` and `ConstituencyParser.Process`,
  so the parser reuses the tagger's charlm outputs (the tagger keeps a sentence only if
  `simplify_punct` left its words unchanged). The charlm is not bitwise batch-invariant, so cached
  values can differ from a fresh computation in the last bits; outputs are still identical.
- Performance: `docs/performance.md` has the benchmark (`samples/StanzaSharp.Benchmark`,
  `tools/benchmark.py`), the method, the latest runner results and remaining ideas.
  Each measurement round is its own file, `docs/performance/round-NN-<slug>.md` (NN zero-padded, the number the
  brief assigns; owner's decision, 2026-10-13, as parallel PRs kept conflicting in one file). A new round adds
  its file and leaves performance.md alone (no index there); only "Run it yourself" changes go there. Open ideas are one file each in `docs/performance/ideas/` (user's decision,
  2026-10-13): a new idea adds a file, finishing or dropping one deletes its file (the round file is the record).
  Round 2 (memory): checkpoints are read tensor by tensor straight into TorchSharp memory (no
  whole-file buffers), depparse's biaffine scorers run a few sentences at a time, POS batches are
  cut at 5000 words like Stanza, and `CharlmCache` keeps at most 32k words (`MaxWords`).
  Round 3 (issue #19, `--memory`): `.pt` files are read tensor by tensor too (the unpickler reads a stream; storages
  are read from the file on demand), and `HighwayLstm` frees each layer's padded tensors as soon as they are used.
  Round 7 (per call, `--per-call N`, `--concurrent T`): one short sentence takes 28 ms managed (Python 189); the rest is
  memory bandwidth (weights streamed per call), not setup. Past ~2 concurrent callers managed throughput drops
  (L3 evictions); `PipelineOptions.MaxConcurrentCalls` (0.6, opt-in) is the gate.
  After a Process call of 1,000+ words, `NativeHeap.Trim` calls glibc's `malloc_trim(0)` on Linux (user's decision,
  2026-10-07: always on, no public API; `PipelineOptions.TrimNativeHeap` is internal, for the benchmark); since the
  decision review of 2026-10-31 only on the TorchSharp backend (`BackendModels.UsesNativeHeap`), as the managed models
  have nothing on the native heap (round 6) and the trim only took glibc's arena locks. On Windows
  libtorch's built-in mimalloc keeps freed pages; only `MIMALLOC_PURGE_DELAY=0` releases them (documented, not set).
  Round 11 (garbage on long texts, managed): constituency's stack states, word vectors and Compose outputs come from a
  thread-static `Recycler` (per call, since Parse is synchronous; emptied when every array is back) and go back when
  `ParserState.Dispose` disposes them; Chu-Liu/Edmonds works on flat rented n × n buffers; `DepparseBatch.RentScores`
  lets the managed net rent the arc/label outputs, which `DependencyParser.Parse` returns. Rent an output only where its
  one caller gives it back; the Cuda nets keep plain arrays. A call now allocates ~18 KB/word instead of ~54 (docs/performance/round-11-garbage-long-texts.md).
  Round 18: the pool's trimming costs nothing (`DOTNET_SYSTEM_BUFFERS_ARRAYPOOL_TRIMSHARED=false`: same bytes); the
  managed charlm's per-character strings (`EnumerateRunes().Select(ToString)`) were ~3 KB/word, now looked up by span
  (`GetAlternateLookup<ReadOnlySpan<char>>`): 15 KB/word. Depparse's forward charlm from a stored "\n\n " state was
  exact but saved 0.13–0.2 ms per sentence, under a call's noise: not shipped.
  Round 22: 11 KB/word (NER's emissions rented through `INerNet.Forward`'s span, Tarjan's per-node arrays, ASCII
  `PyString.Lower`, the `Recycler` keeping `StackState`/`Composed` objects, the tokenizer's `Paragraph` as one string with
  inputs rented per batch). GC allocation ticks over-credit small objects; trust the benchmark's per-processor bytes.
  Round 23: the `ArrayPool` "misses" (`PoolExhausted`) left in the LSTMs are the pool's first fill on a fresh process
  (~0.6 GB, live and reused; a second call allocates 8 KB/word, 7 since round 38: rented Chu-Liu/Edmonds contractions, constituency step buffers), not garbage; no thread, partition or size limit is hit.
  Round 12 (concurrent calls, `--with-long`): `ManagedThreads.Pool` keeps workers asleep while callers in a region plus
  awake workers number max(ProcessorCount, Count) (`_busy`/`_limit`); with Threads = ProcessorCount and 8 callers, spinning
  oversubscription had stalled calls for seconds. Calls don't serialize or starve; >2 callers cost throughput (L3/DRAM,
  same on 1 thread). macOS runner one-sentence slowness was Spotlight indexing (benchmark.yml turns it off). Neither
  physical cores nor all logical processors won everywhere (Ryzen vs the 2-core runners). Round 13: the default is the
  physical cores from 4 up, else all logical processors (Ryzen 8 vs 16: big texts -9%/-6%, one-sentence p90 -16%/-36%,
  2 callers +11% calls/s; runners unchanged at 4/4/4/4/3).
  Round 6 (managed backend, docs/performance/round-06-memory-managed-backend.md): no native heap (malloc_trim returns nothing there); after a call the
  GC heap holds the models plus `ArrayPool<T>.Shared`'s scratch buffers (all eight, 6,761 words: 2.4 GB working set
  then, 1.3 GB since round 10 shrank sentiment's buffers), released only by a gen2 GC a minute later; the README tells services to run an aggressive GC when idle. The library
  runs no GC.
- `Tree.ToString` prints `(`/`)` in labels and words as `-LRB-`/`-RRB-`, like Stanza. The tree
  itself keeps the raw text.
- Measured float drift vs Python: charlm < 1e-7, tokenizer logits ≈ 4e-6, UPOS logits ≈ 2e-5,
  parser scores ≈ 2e-5. Test tolerances are 1e-4 and 1e-3. A 1e-4 tolerance is relative above |100| (owner's decision,
  2026-10-14, round 26): a value passes within 1e-4 × max(1, |expected| / 100) of its reference
  (`TokenizerTests.Allowed`, used by `AssertClose`, `PosTests.AssertUposClose` and `SentimentTests`), since float32
  can't hold 1e-4 above |x| ≈ 300 (one ulp is 3.05e-5 from 256 up). Only UPOS logits exceed |100| in the golden data
  (German up to |380|, English |139|); 1e-3 tolerances stay absolute.
- `StanzaSharp.Pipeline` ties the processors together: `Load(modelDir, "tokenize,mwt,...")`, then
  `Process(text)`. It loads the shared pretrain/charlms once. `Conllu.Write(pipeline.Process(corpus))`
  is byte-identical to `tests/golden/pipeline.conllu`.
  - The `samples/StanzaSharp.Cli` sample writes the same CoNLL-U from a file or stdin.
  - `Conllu.Write` fills a missing HEAD with `id - 1`, like Stanza's writer.
- `Depparse.DependencyParser` (processor `depparse`, requires tokenize, mwt, pos, lemma; runs after
  lemma and constituency, as in Stanza) sets `Word.Head`/`Deprel` with the pipeline's shared pretrain/charlms.
  `Pipeline` with `tokenize,mwt,pos,lemma,depparse` reproduces every `tests/golden/depparse/*.conllu`
  byte for byte (13 files, 8401 words), and so does the parser alone on Stanza's own tags and lemmas.
  Arc/label log-prob drift ≈ 2e-5 (tolerance 1e-4).
  - The parser feeds the charlms `"\n"` + the words (a ROOT word), so every forward state and the backward ROOT
    state differ from the tagger's. But backward, ROOT comes last: the tagger's backward pass is the parser's prefix.
    Since `Gemm.KernelBlocked` a charlm step gives a row the same arithmetic in any batch, so (managed only) the tagger
    keeps each cached sentence's backward final cell state (`CharlmCache.TryAdd`'s `backwardFinalC`, 4 KB per sentence,
    not counted in `MaxWords`) and the parser takes every such sentence's backward rows and continues from the tagger's
    final (h, c) over ROOT only, in its own batches (`CharlmCache.TryGetBackward`: the entry by `Sentence`, else the
    single-sentence table by texts; `ManagedCharLanguageModel.Continue` takes per-sentence start states). Bit for bit
    (`Batches_ContinueTheTaggers...`, `OneSentenceDocument_ContinuesTheTaggers...`; round 20 hashed every path).
  - `CharlmCache`'s single-sentence table (`AddAlone`/`TryGetAlone`, keyed by the texts): a managed charlm call on one
    sentence depends on its texts only, so `ManagedCharLanguageModel.BuildBoth` (pos, sentiment, ner) reuses any earlier
    one on the same texts exactly. Besides depparse, NER takes a multi-word-token sentence's token charlms from sentiment.
    `PipelineTests.OneSentenceDocuments_EqualTheUncachedPipeline` checks the whole pipeline against the cache turned off.
  - A missing lemma is read as `_`, as in Stanza, so without the lemmatizer the parses differ.
  - `Conllu.Write` needs no change: with heads set it matches Stanza (DEPS stays `_`).
- `Lemma.Lemmatizer` (processor `lemma`, requires tokenize, mwt, pos) reproduces `tests/golden/lemma/`
  byte for byte: 13 files, 8401 words, 650 of them through the seq2seq model (868 before the re-upload of 2026-10-30). It mirrors
  LemmaProcessor: dictionary skip, DeltaVocab, `batch_size` 50 batches sorted like `sort_all`, greedy
  decoding, edits, `<UNK>` fallback, and the `Word.lemma` setter turning `_` into null.
- `Ner.NerTagger` (processor `ner`, requires only tokenize: it reads token text, like Stanza) sets
  `Token.Ner` (BIOES) and `Sentence.Entities` (`Entity`: Text, Type, StartChar, EndChar, Tokens;
  `Document.Entities` concatenates them). `tests/golden/ner/` is reproduced byte for byte: 13 files,
  845 sentences, 706 entities, with `tokenize,mwt,pos,lemma,depparse,ner`, with the cache off, and with
  `tokenize,ner` alone. Emission drift ≈ 2.4e-6 (tolerance 1e-4).
  - `Entity.Build` is `build_ents`/`decode_from_bioes`; `Conllu.Read` calls it too (text from the
    tokens, since a read document has no `Text`).
  - `Conllu` writes/reads `ner=` after `end_char` in MISC, on the MWT range line and on single-word
    tokens' word lines, never on MWT words. `multi_ner` is not written by Stanza and not ported (with
    one model it is just `(ner,)`).
  - `CharlmCache`: NER's charlm input (`"\n"`, each token + `" "`, same vocab) equals the tagger's
    when a sentence's tokens are its words (no MWT), so it reuses those; others are computed.
- `Checkpoint.Load` also reads Stanza's original `.pt` files (`Core/TorchCheckpoint.cs` +
  `Core/Pickle.cs`), so `Pipeline.Load("models/stanza/en")` works without Python. For all fourteen
  checkpoints the result equals the converter's: identical JSON and byte-identical tensors.
- `Sentiment.SentimentClassifier` (processor `sentiment`, requires tokenize; runs last, as in Stanza)
  sets `Sentence.Sentiment` (0 negative, 1 neutral, 2 positive). `Conllu` reads and writes
  `# sentiment = N` after `# constituency`, Stanza's comment order. `tokenize,mwt,sentiment`
  reproduces `tests/golden/sentiment/` byte for byte (14 files, 886 sentences), with every label and
  logit (drift ≤ 2.2e-5, tolerance 1e-4), also as one document in two 5000-token batches (`all.json`).
  - Batches must match Stanza's: padding runs through the unpacked biLSTM and the convolutions, so a
    sentence's result depends on its batch (one document vs per file: 173 of 886 labels change in Stanza itself).
  - `CharlmCache`: the classifier reads token texts, so it reuses the tagger's outputs for sentences
    without multi-word tokens (641 of 886). Labels stay identical; logits drift up to ~1e-4 (vs
    Stanza), as the network amplifies the cached charlm's last-bit differences; tested at 1e-3.
- `default_fast` (`PipelineOptions.Package`): `PosTagger`, `DependencyParser` and `NerTagger` switch on the
  checkpoint's `charlm` flag and take null charlms for `_nocharlm` checkpoints. `tests/golden/fast/` is
  reproduced byte for byte (13 files, 845 sentences, 8,401 words, all seven processors). Drift: UPOS
  logits 1.5e-5, arc log-probs 7.6e-6, label log-probs 1.5e-5, NER emissions 1.9e-6 (tolerance 1e-4).
  - No `CharlmCache` there: the nocharlm tagger has no charlm outputs to share (`PosTagger.UsesCharlm`).
  - Stanza silently skips a processor its package lacks (`processors='...,constituency'` with
    `default_fast` loads no parser); we throw instead (user's decision), naming the `default` package.
- Input modes (2026-10-06, the user's approved examples): `Process(IEnumerable<string>)` (bulk) and
  `Process(IEnumerable<IEnumerable<string>>)` (pretokenized). Collection expressions pick the right overload;
  `Process([])` and `Process(null)` are ambiguous (compile errors), never a silent wrong pick.
  - Bulk = Stanza's `bulk_process`: `Tokenizer.Process(texts)` tokenizes `"\n\n".Join(texts)` and `Tokenizer.Split`
    deals sentences out like `TokenizeProcessor.bulk_process` (a sentence goes to the current text while its last
    token ends inside it; offsets shifted; the last token's SpaceAfter and first token's SpacesBefore from the text
    itself). The other processors run on one combined `Document` (UDProcessor.bulk_process), then entities are
    rebuilt per document. Sentence ids continue across documents, as in Stanza.
  - Bulk ≠ one by one, in Stanza too: sentiment batches change labels (172 of 854 in the golden; depparse could too, via
    its padded log-softmax, but no head changed), and sent_ids continue. Without those two processors the output equals one call per text apart from
    sent_id (`Bulk_EqualsProcessingAloneExceptBatchedProcessors`).
  - Pretokenized = `process_pre_tokenized_text` with a list: text = tokens joined by single spaces, offsets into
    it, `IsMwtCandidate` false, so mwt changes nothing ("don't" stays one word, as in Stanza). Empty sentences and
    empty/whitespace tokens throw (Stanza fails on the first two; whitespace tokens give broken CoNLL-U there).
    Stanza's string form (whitespace tokens, newline sentences) is not exposed; make_golden asserts it equals the list form.
  - `Process(Document)` (owner's decision, 2026-10-14) = Stanza's `process_conllu`: `nlp.process(doc, processors=` the
    loaded ones minus tokenize and mwt`)`. Same `Annotate` as the other overloads with `mwt: false`, gated and cancelable
    like them; the document is changed in place and returned. Each processor overwrites what it sets (Stanza does too);
    the rest is kept (a tree in default_fast). Stanza keeps comments in input order, so a document
    whose CoNLL-U had `# sentiment` before a new `# constituency` writes them the other way round there; `Conllu.Write`
    always writes constituency first. `Process(null)` stays ambiguous; `Document` is no collection type, so collection
    expressions never pick it.
  - `PipelineOptions.Pretagged` (owner's decisions, 2026-10-17; docs/pretagged-plan.md) = Stanza's `pretagged=True`:
    lemma, depparse and constituency need no processor at Load (`SelectModels`' `pretagged`; `FilesFor` follows), so
    `Processors = "depparse"` loads the parser without the tagger and parses a `Process(Document)` document's own tags.
    Tokenize and the package's mwt always load (1.2 MB), so the text overloads run and fail in the parsers like Stanza's
    `tokenize,depparse` pretagged pipeline; pos, ner and sentiment keep their requirements. Missing annotations as in
    Stanza (depparse throws for a word with neither UPOS nor XPOS, reads a missing XPOS or lemma as `_`; the lemmatizer
    a missing UPOS as `_`), except that constituency throws for a word without its tag (Stanza parses an empty tag).
    `tests/golden/pretagged/` (`make_golden.py --pretagged-only`, `PRETAGGED_CASES`), `PretaggedTests`.
- 0.4 API (issue #12, items chosen by the owner, 2026-10-06):
  - `PipelineOptions.Threads` (int?): at Load, process-wide, `ManagedThreads.Count` on the managed backend and
    `torch.set_num_threads` on TorchSharp. Managed null = min(current count, `ManagedThreads.Default`), the pool's initial
    count: `CpuCores.DefaultThreads` = the physical cores if ≥ 4, else `Environment.ProcessorCount`, never more than
    ProcessorCount (owner's decision, 2026-10-12; docs/performance/round-13-physical-cores.md). `CpuCores.Physical()` (Nn/Managed;
    the benchmark's core count too) never throws, null → logical: Linux distinct sysfs `thread_siblings_list`, Windows
    GetLogicalProcessorInformationEx records, macOS sysctlbyname `hw.physicalcpu`; no process spawned. A caller that
    sets `ManagedThreads.Count` and then loads with null gets it capped at the default, so the benchmark sample passes
    `Threads` explicitly. The rest of this bullet is TorchSharp's. Null = min(libtorch's current
    value, `Environment.ProcessorCount`), so a container's CPU quota is respected and a lower count the caller set is
    kept. Inter-op threads are untouched (eager mode never uses that pool; libtorch allows setting it only once).
    libtorch's OpenMP gives each calling thread its own team of `Threads` threads (measured: 8 callers × 8 threads
    = 83 OS threads), and a thread that already ran an op keeps its old count after a later Load.
    Measured (benchmark, default, 6,566 words): 1 thread 57.8 s, 2: 33.2 s, 8: 19.1 s, 16: 19.7 s. One loaded pipeline
    ≈ 950 MB private (default), 700 MB (default_fast). VerifyChecksums ≈ +1.3 s / +0.8 s load.
  - `CancellationToken` overloads of all three `Process` (separate overloads, binary compatible). Checked in
    `Pipeline.Step` before each processor and at the head of each batch loop (tokenizer batches and windows, pos,
    lemma, constituency per step, depparse, sentiment, ner; mwt is one batch). Depparse also checks inside a batch:
    after each charlm pass, between `HighwayLstm` layers (its optional token), before each `DeepBiaffine` chunk and
    per Chu-Liu/Edmonds decode; the longest gap is one charlm pass or LSTM layer (~1/8 of a batch). Pos does the same
    (after each charlm pass or the char model, between `HighwayLstm` layers, between the heads), and so does sentiment
    (after each charlm pass, after the LSTM, between the convolutions). Worst measured latency ≈ 0.6 s on the desktop;
    `ConcurrencyTests.MaxLatency` is 5 s for CI. Every check sits outside the inner
    dispose scopes or inside a try/finally that disposes, so nothing leaks (`ConcurrencyTests`).
  - Thread safety: Process keeps all mutable state per call (the Document, `CharlmCache`, the lemmatizer's
    DeltaVocab copy, the parser states); the models are read-only after Load; TorchSharp's dispose scopes are
    thread-static and libtorch's grad mode thread-local. The stress test (8 threads, mixed modes, both packages)
    needed no code change. Keep it that way: never add mutable fields to a processor (the one exception: the batch sizes,
    set only on `WithBatchSize`'s fresh `MemberwiseClone` inside Load, before the pipeline is returned).
  - TorchSharp counts the two undefined index tensors of `pack_padded_sequence(enforce_sorted: true)` (lemmatizer)
    as live forever (`DisposeScopeManager.Statistics`); they hold no memory. Leak tests leave lemma out.
  - `SplitSentences = false` = `tokenize_no_ssplit`: `Tokenizer.Decode` ends sentences only at paragraph ends.
    `tests/golden/no_ssplit/` (`make_golden.py --no-ssplit-only`). Pretokenized input ignores it, as in Stanza.
  - `MaxConcurrentCalls` (int?, owner's decision 2026-10-09: opt-in, null = no limit): a `SemaphoreSlim` in `Pipeline`
    around each Process overload's work (`Gated`); the wait honours the call's token. Managed throughput peaks at ~2 calls
    in flight (cache eviction of the charlm weights; docs/performance/round-07-per-call-short-texts.md).
  - `Processor` constants; `VerifyChecksums` (MD5 of each `.pt` Load reads, `ModelDownloader.Verify`, and the sha256 of each
    transformer file, 2026-10-18; throws for converted models, which have no published MD5); `Logger` (`Microsoft.Extensions.Logging.Abstractions` 10.0.0,
    since 1.0 the StanzaSharp package's only dependency): load times at Information, per-processor times at Debug, nothing measured
    without a logger.

Tokenizer notes:
- `Tokenizer.Predict` batches like Stanza: sort paragraphs by length, batch by 32, pad to max+1,
  and use 1000-char windows. Both LSTMs are packed at the length of each row's raw units. In
  the normal case that is the row's own length + 1, since collate appends one `<PAD>`. In the
  first long-paragraph window, the length is cut to the window. After `advance_old_batch`, every
  row runs at the full window width, so padding then reaches the backward direction.
  - Those later windows run only the unfinished rows (`Tokenizer.LiveRows`, round 15: 2.4x on the benchmark text), plus
    finished (all-padding) rows placed before the last live row until `Gemm.Run`'s single-row kernel (row m − 1 when
    m % 6 is 1) falls on the same row as with the whole batch, so every used logit keeps its bits on every path.
- `FixLabels` runs the e-mail/URL regex (`MaskRe`) only on paragraphs containing one of `MaskLiterals` (`@`, `://`,
  `www.`, `.gov` ... read off the pattern's alternatives; round 16). If the pattern changes, re-derive them; the
  `MaskLiterals_*` tests check every alternative and a fuzz.
- Regexes Stanza runs over a paragraph's code points (`STRUCTURAL_FEATURES`, `MASK_RE`) run on `Paragraph.CodePoints`,
  one char per unit (a non-BMP digit → `٠`, letter/number → `ª`, else U+FFFD), so match indices are unit indices; on
  the UTF-16 text a surrogate pair broke `\b` and counted twice in `{2,}` (found by `tools/text_fuzz.py`).
- Model units are code points, like Python. Offsets are UTF-16 indices, so they differ from
  Stanza's only after non-BMP characters.
- MWT-flagged tokens (`Token.IsMwtCandidate`) carry a single word until the MWT stage runs.
- Two lengths from the config: windows are `max(1000, max_seqlen)` units (output_predictions), and tokens longer
  than `max_seqlen` (200 for English; default 1000) in code points become `<UNK>` (TokenizeProcessor.process: text
  and word text only, offsets and MWT flag kept; bulk too, never pretokenized). `tests/golden/long_token*`
  (`make_golden.py --long-token-only`), both packages.

## Build order

1. Generate golden data with Python, and inspect every checkpoint's config (start with constituency;
   its config decides which optional layers the parser needs).
2. `Core`: document model, safetensors reader, checkpoint JSON loading.
3. Tokenizer (character-level labeling + post-processing into tokens and sentences).
4. MWT: dictionary lookup first, then the character split classifier.
5. Charlm + pretrain in `Nn`; validate against saved hidden states.
6. POS tagger.
7. Constituency parser: in-order transition system, parser state, Tree type, plus the neural scoring.
