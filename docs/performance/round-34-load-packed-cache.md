# Results, round 34: where `Pipeline.Load` goes, and a packed-weight cache

Part of [performance.md](../performance.md)'s measurement history.

The question (owner, 2026-10-16): would caching the managed backend's packed weights on disk cut `Pipeline.Load` time
meaningfully? Measure first.

## Result

Not by enough to be worth a cache, in my view; the decision is the owner's (below).

- **Packing is a fifth of the load's thread time.** Packing (`PackedMatrix`) plus the load-time preparation that a cache
  would also skip (the tagger's `BiaffinePerUpos`, depparse's bilinear transpose, the charlm input tables) take 20-26%
  of the summed thread time of a warm English `default` load from `.pt` files. The rest is parsing the checkpoints, the
  vocabularies, reading the tensors, the JIT, a GC after each model and model setup. None of that would change.
- **A cache cuts a warm load by 13-17% when it reads the cached arrays** (100-130 ms on a 0.7-0.9 s load, four
  runners; 14-24% for `default_fast`). It cuts it by 22-25% when it maps the cache file instead. Mapping only moves the work, though: the page
  faults move into the first `Process` call. Load plus the first one-sentence call is then 9-18% faster, and 8-11%
  faster when reading. German `default` gains 7% reading, 15% mapped.
- **The cost:** 418 MB of cache files for English `default` (252 MB `default_fast`, 400 MB German `default`). That is
  two-thirds the size of the models again. A cache miss makes the load 22-180% slower, since that load writes the
  files. We would also need a file format, invalidation and failure handling to maintain.
- **Faster packing code doesn't help.** A pointer loop without bounds checks, parallel packing per panel and parallel
  `BiaffinePerUpos` were all within ±3% of main on the load. Packing writes 391 MB of freshly allocated memory, and
  that, not the loop, is what it costs. A copy into fresh memory costs as much as the strided pack. So there is no
  packing code change to ship.
- **Two load costs bigger than packing came up:**
  - German's lemma dictionary takes about 0.77 s of the German load's thread time on its own (1.0-1.2 s locally):
    524,460 entries through `JsonSerializer`.
  - The two 250k-word vocabularies (pretrain and NER) are still the biggest parsing item.

  Both are listed below under other findings.

Bits: the CoNLL-U was identical under every setting, on every runner, warm and cold. The packed arrays are the same
on every kernel path except the charlm input tables (details below).

## Method

The prototype is on branch [`perf/load-packed-cache`](https://github.com/bisforboman/StanzaSharp/tree/perf/load-packed-cache)
(not for merge):
- **Instrumentation:** `LoadProfile` adds up thread time per stage: `Checkpoint.Load` (parse, which includes reading
  the pickle or JSON), `Checkpoint.UnitToId`/`Strings` (vocab), the lemmatizer's dictionaries, tensor reads,
  `PackedMatrix`, prep, the GC after each model, and cache reads and writes. Each model also logs its own split.
- **Harness:** `StanzaSharp.Benchmark startup` gained `--package` and `--lang`. `tools/load_profile.py` runs one fresh
  process per setting and config. The settings alternate, their order rotates each round, and medians are reported.
- **Runs:** Benchmark workflow runs [38874428712](https://github.com/bisforboman/StanzaSharp/actions/runs/38874428712)
  and [38875756126](https://github.com/bisforboman/StanzaSharp/actions/runs/38875756126) (the tables below), job
  `load-profile`, on the five free runners: 5 processes per setting warm, 3 cold.
- **Configs:** English `default` and `default_fast` from `.pt` files everywhere. ubuntu-26.04 also runs converted
  models and German `default`, both forms.
- **Cold:** on Linux the OS file cache was dropped before each process (`drop_caches`); on macOS with `purge`.
  Windows has no cold runs.
- **Defaults:** default threads and load lanes 2 (`Pipeline.LoadLanes`).
- **Machine:** the local machine (Ryzen 5800X) was fully loaded by other work during this round, so it only served
  for the bit checks.

Thread times are summed over the two load lanes. The load's wall time is roughly half of the summed thread time.

## Where the load goes

Warm, `.pt` files, main's code ("main loop"): summed thread ms, with each stage's share of the model thread time.

| runner | config | load ms | load CPU | model thread | parse | vocab | lemma dict | tensor read | pack | prep | GC after each | other (JIT, setup) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ubuntu-26.04 | en default | 890 | 1,641 | 1,672 | 415 (25%) | 49 | 79 | 240 (14%) | 305 (18%) | 135 (8%) | 152 (9%) | 297 (18%) |
| ubuntu-26.04 | en default_fast | 797 | 1,357 | 1,410 | 410 (29%) | 49 | 66 | 176 | 199 (14%) | 143 (10%) | 130 | 237 |
| ubuntu-26.04 | de default | 1,505 | 2,652 | 2,880 | 667 (23%) | 77 | **767 (27%)** | 430 | 272 (9%) | 131 | 271 | 265 |
| ubuntu-26.04-arm | en default | 708 | 1,294 | 1,348 | 338 (25%) | 51 | 57 | 160 | 221 (16%) | 108 (8%) | 159 | 254 |
| windows-2025 | en default | 818 | 1,500 | 1,532 | 420 (27%) | 66 | 85 | 169 | 209 (14%) | 116 (8%) | 140 | 327 |
| windows-11-arm | en default | 770 | 1,328 | 1,426 | 426 (30%) | 39 | 58 | 179 | 169 (12%) | 115 (8%) | 131 | 309 |
| macos-26 | en default | 1,248 | 1,903 | 2,357 | 509 | 51 | 107 | 649 | 511 | 195 | 193 | 142 |

macos-26 varied too much between processes to read shares from (main's median was 1,248 ms, while one setting that
changes nothing on English came in at 799). The JIT compiled 200-290 ms of methods during the load, all threads, and
most of it falls inside the stages.

Per model (ubuntu-26.04, English `default`, one process, thread ms):

| model | total | JIT | parse | vocab | tensor read | pack | prep | GC | not in a stage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pretrain/conll17 | 317 | 52 | 201 | 30 | 56 | - | - | 26 | |
| ner/ontonotes-ww-multi_charlm | 233 | 5 | 119 | 7 | 35 | 41 | - | 22 | |
| depparse/combined_charlm | 385 | 16 | 10 | 1 | 82 | 143 | 78 | 13 | |
| constituency/ptb3-revised_charlm | 227 | 32 | 32 | 1 | 49 | 59 | - | 9 | |
| forward_charlm/1billion | 202 | 58 | 63 | 3 | 16 | 23 | - | 26 | its input table (a GEMM) |
| pos/combined_charlm | 174 | 21 | 5 | 1 | 20 | 21 | 75 | 14 | |
| lemma/combined_nocharlm | 133 | 18 | 16 | 0 | 1 | 1 | - | 5 | the dictionaries, 102 |
| sentiment/sstplus_charlm | 102 | 14 | 5 | 0 | 28 | 34 | - | 10 | |

JIT time overlaps the stages (code compiled inside them).

Observations:
- **The critical path is unchanged since round 17:** the pretrain (its 250k-word vocabulary), then its readers. The
  pretrain and NER parses (two 250k-word vocabularies) take 320 ms of thread time. Packing is spread over depparse,
  constituency, sentiment and the charlms, which run beside other models, so removing it saves less wall time than
  its thread share.
- **Packing is bound by fresh memory.** A load packs 391 MB (English `default`), all into newly allocated pinned
  arrays. In a micro-benchmark a plain `CopyTo` of 392 MB into fresh 4 MB arrays took as long as the strided pack. Into
  memory already touched, it took two-thirds of that. The page faults and zeroing of the destination dominate. That
  is why the loop variants below change nothing.
- **German's lemma dictionary** (`dicts`: 2.2 MB gzip, 14.7 MB of JSON, 524,460 entries) takes about 0.77 s of
  thread time in `JsonSerializer.Deserialize` (1.0-1.2 s locally). That is 27% of the German load's thread time, in one model that starts
  in the first lane. English's dictionary is 37,130 entries, about 80 ms.
- **Converted models are no faster than `.pt` any more.** In run 1 they loaded in the same time (590 ms on
  ubuntu-26.04). In run 2 they took longer, with parse times that jumped between about 500 and 1,000 ms from process
  to process. Since round 21, `.pt` vocabularies skip the JSON round trip. The converted ones still build a `JsonNode`
  tree and then re-serialize `_unit2id` (vocab: 160-350 ms against 25-50 ms from `.pt`). CLAUDE.md's "converting saves
  about a second of load time" dates from before round 21.

## The prototype

`Nn/Managed/PackCache.cs` on the branch. `STANZASHARP_PACK_CACHE=DIR` turns it on, and
`STANZASHARP_PACK_CACHE_MODE=map` maps the file.

- **What's cached:** each model loads on one thread and packs in a fixed order. So the cache is the list of arrays
  its load produced, in that order: every `PackedMatrix`, plus three "units" that a hit skips entirely:
  - the tagger's `BiaffinePerUpos` (weights and bias);
  - depparse's bilinear transpose;
  - each charlm's input table. This one is a GEMM, so it is the only path-dependent entry; see the header.
- **What's skipped:** on a miss, every checkpoint tensor read is tracked with a weak reference. After the load and its
  GC, tensors of 256 KB or more that died fed only packing. Their keys go into the file, and a hit doesn't read them
  (their arrays stay uninitialized, which the bit checks would expose if anything read them). Unit outputs that only
  fed a `PackedMatrix` aren't stored either.
- **Header:** format version, the source file's size and last-write time, `PackedMatrix.NR`, and the entries' lengths.
  The file name carries `Gemm.Path` (see below). Data is 64-byte aligned.
- **Reading:** "cache read" reads each entry into a new pinned array (`RandomAccess.Read`). "cache map" maps the file
  read-only and points each `PackedMatrix` into the view (a new `float*` field next to the array), so nothing is
  copied at load.
- **Kernel paths:** cache files were written on Vector256, without AVX-512, on Vector128 (`DOTNET_EnableAVX2=0`) and on
  Scalar (`DOTNET_EnableHWIntrinsic=0`). They were byte-identical for every model except the two charlms. Their input
  table is a GEMM over the embedding, and Vector256 and Vector128/Scalar give it different bits. Everything else is a
  copy, a transpose or a sum in double, the same on every path. A real cache would key the charlm entry by the path,
  or recompute that table: a GEMM over a few hundred characters, a few ms.

### Load with the cache

Warm, run 38875756126. Each cell is the change in load time against main, with the change in load plus the first
one-sentence call in brackets, then the peak working set after the load (MB). "miss" is the process that writes the
cache.

| runner | config | main: load (+ call 1), peak | cache read | cache map | map, no GC after each | miss (writes) |
|---|---|---:|---:|---:|---:|---:|
| ubuntu-26.04 | en default pt | 890 (1,158), 818 MB | -13% (-10%), 853 | -24% (-18%), 478 | -29% (-21%), 573 | +62% (+58%) |
| ubuntu-26.04 | en fast pt | 797 (1,024), 671 MB | -24% (-18%), 556 | -27% (-20%), 368 | -35% (-25%), 434 | +22% (+33%) |
| ubuntu-26.04 | de default pt | 1,505 (1,802), 1,212 MB | -7% (-6%), 1,168 | -15% (-12%), 808 | -19% (-15%), 904 | +31% (+33%) |
| ubuntu-26.04-arm | en default pt | 708 (982), 902 MB | -14% (-10%), 850 | -25% (-18%), 528 | -36% (-25%), 577 | +24% (+27%) |
| ubuntu-26.04-arm | en fast pt | 589 (822), 753 MB | -14% (-9%), 588 | -19% (-12%), 402 | -33% (-18%), 471 | +30% (+35%) |
| windows-2025 | en default pt | 818 (1,120), 855 MB | -13% (-8%), 823 | -22% (-10%), 528 | -24% (-11%), 680 | +104% (+95%) |
| windows-2025 | en fast pt | 700 (948), 658 MB | -16% (-9%), 568 | -21% (-8%), 364 | -30% (-13%), 452 | +70% (+75%) |
| windows-11-arm | en default pt | 770 (1,060), 859 MB | -17% (-11%), 824 | -23% (-9%), 522 | -27% (+1%), 698 | +181% (+185%) |
| windows-11-arm | en fast pt | 692 (934), 654 MB | -18% (-13%), 547 | -16% (-2%), 384 | -19% (-4%), 511 | +137% (+122%) |

Run 38874428712 (another CPU draw, settings not yet rotated) gave the same picture. On ubuntu-26.04 English
`default` went from 590 to 476 ms reading (-19%) and to 398 ms mapped (-33%).

Reading the table:
- **Mapping's lower peak is not a saving.** It is measured right after the load, before the mapped pages are touched,
  and the first call then pays their faults (+5 to +90 ms). Mapped pages are file-backed and shared, so the OS can drop
  and re-read them under memory pressure. Several processes on one machine would share one copy. That is the mapped
  variant's real advantage.
- **"No GC after each"** on a hit saves another 15-80 ms and raises the peak by 50-180 MB. That is the same trade
  round 17 made, and it is independent of the cache.
- **Converted models and macOS:** they gained more (-30 to -40%), but their main-loop medians were the noisy ones
  above, so those percentages are not reliable.

**Cold** (Linux, the file cache dropped before each process): the cache read cut English `default` by 40-41%
(ubuntu-26.04: 1,360 → 814 ms) and `default_fast` by 22-23%. I don't trust that as a property of the cache:
- A hit reads about as many bytes as main: 418 MB of cache plus about 140 MB of tensors it still reads (the pretrain
  and embeddings), against 542 MB of tensors.
- A control setting that read each checkpoint front to back on a background thread ("prefetch", below) gained nothing
  cold.
- The cache files had been written minutes before, so the runner's virtual disk likely still held them in its host
  cache. The `.pt` files had been restored long before, and gigabytes were written after them.

On a real machine whose cache and models both sit on the disk, a cold hit should read about what a cold load reads.

## The simpler alternatives

Same runs, warm, change in load time against main (load plus call 1 in brackets):

| runner | config | faster loop | parallel pack + prep | prefetch | dictionary reader |
|---|---|---:|---:|---:|---:|
| ubuntu-26.04 | en default pt | -1% (-0%) | -0% (+0%) | +3% (+2%) | -2% (-1%) |
| ubuntu-26.04 | de default pt | +1% (+1%) | -3% (-3%) | +6% (+5%) | -6% (-5%) |
| ubuntu-26.04-arm | en default pt | +0% (-0%) | -3% (-2%) | +1% (+2%) | -0% (-1%) |
| windows-2025 | en default pt | -2% (-1%) | -1% (+0%) | +4% (+2%) | -0% (-0%) |
| windows-11-arm | en default pt | +1% (+1%) | -7% (-5%) | +4% (+8%) | -0% (+1%) |

- **Faster loop:** `PackedMatrix` packs through pointers, panel by panel, without bounds checks, into
  `GC.AllocateUninitializedArray` (padding lanes cleared explicitly). Pack thread time fell 12-18%, but the load
  didn't change.
- **Parallel:** panels packed with `Parallel.For` above 1M floats, and `BiaffinePerUpos` per UPOS in parallel. Both are
  exact, since each panel or UPOS row is written by one thread. Pack thread time fell 20-30% (fresh-memory faults
  partly overlap). The load moved 0-7%, within the noise.
- **Prefetch:** each checkpoint read front to back on the thread pool while it is parsed (`FileOptions.SequentialScan`,
  4 MB reads). This was the control for the cold numbers; it gained nothing, warm or cold.
- **Dictionary reader:** the lemmatizer's dictionaries parsed with a `Utf8JsonReader` loop instead of `JsonSerializer`
  (same dictionaries, checked entry by entry). German's dictionary time fell from 767 to 554 ms of thread time and the
  German load by 6%. Most of the cost is building 1M strings and 31 dictionaries, not the parser.

None of these is worth a code change on its own.

## Recommendation

**Don't add a cache now.**
- At best it saves 0.1-0.2 s of a 0.7-0.9 s load. Mapped, a third of that comes back in the first call.
- It costs 250-420 MB of disk per package, an extra slow first load, and a file format and invalidation scheme to
  maintain.
- The rest of the load (parsing, vocabularies, the JIT, the GCs) stays as it is.

For load time, these are better targets than packing, none of which needs a cache:
1. **The German lemma dictionary** (about 0.77 s of thread time). Options:
   - share strings between word and lemma: most lemmas are the word itself or a prefix of it;
   - load it lazily on the first German lemma call;
   - store it in a cheaper form.
   An idea file (`ideas/german-lemma-dictionary.md`) tracks it.
2. **The converted JSON path** is now slower than `.pt` (`ideas/converted-vocab-load.md`). Read `_unit2id` with round 21's approach: a `JsonValue`
   holding a ready dictionary instead of a `JsonObject` that gets re-serialized. Or stop recommending conversion for
   speed.
3. **The GC after each model** (130-270 ms of thread time, 9% of the load; `ideas/gc-after-each-model.md`). It exists because each model reads its
   weights into arrays that it then packs. A model that read straight into its packed layout, or into rented
   buffers, would leave little garbage. The GCs could then go, and the 100-200 MB of peak with them.

If the owner wants the cache anyway (for short-lived processes, where 0.2 s matters), I'd build the mapped variant,
opt-in.

**Owner's decision (2026-10-16): no cache.** The three cache-free ideas stay as idea files.

## What a cache would need decided

- **Opt-in or default.** I'd make it opt-in: a default would write 250-420 MB next to the user's models or into their
  profile without asking, and a model directory can be read-only (containers, shared mounts).
- **Location.**
  - A directory the caller names. This is the simplest, and it fits containers that bake the cache into the image.
  - Next to the model files (`<model>.packed`), if writable.
  - A per-user cache directory (`%LOCALAPPDATA%`, `$XDG_CACHE_HOME`, `~/Library/Caches`).
- **Invalidation.** The prototype checks the source file's size and last-write time, plus a format version and the
  panel width.
  - Checking an MD5 of the source, as `VerifyChecksums` does, would cost about 1 s per load: it reads all 630 MB.
  - A real implementation should also key by the library version, since packing or prep could change between versions
    without changing any file. It should also key by `Gemm.Path`, for the charlm tables, or drop those from the cache.
- **Failure handling.** Treat a corrupt, truncated or foreign file as a miss and rewrite it. The failure contract of
  `Checkpoint.Load` applies to the cache file too. Write to a temporary name and rename, so concurrent writers race
  harmlessly; the prototype does this. Also decide whether a failed write (read-only directory, full disk) logs or
  throws. I'd log.
- **Disk size:** 418 MB for English `default`, 252 MB for `default_fast`, 400 MB for German `default`.
- **API:** a proposal, not added. `PipelineOptions.PackedCacheDirectory` (`string?`, null = off). With
  `PipelineOptions.Logger`, hits and misses are logged at Information. The downloader would ignore it, as it ignores
  the other load options. A `stanzasharp warm-cache DIR` tool command could write the files ahead of time, for
  example in a Dockerfile.
- **Mapped or read.** Mapped saves more and shares pages between processes, but `PackedMatrix` then holds a pointer
  into a view that must outlive the pipeline: the view's lifetime moves to `Pipeline.Dispose`. Read keeps the managed
  arrays and today's lifetime.

## Exactness

- **CoNLL-U:** identical within every config under every setting, on all five runners, warm and cold. That covers the
  cache miss, both hit modes, no GC, every packing variant, prefetch and the dictionary reader.
- **Locally, all golden texts** (`startup --hash`, each processed alone) under main's loop, the faster loop, parallel,
  a miss and all three hit modes: `4FD0F4EE…`, round 17's hash.
- **Dictionaries:** the dictionary reader's output equals `JsonSerializer`'s, entry by entry, for German's 524,460
  entries.
- **Packed arrays:** identical across kernel paths except the charlm tables (above).

Nothing from the prototype is merged.
