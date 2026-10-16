# Changelog

All notable changes to the `StanzaSharp` NuGet package. Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, minor versions may change the public API.

Output is verified against Python Stanza 1.15.0 and its English models (`ModelDownloader.StanzaVersion`).

Unreleased changes are one file each under [`changes/`](changes/README.md), so pull requests don't conflict here;
a release folds them in (`python tools/changelog.py release X.Y.Z`).

## [1.10.0] - 2026-10-16

1.10: ten more NER models and several character-model pairs in one pipeline, both through `PipelineOptions.Models`,
and `download --list`. Additive; no behaviour change.

### Added
- Ten more NER models for `PipelineOptions.Models` (`--model ner=NAME`): `conll03_charlm`, the clinical `radiology`,
  and the biomedical `anatem`, `bc4chemd`, `bc5cdr`, `bionlp13cg`, `jnlpba`, `linnaeus`, `ncbi_disease` and `s800`
  (cased word vectors, PubMed character models), byte-identical to Python Stanza on both backends.
- `stanzasharp download --list`: prints each file the download would fetch, with its MD5, without downloading.
- A pipeline can read several pairs of character language models: `PipelineOptions.Models` may choose a model whose
  pair differs from the package's (e.g. the default package with ner `i2b2`, which reads mimic's), which no longer
  throws; the pipeline loads each pair once (mimic's two charlm files are 19 MB each), and the tagger's
  cached outputs are reused only by models that read the same pair. Byte-identical to Python Stanza on both backends.

## [1.9.0] - 2026-10-16

1.9: Stanza's English treebank and biomedical packages, one model per processor (`PipelineOptions.Models`), Native
AOT and trimming on the managed backend, and fixes found by differential text fuzzing. The new API is additive. One
behaviour change: `tokenize` now brings the package's `mwt` along, as in Stanza, so a model folder downloaded on 1.x
for a list like `tokenize,ner` needs a re-download (see Changed).

### Added
- Stanza's English biomedical packages `craft`, `genia` and `mimic` (tokenize, pos, lemma and depparse, with their own
  word vectors), including the dependency parser's repair of heads with two subjects or objects that their checkpoints
  turn on, byte-identical to Python Stanza on both backends.
- Stanza's English treebank packages: `PipelineOptions.Package` (and `download --package`) accepts `combined`, `ewt`,
  `gum`, `lines`, `partut`, `atis`, `childes` and `eslspok` (tokenize, mwt, pos, lemma and depparse; `eslspok` has no
  mwt and Stanza's identity lemmatizer), byte-identical to Python Stanza on both backends.
- Native AOT and trimming on the managed backend: the StanzaSharp assemblies are `IsAotCompatible` and publish with
  `PublishAot` or `PublishTrimmed` without a warning, with byte-identical output. The lemmatizer read its dictionary
  through reflection-based System.Text.Json, so such apps failed in `Pipeline.Load`; it now uses a source-generated
  context. Publish AOT apps with `<IlcInstructionSet>x86-64-v3</IlcInstructionSet>`: the default x64 target runs the
  128-bit kernels, about 2x slower. Not `StanzaSharp.Cuda` (TorchSharp isn't AOT-safe).
- `PipelineOptions.Models`: one model per processor instead of the package's, like Python Stanza's
  `processors={'ner': 'i2b2'}` (a model's or a package's name per processor; the keys' processors are added to the
  package's), honoured by `ModelDownloader.DownloadAsync(dir, options)` and `--model PROCESSOR=NAME` in the tool's
  download, compare and benchmark. New models: English constituency `ptb3_charlm`, English ner `ontonotes_charlm`,
  `ontonotes_nocharlm`, `combined_charlm` and `i2b2` (clinical, with the `mimic` package), German ner `conll03`, all
  byte-identical to Python Stanza on both backends.

### Changed
- `tokenize` now loads and runs the package's `mwt` too, as Stanza does (`add_mwt`): `Processors = "tokenize,ner"`
  (or `"tokenize,sentiment"`, `"tokenize"`) expands multi-word tokens into words like Stanza's output, and
  `ModelDownloader` and `stanzasharp compare` select the files Stanza loads. A model folder downloaded on 1.x for such a
  list lacks the mwt model: `Pipeline.Load` then throws a `FileNotFoundException` naming the file; re-run
  `stanzasharp download` (or `ModelDownloader.DownloadAsync(dir, options)`), which fetches only the missing files.

### Fixed
- `stanzasharp compare`: a text without sentences (empty or whitespace only) was reported as different, since Stanza's
  side wrote a lone newline where StanzaSharp's CoNLL-U is empty.
- `Pipeline.Load` with a model file missing from the folder threw a bare `FileNotFoundException` from whichever model
  read it first; it now names the file and the `stanzasharp download` command (or `ModelDownloader.DownloadAsync`)
  that fetches the missing files.
- MWT seq2seq models (German, English `partut`) are decoded with a beam search of 5, as Stanza does (its MWT trainer
  never passes its `beam_size` of 1 through), instead of greedily, which could expand a token differently.
- Tokenizer: a character outside the BMP could change tokens, since the regexes Stanza runs over code points ran over
  UTF-16 text: a letter such as `𠀁` before `C:` was no word character for the `labeled_field` feature (`𠀁C` became one
  token), and an emoji counted as two characters for a URL's tail (`example.com.😀` was kept as one token). Found by
  the new differential text fuzzer (`tools/text_fuzz.py`, docs/text-fuzzing.md).

## [1.8.0] - 2026-10-15

1.8: performance only (rounds 27–32, docs/performance/). Sentiment about 22% faster, depparse and constituency a few
percent, and the managed thread pool holds up when other processes keep the cores busy. No API change; output is bit
for bit the same as 1.7.

### Changed
- Managed backend: the constituency parser's stack LSTMs run their cell updates and row copies inside the GEMM's
  parallel tasks instead of on the calling thread, so constituency is 2 to 5% faster at 4 threads on the free x64 and
  Arm runners (more on macOS). Output is bit-for-bit unchanged.
- Managed backend: the dependency parser scores the padding columns of a batch's arc matrix once per row instead of
  once per column, and sums its arc dot products with SIMD. Output, arc log-probs and label scores are bit for bit the
  same.
- Managed backend: when other processes keep the cores busy, the thread pool puts some workers to sleep instead of
  letting every parallel step wait for a preempted one, and wakes them again once the stalls stop. On a loaded machine
  a call took less wall time and about a third of the CPU time. Concurrent calls share the pool as before, and results
  are unchanged (bitwise).
- Managed backend: sentiment computes its biLSTM's first-layer input projection once for all padding rows of a batch
  instead of once per row, and runs its strided (5, 5) filter with SIMD across the 8 channels. Results are bit for bit
  the same; sentiment is faster on short and mid-length documents (docs/performance/round-28-sentiment-padding.md).

## [1.7.0] - 2026-10-14

1.7: JSON output (`Json.Write`/`Json.Read`, like Stanza's `print(doc)`), `Pipeline.Process(Document)` for
annotating a parsed CoNLL-U document, `stanzasharp compare` and `benchmark` take `--lang de`, so German can be checked against Python Stanza and timed
like English; and the TorchSharp backend (`StanzaSharp.Cuda`) parses faster with a lower depparse peak. Outputs are
unchanged bit for bit; the new API is additive.

### Added
- `stanzasharp compare` and `stanzasharp benchmark` take `--lang de` for German: Python Stanza runs its German pipeline
  with the German entry of Stanza 1.15.0's resources.json (shipped with the tool), StanzaSharp loads with
  `Language = "de"`, and `--models` defaults to `models/stanza/de`. `--package` takes German's packages too (`combined`,
  `gsd` and `hdt` run only their processors, tokenize to depparse). `benchmark --lang de` times a built-in German text
  (our German test texts, cut like the English one), and the block names the language. The manual Benchmark workflow
  has a `lang` input.
- `Json.Write(doc)` writes a document as Python Stanza's `str(doc)` does (`json.dumps(doc.to_dict(), indent=2,
  ensure_ascii=False)`), byte for byte, and `Json.Read(json)` reads it back (`Json.Write(Json.Read(json))` equals
  `json`). Like Stanza's JSON, it holds the tokens and words only, not the sentence text, id, tree or sentiment.
- `Pipeline.Process(Document)` (and a `CancellationToken` overload) runs the pipeline's processors on a document
  that already has its tokens, such as one from `Conllu.Read` or `Json.Read`, like Stanza's `process_conllu`:
  tokenize and mwt are skipped, every other processor overwrites what it sets, and the rest is kept.

### Changed
- TorchSharp backend (`StanzaSharp.Cuda`): the constituency parser is about 20% faster on the CPU (each step reads and
  writes its states with a constant number of tensor ops), and the dependency parser's memory peak is about 280 MB
  lower on a 27k-word text (no padded input, no full label-score tensor). Outputs and scores are bit-identical.
- TorchSharp backend (`StanzaSharp.Cuda`): the tagger's UPOS head (`upos_clf`) sums in double, as the managed backend
  does, so its logits come closer to Stanza's float64 values and no longer depend on libtorch's CPU kernels in the last
  layer (German `default_fast` on an Intel AVX-512 CPU: 1.2e-4 from Stanza before). Tags are unchanged.
- The `StanzaSharp` package depends on `Microsoft.Extensions.Logging.Abstractions` 10.0.13 (was 10.0.0).

### Fixed
- `stanzasharp --help` and the tool's README said German models could be downloaded but not loaded by `Pipeline.Load`;
  they load since 1.6.0.

## [1.6.0] - 2026-10-13

1.6: German. `PipelineOptions.Language = "de"` runs Stanza's German models byte-identical to Python Stanza 1.15.0,
on both backends. English is unchanged; the new option defaults to `"en"`, so existing code needs no change.

### Added
- German: `Pipeline.Load(dir, new PipelineOptions { Language = "de" })` runs Stanza's German models, all eight processors,
  byte-identical to Python Stanza 1.15.0 on both backends, with the packages `default`, `default_fast` and the treebank
  packages `combined`, `gsd` and `hdt` (tokenize to depparse; gsd's and hdt's taggers are Stanza's older checkpoint
  format, read as Stanza reads it). German MWT is a character seq2seq model (shared with the
  lemmatizer), and German NER (GermEval 2014: `PER`, `LOC`, `ORG`, `MISC`) reads fastText's German vectors, so a German
  pipeline loads two word-vector files. The samples' `StanzaSharp.Cli` takes `--lang de`.
- `PipelineOptions.Language` (Stanza's language code, default `"en"`). German (`"de"`) models can be downloaded:
  `ModelDownloader.DownloadAsync(dir, new PipelineOptions { Language = "de" })` or `stanzasharp download --lang de`
  (into `models/stanza/de` by default), with the packages `default`, `default_fast`, `combined`, `gsd` and `hdt`.
  An unknown language throws with the supported list, and so does `default_accurate`, whose transformer models
  StanzaSharp doesn't run. The `DownloadAsync(dir)` and `DownloadAsync(dir, processors)` overloads stay English.

## [1.5.0] - 2026-10-13

1.5: the dependency parser is 22–29% faster (the whole pipeline 6–9%), loading Stanza's `.pt` files is 35–48%
faster, and long texts allocate 28% less. Outputs are unchanged bit for bit; no API changes.

### Changed
- Managed backend: the dependency parser reuses the tagger's backward character-LM outputs for every sentence the
  charlm cache holds, not only for one-sentence documents, and runs only ROOT's two characters on from the tagger's
  final state. Output is bit for bit the same. The cache keeps about 4 KB more per sentence for it (not counted in
  `CharlmCacheOptions.MaxWords`).
- Long texts allocate less garbage (managed backend): the tokenizer keeps each paragraph as one string and builds the
  model's inputs per batch, constituency reuses its stack-state objects, dependency decoding no longer allocates an
  array per tree node, NER's emission scores are rented, and lowercasing an ASCII word allocates nothing when it has no
  capital. Outputs are bit-identical.
- `Pipeline.Load` from Stanza's own `.pt` files reads the large vocabularies straight from the unpickled checkpoint
  instead of mirroring them into JSON and parsing that back: 35–48% less load time on GitHub's free runners (e.g.
  1,993 → 1,046 ms on ubuntu-26.04), with a load peak mostly lower. Converted models load as before, and the output is
  unchanged, bit for bit.

### Fixed
- Converted `.json` checkpoints: invalid UTF-8, or an escaped lone surrogate, in a property name or string loaded
  without error and threw `InvalidOperationException` later, when a model read it; loading now throws
  `InvalidDataException`.

## [1.4.0] - 2026-10-13

1.4: faster startup for short-lived processes, faster AVX-512 LSTMs (now ahead of AVX2 on every CPU measured,
Intel included) and less garbage on long texts. Outputs are unchanged bit for bit; no API changes.

### Changed
- Managed backend: the character language models look up their character ids without a string per character, so a
  long `Process` call allocates about 17% less (201,313 words, all eight processors: 3.7 → 3.1 GB, 18 → 15 KB per
  word). The peak memory and the output are unchanged, bit for bit.
- Managed backend, startup: `Pipeline.Load` loads two models at a time (21–37% less time on GitHub's free runners, a
  load peak 25–89 MB higher), and the sigmoid/tanh exponential and the thread pool's per-step code are compiled
  optimized from the first call instead of starting at JIT tier 0. A fresh process that loads and makes six
  one-sentence calls finishes 18–29% sooner; the first call is 5–11% faster, the next few 17–52%. Warmed speed is
  unchanged, and so is the output, bit for bit. README has advice for short-lived processes (tiering, ReadyToRun).
- Managed backend: on AVX-512 CPUs each LSTM time step now computes two 16-column panels per kernel call, as the other
  matrix products already did. On GitHub's AVX-512 runners the full pipeline got 11–15% faster on an AMD EPYC 9V45
  (Zen 5), 5–9% on an Intel Xeon Platinum 8573C (Emerald Rapids) and 11–12% on a Xeon Platinum 8370C (Ice Lake), where
  the AVX-512 kernels had been 5–7% slower than the AVX2 ones and are now 4–6% faster. The output is unchanged, bit for
  bit.

## [1.3.0] - 2026-10-12

1.3: faster on modern CPUs (AVX-512 kernels, up to 53% on Zen 5) and in the tokenizer (now ahead of Python Stanza on
every tested runner), with outputs unchanged bit for bit and no API changes. `stanzasharp benchmark` now warms up on
the whole text, so its numbers aren't directly comparable with earlier versions.

### Added
- Managed backend: AVX-512 kernels (`Vector512`), used automatically on CPUs where .NET accelerates `Vector512`, with
  results identical bit for bit to the AVX2 kernels. On GitHub's AVX-512 runners the full pipeline ran 48–53% faster on
  an AMD EPYC 9V45 (Zen 5) and 3–10% faster on an EPYC 9V74 (Zen 4). `DOTNET_PreferredVectorBitWidth=256` turns them off.

### Changed
- `stanzasharp benchmark` warms up on the whole text, repeating until each side's stage times settle (2 to 5 runs; the
  block says how many), instead of one run on a tenth of the text. StanzaSharp's numbers now show a warmed service:
  on the same runners about 5% more words/s, with tokenize and lemma 30–45% faster; Python Stanza's moved about 1%.
  Results from earlier versions aren't directly comparable. A run takes about twice as long.
- Managed backend: each LSTM time step is split into four tasks per thread instead of one, so a thread that runs late
  holds less of the step. On GitHub's free runners the tokenizer got up to 29% faster (3-vCPU macOS) and the whole
  pipeline 2–4% (the constituency parser up to 10%, depparse up to 6%). One-sentence `tokenize,mwt` calls take ~0.1 ms
  longer. The output is unchanged, bit for bit.
- The tokenizer is about 2.4x faster on a text with a paragraph over 1,000 characters (the benchmark's 2,009 words:
  144 → 60 ms on 8 threads, 257 → 145 ms on 1): after such a paragraph's first window it no longer runs the other,
  finished paragraphs of its batch through the network as padding. The output is unchanged, bit for bit.
- The tokenizer skips Stanza's e-mail/URL pattern on paragraphs that contain none of the literals every match needs
  (`@`, `://`, `www.`, `.gov`, `.org`, `.edu`, `.net`, `.co`). The output is unchanged.

## [1.2.0] - 2026-10-12

1.2: the managed backend now defaults to the physical core count on machines with at least 4 cores (fewer stalls
on single sentences and concurrent calls), and `stanzasharp compare` accepts a models folder of any name. Outputs
are unchanged; no API changes.

### Changed
- `stanzasharp compare` and `benchmark`: `--models` accepts a folder of any name holding Stanza's `.pt` files; it no
  longer has to be named `en`.
- Managed backend: with `PipelineOptions.Threads` null the thread pool now uses the physical cores on machines with at
  least 4 of them (all logical processors before) and still all logical processors on smaller ones, never more than
  `Environment.ProcessorCount`. On an 8-core/16-thread desktop that cuts the slowest one-sentence calls (p90 16-36%
  lower) and gives two concurrent callers 11% more calls/s, while long texts run up to 9% slower. Output is unchanged.

## [1.1.0] - 2026-10-12

1.1: about 1 GB less peak memory with all eight processors, hardened model loading, faster concurrent calls, and two
new `stanzasharp` commands to check StanzaSharp against Python Stanza on your own text and machine. Outputs are
unchanged; no API changes.

### Added
- `stanzasharp benchmark [--package NAME] [--processors LIST] [--models DIR] [--threads N] [--words N] [--python PATH]
  [--quick] [--json FILE]` (`StanzaSharp.Tool`): times StanzaSharp on a built-in English text (load, each processor,
  words/s, one-sentence calls with median and p90, peak memory) and, if Python Stanza 1.15.0 is available, Stanza on
  the same text with the same models, runs alternating, and prints a Markdown block with the machine (CPU model,
  cores, RAM, OS, .NET, SIMD path), both sides and the ratio. A manual workflow (`benchmark.yml`) runs it on every free
  GitHub-hosted runner and combines the results into one table.
- `stanzasharp compare FILE [--package NAME] [--processors LIST] [--models DIR] [--python PATH]` (`StanzaSharp.Tool`):
  runs Python Stanza 1.15.0 and StanzaSharp on your own text with the same, MD5-checked model files and reports
  whether the CoNLL-U output is identical, or the first difference (sentence, both lines), with sentence and word
  counts and both sides' load and processing times. Offsets after characters outside the BMP are converted to code
  points (Stanza's) before comparing. Exit code 0 identical, 1 different, 2 usage or setup error.

### Changed
- Less memory on long texts (managed backend): the parsers and classifiers reuse their per-batch arrays instead of
  leaving them to the garbage collector (constituency recycles its parser states, the dependency parser decodes in
  rented buffers), so a call allocates about 18 KB per word instead of 54, and one call on 201,313 words with all eight
  processors peaks at about 1.9 GB instead of 2.2–2.3 GB. Outputs are unchanged.
- README: guidance for very long texts, including what splitting a text into parts changes (sentiment labels).
- Less memory with sentiment (managed backend): its LSTM and convolutions no longer hold whole-batch buffers for the
  padded rows, so one call on 6,761 words peaks at 1.3 GB instead of 2.3 GB with all eight processors (1.1 GB instead of
  2.0 GB with `default_fast`). Outputs are unchanged, bit for bit.
- Less memory on long texts: the returned `Document` takes about 19% less (482 instead of 594 bytes per word with all
  eight processors: tokens hold room for one word, `SpaceAfter` and equal feats strings are shared, tree children are
  arrays), and NER frees the charlm outputs the tagger kept for it (up to 256 MB) as it goes. One call on 670,000
  words peaked at 3.0 GB instead of 3.6 GB. Outputs are unchanged. `Tree.Children` of a parsed tree is now an array
  (still an `IReadOnlyList<Tree>`).

### Fixed
- Managed backend: concurrent `Process` calls with `Threads` near the number of logical processors (the default) no
  longer oversubscribe the CPU. The pool's workers stay asleep while callers and workers already fill every logical
  processor; before, 8 callers on 16 threads got 3-10 calls/s and single calls stalled for up to 20 s, now 13-17
  calls/s (docs/performance/round-12-concurrent-calls-thread-pool.md). Output is unchanged.
- `Pipeline.Load`'s error for a missing model directory now suggests `stanzasharp download` (the .NET tool) instead of the sample CLI.
- `.pt` loading: an integer pickled as LONG1 that fits in 64 bits (such as a storage of 2^31 elements or more) was kept
  as a big integer, so such a checkpoint failed to load.

### Security
- Loading a malformed or hostile model file now always fails with `InvalidDataException` (`NotSupportedException` for
  a big-endian `.pt`), in time and memory bounded by the file's size. Found by a new fuzz test of the `.pt` and
  `.json` + `.safetensors` loaders:
  - A pickle with many nested MARKs took quadratic time (each MARK copied the stack): 400 KB could hang the load.
  - Deeply nested or self-referencing lists threw `InvalidOperationException` or `JsonException` after recursing up
    to 1000 levels (a stack overflow on a small thread stack); a dict key of deeply nested tuples overflowed the stack
    when hashed, crashing the process.
  - An object referenced many times through the pickle memo was copied into the JSON each time (a 264 KB file asked for
    gigabytes).
  - Storage and tensor sizes could overflow: a storage of 2^62 floats passed the "fits in the file" check, a storage
    view at a huge offset passed its bounds check, and huge strides wrapped a tensor's extent negative. A stride-0
    tensor could claim terabytes, and a negative size next to a zero was accepted.
  - A zip `.pt` whose `data.pkl` header claimed 2 GB allocated it before reading; compressed (zip bomb) entries are
    now rejected, since torch.save stores them uncompressed.
  - safetensors headers were not validated: an unknown dtype, a shape that disagrees with the byte size, overlapping
    or missing data, or a malformed header threw `FormatException`, `KeyNotFoundException`, `JsonException` or
    `InvalidOperationException`, or loaded.

## [1.0.0] - 2026-10-10

1.0: the `StanzaSharp` package is fully managed, with no native or TorchSharp dependency; GPU and TorchSharp users add
`StanzaSharp.Cuda`. From here on the public API follows Semantic Versioning: breaking changes only in a new major version.

### Changed
- **The TorchSharp backend moved to its own package, `StanzaSharp.Cuda`.** `StanzaSharp` no longer depends on
  TorchSharp at all: the default install has no native code and no TorchSharp assemblies. For a GPU, or libtorch on the
  CPU, add `StanzaSharp.Cuda` and the native libtorch (`TorchSharp-cuda-windows`, `TorchSharp-cuda-linux` or
  `TorchSharp-cpu`):
  - `PipelineBackend.TorchSharp` → `CudaBackend.Cpu`.
  - `PipelineOptions { Device = torch.CUDA, DisableTf32 = true }` → `PipelineOptions { Backend = CudaBackend.Create(disableTf32: true) }`
    (`deviceIndex` picks another GPU).
  - `PipelineOptions.Device` and `PipelineOptions.DisableTf32` (obsolete since 0.5) are removed.
  - The `STANZA001` version check and `StanzaSharpTrimNative` come with `StanzaSharp.Cuda` now.
- Managed backend: each LSTM step now sums its recurrent matrix product in blocks of 128 terms, then adds the input
  projection, instead of one long chain. This halves the charlm's rounding error (now as close to float64 as TorchSharp)
  and the worst-case score gap between the backends on the golden parses (constituency margins: 7.3e-5 → 4.0e-5), at
  about 3–4% of the managed charlm's time. Outputs are unchanged.

## [0.6.0] - 2026-10-10

### Added
- `PipelineOptions.MaxConcurrentCalls` (opt-in; null, the default, means no limit): at most this many `Process` calls
  run on the pipeline at once, and further callers wait their turn (the wait honours the call's cancellation token).
  On the managed backend concurrent calls on short texts evict each other's model weights from the CPU cache, so with
  many callers 2 gives the most throughput (8 callers, one sentence each: 23.6 calls/s unlimited, 36.4 with one at a
  time). Output never depends on it.

### Changed
- Faster `Process` on a single sentence, the usual call of a service, on the managed backend (`default` package): the
  processors now share more of the character language models' work for a one-sentence text (the dependency parser
  continues the tagger's backward pass; NER reuses sentiment's on multi-word tokens). The output is the same bit for
  bit. One call on a 5–25-word sentence takes 27.9 ms instead of 30.0 (p90 44.6 instead of 50.7) on 8 threads, and
  83 ms instead of 94 (p90 141 instead of 170) on 1 thread, on a Ryzen 7 5800X; Python Stanza takes 189 and 319 ms.
  Details, and guidance on bulk processing and concurrent callers, in docs/performance/round-07-per-call-short-texts.md.
- The TorchSharp backend's code moved into its own assembly, `StanzaSharp.TorchSharp.dll`, which the `StanzaSharp`
  package carries next to the others. Nothing else changes; this prepares the opt-in `StanzaSharp.Cuda` package of 1.0.
- The managed backend is about 30% faster on Arm64 (Windows on Arm, Linux Arm64, Apple Silicon): its matrix kernel
  now multiplies by vector element instead of broadcasting each value first, reaching 86–89% of the NEON peak instead
  of 58–60% on a Neoverse N2. Results are bitwise unchanged.

## [0.5.1] - 2026-10-09

### Fixed
- Loading Stanza's `.pt` files from a slow filesystem, such as a Docker Desktop bind mount of a Windows folder, took
  minutes instead of seconds (issue #48), on both backends. The checkpoint reader asked the file for its length once
  per pickle opcode, millions of `fstat` calls for the pretrain's vocabulary, each about 0.3 ms on such a mount. It
  now reads the length once per pickle. Loading `tokenize,mwt,pos,constituency` from a bind mount on a busy machine
  took 13–26 minutes and now takes 8–9 s; from the container's own disk, 12 s became 6 s.

## [0.5.0] - 2026-10-09

### Added
- `PipelineOptions.Backend` and `PipelineBackend`: the implementation that runs the models.
  `PipelineBackend.Managed` (the new default) or `PipelineBackend.TorchSharp` (libtorch, as up to 0.4). Both give the
  same tags, lemmas, parses, entities and sentiment.
- Alpine (musl) and Linux on Arm64 run StanzaSharp now, on the managed backend, and CI checks both: the Docker sample
  on `runtime:10.0-alpine` reproduces the golden CoNLL-U byte for byte, and Linux Arm64 runs the managed tests and the
  package check.

### Changed
- **The models run on a managed backend by default**: C# kernels using the CPU's SIMD instructions (AVX2/FMA on x64,
  NEON on Arm64), with no libtorch or other native library. `dotnet add package StanzaSharp` is the whole install;
  `TorchSharp-cpu` is needed only for `PipelineBackend.TorchSharp`. The output is byte-identical to the TorchSharp
  backend and to Python Stanza on all the golden data (both packages). On a Ryzen 7 5800X with 8 threads all eight
  processors run about 2.2x faster (26,264 words: 25.9 s against 56.4 s); on 1 thread every processor is faster too,
  by 13–52%. Memory: one `Process` call on 8,000 words peaks at 2.5 GB instead of 3.8–4.3 GB; loading peaks at about
  850 MB (TorchSharp: 820 MB). Details in docs/backends.md.
- `Threads` sets the managed backend's thread pool (process-wide, shared by concurrent `Process` calls) or, on
  TorchSharp, libtorch's intra-op threads as before. A managed pipeline never calls into libtorch, so `Load` no longer
  changes libtorch's thread count unless TorchSharp is selected.
- The Docker sample references only `StanzaSharp`: no libtorch in the image, which now works on any .NET 10 runtime
  image, including chiseled and Alpine. The image is 209 MB on `runtime:10.0` (was 707 MB) and 103 MB on
  `runtime:10.0-noble-chiseled` (was 601 MB); 100 MB on `runtime:10.0-alpine`.
- Cancellation inside the POS tagger and the sentiment classifier is faster: the token is now also checked inside
  each batch (the tagger: after each charlm pass or the character model, between the LSTM layers and between the
  heads; sentiment: after each charlm pass, after the LSTM and between the convolutions), so a call stops within one
  such step instead of up to a whole batch (0.8 s for POS, 1.2 s for sentiment on an 8-core machine). Any call now
  stops within about half a second there. Output is unchanged.

### Deprecated
- `PipelineOptions.Device` and `PipelineOptions.DisableTf32` are `[Obsolete]`; GPU support moves to a separate
  `StanzaSharp.Cuda` package, and both options leave the main package in 1.0, with `PipelineBackend.TorchSharp`. They
  keep working: with `Backend` unset, setting either selects the TorchSharp backend, so existing GPU code runs as
  before (with a warning). Setting them together with `Backend = PipelineBackend.Managed` throws an
  `ArgumentException`.

### Removed
- The platform packages `StanzaSharp.Cpu.Linux`, `.Windows`, `.WindowsArm64` and `.MacOS` are no longer published
  (0.4.2 is their last version): the default backend needs no libtorch. For the TorchSharp backend, reference
  `TorchSharp-cpu`, or one platform's `libtorch-cpu-<rid>` package in version 2.10.0 (`libtorch-cpu-linux-x64`,
  `-win-x64`, `-win-arm64`, `-osx-arm64`), next to `StanzaSharp`, and set `Backend = PipelineBackend.TorchSharp`.
  `STANZA001` and `StanzaSharpTrimNative` still apply to those packages.

### Fixed
- Tokens longer than the tokenizer's `max_seqlen` (200 characters for English, read from the model's config) now
  become `<UNK>`, as in Stanza; before, only tokens over 1,000 characters did, so tokens of 201–1,000 characters
  (long URLs) kept their text and could get different tags, lemmas, parses and entities than in Stanza. Only the text
  changes (offsets stay), in bulk input too; pretokenized input keeps its tokens, as in Stanza.

## [0.4.2] - 2026-10-07

### Changed
- On Linux (glibc), a `Process` call of at least 1,000 words now returns the memory it freed to the OS
  (`malloc_trim`). glibc kept it for reuse, so a long-running process stayed near its peak: with
  `tokenize,mwt,pos,constituency` the RSS after a 6,761-word call falls from 930 to 650 MB (after loading: 590 MB;
  Python Stanza: 890 MB). The trim takes 25–40 ms after such a call; output is unchanged. Other platforms are
  unchanged; on Windows, the environment variable `MIMALLOC_PURGE_DELAY=0` has the same effect for about 20% more
  time (see the README).
- The POS tagger no longer pads its batches: its input and its highway LSTM layers work on the real words only, so
  one long sentence in a batch of 250 no longer makes every sentence as long. With `tokenize,mwt,pos,constituency`
  the peak on 5,000 words falls from 1.57 to 1.02 GB (15,000 words: 1.66 to 1.25 GB), and pos is about 20% faster
  on such text. Tags, lemmas and parses are unchanged; logits move by at most 5e-5 (docs/performance/round-04-highway-lstm-packed-rows.md).
- Cancellation inside the dependency parser is about 8x faster: the token is now also checked inside each 5,000-word
  batch (after each charlm pass, between the LSTM layers, between the scorers' chunks and between the sentences' tree
  decodes), so a call stops within about an eighth of a batch (one charlm pass or LSTM layer) instead of up to a
  whole batch (1.7 s on an 8-core machine, 6-7 s on CI runners). Output is unchanged.

## [0.4.1] - 2026-10-07

### Changed
- Loading Stanza's `.pt` files no longer reads each file whole and copies its tensors into a second buffer: tensors
  are read from the file as the models load, as for converted models. With `tokenize,mwt,pos,constituency` the load
  peak falls from 970 MB to 540 MB (1,090 to 510 MB with Server GC), and the memory no longer waits for a GC
  (issue #19).
- The POS tagger and the dependency parser free their padded batch tensors as soon as they are used. A batch padded
  to one long sentence peaked much higher: on 5,000 words of the validation texts the peak falls from 2.36 to
  1.57 GB (Python Stanza: 2.36 GB). Output is unchanged.
- The README describes what sets the memory peak and how to bound it (parts of about 1,000 words per call).

## [0.4.0] - 2026-10-06

### Added
- `PipelineOptions.Threads`: libtorch's intra-op thread count, set at `Pipeline.Load` (process-wide, like
  `torch.set_num_threads`). By default it is now capped at `Environment.ProcessorCount`, which .NET limits to a
  container's CPU quota; before, libtorch used one thread per physical core of the host.
- Cancellation: `Pipeline.Process(text, cancellationToken)`, and the same for bulk and pretokenized input. The token
  is checked between processors and between batches inside them; a canceled call throws `OperationCanceledException`,
  returns no document, and leaves the pipeline usable.
- `Pipeline.Process` is thread-safe: one pipeline can serve concurrent calls from many threads, with each call's output
  identical to a sequential call's (verified by a stress test). One loaded pipeline is about 950 MB of private memory
  (`default`) or 700 MB (`default_fast`), plus each call's working memory; see the README.
- `PipelineOptions.SplitSentences`: false gives Stanza's `tokenize_no_ssplit`, one sentence per paragraph, with output
  identical to Stanza's, also in bulk.
- `Processor` constants (`Processor.Tokenize`, ..., `Processor.Ner`) for the processor list.
- `PipelineOptions.VerifyChecksums`: checks every model file against Stanza's published MD5 before loading.
  Only Stanza's `.pt` files have checksums; with converted models it throws.
- `PipelineOptions.Logger` (`ILogger`): model load times and the thread count at Information, each processor's time
  per call at Debug. StanzaSharp now depends on `Microsoft.Extensions.Logging.Abstractions` (10.0.0, which brings
  `Microsoft.Extensions.DependencyInjection.Abstractions`; about 130 KB of assemblies, already in ASP.NET Core).
- Build warning `STANZA001` when the project resolves TorchSharp, `TorchSharp-cpu`/`TorchSharp-cuda-*` or a
  `libtorch-*` package in another version than StanzaSharp was built with (TorchSharp 0.107.0, libtorch 2.10.0).
- `<StanzaSharpTrimNative>true</StanzaSharpTrimNative>` leaves libtorch's Python bindings and test/mobile backends,
  which StanzaSharp never loads, out of build and publish output (CPU libtorch only): 35 MB less on Linux x64, 29 MB on macOS. The Docker
  sample uses it.
- The `stanzasharp` .NET tool (package `StanzaSharp.Tool`): `stanzasharp download [DIR] [--package NAME]
  [--processors LIST]` downloads the models without native libraries, e.g. in a Dockerfile.

### Changed
- `Sentence.Text` documents that every `Process` overload sets it; it is null only for CoNLL-U read without `# text`.

### Fixed
- A possible crash or wrong result when a garbage collection ran during a model call, more likely with concurrent
  calls and on Arm64: TorchSharp frees temporary Scalars (such as the hidden `alpha = 1` of a tensor addition) in a
  finalizer, which could run while libtorch was still reading them. StanzaSharp now keeps every Scalar it passes alive.

## [0.3.0] - 2026-10-06

### Added
- Platform packages `StanzaSharp.Cpu.Linux`, `StanzaSharp.Cpu.Windows` and `StanzaSharp.Cpu.MacOS`. Each is
  StanzaSharp plus one platform's CPU libtorch, for deployments that don't want all three (as `TorchSharp-cpu`
  restores them). Each is verified in CI on its own OS.
- Platform package `StanzaSharp.Cpu.WindowsArm64`: Windows on Arm64, which `TorchSharp-cpu` doesn't cover. Verified
  in CI on a Windows Arm64 runner.
- Supported platforms in the README and package readme: Linux x64 (glibc), Windows x64 and Arm64, macOS on Apple
  Silicon. Alpine (musl), Linux Arm64 and Intel Macs are not supported: TorchSharp and libtorch have no builds for them.
- A sample Dockerfile (`samples/docker`) for a Linux x64 container on the .NET runtime image (plain or chiseled),
  verified in CI to give output identical to Stanza's.
- Bulk processing: `Pipeline.Process(IEnumerable<string>)` returns one `Document` per text, batching all their
  sentences together, like Stanza's `bulk_process`, and with identical output (which, as in Stanza, can differ
  slightly from processing each text alone: sentiment labels depend on their batch, and sentence ids continue
  across the documents). On 2,000 one-sentence texts it is 7.7× faster than one call per text (9× with `default_fast`).
- Pretokenized input: `Pipeline.Process(IEnumerable<IEnumerable<string>>)` takes sentences of tokens and skips the
  tokenizer, like Stanza's `tokenize_pretokenized=True`, with identical output.

### Changed
- Verified on Windows and macOS (Apple Silicon) in CI, besides Linux. On macOS, `brew install libomp` is
  required: TorchSharp-cpu's libtorch links OpenMP from Homebrew's path.

## [0.2.0] - 2026-10-06

### Added
- Stanza's English `default_fast` package: `PipelineOptions.Package = "default_fast"` (default
  `"default"`, also `Pipeline.DefaultPackage`). It runs tokenize, mwt, pos, lemma, depparse, sentiment and
  ner, with pos, depparse and ner models that have their own small character LSTMs instead of the large
  character language models. Output is identical to Python Stanza's `package='default_fast'`. On 8 CPU
  threads it is about 2.3× faster than the default package (whose output it does not match: these are
  different models).
  - An unknown package name throws `ArgumentException`, and so does a processor the package lacks, such as
    `constituency` in `default_fast`.
- `ModelDownloader.DownloadAsync(dir, PipelineOptions)` downloads exactly what `Pipeline.Load` reads with the
  same options (package and processors).
- The CLI's `--package NAME`, for running and for `download`.
### Changed
- `PipelineOptions.Processors` is now `string?` and null by default, meaning all processors of the package
  (for `default`, the same eight as before).
- The POS tagger no longer scores batch padding: pos is about 25% faster with the default package (16.1 →
  12.1 s on the 26k-word benchmark), with identical output.

## [0.1.0] - 2026-10-06

First stable release.

### Changed
- The public API is now only the pipeline and its results:
  - `Pipeline`, `PipelineOptions`, `CharlmCacheOptions` and `ModelDownloader`;
  - `Document`, `Sentence`, `Token`, `Word`, `Entity`, `Tree` and `Conllu`.

  The processor classes (`PosTagger`, `Tokenizer`, …), the shared model classes (`Pretrain`,
  `CharLanguageModel`, `CharlmCache`, `Weights`, …), `Checkpoint`, `SafeTensorFile` and `PyString` are
  now internal. Use `PipelineOptions.Processors` to run a subset of processors.
- `Entity.Build` and `Token.IsMwtCandidate` are internal.

### Added
- XML documentation for every public member.
- `samples/StanzaSharp.Example`, a commented tour of the API.
- Issue templates for bug reports and feature requests.

## [0.1.0-alpha.2] - 2026-10-06

### Added
- Named-entity recognition (`ner`, Stanza's OntoNotes model):
  - `Token.Ner` holds the BIOES tag;
  - `Sentence.Entities` and `Document.Entities` hold the spans;
  - CoNLL-U output gets `ner=` in MISC.
- Sentiment (`sentiment`): `Sentence.Sentiment` (0 negative, 1 neutral, 2 positive) and the `# sentiment`
  comment in CoNLL-U.
- `ModelDownloader.DownloadAsync(dir, processors)` downloads only the models those processors need.

### Changed
- The default pipeline runs all eight processors, in Stanza's order, like Stanza's English default:
  tokenize, mwt, pos, lemma, constituency, depparse, sentiment, ner.
- The full model download is about 600 MB.

## [0.1.0-alpha.1] - 2026-10-06

First release. Stanza's English pipeline in .NET, running Stanza's own models through TorchSharp, with
output byte-identical to Python Stanza on the golden test data.

### Added
- Processors: tokenize (with sentence splitting), mwt, pos (UPOS, XPOS, features), lemma, depparse
  and constituency.
- `Pipeline.Load(dir, PipelineOptions)` with:
  - `Processors`;
  - `Device`, for CUDA;
  - `DisableTf32`, for GPU output identical to the CPU;
  - `CharlmCache`, to share character-model outputs between processors.
- Loads Stanza's original `.pt` model files directly, with no Python needed, through a restricted
  unpickler that never runs code.
- `ModelDownloader` fetches the models from Stanza's Hugging Face repository and checks their MD5s.
- `Conllu.Read` and `Conllu.Write`, in Stanza's CoNLL-U dialect.

[Unreleased]: https://github.com/bisforboman/StanzaSharp/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.6.0...v1.0.0
[0.6.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/bisforboman/StanzaSharp/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/bisforboman/StanzaSharp/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/bisforboman/StanzaSharp/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.2...v0.1.0
[0.1.0-alpha.2]: https://github.com/bisforboman/StanzaSharp/compare/v0.1.0-alpha.1...v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/bisforboman/StanzaSharp/releases/tag/v0.1.0-alpha.1
