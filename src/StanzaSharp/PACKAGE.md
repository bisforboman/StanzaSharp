# StanzaSharp

[Stanza](https://github.com/stanfordnlp/stanza)'s English, German, Spanish, French, Italian, Portuguese and Dutch NLP pipelines in .NET:
- tokenization and sentence splitting
- multi-word token expansion ("don't" → "do" + "n't", "zum" → "zu" + "dem")
- part-of-speech tags and morphological features
- lemmas
- dependency parsing
- named entities (English: OntoNotes types, PERSON, ORG, GPE, DATE, ...; German, Spanish, French, Dutch: PER, LOC, ORG,
  MISC; Italian: PER, LOC, ORG)
- constituency parsing
- sentence sentiment (negative, neutral, positive)

It runs Stanza's own pretrained models in plain .NET code, with no Python and no native libraries. The output is
identical to Python Stanza 1.15.0 on its golden test data (845 English and 181 German sentences, about 10 in each other
language), on Linux, Windows and
macOS, x64 and Arm64;
[how that is checked](https://github.com/bisforboman/StanzaSharp/blob/main/docs/verification.md).

## Install

```
dotnet add package StanzaSharp
```

That is all: the models run on StanzaSharp's managed backend, C# kernels using the CPU's SIMD instructions. On
the full pipeline it is about twice as fast as the TorchSharp backend on 8 threads and needs less memory, with the
same output.

**TorchSharp backend.** In its own package since 1.0, `StanzaSharp.Cuda` (also the way to a GPU, below). Add it and
the native libtorch yourself, `TorchSharp-cpu` (or one platform's `libtorch-cpu-<rid>` 2.10.0 package; on Windows on
Arm64 `libtorch-cpu-win-arm64`, since `TorchSharp-cpu` has none), and select it:

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Backend = CudaBackend.Cpu });
```

- Its version must match the `TorchSharp` version StanzaSharp.Cuda depends on; otherwise the build warns with `STANZA001`.
- On macOS (Apple Silicon), also run `brew install libomp`: libtorch loads OpenMP from Homebrew's path.
- `<StanzaSharpTrimNative>true</StanzaSharpTrimNative>` leaves out the libtorch files StanzaSharp never loads (35 MB
  on Linux x64, 29 MB on macOS).
- The `StanzaSharp.Cpu.*` platform packages ended with 0.4; reference `TorchSharp-cpu` or a `libtorch-cpu-<rid>`
  package instead.

## Supported platforms

The managed backend runs wherever .NET 10 runs. CI runs it against the golden data on:

| Platform | Managed backend (default) | TorchSharp backend |
|---|---|---|
| Linux x64 (glibc) | Yes | Yes |
| Windows x64 | Yes | Yes |
| Windows on Arm64 | Yes | Yes, with `libtorch-cpu-win-arm64` |
| macOS on Apple Silicon | Yes | Yes, after `brew install libomp` |
| Linux Arm64 (glibc) | Yes (verified in CI) | No: no libtorch or TorchSharp build |
| Alpine and other musl Linux | Yes (verified in CI, Docker) | No: libtorch is built for glibc only |
| macOS on Intel (x64) | Not tested | No |

**Docker:** any .NET 10 runtime image works, including the chiseled `runtime:10.0-noble-chiseled` and
`runtime:10.0-alpine` (publish with `-r linux-musl-x64` there): the image needs nothing but the .NET runtime. Mount
the models read-only rather than copying them into the image. A sample Dockerfile is in the repository's
[samples/docker](https://github.com/bisforboman/StanzaSharp/tree/main/samples/docker).

## Native AOT and trimming

On the managed backend, apps can be published with Native AOT or trimmed (`PublishTrimmed`, self-contained): no trim or
AOT warning, byte-identical output. `StanzaSharp.Cuda` is not supported there (TorchSharp isn't AOT-safe).

```xml
<PropertyGroup>
  <PublishAot>true</PublishAot>
  <!-- AVX2/FMA: the 256-bit kernels. Without it AOT runs the 128-bit ones, ~2x slower. -->
  <IlcInstructionSet>x86-64-v3</IlcInstructionSet>
</PropertyGroup>
```

Without `IlcInstructionSet` AOT runs the 128-bit kernels, about 2x slower; `x86-64-v3` needs AVX2/FMA (Intel from Haswell,
AMD from Excavator), `x86-64-v4` gives the AVX-512 kernels on AVX-512 CPUs only. Arm64 needs nothing: NEON is its baseline. AOT
needs `clang` and `zlib1g-dev` on Linux, Visual Studio's C++ workload on Windows.

## Download the models

The models (about 600 MB) come from Stanza's Hugging Face repository. Download them once:

```csharp
await ModelDownloader.DownloadAsync("models/stanza/en");                            // all, ~600 MB
await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
```

Each file is checked against its published MD5, and files already present are kept. `Pipeline.Load` itself
never touches the network.

**Command line:** the `stanzasharp` .NET tool (package `StanzaSharp.Tool`, about 0.5 MB, no native libraries) downloads
them too, e.g. in a Dockerfile or CI:

```
dotnet tool install -g StanzaSharp.Tool
stanzasharp download models/stanza/en --processors tokenize,mwt,pos,lemma   # --package default_fast for that package
stanzasharp download --lang de                                              # German, into models/stanza/de
```

**Check it yourself:** `stanzasharp compare my-text.txt` runs Python Stanza 1.15.0 (`pip install stanza==1.15.0`) and
StanzaSharp on your text with the same model files and reports whether the CoNLL-U output is identical, or shows the
first difference, with both sides' timings. `stanzasharp benchmark` times StanzaSharp (and Python Stanza, if installed)
on a built-in text on your machine and prints a shareable Markdown block with the hardware and both sides' results.
Both take `--lang de` for German.

## Use

```csharp
using StanzaSharp;

using var nlp = Pipeline.Load("models/stanza/en");
var doc = nlp.Process("Barack Obama was born in Hawaii. He was elected president in 2008.");

foreach (var sentence in doc.Sentences)
{
    foreach (var word in sentence.Words)
        Console.WriteLine($"{word.Id}\t{word.Text}\t{word.Lemma}\t{word.Upos}\t{word.Feats}\t{word.Head}\t{word.Deprel}");
    Console.WriteLine(sentence.Constituency); // (ROOT (S (NP (NNP Barack) (NNP Obama)) ...))
}
```

- `Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,mwt" })` runs only the listed processors; each needs the ones before it. The default is all eight, like Stanza's English default: `tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner`.
- Add `ner` to the list for named entities: `doc.Entities` (text, type, character offsets) and a BIOES tag per token (`Token.Ner`).
  The types depend on the language (table below).
- Add `sentiment` to the list for `sentence.Sentiment`: 0 negative, 1 neutral, 2 positive.
- `Conllu.Write(doc)` gives CoNLL-U in Stanza's format; `Json.Write(doc)` gives Python Stanza's `str(doc)` JSON, byte for
  byte, and `Json.Read` reads it back.
- `nlp.Process(texts)` with any `IEnumerable<string>` processes many texts at once, like Stanza's `bulk_process`:
  one `Document` per text, much faster for short texts.
- `nlp.Process(new[] { new[] { "Hello", "world", "." } })` takes text that is already split into sentences and
  tokens, like Stanza's `tokenize_pretokenized=True`.
- `nlp.Process(Conllu.Read(conllu))` annotates a document that already has its tokens, like Stanza's `process_conllu`:
  tokenize and mwt are skipped, and the other processors overwrite what they set.
- `Process` is thread-safe: share one pipeline between threads. Every overload takes a `CancellationToken`.
  One call on a short sentence takes about 28 ms on 8 threads (`default_fast`: 14 ms). On short texts, about two calls
  in flight per pipeline give the most throughput; more evict each other's weights from the CPU cache. Set
  `MaxConcurrentCalls = 2` in `PipelineOptions` to make extra callers wait their turn.
- Memory: a call's peak is set mostly by its largest batches, not by the length of the text: all eight processors peak
  at about 1.3 GB for 20,000 words and 1.9 GB for 200,000 (a book in one call is fine). What grows is the returned
  `Document`, about 0.5 KB per word, and garbage the GC collects when memory runs short (a container's memory limit
  caps it). To bound memory further, call `Process` on parts split at blank lines and drop each part's `Document`
  after use: sentence ids restart, offsets are each part's own, and sentiment labels can change (the classifier's
  batches span the whole call, as in Stanza: 24% of them with 1,000-word parts); everything else stays the same.
- Memory between calls: on the managed backend (the default) the models and the scratch buffers of a call's largest
  batches stay on the .NET GC heap for the next call, so the working set stays near the call's peak. To give it back
  when a service goes idle, call `GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true)` a
  minute or more after the last call (pooled buffers are released only then); the process drops to about its size
  after loading. On the TorchSharp backend, on Linux (glibc), a call of 1,000 words or more returns the memory it
  freed to the OS (`malloc_trim`); on Windows libtorch's allocator keeps it, and the environment variable
  `MIMALLOC_PURGE_DELAY=0`, set before libtorch loads, returns it for about 20% more time.
- Short-lived processes (CLI, serverless): loading takes about 1–1.5 s on 2–4 cores, and the first calls run while
  .NET's JIT is still at tier 0. `DOTNET_TieredCompilation=0` (or `<TieredCompilation>false</TieredCompilation>` in
  your project) compiles everything optimized once: the first call is slower, every later one as fast as in a warmed
  service. `PublishReadyToRun` makes the first call about a third faster; don't combine it with tiering off on Arm64
  for long texts.
- `PipelineOptions`:
  - `Backend`: `PipelineBackend.Managed` (the default), or `CudaBackend.Create()` / `CudaBackend.Cpu` from
    `StanzaSharp.Cuda`.
  - `Threads`: threads per operation, set process-wide at Load: the managed backend's one thread pool (shared by
    concurrent calls), or libtorch's intra-op threads. By default the physical cores when there are at least 4,
    else all logical processors (managed), or libtorch's own default (TorchSharp); never more than
    `Environment.ProcessorCount`, which respects a container's CPU quota.
  - `SplitSentences = false`: one sentence per paragraph (Stanza's `tokenize_no_ssplit`).
  - `Pretagged`: lemma, depparse and constituency load without the tagger (Stanza's `pretagged`), for
    `Process(Document)` on documents that already carry tags.
  - `NerTagSet`, `LemmaDictionaryOnly`, `MwtDictionaryOnly`, `LemmaBeamSize`: Stanza's `ner_predict_tagset`,
    `lemma_dict_only`, `mwt_dict_only` and `lemma_beam_size`.
  - `PosBatchSize`, `DepparseBatchSize`, `SentimentBatchSize` and the other batch sizes: Stanza's
    `{processor}_batch_size` options. Sentiment's (and in principle depparse's) changes results, as in Stanza.
  - `ModelPaths`: models of your own by file (`ModelPath`: the model, and optionally the word vectors and character
    models it reads), Stanza's `{processor}_model_path`.
  - `VerifyChecksums`: checks the `.pt` model files against Stanza's MD5s before loading.
  - `Logger`: an `ILogger` for load and processing times.
- `Processor.Tokenize`, ..., `Processor.Ner` are constants for the processor list.

## More accurate: the default_accurate package

`new PipelineOptions { Package = "default_accurate" }` selects Stanza's `default_accurate` package: all eight
processors, with pos, constituency, depparse, ner and sentiment reading the transformer
`google/electra-large-discriminator` (1.3 GB from Hugging Face, fetched by `DownloadAsync(dir, options)`; sentiment
through its LoRA adapters). Its output is identical to Python Stanza's `package='default_accurate'`; it is about 5.5 times
slower than `default` and needs about 4.2 GB of memory instead of 1.1 GB. German's (`Language = "de"`) reads
`german-nlp-group/electra-base-german-uncased` (445 MB) under pos, depparse, constituency and sentiment (the last two
through their LoRA adapters).

## Faster: the default_fast package

`new PipelineOptions { Package = "default_fast" }` selects Stanza's `default_fast` package: no constituency
parser, and pos, depparse and ner with their own small character models instead of the large character
language models. It is faster and smaller, and its output is identical to Python Stanza's
`package='default_fast'`.

```csharp
var options = new PipelineOptions { Package = "default_fast" };
await ModelDownloader.DownloadAsync("models/stanza/en", options); // what Load reads with these options
using var nlp = Pipeline.Load("models/stanza/en", options);
```

Stanza's English treebank packages run too, each trained on one treebank, with tokenize, mwt, pos, lemma and depparse
only: `combined`, `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`, `eslspok` (no mwt; its lemma is the word), and
the biomedical `craft`, `genia` and `mimic` (no mwt).

## One model per processor

`PipelineOptions.Models` replaces single models of the package, like Python Stanza's `processors={'ner': 'i2b2'}`
(processor → a model's or a package's name). The keys' processors are added to the package's when `Processors` is null.

```csharp
var clinical = new PipelineOptions { Package = "mimic", Models = new Dictionary<string, string> { [Processor.Ner] = "i2b2" } };
await ModelDownloader.DownloadAsync("models/stanza/en", clinical);
using var nlp = Pipeline.Load("models/stanza/en", clinical);
```

Beyond the packages' models: English constituency `ptb3_charlm`; English ner `ontonotes_charlm`, `ontonotes_nocharlm`,
`combined_charlm`, `conll03_charlm`, the clinical `i2b2` and `radiology` and the biomedical `anatem`, `bc4chemd`, `bc5cdr`, `bionlp13cg`, `jnlpba`, `linnaeus`, `ncbi_disease`, `s800`
(they read their own character language models, loaded next to the package's when they differ); English pos and depparse
`combined_electra-large` (and the treebanks' `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`, `eslspok` +
`_electra-large`), ner `ontonotes-ww-multi_electra-large` and `ontonotes_electra-large`, constituency `ptb3-revised_electra-large`, constituency `ptb3_electra-large` (top-down transitions) and sentiment
`sstplus_electra-large` (both with LoRA adapters, applied at run time), which read the transformer
`google/electra-large-discriminator` (1.3 GB from Hugging Face; `DownloadAsync(dir, options)` fetches it); German ner `conll03`;
the treebanks' `_nocharlm` taggers and parsers: English pos and depparse `ewt`, `gum`, `lines`, `partut`, `atis`,
`childes`, `eslspok` + `_nocharlm`, depparse `mimic_nocharlm`, German pos and depparse `gsd_nocharlm`, `hdt_nocharlm`;
the treebanks' `_charlm` tokenizers (English `combined`, `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`, `eslspok`;
German `combined`, `gsd`, `hdt`) and lemmatizers (English `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`; German
`combined`, `gsd`, `hdt`). Unknown or unsupported names throw `ArgumentException`.

## Language identification

`LanguageIdentifier` is Stanza's language identifier (68 languages, a 9 MB model in Stanza's `multilingual` folder):

```csharp
await ModelDownloader.DownloadLanguageIdentifierAsync("models/stanza/multilingual");
using var identifier = LanguageIdentifier.Load("models/stanza/multilingual");
string language = identifier.Identify("Das ist ein deutscher Satz."); // "de"
```

`MultilingualPipeline` routes each text to its language's pipeline (English and German by default, loaded at `Load` from
`en/`, `de/` and `multilingual/` under one folder) and sets `Document.Lang`; a text in another language comes back with
only `Text` and `Lang` (or, with `Restrict = true`, is processed as one of the configured languages):

```csharp
using var nlp = MultilingualPipeline.Load("models/stanza");
var docs = nlp.Process(["I don't think she's coming to the meeting.",
    "Die Bundesregierung hat am Mittwoch neue Regeln beschlossen."]); // Lang "en", "de"
```

## German

`Language = "de"` selects Stanza's German models: the same processors and packages as English (`default`,
`default_fast`), plus the German treebank packages `combined`, `gsd` and `hdt` (tokenize to depparse).

```csharp
var german = new PipelineOptions { Language = "de" };
await ModelDownloader.DownloadAsync("models/stanza/de", german); // ~900 MB
using var nlp = Pipeline.Load("models/stanza/de", german);
var doc = nlp.Process("Angela Merkel wurde in Hamburg geboren. Sie zog im Jahr 1954 in die DDR.");
// "im" is one token of two words, "in" + "dem"; doc.Entities: Angela Merkel/PER, Hamburg/LOC, DDR/LOC
```

Spanish (`es`), French (`fr`), Italian (`it`), Portuguese (`pt`) and Dutch (`nl`) work the same way, with Stanza's
`default` package of each language (and `default_fast`, but not in Spanish):

```csharp
var french = new PipelineOptions { Language = "fr" };
await ModelDownloader.DownloadAsync("models/stanza/fr", french); // ~380 MB
using var nlp = Pipeline.Load("models/stanza/fr", french);
```

Named entity types per language:

| Language | Model | Entity types (`Entity.Type`; `Token.Ner` is BIOES, e.g. `B-PER`) |
|---|---|---|
| English (`en`) | OntoNotes (`ontonotes-ww-multi`) | 18: `PERSON`, `NORP`, `FAC`, `ORG`, `GPE`, `LOC`, `PRODUCT`, `EVENT`, `WORK_OF_ART`, `LAW`, `LANGUAGE`, `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL`, `CARDINAL` |
| German (`de`) | GermEval 2014 (`germeval2014_charlm`) | 4: `PER`, `LOC`, `ORG`, `MISC` |
| Spanish (`es`), Dutch (`nl`) | CoNLL 2002 (`conll02`) | 4: `PER`, `LOC`, `ORG`, `MISC` |
| French (`fr`) | WikiNER (`wikinergold_charlm`) | 4: `PER`, `LOC`, `ORG`, `MISC` |
| Italian (`it`) | FBK (`fbk`) | 3: `PER`, `LOC`, `ORG` |

Portuguese has no NER model in Stanza.

## GPU

Reference `StanzaSharp.Cuda` and `TorchSharp-cuda-windows` (or `TorchSharp-cuda-linux`), then
`Pipeline.Load(dir, new PipelineOptions { Backend = CudaBackend.Create(disableTf32: true) })`.
`disableTf32` gives output identical to the CPU by turning TF32 off process-wide; with TF32 on (libtorch's
default), a few near-tie decisions can differ. See the repository's docs/gpu.md.

## License

Apache 2.0, like Stanza, whose code this ports (see NOTICE). Stanza's models are not included. Their
licenses vary with the training data.

Source, issues and docs: https://github.com/bisforboman/StanzaSharp
