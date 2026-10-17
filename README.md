# StanzaSharp

[![CI](https://img.shields.io/github/actions/workflow/status/bisforboman/StanzaSharp/ci.yml?branch=main&label=CI)](https://github.com/bisforboman/StanzaSharp/actions/workflows/ci.yml?query=branch%3Amain)
[![NuGet](https://img.shields.io/nuget/v/StanzaSharp)](https://www.nuget.org/packages/StanzaSharp)
[![License](https://img.shields.io/github/license/bisforboman/StanzaSharp)](LICENSE)

C# port of Stanza's English and German inference pipelines (tokenize, mwt, pos, lemma, depparse, ner, sentiment,
constituency).
It runs Stanza's own pretrained models in plain .NET code, with no native dependencies. On the golden test texts
(845 English and 181 German sentences), the output is byte-identical to Python Stanza 1.15.0, on Linux, Windows and
macOS, x64 and Arm64.
[docs/verification.md](docs/verification.md) shows what is compared, how, and how to check it on your own text.

Contributors: CLAUDE.md holds the scope, layout, decisions and conventions.

## Install

```
dotnet add package StanzaSharp
```

That is the whole install. Requires .NET 10. The models run on StanzaSharp's **managed backend**: C# kernels using the
CPU's SIMD instructions (AVX-512, AVX2 or Arm64 NEON where present), with no libtorch or other native library. On the full
pipeline it is about twice as fast as the TorchSharp backend on 8 threads and peaks lower in memory (see
[Performance](#performance)), and its output is the same, byte for byte.

**The TorchSharp backend** (libtorch) is in its own package, `StanzaSharp.Cuda`, since 1.0: it is the way to a GPU
(see [GPU](#gpu)) and also runs on libtorch's CPU kernels. Add it and the native libtorch yourself:

```
dotnet add package StanzaSharp.Cuda
dotnet add package TorchSharp-cpu
```

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Backend = CudaBackend.Cpu });
```

- `TorchSharp-cpu`'s version must match the `TorchSharp` version StanzaSharp.Cuda depends on (0.107.0); otherwise the build
  warns with `STANZA001`. Silence it with `<NoWarn>STANZA001</NoWarn>`.
- `TorchSharp-cpu` restores libtorch for Linux x64, Windows x64 and macOS (about 265 MB of downloads). To restore one
  platform's only, reference the matching `libtorch-cpu-<rid>` 2.10.0 package (`libtorch-cpu-linux-x64`,
  `-win-x64`, `-win-arm64`, `-osx-arm64`) instead; on Windows on Arm64 that is the only way, since `TorchSharp-cpu` has
  no Arm64 libtorch. (The `StanzaSharp.Cpu.*` platform packages did this up to 0.4 and are no longer published.)
- On macOS (Apple Silicon), also run `brew install libomp`: libtorch loads OpenMP from Homebrew's path.
- `<StanzaSharpTrimNative>true</StanzaSharpTrimNative>` in your project drops the libtorch files StanzaSharp never
  loads from build and publish output: the Python bindings (`libtorch_python`, `libshm`) and test and mobile backends
  (`libtorchbind_test`, `libjitbackend_test`, `libbackend_with_compiler`, `libaoti_custom_ops`, `libnnapi_backend`).
  That is 35 MB less on Linux x64 (a `-r linux-x64` publish goes from 503 to 468 MB) and 29 MB on macOS; the Windows
  libtorch packages ship none of them. Nothing TorchSharp loads links to them. It applies to the CPU libtorch only;
  CUDA builds are left whole.

Smaller models help either way: `Package = "default_fast"` and downloading only the processors you use.

### Supported platforms

The managed backend runs wherever .NET 10 runs. These run it in CI, against the golden data:

| Platform | Managed backend (default) | TorchSharp backend |
|---|---|---|
| Linux x64 (glibc: Ubuntu, Debian, RHEL, ...) | Yes, full suite in CI | Yes: `TorchSharp-cpu` or `libtorch-cpu-linux-x64` |
| Windows x64 | Yes, full suite in CI | Yes: `TorchSharp-cpu` or `libtorch-cpu-win-x64` |
| Windows on Arm64 | Yes, full suite in CI | Yes: `libtorch-cpu-win-arm64` only (`TorchSharp-cpu` has no Arm64 libtorch) |
| macOS on Apple Silicon | Yes, full suite in CI | Yes, after `brew install libomp`: `TorchSharp-cpu` or `libtorch-cpu-osx-arm64` |
| Linux Arm64 (glibc) | Verified in CI: the managed tests and the package | No: no `libtorch-cpu-linux-arm64` package, and TorchSharp has no `linux-arm64` native layer |
| Alpine and other musl Linux | Verified in CI: the Docker check on `runtime:10.0-alpine` | No: libtorch and TorchSharp are built for glibc only. Not even with `gcompat`: libtorch needs glibc-only symbols (`__memcpy_chk`, `backtrace`, `fcntl64`, …) it doesn't provide |
| macOS on Intel (x64) | Not tested | No: TorchSharp has no `osx-x64` native layer, and `libtorch-cpu-osx-x64` stops at 2.2 |

GPU: see [GPU](#gpu) (CUDA on Windows and Linux x64, TorchSharp backend).

### Docker

Any .NET 10 runtime image works: the image needs nothing but the .NET runtime, so `mcr.microsoft.com/dotnet/runtime:10.0`
(Ubuntu), the chiseled `runtime:10.0-noble-chiseled` and Alpine's `runtime:10.0-alpine` (publish with
`-r linux-musl-x64` there) all do. [samples/docker](samples/docker/Dockerfile) is a small app that reads text on stdin
and writes CoNLL-U:

```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY . .
RUN dotnet publish -c Release -r linux-x64 --no-self-contained -o /app

FROM mcr.microsoft.com/dotnet/runtime:10.0
WORKDIR /app
COPY --from=build /app .
ENTRYPOINT ["dotnet", "StanzaSharpDocker.dll", "/models"]
```

```
docker build -t stanzasharp-sample samples/docker
docker run -i --rm -v /path/to/models/stanza/en:/models:ro stanzasharp-sample < input.txt
```

Mount the models rather than copying them into the image: they are about 600 MB, several times the image itself
(209 MB on `runtime:10.0`, 103 MB on `runtime:10.0-noble-chiseled`, 100 MB on `runtime:10.0-alpine`; with
libtorch, up to 0.4, they were 707 and 601 MB). CI builds this image on the plain, chiseled and Alpine base images, checks that no libtorch is in it,
and that its output is byte-identical to the golden file. Since 1.0 nothing of TorchSharp is in it at all.

To download the models in a Dockerfile (on the SDK image):

```dockerfile
RUN dotnet tool install --tool-path /tools StanzaSharp.Tool && /tools/stanzasharp download /models --processors tokenize,mwt,pos
```

### Native AOT and trimming

On the managed backend an app can be published with Native AOT (`PublishAot`) or trimmed (`PublishTrimmed`, self-contained):
the StanzaSharp assemblies are `IsAotCompatible`, publish with no trim or AOT warning, and give byte-identical output.
`StanzaSharp.Cuda` is not supported there: TorchSharp isn't trim- or AOT-safe.

```xml
<PropertyGroup>
  <PublishAot>true</PublishAot>
  <!-- AVX2/FMA: the 256-bit kernels. Without it AOT runs the 128-bit ones, ~2x slower. -->
  <IlcInstructionSet>x86-64-v3</IlcInstructionSet>
</PropertyGroup>
```

`dotnet publish -c Release -r linux-x64` (or `win-x64`) then writes one self-contained executable of about 6 MB that loads
the models about 0.6 s faster and answers its first call about 3x sooner than the JIT, at the JIT's speed afterwards.
AOT fixes the SIMD width at compile time, unlike the JIT, which picks it per CPU:

- no `IlcInstructionSet` (x86-64-v2): the 128-bit kernels on every CPU, about 2x slower;
- `x86-64-v3` (recommended): AVX2 and FMA: Intel from Haswell (2013) and AMD from Excavator (2015), but not
  some Atom, Celeron and Pentium models (the binary won't start without them);
- `x86-64-v4`: the AVX-512 kernels, for fleets of AVX-512 CPUs only;
- Arm64 (`linux-arm64`, `win-arm64`): NEON is part of the Arm64 baseline, so nothing to set (not tested in CI yet).

AOT needs the platform's linker: `clang` and `zlib1g-dev` on Linux, Visual Studio's "Desktop development with C++" on
Windows. CI publishes [tests/StanzaSharp.ManagedCheck](tests/StanzaSharp.ManagedCheck/Program.cs) both ways on Linux and
Windows x64 and compares its output with the golden files ([tools/verify-aot.sh](tools/verify-aot.sh)). Details and
measurements: [docs/aot.md](docs/aot.md).

## Models

Download Stanza's English models (about 600 MB, from Stanza's Hugging Face repository) once. Every file is
checked against its published MD5; `Pipeline.Load` itself never touches the network.

```csharp
await ModelDownloader.DownloadAsync("models/stanza/en");                            // all, ~600 MB
await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Package = "default_fast" });
```

or from the command line with the `stanzasharp` tool (package `StanzaSharp.Tool`, no native libraries):
`dotnet tool install -g StanzaSharp.Tool`, then `stanzasharp download models/stanza/en`
(add `--processors LIST` for a subset, `--package default_fast` for that package, `--model ner=i2b2` for single models, `--lang de` for German, which goes to
`models/stanza/de` by default). In this repository,
`dotnet run --project samples/StanzaSharp.Cli -- download models/stanza/en` does the same.
`DownloadAsync(dir, options)` fetches exactly what `Pipeline.Load(dir, options)` reads. A subset includes the models of the processors it requires and
the shared word vectors and character models when used.
A directory you already have from Python Stanza (`<processor>/<name>.pt`, e.g. `~/stanza_resources/en`)
works too.

Model files are read without Python and without running any code from them: the `.pt` reader accepts only the plain
data and tensor types Stanza's checkpoints use (like torch's `weights_only=True`). A malformed, truncated or hostile
file fails with an `InvalidDataException` in time and memory bounded by its size; a fuzz test of both file formats
runs in CI.

### Check it yourself

`stanzasharp compare FILE` runs Python Stanza 1.15.0 and StanzaSharp on your own text, with the same model files
(MD5-checked), and tells you whether the CoNLL-U output is identical or shows the first difference. It needs Python
with `pip install stanza==1.15.0`:

```
stanzasharp download models/stanza/en
stanzasharp compare my-text.txt                   # --lang de, --package, --processors, --models DIR, --python PATH
```
```
Stanza 1.15.0 (Python) and StanzaSharp on my-text.txt
  language:    en
  package:     default (tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner)
  models:      /home/me/models/stanza/en: the same 11 files on both sides, MD5s match Stanza 1.15.0's
  text:        10 sentences, 140 words
  Stanza:      loaded in 10.4 s, processed in 25.3 s
  StanzaSharp: loaded in 5.8 s, processed in 14.4 s

Identical: the CoNLL-U output is the same, byte for byte.
```

For German, `stanzasharp download --lang de` and `stanzasharp compare my-text.txt --lang de` (models from
`models/stanza/de`; Stanza runs its German pipeline). Exit code 0 means identical, 1 different, 2 a usage or setup
problem (no Python, wrong Stanza version, missing models).
After characters outside the BMP (emoji, for example), StanzaSharp's `start_char`/`end_char` count UTF-16 code units
and Stanza's count code points. `compare` converts StanzaSharp's offsets to code points before comparing and says
that it did.

`stanzasharp benchmark` measures speed on your machine: load time, each processor, words/s, one-sentence calls (median
and p90) and peak memory on a built-in English text (2,009 words; `--lang de`: a German one), and, if Python Stanza 1.15.0 is installed, Stanza on the same text
with the same models, so you see the ratio on your own hardware. It prints a Markdown block with the CPU, cores, RAM,
OS, .NET and SIMD path, ready to paste into an issue. About two minutes with Python on 4 cores; `--quick` for a smoke
test. See [docs/performance.md](docs/performance.md#run-it-yourself).

## Usage

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

`Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,mwt" })` runs only the listed processors; each
needs the ones before it. `PipelineOptions` also holds `Backend` (`PipelineBackend.Managed`, the default, or
`CudaBackend.Create()` / `CudaBackend.Cpu` from `StanzaSharp.Cuda`; see [Install](#install)) and `CharlmCache`
(`IsEnabled`, `MaxWords`: the tagger's character-model outputs reused by the constituency parser and sentiment, at most
32,768 words / ~256 MB by default; output is identical either way).
The default is all eight processors, like Stanza's English default:
`tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner`. List fewer to run less; `ner` and `sentiment`
need only `tokenize`. NER gives named entities: `Token.Ner` holds the BIOES tag (`B-PERSON`, `O`, ...), and `sentence.Entities` /
`doc.Entities` the spans (`Text`, `Type`, `StartChar`, `EndChar`); the types depend on the language's model (see
[German](#german)).
Sentiment gives `sentence.Sentiment`: 0 negative, 1 neutral, 2 positive.
`Conllu.Write(doc)` gives Stanza-style CoNLL-U, with `ner=` in MISC when NER ran. `Json.Write(doc)` gives Python
Stanza's `str(doc)` byte for byte: per sentence, a list of token and word objects (`id`, `text`, `lemma`, `upos`, `xpos`,
`feats`, `head`, `deprel`, `start_char`, `end_char`, `ner`, `multi_ner`, `misc`); like Stanza's, it has no sentence text,
tree or sentiment. `Json.Read` reads it back.

### German

`PipelineOptions.Language = "de"` selects Stanza's German models (the same object downloads and loads them):

```csharp
var german = new PipelineOptions { Language = "de" };
await ModelDownloader.DownloadAsync("models/stanza/de", german); // ~900 MB
using var nlp = Pipeline.Load("models/stanza/de", german);
var doc = nlp.Process("Angela Merkel wurde in Hamburg geboren. Sie zog im Jahr 1954 in die DDR.");
// "im" is one token of two words, "in" + "dem"; doc.Entities: Angela Merkel/PER, Hamburg/LOC, DDR/LOC
```

The same eight processors and both packages (`default`, `default_fast`) as English, plus Stanza's German treebank packages
`combined`, `gsd` and `hdt` (tokenize, mwt, pos, lemma and depparse only); `default_accurate` needs transformers and
throws. German differs from English in its models, not in the API:

- MWT is a character seq2seq model (contractions such as `zum` → `zu dem`, `ins` → `in das`), after Stanza's dictionary.
- The constituency parser reads UPOS and writes the SPMRL labels (`(ROOT (VROOT (S ...)))`).
- NER uses fastText's 300-dimensional German vectors, a second word-vector file, and the GermEval 2014 types:

| Language | Model | Entity types (`Entity.Type`; `Token.Ner` is BIOES, e.g. `B-PER`) |
|---|---|---|
| English (`en`) | OntoNotes (`ontonotes-ww-multi`) | 18: `PERSON`, `NORP`, `FAC`, `ORG`, `GPE`, `LOC`, `PRODUCT`, `EVENT`, `WORK_OF_ART`, `LAW`, `LANGUAGE`, `DATE`, `TIME`, `PERCENT`, `MONEY`, `QUANTITY`, `ORDINAL`, `CARDINAL` |
| German (`de`) | GermEval 2014 (`germeval2014`) | 4: `PER`, `LOC`, `ORG`, `MISC` |

### Many texts, pretokenized text, and annotating a document

```csharp
// Bulk: one Document per text, in order; sentences of all texts are batched together (Stanza's bulk_process).
List<Document> docs = nlp.Process(tweets); // any IEnumerable<string>

// Pretokenized: sentences of tokens; the tokenizer model is skipped and the tokens kept as given
// (Stanza's tokenize_pretokenized=True). doc.Text is the tokens joined by spaces: "Hello world . Bye ."
var doc = nlp.Process(new[] { new[] { "Hello", "world", "." }, new[] { "Bye", "." } });

// A document that already has its tokens, e.g. read from CoNLL-U or JSON (Stanza's process_conllu): tokenize and mwt
// are skipped, the other processors overwrite what they set (tags, lemmas, parses, NER, sentiment), the rest stays.
var annotated = nlp.Process(Conllu.Read(File.ReadAllText("tokens.conllu")));
```

Documents that already carry tags: `Pretagged = true` is Stanza's `pretagged=True`. Lemma, depparse and constituency
then load without the processors they require, so a pipeline with `Processors = "depparse"` parses your own UPOS, XPOS
and lemmas without loading the tagger (tokenize and mwt still load, about 1 MB):

```csharp
using var parser = Pipeline.Load("models/stanza/en", new PipelineOptions { Processors = "depparse", Pretagged = true });
var parsed = parser.Process(Conllu.Read(File.ReadAllText("tagged.conllu")));
```

Both match Python Stanza byte for byte. Bulk is much faster than one call per text for short texts: 7.7× on
2,000 one-sentence texts ([docs/performance/bulk-processing.md](docs/performance/bulk-processing.md)). One call per
text costs about 28 ms for a short sentence on 8 threads (`default_fast`: 14 ms; Python Stanza: 189 and 73 ms), mostly
reading the models' weights from memory ([round 7](docs/performance/round-07-per-call-short-texts.md)). Bulk output
can differ slightly from processing each text alone, exactly as in Stanza: the sentiment classifier sees its batch's padding (172 of 854
golden labels change), and sentence ids continue across the documents. Pretokenized tokens are never split into
multi-word tokens (`"don't"` stays one word), as in Stanza. `Process(Document)` changes the document in place and
returns it; its tokens and words are never split or merged.

### Packages

`PipelineOptions.Package` picks one of Stanza's packages by its Stanza name (German's: [German](#german)):

- `"default"` (the default): all eight processors above.
- `"default_fast"`: tokenize, mwt, pos, lemma, depparse, sentiment and ner, without constituency. Its pos,
  depparse and ner models have their own small character LSTMs instead of the large character language
  models, so they are faster and need less memory, at slightly lower accuracy. Output is byte-identical to
  Python Stanza's `package='default_fast'`.

```csharp
var options = new PipelineOptions { Package = "default_fast" };
await ModelDownloader.DownloadAsync("models/stanza/en", options);
using var fast = Pipeline.Load("models/stanza/en", options);
```

- The treebank packages, each trained on one English treebank: `"combined"` (the default package's models), `"ewt"`,
  `"gum"`, `"lines"`, `"partut"`, `"atis"`, `"childes"` and `"eslspok"`, and the biomedical `"craft"`, `"genia"` and
  `"mimic"` (clinical notes). They have tokenize, mwt, pos, lemma and depparse only; `eslspok` and the biomedical ones
  have no mwt, and `eslspok`'s lemma is the word itself (Stanza's identity lemmatizer). Each downloads about 330 MB, of
  which 150 MB (the word vectors and character language models) is shared with the default package; the biomedical
  ones bring their own word vectors (`craft`, `genia`: about 220 MB; `mimic`: 310 MB with its character models).

An unknown package name throws, and so does listing a processor the package lacks (`constituency` with
`default_fast`). See [docs/performance.md](docs/performance.md) for the speed of both.

### One model per processor

`PipelineOptions.Models` replaces single models of the package, like Python Stanza's `processors={'ner': 'i2b2'}`. Each
key is a processor, each value a model's name or a package's name (that package's model). Every other processor keeps
the package's model. With `Processors` null, the keys' processors are added to the package's (`mimic` has no ner);
with `Processors` set, every key must be in it.

```csharp
// Clinical entities (PROBLEM, TEST, TREATMENT) on the mimic package, whose character models i2b2 shares
var clinical = new PipelineOptions { Package = "mimic", Models = new Dictionary<string, string> { [Processor.Ner] = "i2b2" } };
await ModelDownloader.DownloadAsync("models/stanza/en", clinical);
using var nlp = Pipeline.Load("models/stanza/en", clinical);
```

The models beyond the packages':

- English constituency: `ptb3_charlm` (the original Penn Treebank labels; the default is `ptb3-revised_charlm`).
- English ner: `ontonotes_charlm` and `ontonotes_nocharlm` (OntoNotes only, with 300-dimensional fastText vectors, 123
  MB), `combined_charlm` (three tag sets, of which Stanza uses OntoNotes'), `conll03_charlm` (PER, LOC, ORG, MISC, with
  GloVe vectors), the clinical `i2b2` (problems, tests, treatments) and `radiology` (anatomy, observations, uncertainty),
  and the biomedical `anatem` (anatomy), `bc4chemd` (chemicals), `bc5cdr` (chemicals, diseases), `bionlp13cg` (cancer
  genetics, 16 types), `jnlpba` (proteins, DNA, RNA, cell lines and types), `linnaeus` and `s800` (species) and
  `ncbi_disease` (diseases), which read biomedical word vectors and PubMed character models (about 470 MB together).
- English models with a transformer (Stanza's `default_accurate` models): pos and depparse `combined_electra-large`, ner
  `ontonotes-ww-multi_electra-large`, constituency `ptb3-revised_electra-large` and sentiment `sstplus_electra-large`.
  They read `google/electra-large-discriminator`, 1.3 GB from Hugging Face, which `DownloadAsync(dir, options)` fetches
  into `transformer/`. The first four read it frozen and share one pass per sentence; sentiment applies its LoRA
  adapters at run time, a second pass. A pass adds about 70 ms to a short sentence on 4 cores.
- German ner: `conll03` (PER, LOC, ORG, MISC).
- The treebanks' taggers and parsers without character language models (their own character LSTM, like
  `default_fast`'s): English pos and depparse `ewt_nocharlm`, `gum_nocharlm`, `lines_nocharlm`, `partut_nocharlm`,
  `atis_nocharlm`, `childes_nocharlm`, `eslspok_nocharlm`, depparse `mimic_nocharlm`; German pos and depparse
  `gsd_nocharlm`, `hdt_nocharlm` (e.g. `Package = "ewt"` with `[Processor.Pos] = "ewt_nocharlm"` and
  `[Processor.Depparse] = "ewt_nocharlm"`).
- The treebanks' tokenizers and lemmatizers with character language models (they read the charlm's state at every
  character): English tokenize `combined_charlm`, `ewt_charlm`, `gum_charlm`, `lines_charlm`, `partut_charlm`,
  `atis_charlm`, `childes_charlm`, `eslspok_charlm` and lemma `ewt_charlm`, `gum_charlm`, `lines_charlm`,
  `partut_charlm`, `atis_charlm`, `childes_charlm`; German tokenize and lemma `combined_charlm`, `gsd_charlm`,
  `hdt_charlm`.
- English lemma `combined_charlm`, whose contextual classifiers choose the lemma of `'s` (be or have) and `her` (she or
  her) from the whole sentence.

Their output is byte-identical to Python Stanza's on both backends. Names are exact, as in Stanza (`ptb3` is not
`ptb3_charlm`). An unknown, transformer or not yet supported name throws. A model may read other character language
models than the package's: `i2b2` reads mimic's, so with the default package the pipeline loads both pairs (mimic's two
files are 19 MB each); with `Package = "mimic"` they are the tagger's, whose outputs it then reuses. The tool takes the same as
`--model ner=i2b2`.
docs/per-processor-models-plan.md lists every other model and what it needs.

### Threads, concurrency and cancellation

```csharp
using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions
{
    Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos}", // or "tokenize,mwt,pos"
    Threads = 2,              // threads per operation; null: physical cores (from 4 up) or logical processors
    VerifyChecksums = true,   // check the .pt files against Stanza's MD5s first
    Logger = loggerFactory.CreateLogger("StanzaSharp"), // load and processing times
});
var doc = nlp.Process(text, cancellationToken); // OperationCanceledException within about one batch
```

- **Threads.** On the managed backend `Threads` is the size of StanzaSharp's thread pool; on the TorchSharp backend it
  is libtorch's `set_num_threads`. Both are **process-wide**: the last pipeline loaded sets it. Null (the default)
  means, on the managed backend, the physical cores on machines with at least 4 of them (fewer slow calls and more calls/s
  with concurrent callers than all logical processors, up to 9% slower on long texts) and all logical processors on smaller ones; on
  TorchSharp, libtorch's own default (the physical cores). Either way at most `Environment.ProcessorCount`, which .NET
  limits to a container's CPU quota, and a lower count set earlier is kept. (libtorch applies a new count to the loading thread and to threads that haven't run a
  tensor operation yet, so with TorchSharp load before processing starts.)
- **Concurrency.** `Process` is thread-safe: one pipeline can serve many threads, and each call's output is identical
  to a sequential call's. The models are only read; everything a call changes is its own. On the managed backend
  concurrent calls share the one pool: each call works its own part and idle threads help, so N callers don't start
  N teams. On the TorchSharp backend each call runs its operations on up to `Threads` threads of its own, so N
  callers on C cores do best with `Threads` about C / N. Throughput is bounded by the CPU, not the number of callers.
  On the managed backend it even drops past about two callers in flight on short texts (8 callers: 24 calls/s, 2: 36),
  as concurrent calls evict each other's weights from the CPU cache: limit a service to about two concurrent
  `Process` calls per pipeline (`new PipelineOptions { MaxConcurrentCalls = 2 }`: extra calls wait their turn), or batch
  texts that arrive together into one bulk call.
  Sharing one pipeline saves memory: a loaded pipeline takes about 0.8–1 GB (`default`; about 0.7 GB
for `default_fast`), and each call in flight adds
  its own working memory on top (a few hundred MB for a page of text, more for long documents).
- **Memory.** The numbers below were measured on the TorchSharp backend; the managed backend peaks lower (all eight
  processors, one call on 6,761 words: 1.3 GB against 4.3 GB;
  [round 10](docs/performance/round-10-sentiment-memory.md) and
  [round 6](docs/performance/round-06-memory-managed-backend.md)). A call's peak is set
  by its largest batch more than by the length of the text: the tagger pads up to
  250 sentences to the longest one among them. With `tokenize,mwt,pos,constituency` (8 threads) loading peaks at about
  0.5 GB, and one call on 500, 5,000 and 15,000 words peaks at 0.65, 1.6 and 1.7 GB (Python Stanza: 0.75, 2.4 and
  2.5 GB). To bound it in a memory-limited container, call `Process` on parts of about 1,000 words, split at blank
  lines: on 15,000 words that peaks at 1.2 GB instead of 1.7 GB and takes about 15% longer. With these processors the
  annotations are the same; only sentence ids, offsets (each part's own) and the whitespace at the cuts differ. With
  `sentiment` the labels can change too (see "Very long texts" below). One call per paragraph
  peaks at 0.6 GB but is 5× slower. Bulk `Process(texts)` batches all texts together, so it peaks like one call.
  `CharlmCache` changes the peak by less than 50 MB up to its 32k-word cap (turning it off costs 30–40% more time),
  and neither the GC mode nor converting the models changes it much. Measurements: [round 3](docs/performance/round-03-memory-short-lived-process.md).
- **Very long texts** (a book in one call). The peak grows slowly with the length of the text: all eight processors on
  the managed backend peak at about 1.3 GB for 20,000 words and 1.9 GB for 200,000 (8 threads, Windows;
  [round 11](docs/performance/round-11-garbage-long-texts.md)). 670,000 words peaked at
  3.0 GB in round 8, before rounds 10 and 11 cut sentiment's buffers and the garbage of long texts, and haven't been
  measured since. Most of it does not depend on the length (the models, and the scratch buffers of the largest batches,
  which are at most 5,000 words); what grows is the returned `Document` itself, about 0.5 KB per word (310 MB for
  670,000 words), plus garbage the .NET GC lets build up on a larger heap. That garbage is not needed: with the GC heap
  capped (`DOTNET_GCHeapHardLimit`, or a container's memory limit, of which the GC takes 75%) 670,000 words ran in a
  2.3 GB heap and peaked at 2.2 GB. Splitting the text at blank lines into parts of about 1,000 words and calling
  `Process` per part saves about 0.2 GB more, plus the `Document`s you drop, but changes some output: the sentiment
  classifier's batches depend on all of a call's sentences, as in Stanza, so about a quarter of the sentiment labels
  change (508 of 2,091 sentences on 20,000 words); sentence ids restart and offsets are each part's own. Every other
  annotation was the same. Bulk `Process(texts)` batches like one call.
  Measurements: [round 8](docs/performance/round-08-very-long-text.md).
- **Memory between calls** (long-running services). On the managed backend (the default) everything lives on the
  .NET GC heap: the models (about 370 MB for `tokenize,mwt,pos,constituency`, 660 MB for all eight) and the scratch
  buffers of the call's largest batches, which `ArrayPool<T>.Shared` keeps for the next call. So the working set
  stays near the call's peak (Windows, 6,761 words: 0.7 GB with those four processors, 1.3 GB with all eight; the same
  on Linux), and later calls reuse it. .NET gives pooled arrays back only at a gen2 GC at least a minute after their
  last use, and keeps freed memory committed until an aggressive GC, so to return it when a service goes idle, call
  `GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true)` a minute or more after the last call
  (50–250 ms): the working set drops to about the size after loading (0.4 GB and 0.7–0.8 GB). Right after a call it
  frees little, since the pool still holds the buffers. `malloc_trim` and
  `MIMALLOC_PURGE_DELAY` play no part on this backend.
  On the TorchSharp backend tensors live in libtorch's native heap instead. On Linux (glibc) a call of 1,000 words or
  more gives the memory it freed back to the OS (`malloc_trim`, 25–40 ms), so with `tokenize,mwt,pos,constituency` the
  RSS drops back to about 650 MB after each call (590 MB after loading; Python Stanza: 850–950 MB). No `MALLOC_*`
  setting is needed in a container, and `MALLOC_ARENA_MAX` does not help. On Windows libtorch's built-in allocator
  (mimalloc) keeps freed memory committed, so the working set stays near the peak (1.4–1.6 GB). The environment
  variable `MIMALLOC_PURGE_DELAY=0` makes it give memory back as it is freed (about 580 MB after a call, and a 370 MB
  lower peak) for about 20% more time. It must be set before libtorch loads: in the environment, or with
  `Environment.SetEnvironmentVariable` before the first TorchSharp call. Measurements:
  [round 6](docs/performance/round-06-memory-managed-backend.md) (managed) and
  [round 5](docs/performance/round-05-memory-linux-windows.md) (TorchSharp).
- **Cancellation.** Every `Process` overload takes a `CancellationToken`. It is checked between processors and
  between batches inside each (inside the tagger's, dependency parser's and sentiment classifier's 5,000-word batches
  too, between their stages), so a call stops within about half a second on an 8-core machine; no document is returned and
  the pipeline stays usable.
- **One sentence per paragraph.** `SplitSentences = false` is Stanza's `tokenize_no_ssplit`: the tokenizer still
  splits tokens, but each paragraph (text between blank lines) is one sentence. Bulk input works the same way;
  pretokenized input keeps its sentences, as in Stanza.
- **Checksums.** `VerifyChecksums` reads every model file once more before loading (about 1.3 s for `default`'s 700 MB, 0.8 s for `default_fast`, with the files in the OS cache) and throws an
  `InvalidDataException` naming a file that doesn't match. Only Stanza's `.pt` files have published checksums, so with
  converted models it throws instead of skipping the check.
- **Short-lived processes** (a CLI run, a serverless function, a one-off job). `Pipeline.Load` takes about 1–1.5 s
  on GitHub's free runners (2–4 cores; two models load at a time), and the first calls run while .NET's JIT is still
  at tier 0: a one-sentence call takes about 180–300 ms the first time and 70–120 ms the next few (60–140 ms once
  warm). With
  `DOTNET_TieredCompilation=0` in the environment (or `<TieredCompilation>false</TieredCompilation>` in your project)
  everything is compiled optimized once: the first call is about 40–50% slower, every later one as fast as in a
  warmed service, and a process that loads and makes six one-sentence calls finishes up to 16% sooner, long texts up
  to 6%. Publishing your app with `PublishReadyToRun` precompiles our assemblies too and makes the first call 23–38%
  faster; combined with tiering off it keeps the precompiled code, which on Arm64 made long texts 20–60% slower.
  Measurements: [round 17](docs/performance/round-17-startup.md).
- **Logging.** `Logger` gets each model's load time and the thread count at Information, and each processor's time
  per call at Debug. Without a logger nothing is measured.
- `Sentence.Text` is nullable because CoNLL-U read without `# text` has none, but `Process` always sets it.

`samples/StanzaSharp.Example` is a commented tour of the whole API: `dotnet run --project samples/StanzaSharp.Example`.

From the command line, this writes CoNLL-U for a file (or standard input):

```powershell
dotnet run --project samples/StanzaSharp.Cli -- input.txt
dotnet run --project samples/StanzaSharp.Cli -- --processors tokenize,mwt,pos input.txt
dotnet run --project samples/StanzaSharp.Cli -- --package default_fast input.txt
dotnet run --project samples/StanzaSharp.Cli -- --backend torch input.txt
```

## Development setup

The .NET solution needs only the .NET 10 SDK. Python 3 is used for the reference environment: Stanza
itself (to generate golden data), the optional converter to safetensors + JSON (slightly faster to load
than `.pt`), and the benchmark's Python side.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Models
```

`-Models` creates `tools\.venv` with Stanza, downloads the English models to `models\stanza` and
converts them to `models\converted\en`. Use `-Python` for the environment only, or no switch for just the
.NET solution. Add `-Cuda` for GPU libtorch (see GPU below).

## Performance

On GitHub's free hosted runners (2–4 cores), the default pipeline ran 2.1–3.8x faster than Python Stanza 1.15.0 on the
same text and models, peaked at about a third of its memory (0.9–1.1 GB against 2.2–3.7 GB), and gave identical
CoNLL-U ([docs/verification.md](docs/verification.md#speed-on-the-free-runners)). `stanzasharp benchmark` measures
your own machine (see [Check it yourself](#check-it-yourself)).

The managed backend runs all eight processors about 2.2x faster than the TorchSharp backend on 8 threads (26,264
words on a Ryzen 7 5800X: 25.9 s against 56.4 s), and peaks lower (one call on 6,761 words: 1.3 GB against 4.3 GB),
with byte-identical output. On 1 thread every processor is faster too, by 13–52%. Loading takes about as long (1.1 s)
and peaks at about 850 MB (TorchSharp: 820 MB). Details per processor:
[docs/backends.md](docs/backends.md#phase-2-progress).

On the TorchSharp backend, on 8 CPU threads the six-processor pipeline is about 1.7x faster than Python Stanza and peaks
at 2.7 GB of memory (Python: 4.3 GB); see [docs/performance.md](docs/performance.md). The `default_fast` package runs
its seven processors about 2.3x faster than the default's eight, and 1.5x faster than Python's `default_fast`. On a
machine busy with other work, fewer threads (`Threads`) are often faster.

## GPU

The GPU runs on the TorchSharp backend, in the `StanzaSharp.Cuda` package:
`Pipeline.Load(dir, new PipelineOptions { Backend = CudaBackend.Create() })` runs every model on an NVIDIA GPU
(`CudaBackend.Create(deviceIndex: 1)` for the second one). Reference a CUDA libtorch package such as
`TorchSharp-cuda-windows` (several GB), in the same version as StanzaSharp.Cuda's `TorchSharp` (0.107.0). Up to 0.5
this was `PipelineOptions.Device` and `DisableTf32`, which 1.0 removed.

For output identical to the CPU (and so to Python Stanza), turn off TF32, which libtorch enables for cuDNN by
default on Ampere and newer GPUs:

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Backend = CudaBackend.Create(disableTf32: true) });
```

`disableTf32` sets `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` to false. These
are process-wide torch settings, so it is opt-in and affects all TorchSharp code in the process. With TF32 off, the GPU matched
every golden file exactly on an RTX 3080; with TF32 on, 3 of 845 sentences differed (near-ties). Results
are deterministic from run to run either way. The pipeline is roughly 4x faster on an RTX 3080 than on 8 CPU
threads (Ryzen 7 5800X), mostly in pos and depparse; the constituency parser gains least. Details in
[docs/gpu.md](docs/gpu.md).

In this repository, set `STANZASHARP_CUDA=1` to build the Cli, Benchmark and Tests with
`TorchSharp-cuda-windows`; that also compiles `GpuTests`, which skip without a CUDA device:

```powershell
$env:STANZASHARP_CUDA = '1'
dotnet test tests/StanzaSharp.Tests --filter FullyQualifiedName~GpuTests --logger "console;verbosity=detailed"
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --device cuda --no-tf32
```

Unset it (and let `dotnet` restore again) to go back to the CPU build.

## Tests

```powershell
dotnet test -c Release
```

Tests that need models skip when `models/converted/en` (or, for the `.pt` loader tests, `models/stanza/en`) is
missing; `STANZASHARP_MODELS` points them elsewhere. Build Release: the managed kernels are many times slower in Debug.
To regenerate the golden data, run `tools\.venv\Scripts\python tools\make_golden.py`.

## Code style

[StyleBro](https://github.com/bisforboman/stylebro) and the .NET code-style rules in `.editorconfig` run on every
build; `dotnet tool install --global StyleBro.Migrate --version 0.5.0-alpha.1`, then `stylebro-migrate format
StanzaSharp.slnx` fixes what they report.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` (a release waits for that run) and on pull requests (.NET 10; Ubuntu 26.04
x64 and Arm64, Windows x64 and Arm64, macOS 26 on Apple Silicon). [docs/verification.md](docs/verification.md#where-it-is-checked)
summarizes what each job proves.

- `build-test` builds the solution with warnings as errors, checks the code style (`stylebro-migrate format
  --verify-no-changes`, see above) and runs the tests without models (the model tests skip).
- `golden` downloads the English models with `ModelDownloader` (through the CLI), converts them with
  `tools/stanza_convert.py` (CPU-only torch wheel), and runs the full suite (both backends) in three shards
  (`golden-shard`; `golden` passes only if all three did), with the German tests on Stanza's German `.pt` files (cached;
  the other jobs stay English). It fails if any test was skipped. Then `tools/verify-package.ps1` packs StanzaSharp and StanzaSharp.Cuda and runs two fresh apps that
  reference them: one with only StanzaSharp (the default install; it must reproduce the golden CoNLL-U with nothing of
  TorchSharp in it), one with StanzaSharp.Cuda and `TorchSharp-cpu` on `CudaBackend.Cpu` (no `STANZA` build warning allowed, with `StanzaSharpTrimNative`). Then
  `tools/verify-tool.ps1` installs the `stanzasharp` tool from a local feed, downloads `tokenize,mwt` with it and
  checks `compare`'s setup errors. Golden shard 2 also installs Python Stanza and runs `stanzasharp compare` and
  `benchmark` against it, in English and German (`Run_PythonStanza_*`); the other jobs, without Python, allow those to skip. The
  original and converted models are cached, keyed on `ModelDownloader.cs`, `tools/requirements.txt` and
  `tools/stanza_convert.py`.
- `docker` runs `tools/verify-docker.sh`: it packs StanzaSharp, builds [samples/docker](samples/docker/Dockerfile) on
  `mcr.microsoft.com/dotnet/runtime:10.0` and on its chiseled variant, and checks that no libtorch is in the image and
  that each container turns `corpus.txt` into `pipeline.conllu` byte for byte. It reports the image sizes in the job
  summary. `alpine` does the same on `runtime:10.0-alpine` (musl).
- `cross-os (windows-2025)`, `cross-os (macos-26)` (Apple Silicon) and `cross-os (windows-11-arm)` run the same
  full suite, the no-skip check and both `tools/verify-package.ps1` runs (on Windows Arm64 only the managed one:
  `TorchSharp-cpu` has no Arm64 libtorch), against Stanza's `.pt` files without converting them (no Python). Like
  `golden`, each OS runs the suite in three shards (`cross-os-shard`, with golden's split; one shard for a pull request's
  subset), and `cross-os (<os>)` passes only if all of that OS's shards did. Their models are cached per OS, keyed on
  `ModelDownloader.cs`.
- Native AOT and trimming: golden shard 1 (linux-x64) and `cross-os (windows-2025)` shard 3 run `tools/verify-aot.sh`, which
  publishes `tests/StanzaSharp.ManagedCheck` with `PublishAot` (`x86-64-v3`) and with `PublishTrimmed`, fails on any
  IL2xxx/IL3xxx warning or torch file, and runs both binaries against the golden files. Full runs only (in golden also
  pull requests that touch CI or the backends' shared files).
- `linux-arm64` (`ubuntu-26.04-arm`): no libtorch exists for it, so the tests build without one and it runs those
  that need none (the managed cases of the both-backend theories, and the tests marked
  `[Trait("Backend", "Managed")]`), the no-skip check and the managed `verify-package.ps1`.

How much runs depends on the event: a draft pull request, or one that changes only documentation, runs `build-test` and
the other jobs with their steps skipped (they pass in about a minute); a ready pull request runs `build-test`, `golden`, `docker` and `alpine` in full but
only the pipeline and kernel tests on `cross-os` and `linux-arm64`, with no package checks there; a push to `main` and
a release run everything. A pull request that touches no TorchSharp-backend file runs only a smoke subset of the
TorchSharp-backend tests in `golden`.

All eight jobs above are required checks on `main`. `build-test` and `golden` stay separate jobs, not a matrix, so their
check names stay as they are. All upload their `.trx` test results. To reproduce `golden` locally, run `setup.ps1 -Models`, then
`dotnet test -c Release --logger trx --results-directory TestResults` and check that nothing was skipped.

`.github/workflows/benchmark.yml` runs only by hand (Actions > Benchmark > Run workflow): `stanzasharp benchmark`
with Python Stanza on each free hosted runner, in English or German (input `lang`), combined into one table (job summary and artifact `benchmarks`).
Results: [docs/performance.md](docs/performance.md#run-it-yourself).

## Releasing

Unreleased changes are one file each under [`changes/`](changes/README.md). `python tools/changelog.py release X.Y.Z`
folds them into CHANGELOG.md under `## [X.Y.Z] - <date>`; merge that pull request. `.github/workflows/release.yml`
sees a version without a tag, waits for `main`'s CI run of that commit and requires it to have passed (if there is
none, it runs all CI checks itself), waits for approval, packs that version (`StanzaSharp`,
`StanzaSharp.Cuda`, `StanzaSharp.Tool`) and publishes it to nuget.org through NuGet Trusted Publishing. Only then does
it create the tag (`vX.Y.Z`) and a GitHub release (a prerelease when the version has a `-`).

## Changes

See [CHANGELOG.md](CHANGELOG.md), and [`changes/`](changes) for what is merged but not yet released.

## License

Apache License 2.0, the same as Stanza, which this ports. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Stanza's pretrained models are not part of this repository; their licenses vary with the training
data, so check them before redistributing models.
