# StanzaSharp

[Stanza](https://github.com/stanfordnlp/stanza)'s English NLP pipeline in .NET:
- tokenization and sentence splitting
- multi-word token expansion ("don't" → "do" + "n't")
- part-of-speech tags and morphological features
- lemmas
- dependency parsing
- named entities (OntoNotes types: PERSON, ORG, GPE, DATE, ...)
- constituency parsing
- sentence sentiment (negative, neutral, positive)

It runs Stanza's own pretrained models in plain .NET code, with no Python and no native libraries. The output is
identical to Python Stanza 1.15.0 on its golden test data.

## Install

```
dotnet add package StanzaSharp
```

That is all: the models run on StanzaSharp's managed backend, C# kernels using the CPU's SIMD instructions. On
the full pipeline it is about twice as fast as the TorchSharp backend on 8 threads and needs less memory, with the
same output.

**TorchSharp backend.** Still selectable in 0.5 (and the way to a GPU, below); it leaves the main package in 1.0.
Add the native libtorch yourself, `TorchSharp-cpu` (or one platform's `libtorch-cpu-<rid>` 2.10.0 package; on
Windows on Arm64 `libtorch-cpu-win-arm64`, since `TorchSharp-cpu` has none), and select it:

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Backend = PipelineBackend.TorchSharp });
```

- Its version must match the `TorchSharp` version StanzaSharp depends on; otherwise the build warns with `STANZA001`.
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

## Download the models

The models (about 600 MB) come from Stanza's Hugging Face repository. Download them once:

```csharp
await ModelDownloader.DownloadAsync("models/stanza/en");                            // all, ~600 MB
await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
```

Each file is checked against its published MD5, and files already present are kept. `Pipeline.Load` itself
never touches the network.

**Command line:** the `stanzasharp` .NET tool (package `StanzaSharp.Tool`, about 1 MB, no native libraries) downloads
them too, e.g. in a Dockerfile or CI:

```
dotnet tool install -g StanzaSharp.Tool
stanzasharp download models/stanza/en --processors tokenize,mwt,pos,lemma   # --package default_fast for that package
```

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
- Add `sentiment` to the list for `sentence.Sentiment`: 0 negative, 1 neutral, 2 positive.
- `Conllu.Write(doc)` gives CoNLL-U in Stanza's format.
- `nlp.Process(texts)` with any `IEnumerable<string>` processes many texts at once, like Stanza's `bulk_process`:
  one `Document` per text, much faster for short texts.
- `nlp.Process(new[] { new[] { "Hello", "world", "." } })` takes text that is already split into sentences and
  tokens, like Stanza's `tokenize_pretokenized=True`.
- `Process` is thread-safe: share one pipeline between threads. Every overload takes a `CancellationToken`.
  One call on a short sentence takes about 28 ms on 8 threads (`default_fast`: 14 ms). On short texts, about two calls
  in flight per pipeline give the most throughput; more evict each other's weights from the CPU cache. Set
  `MaxConcurrentCalls = 2` in `PipelineOptions` to make extra callers wait their turn.
- Memory: a call's peak is set by its largest batch (the tagger pads up to 250 sentences to the longest), so to
  bound it, call `Process` on parts of about 1,000 words split at blank lines. The annotations stay the same; with
  `tokenize,mwt,pos,constituency` on 15,000 words the peak drops from 1.7 to 1.2 GB for about 15% more time.
- Memory between calls: on the managed backend (the default) the models and the scratch buffers of a call's largest
  batches stay on the .NET GC heap for the next call, so the working set stays near the call's peak. To give it back
  when a service goes idle, call `GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true)` a
  minute or more after the last call (pooled buffers are released only then); the process drops to about its size
  after loading. On the TorchSharp backend, on Linux (glibc), a call of 1,000 words or more returns the memory it
  freed to the OS (`malloc_trim`); on Windows libtorch's allocator keeps it, and the environment variable
  `MIMALLOC_PURGE_DELAY=0`, set before libtorch loads, returns it for about 20% more time.
- `PipelineOptions`:
  - `Backend`: `PipelineBackend.Managed` (the default) or `PipelineBackend.TorchSharp`.
  - `Threads`: threads per operation, set process-wide at Load: the managed backend's one thread pool (shared by
    concurrent calls), or libtorch's intra-op threads. By default at most `Environment.ProcessorCount`, which
    respects a container's CPU quota.
  - `SplitSentences = false`: one sentence per paragraph (Stanza's `tokenize_no_ssplit`).
  - `VerifyChecksums`: checks the `.pt` model files against Stanza's MD5s before loading.
  - `Logger`: an `ILogger` for load and processing times.
- `Processor.Tokenize`, ..., `Processor.Ner` are constants for the processor list.

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

## GPU

On the TorchSharp backend: reference `TorchSharp-cuda-windows` (or `TorchSharp-cuda-linux`), then
`Pipeline.Load(dir, new PipelineOptions { Backend = PipelineBackend.TorchSharp, Device = torch.CUDA, DisableTf32 = true })`.
`DisableTf32` gives output identical to the CPU by turning TF32 off process-wide; with TF32 on (libtorch's
default), a few near-tie decisions can differ. `Device` and `DisableTf32` are obsolete: GPU support moves to a
separate `StanzaSharp.Cuda` package, and both leave this package in 1.0. See the repository's docs/gpu.md.

## License

Apache 2.0, like Stanza, whose code this ports (see NOTICE). Stanza's models are not included. Their
licenses vary with the training data.

Source, issues and docs: https://github.com/bisforboman/StanzaSharp
