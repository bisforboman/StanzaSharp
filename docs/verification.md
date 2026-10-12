# Verification

StanzaSharp claims that it gives the same output as Python Stanza 1.15.0 with the same English models. This page
shows how that is checked, on which platforms, and how you can check it on your own text.

## Same output as Python Stanza

The reference output in [`tests/golden/`](../tests/golden) was made by Python Stanza 1.15.0 (torch 2.14.1, CPU;
versions pinned in `tools/requirements.txt`) with `tools/make_golden.py`, from the same model files that
`stanzasharp download` fetches. The downloader checks each file against the MD5 in Stanza 1.15.0's `resources.json`.

The tests compare StanzaSharp's output with it **byte for byte**, with no tolerance:

| What | Input | Compared |
|---|---|---|
| Full pipeline, `default` package (all eight processors) | `corpus.txt` and 12 hand-written `validation*.txt` texts: 13 files, 845 sentences, 8,401 words | the whole CoNLL-U: tokens, sentences, offsets, multi-word tokens, lemmas, UPOS/XPOS/features, heads and relations, constituency trees, sentiment, NER tags |
| Full pipeline, `default_fast` package (seven processors) | the same 13 files | the whole CoNLL-U |
| Each processor on its own | the same 13 files; sentiment also on `sentiment/reviews.txt` (14 files, 886 sentences) | lemmas (868 words through the seq2seq model), dependency parses, entities (706), sentiment labels |
| Input modes, both packages | bulk processing (Stanza's `bulk_process`), pretokenized text, one sentence per paragraph (`tokenize_no_ssplit`), very long tokens | the CoNLL-U |
| Tokenizer edge cases | a paragraph over 1,000 characters, more paragraphs than one batch | tokens, sentences and offsets |
| Parts | MWT expansions of a word list (dictionary and classifier); the dependency parser's tree decoding on tie-heavy score matrices | the expansions and trees |
| Model loading | Stanza's 14 `.pt` checkpoints, read by StanzaSharp's own unpickler | every tensor byte-identical to torch's |

The validation texts are news, academic writing, instructions, social media, dialogue and poetry, technical text,
multilingual names and quotes, contractions, long sentences, unusual whitespace (CRLF, tabs) and characters outside
the BMP (emoji). One known difference is by design: after a character outside the BMP, `start_char`/`end_char` count
UTF-16 code units in StanzaSharp and code points in Stanza, so `validation_nonbmp` is compared without offsets.

Every test above runs on both backends: the managed backend (the default) and the TorchSharp backend
(`StanzaSharp.Cuda`). The managed backend's kernels are also tested on each SIMD path a CPU can take (Vector256 with
AVX2, Vector128, Scalar), so a machine without AVX2 or an Arm64 machine runs code that was checked too.

## Scores and tolerances

The discrete output (every token, tag, lemma, head, tree and label) must match exactly. The scores behind it are
float32 numbers summed in a different order than in PyTorch (StanzaSharp's SIMD kernels, or libtorch's), so their last
bits differ. The tests compare the scores Stanza saved (for the first three corpus sentences; for sentiment, every
sentence) with a tolerance. Largest differences from Python Stanza measured on the managed backend:

| Scores | Largest difference | Tolerance |
|---|---:|---:|
| tokenizer log-probabilities | 1.1e-5 | 1e-4 |
| UPOS logits | 6.1e-5 | 1e-4 |
| dependency arc / label log-probabilities | 2.3e-5 / 3.4e-5 | 1e-4 |
| NER emissions | 1.1e-5 | 1e-4 |
| sentiment logits | 8.2e-5 | 1e-4 (of Stanza's float32 or float64 logits) |
| constituency transition scores | 3.4e-5 (at most 4.6e-5 on any SIMD path) | 1e-3 |

- **Sentiment** is compared with Stanza's float64 logits as well: the classifier is ill-conditioned, and on one
  sentence Stanza's own float32 logits are 1.49e-4 from float64. With the tagger's character-model outputs reused
  (`CharlmCache`, on by default), logits move by up to about 1e-4 more, so those tests use 1e-3; the labels stay
  identical.
- **The lemmatizer**'s log-probabilities are reported, not asserted: no backend comes within 1e-4 of float64 there,
  Stanza's float32 included. The lemmas, the decoding and the edit decisions are compared exactly, and the smallest
  gap between the best and second-best character is 5.6e-3, far above the drift.
- **Near-ties.** A small drift could still flip a decision whose two best options are almost equal. The constituency
  parser's smallest decision margin on the golden sentences is 3.6e-4 over 22,569 steps, against 4.0e-5 between the
  two backends; `ConstituencyTests.NearTies_StayClearOfBackendDrift` fails if a decision differs or the margin drops
  under 3× the drift. On UD English EWT (254,589 words, 684,407 parser decisions; not in the repository) no tree and no
  lemma differed between Python Stanza, the TorchSharp backend and the managed backend.

Per-processor details: [backends.md](backends.md#phase-2-progress). On a GPU the output is identical with TF32 turned
off (`disableTf32: true`); with TF32 on, 3 of 845 golden sentences differed on an RTX 3080 ([gpu.md](gpu.md)).

## Where it is checked

[CI](../.github/workflows/ci.yml) runs on free GitHub-hosted runners. All eight jobs are required checks on `main`.
Test counts are xUnit test cases (theories count once per case) in `main`'s run of 2026-10-11:

| Job | Runner | What it checks |
|---|---|---|
| `build-test` | ubuntu-26.04 (x64) | builds with warnings as errors, checks the code style, runs the 179 tests that need no models |
| `golden` (3 shards) | ubuntu-26.04 (x64) | the whole suite on both backends against downloaded models, original and converted (360 tests); then packs the packages and builds fresh apps from them: `StanzaSharp` alone must reproduce the golden CoNLL-U with nothing of TorchSharp loaded, and `StanzaSharp.Cuda` with `TorchSharp-cpu` must too; then installs the `stanzasharp` tool and downloads models with it |
| `cross-os (windows-2025)` | Windows x64 | the whole suite on both backends, reading Stanza's `.pt` files directly (360 tests), and both package checks |
| `cross-os (macos-26)` | macOS, Apple Silicon | the same (338 tests: Arm64 has no Vector256 path) |
| `cross-os (windows-11-arm)` | Windows on Arm64 | the same (338 tests), and the managed package check |
| `linux-arm64` | ubuntu-26.04-arm | every managed-backend test (219; there is no libtorch for Linux Arm64) and the managed package check |
| `docker` | ubuntu-26.04 | the [Docker sample](../samples/docker/Dockerfile) on `runtime:10.0` and `runtime:10.0-noble-chiseled`: no libtorch in the image, and the container turns `corpus.txt` into the golden CoNLL-U byte for byte |
| `alpine` | ubuntu-26.04 | the same on `runtime:10.0-alpine` (musl) |

Every job that runs the model tests fails if any of them was skipped. The only exceptions are two tests that run
`stanzasharp compare` and `benchmark` against Python Stanza, which CI doesn't install.

A pull request runs a subset on the other operating systems; every push to `main` and every release runs everything
above. Releases are built from a commit that passed all of it.

## Check it yourself

`stanzasharp compare` runs Python Stanza 1.15.0 and StanzaSharp on your own text with the same model files and says
whether the CoNLL-U is identical, or shows the first difference:

```
dotnet tool install -g StanzaSharp.Tool
pip install stanza==1.15.0
stanzasharp download models/stanza/en
stanzasharp compare my-text.txt          # --package default_fast, --processors LIST
```

It checks the model files' MD5s on both sides and converts the offsets after non-BMP characters before comparing.
Exit code 0 means identical. More: [the tool's README](../src/StanzaSharp.Tool/README.md#check-it-yourself).

## Speed on the free runners

`stanzasharp benchmark` times StanzaSharp and Python Stanza on the same text with the same models, on your machine. The
manual [Benchmark workflow](../.github/workflows/benchmark.yml) runs it on every free GitHub-hosted runner. Its first
run ([38150941695](https://github.com/bisforboman/StanzaSharp/actions/runs/38150941695), 2026-10-11, all eight
processors on 2,009 words, each side's default threads) gave identical CoNLL-U on every runner with Python Stanza, and:

| Runner | StanzaSharp words/s | Python Stanza words/s | Speed-up | Per call (one sentence), median | Peak memory |
|---|---:|---:|---:|---:|---:|
| ubuntu-24.04 (x64, 2 cores) | 246 | 75 | 3.3× | 69 / 249 ms | 1,143 / 3,471 MB |
| ubuntu-24.04-arm (4 cores) | 228 | 81 | 2.8× | 81 / 678 ms | 1,166 / 3,643 MB |
| windows-2025 (x64, 2 cores) | 238 | 63 | 3.8× | 74 / 327 ms | 1,120 / 3,600 MB |
| windows-11-arm (4 cores) | 221 | (install failed) | | 86 ms | 1,016 MB |
| macos-15 (Apple M1, 3 vCPUs) | 183 | 60 | 3.0× | 766 / 339 ms | 877 / 2,454 MB |

The macOS per-call time was the runner's Spotlight indexing, not StanzaSharp: with indexing off it is 94 ms
([performance.md, round 12](performance.md#results-round-12-concurrent-calls-and-the-managed-thread-pool)), and the
workflow now turns it off. That run predates CI's move to ubuntu-26.04 and macos-26. The runners are shared virtual
machines, so compare numbers within one run. More, including how to run it yourself:
[performance.md](performance.md#run-it-yourself).
