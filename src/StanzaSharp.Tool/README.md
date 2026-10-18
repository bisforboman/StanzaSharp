# StanzaSharp.Tool

The `stanzasharp` command for [StanzaSharp](https://www.nuget.org/packages/StanzaSharp): it downloads Stanza's English
and German models (for example in a Dockerfile or a CI job), compares StanzaSharp's output with Python Stanza's on your own text,
and benchmarks both on your machine. Each downloaded file is checked against its published MD5, and files already
present are kept. It needs no native libtorch.

```
dotnet tool install -g StanzaSharp.Tool
stanzasharp download models/stanza/en
stanzasharp download models/stanza/en --package default_fast --processors tokenize,mwt,pos
stanzasharp download --lang de                     # German, into models/stanza/de
```

- `DIR`: where to put the models (default `models/stanza/<lang>`), in the layout `Pipeline.Load(DIR)` reads.
- `--lang CODE`: `en` (the default) or `de` (German), for `download`, `compare` and `benchmark` alike.
- `--package NAME`: `default` (the default) or `default_fast`; or a treebank package (tokenize to depparse only):
  English `combined`, `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`, `eslspok`, `craft`, `genia` and `mimic`, German `combined`, `gsd` and
  `hdt`.
- `--processors LIST`: only the models these processors need, e.g. `tokenize,mwt,pos,lemma`.

In a Dockerfile, `dotnet tool install --tool-path /tools StanzaSharp.Tool` avoids touching the global tool path.

## Check it yourself

`stanzasharp compare FILE` runs Python Stanza 1.15.0 and StanzaSharp on your own text with the same model files and
reports whether their CoNLL-U output is identical. It needs Python with `pip install stanza==1.15.0`.

```
stanzasharp compare my-text.txt [--lang CODE] [--package NAME] [--processors LIST] [--models DIR] [--python PATH]
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

A difference shows the sentence and the two lines:

```
Different: 1 of 68 sentences differ. The first difference, line 384 of the CoNLL-U:
  # sent_id = 18
  # text = "That said, we aren't assuming it'll last forever."
  Stanza:      3	said	said	VERB	VBD	Mood=Ind|Person=3|Tense=Past|VerbForm=Fin	8	parataxis	_	SpaceAfter=No|start_char=1462|end_char=1466|ner=O
  StanzaSharp: 3	said	say	VERB	VBD	Mood=Ind|Person=3|Tense=Past|VerbForm=Fin	8	parataxis	_	SpaceAfter=No|start_char=1462|end_char=1466|ner=O
```

- `--lang CODE`: the language, `en` (default) or `de`. Stanza runs `stanza.Pipeline(CODE)` with that language's entry
  of Stanza 1.15.0's resources.json (shipped with the tool, so nothing is downloaded), StanzaSharp loads with
  `Language = CODE`. A treebank package (e.g. `--package ewt`, or German `gsd`) runs only its processors.
- `--models DIR`: Stanza's `.pt` files as `download` writes them (default `models/stanza/<lang>`), in a folder of any
  name. Both sides load the same files, and
  StanzaSharp checks each against Stanza 1.15.0's MD5.
- `--python PATH`: the Python with Stanza installed (default `python`). The transformer models (`--package
  default_accurate`, the `_electra-large` models) also need `pip install transformers==5.20.1 tokenizers==0.23.3
  safetensors==0.8.0 peft==0.21.2`; Stanza reads the transformer from `DIR/transformer/`, where `download` puts it.
  Without them `compare` says what to install (exit code 2) and `benchmark` runs StanzaSharp alone.
- Both sides read the file as the same string (UTF-8, or the encoding its byte order mark names) and process it as one
  document.
- After characters outside the BMP (emoji, for example), StanzaSharp's `start_char`/`end_char` count UTF-16 code units
  and Stanza's count code points. `compare` converts StanzaSharp's offsets to code points before comparing and says so.
- Exit code: 0 identical, 1 different, 2 a usage or setup problem (no Python or Stanza, another Stanza version,
  missing models).

## Benchmark it on your machine

`stanzasharp benchmark` times StanzaSharp on a built-in English text (our own test texts: 1,500 words by whitespace,
2,009 after tokenizing), or with `--lang de` a German one (our German test texts, cut the same way) and, if Python Stanza 1.15.0 is available, Stanza on the same text with the same models,
alternating their runs (warm-up runs on the whole text until each side's times settle, at most 5, then the median of 3). It reports load time, each processor, words/s, single-sentence calls (median and p90) and
peak memory, and prints a Markdown block with the CPU, cores, RAM, OS, .NET version, SIMD path and StanzaSharp version,
ready to paste into an issue. With Python it takes about four minutes on 4 cores. Results from GitHub's free runners:
[docs/performance.md](https://github.com/bisforboman/StanzaSharp/blob/main/docs/performance.md#run-it-yourself).

```
stanzasharp benchmark [--lang CODE] [--package NAME] [--processors LIST] [--models DIR] [--threads N] [--words N] [--python PATH] [--quick] [--json FILE]
```

- `--threads N`: the same thread count on both sides. By default each side uses its own default (StanzaSharp all
  logical processors, torch the physical cores); the block shows both.
- `--words N`: a longer or shorter text, counted by whitespace before tokenizing (the block counts words after it).
  `--quick`: 300 words, one run and 10 calls, a smoke test.
- Without Python Stanza (or with converted models, which Stanza can't read), only StanzaSharp's column is shown, with
  the reason. With it, the block also says whether both sides' CoNLL-U was identical.
- `--json FILE`: the results as JSON too.

Source, issues and docs: https://github.com/bisforboman/StanzaSharp
