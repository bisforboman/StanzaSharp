# StanzaSharp.Tool

The `stanzasharp` command downloads Stanza's English models for [StanzaSharp](https://www.nuget.org/packages/StanzaSharp),
for example in a Dockerfile or a CI job. Each file is checked against its published MD5, and files already present are
kept. It needs no native libtorch.

```
dotnet tool install -g StanzaSharp.Tool
stanzasharp download models/stanza/en
stanzasharp download models/stanza/en --package default_fast --processors tokenize,mwt,pos
```

- `DIR`: where to put the models (default `models/stanza/en`), in the layout `Pipeline.Load(DIR)` reads.
- `--package NAME`: `default` (the default) or `default_fast`.
- `--processors LIST`: only the models these processors need, e.g. `tokenize,mwt,pos,lemma`.

In a Dockerfile, `dotnet tool install --tool-path /tools StanzaSharp.Tool` avoids touching the global tool path.

## Check it yourself

`stanzasharp compare FILE` runs Python Stanza 1.15.0 and StanzaSharp on your own text with the same model files and
reports whether their CoNLL-U output is identical. It needs Python with `pip install stanza==1.15.0`.

```
stanzasharp compare my-text.txt [--package NAME] [--processors LIST] [--models DIR] [--python PATH]
```
```
Stanza 1.15.0 (Python) and StanzaSharp on my-text.txt
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

- `--models DIR`: Stanza's `.pt` files as `download` writes them (default `models/stanza/en`). The folder must be
  named `en`, since Python Stanza looks for `<dir>/en/<processor>/<name>.pt`. Both sides load the same files, and
  StanzaSharp checks each against Stanza 1.15.0's MD5.
- `--python PATH`: the Python with Stanza installed (default `python`).
- Both sides read the file as the same string (UTF-8, or the encoding its byte order mark names) and process it as one
  document.
- After characters outside the BMP (emoji, for example), StanzaSharp's `start_char`/`end_char` count UTF-16 code units
  and Stanza's count code points. `compare` converts StanzaSharp's offsets to code points before comparing and says so.
- Exit code: 0 identical, 1 different, 2 a usage or setup problem (no Python or Stanza, another Stanza version,
  missing models).

Source, issues and docs: https://github.com/bisforboman/StanzaSharp
