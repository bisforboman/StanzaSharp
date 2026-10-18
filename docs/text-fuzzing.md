# Differential text fuzzing

The golden tests compare StanzaSharp with Python Stanza on hand-written texts. `tools/text_fuzz.py` looks for the
inputs nobody thought to write: it generates adversarial texts from a seed, runs each through Python Stanza 1.15.0 and
StanzaSharp, and reports every text whose CoNLL-U is not byte-identical, or on which either side throws.

## Method

**Texts.** Each text starts from real sentences (the lines of `tests/golden/*.txt`, or `tests/golden/de/*.txt` for
German) and splices fragments into them, at word boundaries mostly and inside words sometimes:

- control characters (`\x00`, `\x1b`, `\x7f`, U+001C..U+001F, U+0080..U+009F, BOM, U+FFFE/U+FFFF), CR, CRLF, lone
  CR, NEL, U+2028/U+2029, every kind of space (NBSP, thin, ideographic, U+180E) and paragraph separators with
  whitespace between them;
- zero-width characters (ZWSP, ZWJ, ZWNJ, word joiner, soft hyphen, CGJ), emoji with ZWJ sequences, skin tones, flags,
  tag sequences, variation selectors and keycaps;
- bidi marks and embeddings, Hebrew and Arabic words, combining marks (stacked too), NFD/NFKD/NFKC forms of
  precomposed words, characters outside the BMP (Gothic, mathematical letters and digits, CJK Extension B, Deseret,
  hieroglyphs, plane 14/16), mixed scripts (Cyrillic, CJK, Hangul, Thai, Devanagari, Georgian, Greek, Armenian,
  Ethiopic, Cherokee, a homoglyph), fullwidth forms, No/Nl characters next to words (`x²`, `Ⅻ`, `①`);
- URLs, e-mail addresses and near misses for `MASK_RE` (`www.`, `.gov`, `://`, `a@b`), Windows and Unix paths,
  abbreviations (`e.g.`, `U.S.A.`, `z.B.`);
- punctuation runs for `simplify_punct` (`?!?!?!`, 2 to 30 repeats of any punctuation), quotes of every style,
  contractions and MWT candidates (English, or German `zum`/`im`/`geht's`), numbers, dates, times, currencies;
- one random code point from any plane (assigned or not; no lone surrogates, which UTF-8 text can't hold).

A text is then one of: spliced sentences in 1 to 5 paragraphs (with `\n\n`, or any of the newline variants between
them); real sentences around a token of `max_seqlen` ± 2 code points (100 for most treebank tokenizers, 200 English,
300 `partut`, 400 German) or of 1,000 (letters, mixed case and digits, a URL, a non-BMP mix, hyphenated); one
paragraph of 1,000 to 4,000 characters (past the 1,000-unit window); 30 to 80 paragraphs with empty, whitespace-only
and one-character ones among them; or fragments alone. The generator is seeded (`random.Random(f"{lang}:{seed}")`),
so a seed always gives the same texts.

**Running.** One Stanza pipeline is loaded in the script's process and one StanzaSharp pipeline in a child process
(`samples/StanzaSharp.Benchmark text-fuzz`, one JSON string per stdin line, one JSON line back), so each text costs
one `Process` call on each side and nothing is loaded per text. A C# exception is a result (`C# error`, with its stack
trace); a crash of the child process (stack overflow, fail-fast) too, and the child is restarted. StanzaSharp's
`start_char`/`end_char` are converted from UTF-16 indices to code points before comparing, so texts outside the BMP
are compared with their offsets, byte for byte.

**Shrinking.** A differing text is shrunk by delta debugging over its code points (both sides rerun on every
candidate) while the first differing field (FORM, UPOS, `start_char`, a comment, ...) stays the same, so the report
gives a small repro next to the original. The kind is kept, not the exact lines, so a shrunk text can show the same
bug at a different place.

## Results (2026-10-16)

Managed backend, after the fixes below (the first tokenizer run, before them, found finding 1 at its 314th text):

| Language, package | Processors | Seeds | Texts | Characters | Differences |
|---|---|---|---|---|---|
| English `default` | tokenize, mwt | 3, 10, 41 | 4,700 | 5.06 M | 0 |
| English `default` | all eight | 1, 2, 14, 40 | 418 | 0.46 M | 0 |
| English `default_fast` | all seven | 13 | 80 | 0.08 M | 0 |
| English `ewt` | tokenize, mwt | 15 | 800 | 0.86 M | 0 |
| English `ewt` | tokenize to depparse | 11 | 200 | 0.22 M | 0 |
| English `partut` | tokenize, mwt | 16 | 800 | 0.86 M | 1: finding 2 |
| English `partut` | tokenize to depparse | 12 | 200 | 0.21 M | 0 |
| German `default` | tokenize, mwt | 20, 44 | 3,000 | 3.48 M | 0 |
| German `default` | all eight | 21 | 60 | 0.08 M | 0 |
| German `default_fast` | all seven | 45 | 100 | 0.11 M | 0 |
| German `gsd` | tokenize to depparse | 42 | 300 | 0.36 M | 6: finding 3 |
| German `hdt` | tokenize to depparse | 43 | 300 | 0.40 M | 0 |

10,958 texts, 12.2 million characters. Neither side crashed or threw on any text. What the fuzzer found:

1. **Characters outside the BMP in the tokenizer's regexes (port bug, fixed).** Stanza runs its `STRUCTURAL_FEATURES`
   (English `labeled_field`, `phone_id`, `date_pattern`, `currency`) and `MASK_RE` (e-mails and URLs) over the
   paragraph's code points; StanzaSharp ran them over the UTF-16 string. A surrogate pair is no word character to
   .NET, so in `𠀁C:` (CJK Extension B, a letter to Python) `\b` matched before `C` and `labeled_field` marked `C:`,
   which made the model keep `𠀁C` as one token (Stanza: `𠀁`, `C`, `:`). The same root cause had a second effect,
   found by hand-written texts once the first was understood: an emoji is two UTF-16 units, so it satisfied a URL's
   `[^\s"]{2,}` tail alone, and `example.com.😀` stayed one token where Stanza splits `example.com`, `.`, `😀`
   (English and German). The regexes now run over one char per code point (`Tokenizer.Paragraph.CodePoints`:
   a non-BMP digit becomes `٠`, another letter or number `ª`, anything else U+FFFD, which every pattern treats as it
   treats the original), and a match's index is a unit index. Regression: `tests/golden/fuzz.json` and
   `tests/golden/de/fuzz.json` (`make_golden.py --fuzz-only`), checked by `TokenizerTests.Pipeline_FuzzFindings_MatchGolden`
   for both packages and backends.
2. **A text without sentences (harness, fixed).** For an empty or whitespace-only text (`"  "`), Stanza's side
   wrote `"{:C}\n"`, a lone newline, where `Conllu.Write` (like Stanza's CoNLL file writer) gives an empty string. Not a
   port difference, but `stanzasharp compare` reported it as one too; `compare.py` now writes an empty CoNLL-U then
   (`CompareCommandTests.Run_PythonStanza_BlankTextIsIdentical`).
3. **float32 noise on degenerate sentences (not a port bug; accepted).** Six German `gsd` texts differ in tags, heads or
   relations, all of them sentences of 100 to 700 tokens made of a hyphenated run (`ab-ab-ab-…`, the long-token
   generator's hyphen kind). Tokenization is identical; the tagger and parser are not. The TorchSharp backend differs
   from Stanza on the same six texts (on the originals in 8 to 405 lines, the managed backend in 32 to 408), each
   backend in other places. On the first one, word 130's UPOS logits (`abela`: NOUN over PROPN) are 1.9e-3 apart in
   Stanza's float64, and Stanza's own float32 result is 3.0e-3 from its float64 one, 150 times its float32 error on
   ordinary text (2e-5): on such input any other summation order, Stanza's on another CPU included, can flip a
   near-tie. The managed result is 3.2e-3 from float64 (the other
   way round), TorchSharp's 6e-5. No other package showed it (`hdt`, `ewt` and `partut` ran the same kinds of text).
   Owner's decision (2026-10-16): accepted and documented, nothing changed; the fuzzer stays local (no CI run), its
   findings live in `fuzz.json`, which every golden run checks.

## Results (2026-10-17): the models of 1.10 and 1.11

Every model 1.10 and 1.11 added through `PipelineOptions.Models` (`--model`), plus `PipelineOptions.Pretagged`
(`--pretagged`), on a 4-core cloud Linux container (Intel Xeon with AVX-512, so the managed backend ran Vector512),
two runs side by side with 2 threads each. Stanza's torch was PyPI's 2.14.1 build (CUDA, run on the CPU). Each row
has a new seed. The domain NER models ran tokenize, mwt and ner (ner reads only tokens). The lemmatizer rows used their
treebank's `_charlm` tokenizer too, and `lemma=combined_charlm` ran with `--words` (`'s`, `'S`, `’s`, `her`, `Her`,
`HER`, `She's`, `her's`, `it's`: about 2,900 `'s` and 830 `her` in its 400 texts, for the contextual classifiers).

| Model | Processors | Backend | Seed | Texts | Characters | Differences |
|---|---|---|---|---|---|---|
| English `default` + `ner=conll03_charlm` | tokenize, mwt, ner | managed | 101 | 200 | 0.21 M | 0 |
| English `default` + `ner=radiology` | tokenize, mwt, ner | managed | 102 | 200 | 0.22 M | 0 |
| English `default` + `ner=anatem` | tokenize, mwt, ner | managed | 103 | 200 | 0.24 M | 0 |
| English `default` + `ner=bc4chemd` | tokenize, mwt, ner | managed | 104 | 200 | 0.24 M | 0 |
| English `default` + `ner=bc5cdr` | tokenize, mwt, ner | managed | 105 | 200 | 0.22 M | 0 |
| English `default` + `ner=bionlp13cg` | tokenize, mwt, ner | managed | 106 | 200 | 0.19 M | 0 |
| English `default` + `ner=jnlpba` | tokenize, mwt, ner | managed | 107 | 200 | 0.20 M | 0 |
| English `default` + `ner=linnaeus` | tokenize, mwt, ner | managed | 108 | 200 | 0.24 M | 0 |
| English `default` + `ner=ncbi_disease` | tokenize, mwt, ner | managed | 109 | 200 | 0.21 M | 0 |
| English `default` + `ner=s800` | tokenize, mwt, ner | managed | 110 | 200 | 0.20 M | 0 |
| English `default` + `ner=i2b2` | tokenize, mwt, pos, lemma, constituency, depparse, sentiment, ner | managed | 111 | 100 | 0.12 M | 0 |
| English `default` + `tokenize=combined_charlm` | tokenize, mwt | managed | 121 | 600 | 0.65 M | 0 |
| English `ewt` + `tokenize=ewt_charlm` | tokenize, mwt | managed | 122 | 500 | 0.52 M | 0 |
| English `gum` + `tokenize=gum_charlm` | tokenize, mwt | managed | 123 | 500 | 0.54 M | 0 |
| English `lines` + `tokenize=lines_charlm` | tokenize, mwt | managed | 124 | 500 | 0.50 M | 0 |
| English `partut` + `tokenize=partut_charlm` | tokenize, mwt | managed | 125 | 500 | 0.54 M | 0 |
| English `atis` + `tokenize=atis_charlm` | tokenize, mwt | managed | 126 | 500 | 0.51 M | 0 |
| English `childes` + `tokenize=childes_charlm` | tokenize, mwt | managed | 127 | 500 | 0.59 M | 0 |
| English `eslspok` + `tokenize=eslspok_charlm` | tokenize | managed | 128 | 500 | 0.53 M | 0 |
| English `default` + `lemma=combined_charlm` | tokenize, mwt, pos, lemma | managed | 131 | 400 | 0.43 M | 0 |
| English `ewt` + `tokenize=ewt_charlm,lemma=ewt_charlm` | tokenize, mwt, pos, lemma | managed | 132 | 200 | 0.25 M | 0 |
| English `gum` + `tokenize=gum_charlm,lemma=gum_charlm` | tokenize, mwt, pos, lemma | managed | 133 | 200 | 0.20 M | 0 |
| English `lines` + `tokenize=lines_charlm,lemma=lines_charlm` | tokenize, mwt, pos, lemma | managed | 134 | 200 | 0.22 M | 0 |
| English `partut` + `tokenize=partut_charlm,lemma=partut_charlm` | tokenize, mwt, pos, lemma | managed | 135 | 200 | 0.21 M | 0 |
| English `atis` + `tokenize=atis_charlm,lemma=atis_charlm` | tokenize, mwt, pos, lemma | managed | 136 | 200 | 0.22 M | 0 |
| English `childes` + `tokenize=childes_charlm,lemma=childes_charlm` | tokenize, mwt, pos, lemma | managed | 137 | 200 | 0.22 M | 0 |
| English `ewt` + `pos=ewt_nocharlm,depparse=ewt_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 141 | 150 | 0.18 M | 0 |
| English `gum` + `pos=gum_nocharlm,depparse=gum_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 142 | 150 | 0.16 M | 0 |
| English `lines` + `pos=lines_nocharlm,depparse=lines_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 143 | 150 | 0.16 M | 0 |
| English `partut` + `pos=partut_nocharlm,depparse=partut_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 144 | 150 | 0.16 M | 0 |
| English `atis` + `pos=atis_nocharlm,depparse=atis_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 145 | 150 | 0.15 M | 0 |
| English `childes` + `pos=childes_nocharlm,depparse=childes_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 146 | 150 | 0.16 M | 0 |
| English `eslspok` + `pos=eslspok_nocharlm,depparse=eslspok_nocharlm` | tokenize, pos, lemma, depparse | managed | 147 | 150 | 0.15 M | 0 |
| English `mimic` + `depparse=mimic_nocharlm` | tokenize, pos, lemma, depparse | managed | 148 | 150 | 0.15 M | 1: finding 5 |
| German `default` + `ner=conll03` | tokenize, mwt, ner | managed | 151 | 200 | 0.24 M | 0 |
| German `default` + `tokenize=combined_charlm` | tokenize, mwt | managed | 152 | 500 | 0.60 M | 0 |
| German `gsd` + `tokenize=gsd_charlm` | tokenize, mwt | managed | 153 | 500 | 0.57 M | 0 |
| German `hdt` + `tokenize=hdt_charlm` | tokenize, mwt | managed | 154 | 500 | 0.62 M | 0 |
| German `combined` + `tokenize=combined_charlm,lemma=combined_charlm` | tokenize, mwt, pos, lemma | managed | 155 | 200 | 0.24 M | 0 |
| German `gsd` + `tokenize=gsd_charlm,lemma=gsd_charlm` | tokenize, mwt, pos, lemma | managed | 156 | 200 | 0.25 M | 0 |
| German `hdt` + `tokenize=hdt_charlm,lemma=hdt_charlm` | tokenize, mwt, pos, lemma | managed | 157 | 200 | 0.25 M | 0 |
| German `gsd` + `pos=gsd_nocharlm,depparse=gsd_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 158 | 150 | 0.17 M | 1: finding 4 |
| German `hdt` + `pos=hdt_nocharlm,depparse=hdt_nocharlm` | tokenize, mwt, pos, lemma, depparse | managed | 159 | 150 | 0.21 M | 0 |
| English `default` (pretagged) | depparse | managed | 161 | 100 | 0.11 M | 0 |
| English `default` (pretagged) | lemma, depparse, constituency | managed | 162 | 100 | 0.11 M | 0 |
| English `default_fast` (pretagged) | lemma, depparse | managed | 163 | 100 | 0.12 M | 0 |
| German `default` (pretagged) | lemma, depparse, constituency | managed | 164 | 60 | 0.07 M | 0 |
| English `default` + `ner=bc5cdr` | tokenize, mwt, ner | TorchSharp | 171 | 60 | 0.06 M | 0 |
| English `ewt` + `tokenize=ewt_charlm` | tokenize, mwt | TorchSharp | 172 | 150 | 0.18 M | 0 |
| English `default` + `lemma=combined_charlm` | tokenize, mwt, pos, lemma | TorchSharp | 173 | 80 | 0.10 M | 0 |
| German `gsd` + `pos=gsd_nocharlm,depparse=gsd_nocharlm` | tokenize, mwt, pos, lemma, depparse | TorchSharp | 174 | 40 | 0.05 M | 0 |

51 runs, 12,290 texts, 13.6 million characters. Neither side crashed or threw on any text. Two texts differ, both float32 noise on degenerate input
(findings 4 and 5). Three harness bugs, all in the new options or package defaults, were fixed along the way:
mwt-less packages (`eslspok`, `craft`, `genia`, `mimic`) got mwt in their default processor list, `--pretagged`
converted the input document's offsets (already code points) as if they were UTF-16, and `--pretagged` dropped XPOS
where the constituency parser reads it. StanzaSharp throws for a missing tag there where Stanza parses an empty one, a
deliberate difference (CLAUDE.md, "Pretagged"); the generator now keeps XPOS when constituency runs.

4. **German `gsd` + `pos=gsd_nocharlm` (float32 noise; not a port bug, nothing changed).** One text of 150: a
   sentence starting with a 97-letter random word (the long-token generator's letters). Its FEATS `Number` is `Plur`
   in Stanza and TorchSharp, `Sing` in the managed backend. Shrunk:
   `arfyordihepegvuujnueiarradbquujtpfyybmsjgyllchutzgystzvjyeawbczmvzgekvgbabdwdahmmtltdxpcznnxterss, unProtokoll:
   sagt man oft so.` The tagger's character LSTM (`Nn.CharacterModel`, unidirectional, hidden 400) runs 97 steps
   over that word, and its output is 0.58 away from float64 in Stanza's own float32 (ordinary words: under 1e-4).
   That error reaches the feats logits: Stanza float32 vs float64 differ by 2.2 there, while the `Plur`/`Sing`
   margin is 0.39 (float64) to 0.67 (float32). Any other summation order can flip it. On the unshrunk text the
   managed result (`Hinters`: `Case=Nom|Number=Sing`, no `Gender`) is Stanza's own float64 result, and Stanza's
   float32 `Gender=Neut` has a margin of 0.07. Same class as finding 3, on the same treebank.
5. **English `mimic` (float32 noise; not a port bug, nothing changed).** One text of 150: a sentence around a
   300-character token (`https://example.com/aaa…`), whose UPOS is `X` in Stanza and `PROPN` here. It comes from
   `mimic`'s own `_charlm` tagger (in the package since 1.10, fuzzed here for the first time), not from the new
   `mimic_nocharlm` parser. The mimic charlms (1024 hidden, 101 characters) run over a run of a hundred and more
   `a`s. On that sentence Stanza's float32 UPOS logits are 4.0 to 5.8 from float64 (median over ordinary words:
   7e-6), and Stanza's own float64 tags it `PROPN`, as the managed backend does. On the shrunk text
   (`ro tpsec` + 150 `a`s) TorchSharp differs from Stanza too, at the other word; float64 differs from float32 at a
   third place.

The four TorchSharp runs, the four pretagged runs and every `_charlm` tokenizer and lemmatizer found nothing.
Pretagged input is only Stanza-tagged text with some fields removed: documents a person wrote by hand (odd tags,
empty sentences) are not generated.

Not a port difference, and not seen in these runs, but possible with random code points: Python 3.10's `unicodedata`
is Unicode 13, .NET 10's is newer, so a character assigned since then is a letter (or upper case, or a space) on one
side only. Such a difference would be the platform's, not the port's.

## Run it yourself

It needs `tools/.venv` (tools/requirements.txt) and Stanza's `.pt` files in `models/stanza/<lang>` (the same files
`stanzasharp download` fetches, for the packages you fuzz):

```
dotnet build samples/StanzaSharp.Benchmark -c Release
tools/.venv/Scripts/python tools/text_fuzz.py run --seed 1 --count 200 --processors tokenize,mwt --report fuzz.json
tools/.venv/Scripts/python tools/text_fuzz.py run --lang de --package default --seed 2 --count 50
tools/.venv/Scripts/python tools/text_fuzz.py run --package ewt --seed 3 --count 100 --backend torch
tools/.venv/Scripts/python tools/text_fuzz.py generate --seed 1 --count 5        # look at the texts only
tools/.venv/Scripts/python tools/text_fuzz.py texts cases.json --no-shrink       # a JSON list of your own texts
tools/.venv/Scripts/python tools/text_fuzz.py run --processors tokenize,ner --model ner=bc5cdr --seed 4 --count 100
tools/.venv/Scripts/python tools/text_fuzz.py run --processors tokenize,mwt,pos,lemma --model lemma=combined_charlm --words "'s,'S,her,Her,HER"
tools/.venv/Scripts/python tools/text_fuzz.py run --pretagged --processors lemma,depparse,constituency --seed 5 --count 50
```

On Linux and macOS the venv's Python is `tools/.venv/bin/python`. The Stanza side reads `resources.json` from the
models folder when Stanza downloaded it, else the copy `stanzasharp compare` ships
(`src/StanzaSharp.Tool/stanza_resources_<lang>.json`), so files fetched by `stanzasharp download` are enough.

- `--model P=NAME` (repeatable or comma-separated) replaces a processor's model, as `PipelineOptions.Models` and
  `stanzasharp compare --model` do: Stanza gets the processor → model map StanzaSharp loaded as its processors dict.
- `--words W,W,...` splices these words in as about four fragments in ten (for a model that decides on particular
  words, such as `lemma=combined_charlm`'s `'s` and `her`). Without it a seed gives the texts it always gave.
- `--pretagged` fuzzes `PipelineOptions.Pretagged` (Stanza's `pretagged=True`): each text is tagged by Stanza's
  tokenize, mwt, pos and lemma (the language's `default` package). About one word in ten then loses its lemma and one
  in twenty its XPOS (seeded by the text; UPOS stays). Both sides then annotate that CoNLL-U document with
  `--processors` (Stanza: `CoNLL.conll2doc` and `nlp.process`; StanzaSharp: `Conllu.Read` and `Process(Document)`).

`--processors` defaults to the package's processors (all eight for `default`). Each side uses 4 threads
(`--threads`; 0 for their defaults). The exit code is 1 if anything differed; `--report` writes every finding with its
original and shrunk text (`--shrink-budget`, 300 candidates by default, bounds the shrinking: a long text through the
whole pipeline costs seconds a candidate). Speed on a Ryzen 7 5800X with another run beside it: tokenize and mwt 8 to
12 texts (9,000 to 14,000 characters) a second, Stanza taking 90% of the time; the treebank packages about 3 s a text,
`default_fast` 1.6 s, all eight processors about 5 s, mostly Stanza's.

No run is in the pull-request CI: Python Stanza runs only in golden-shard 2, and a run long enough to find anything
new costs minutes there. The nightly workflow (`.github/workflows/nightly.yml`) runs it on main with the run number as
seed: 1,000 texts through tokenize and mwt, 60 through `default` and 60 through `default_fast`; a difference fails the
night, and its reports are the `fuzz` artifact. Findings go into `fuzz.json`, which every golden run checks.
