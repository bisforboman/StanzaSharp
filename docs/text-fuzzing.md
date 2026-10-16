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
```

`--processors` defaults to the package's processors (all eight for `default`). Each side uses 4 threads
(`--threads`; 0 for their defaults). The exit code is 1 if anything differed; `--report` writes every finding with its
original and shrunk text (`--shrink-budget`, 300 candidates by default, bounds the shrinking: a long text through the
whole pipeline costs seconds a candidate). Speed on a Ryzen 7 5800X with another run beside it: tokenize and mwt 8 to
12 texts (9,000 to 14,000 characters) a second, Stanza taking 90% of the time; the treebank packages about 3 s a text,
`default_fast` 1.6 s, all eight processors about 5 s, mostly Stanza's.

A short seeded run is not in CI: Python Stanza runs only in golden-shard 2, and a run long enough to find anything
new costs minutes there. Findings go into `fuzz.json` instead, which every golden run checks.
