# Subtitle line-break check

Checks where two-line subtitles break against the grammar of their sentences (#267). A break that separates what
subtitle style guides keep on one line is printed with the dependency relation it splits and the best-scoring break
instead. English and Swedish; tokenize, mwt, pos, lemma and depparse on the managed backend (no libtorch).

```
dotnet run -c Release --project samples/StanzaSharp.Subtitles -- samples/StanzaSharp.Subtitles/example.en.srt
dotnet run -c Release --project samples/StanzaSharp.Subtitles -- samples/StanzaSharp.Subtitles/example.sv.srt --lang sv --json breaks.json
```

The first run downloads the models into `models/stanza/LANG` (`--models DIR` to change it). `--max-chars` sets the line
limit (42). Build it in Release: a Debug build runs the managed kernels unoptimized, about 12 times slower (two minutes
instead of ten seconds for the 8,600 words of the measurement below). A file not ending in `.srt` is read as tagged
text, MuST-Cinema's format: one sentence per line, `<eol>` where a subtitle breaks into its second line and `<eob>`
where it ends (`Tagged.cs`). Output, for each flagged break:

```
#1  00:00:01,000 --> 00:00:03,500
    We drove all night to get to the  |
    old house by the lake.
  splits case (to → house): keep a preposition with its noun (break before the preposition)
  splits det (the → house): keep an article or determiner with its noun
  suggest: We drove all night to get
           to the old house by the lake.
```

`--json` writes the same (cue, timing, the break's line, the lines, the splits, whether the text fits on one line, the
suggestion) for a tool.

## The rules

Netflix's timed text style guides ask for a break after punctuation and before conjunctions and prepositions, and say
not to separate a noun from its article or adjective, a first name from a last name, a verb from its subject pronoun or
a prepositional verb from its preposition; text stays on one line unless it exceeds the limit, never more than two
lines, and a bottom-heavy pyramid is preferred. Each rule is a relation whose two words should share a line:

| relation | rule |
|---|---|
| `det`, `det:poss`, `nmod:poss` before the noun | keep an article, determiner or possessive with its noun |
| `amod`, `nummod` before the noun | keep an adjective or number with its noun |
| `flat`, `flat:name` | keep a first name with the last name |
| `case` | keep a preposition with its noun (break before the preposition) |
| `compound:prt` | keep a verb with its particle ("given \| up", "gett \| upp") |
| `compound`, `fixed` | keep a compound or fixed expression together |
| `cc`, `mark` ending the top line | break before a conjunction, not after it |
| `nsubj` pronoun next to its verb or auxiliary | keep a subject pronoun with its verb ("we \| will call", "ska \| vi ringa") |

A suggestion is the cue's best two-line break: the weights of the relations it splits, plus the lines' imbalance (a
top-heavy one costs a little more, a lone word on top more), plus 5 per character over the limit, less a bonus for
breaking after punctuation or a sentence's end, and a smaller one for breaking before a conjunction or preposition.
The weights are a first guess, not tuned.

Netflix's own pages (backlothelp.netflix.com) could not be reached from the session that wrote this (2026-11-09): the
rules above are from search results quoting them and from Rev's summary of Netflix's requirements; check the English
and Swedish guides before relying on the details.

## What it does

- Reads SRT (`Srt.cs`): formatting tags (`<i>`, `{\an8}`) are dropped; a cue's lines starting with a dash are a
  dialogue, whose breaks are the format and not checked.
- Joins every cue's lines into one text (`Transcript.cs`), so a sentence running over several cues is parsed whole,
  and maps each line break back to its place in the parse.
- Checks only relations whose two words are in the same cue (`BreakChecker.cs`): "…and he \| said…" across two cues
  is a cue boundary, not a line break.

## Measurement

On human-made subtitles the flag rate should be low, and on breaks made at the midpoint of the same subtitles high.
`--measure FILE` compares the two on every two-line cue that is not a dialogue (`Measurement.cs`): the midpoint break is
the space that leaves the two lines' lengths closest (the shorter top line on a tie), which is what a tool with no
grammar does. `measure.sh` downloads the data and runs it:

```
samples/StanzaSharp.Subtitles/measure.sh
```

The data is MuST-Cinema's English Amara test set (Karakanta et al., LREC 2020), as EvalSubtitle ships it
([fyvo/EvalSubtitle](https://github.com/fyvo/EvalSubtitle), pinned commit and SHA-256 in `measure.sh`; CC BY-NC-SA 4.0,
so it is downloaded into `models/subtitles/` and never committed): 544 sentences of TED-style talks, 8,600 words, 1,120
subtitles made by people, 584 of them on two lines, none over 42 characters. Results (2026-11-13, English default
package, managed backend):

| | own breaks | midpoint breaks |
|---|---|---|
| flagged, all 584 two-line subtitles | 61 (10.4%) | 242 (41.4%) |
| flagged, the 304 where the two breaks differ | 22 (7.2%) | 203 (66.8%) |
| the checker's best-scoring break is this one | 417 (71.4%) | 280 (47.9%) |

Split relations, counted once per break: `case` 22 own / 83 midpoint, `det` 27 / 79, `amod` 36 / 57, `mark` 4 / 28,
`cc` 0 / 19, `nmod:poss` 3 / 18, `nsubj` 1 / 11, `compound` 3 / 10, `compound:prt` 2 / 8, `nummod` 3 / 7, `det:predet`
1 / 2, `flat` 0 / 1, `nsubj:pass` 0 / 1.

- The rules separate the two: where the human and the midpoint break differ, the checker flags the midpoint one nine
  times as often.
- The flagged human breaks are mostly real departures from the Netflix rules, not parse errors: almost all split a noun
  phrase between a determiner or adjective and the noun ("He launched an aggressive \| marketing campaign,", "very
  expensive \| medical imaging technologies"). These subtitles were not made to Netflix's guides, so 10% is not a
  false-alarm rate; a set made to them should score lower.
- `amod` is the rule these subtitlers keep least (36 of the 61); `cc`, `nsubj` and `flat` they keep almost always.
- The weights were left as they were. Counting every dependency the break crosses (2 points each), which favours
  breaks between phrases, raised the agreement with the human breaks from 69.5% to 75.3% on the first half of the
  sentences, which it was tuned on, but only from 73.1% to 73.8% on the second half: no real gain.
- Blender's open films are CC BY but too small: Tears of Steel's English subtitles have 7 two-line cues. No openly
  licensed Swedish subtitles were found, so Swedish is unmeasured; MuST-Cinema has no Swedish.

## Not yet

- WebVTT, cue boundaries (a sentence split between cues at a bad place), NER for names the parser misses.
- A measurement on subtitles made to Netflix's guides, and on Swedish ones: neither found openly licensed.
