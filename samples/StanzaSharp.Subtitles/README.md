# Subtitle line-break check

Checks where two-line subtitles break against the grammar of their sentences (#267). A break that separates what
subtitle style guides keep on one line is printed with the dependency relation it splits and the best-scoring break
instead. English and Swedish; tokenize, mwt, pos, lemma and depparse on the managed backend (no libtorch).

```
dotnet run --project samples/StanzaSharp.Subtitles -- samples/StanzaSharp.Subtitles/example.en.srt
dotnet run --project samples/StanzaSharp.Subtitles -- samples/StanzaSharp.Subtitles/example.sv.srt --lang sv --json breaks.json
```

The first run downloads the models into `models/stanza/LANG` (`--models DIR` to change it). `--max-chars` sets the line
limit (42). Output, for each flagged break:

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

## Not yet

- WebVTT, cue boundaries (a sentence split between cues at a bad place), NER for names the parser misses.
- A measurement: on professionally made subtitles the flag rate should be low, on breaks made at the midpoint high.
  That needs openly licensed subtitle files, still to find.
