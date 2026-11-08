# Labelled texts for the masking sample

Invented Swedish documents with every span that must be masked labelled inline, for measuring the sample's suggestions:

```
dotnet run --project samples/StanzaSharp.Masking -c Release -- --eval samples/StanzaSharp.Masking/eval/test --verbose
```

It scores two runs on the same NER output: the sample as it was (NER + patterns) and with `--link` (`NameLinker`).
`--verbose` lists each leak, each mask that overlaps no label, and each person split over placeholders or sharing one.

## Format

`{TYPE:text}` marks a span, `{PER#id:text}` a person's, with `id` naming the person within the file
(`{PER#karin:Karin Lindqvist}` … `{PER#karin:Karins}` … `{PER#karin:K. Lindqvist}`). No nesting, no other braces.

| type | what |
|---|---|
| PER | every name of a person (officials too): first names, surnames, nicknames, genitives (the `s` in the span), initials with the surname |
| LOC | towns, districts, countries, places |
| ORG | organisations that could identify someone: employers, companies, schools, landlords, associations; not public authorities acting as such (Försäkringskassan, Socialnämnden) |
| PNR, ORGNR, PHONE, EMAIL | numbers and addresses |
| ADDRESS | a street with its number (and `lgh`, floor), and separately a postal code with its town |

Dates, amounts, case numbers and law references are not labelled.

## The scores

- **leaked**: a labelled span with any letter or digit left unmasked, whatever type the mask has.
- **overlapping no label**: a suggested mask that touches no labelled span (over-masking; a reviewer removes it).
- **same person, same placeholder**: over the pairs of fully masked mentions of one person, those with one
  placeholder; **shared by different people**: pairs of mentions of two people under one placeholder (worse than a split:
  it tells the reader two people are one).

## The sets

- `dev/` (4 texts, 110 spans): written while building `--link` and used to tune it. `01-example.txt` is
  `../example.txt` labelled.
- `test/` (6 texts, 225 spans): written by a separate agent that was told nothing about how `--link` works (genres,
  name forms and labelling rules only), then scored once with the design frozen.

Everything is invented. Personnummer are Skatteverket's test personnummer (open data, `testpersonnummer`, never issued),
phone numbers are in PTS's ranges for fiction (070-174 06 05–99, 08-465 004 00–99, 031-390 06 00–99,
040-628 04 00–99; `08-555 123 45` is `example.txt`'s), organisationsnummer are synthetic, e-mail addresses are under
`example.se/.com/.org`.

## Results (2026-11-08)

Swedish `ner/suc3shuffle`, managed backend. The held-out run is the first one on `test/`; it showed three bugs, fixed
afterwards (names matched inside words such as `NA1b`; a father's name `Abdirahman Nur` merged into his son's
`Yusuf Abdirahman Nur`; a two-column signature tagged as one name), so the second `test/` column is no longer held-out.

| | NER + patterns | `--link`, held-out | `--link`, after the fixes | dev: NER + patterns | dev: `--link` |
|---|---|---|---|---|---|
| leaked spans | 12 of 225 | 11 of 225 | 11 of 225 | 3 of 110 | 2 of 110 |
| leaked PER | 5 of 136 | 5 of 136 | 5 of 136 | 2 of 75 | 1 of 75 |
| masks overlapping no label | 18 of 239 | 21 of 246 | 21 of 247 | 5 of 113 | 3 of 113 |
| same person, same placeholder | 61 of 174 (35%) | 160 of 174 (92%) | 165 of 174 (95%) | 17 of 74 (23%) | 71 of 78 (91%) |
| pairs of different people under one placeholder | 1 | 4 | 0 | 0 | 0 |
| people with one placeholder of their own | 2 of 35 | 28 of 35 | 33 of 35 | 3 of 20 | 18 of 20 |

What it shows:

- **Consistency is the gain.** Without linking, a person's mentions mostly get different placeholders (`Karins`,
  `K. Lindqvist`, `LINDQVIST, KARIN` are different texts), so a reader can't follow who is who; with it, nearly all do.
- **Recall barely moves.** On these texts NER already finds almost every name it can; propagation recovered one span
  (an organisation's second mention) and the initial in front of a surname. What leaks is what it never sees: names in
  lowercase in informal e-mail (`annika`, `jocke`, `sten-åke`; deliberately not propagated, since `sten`, `ek`, `rosa` are
  words), organisations NER never tagged, a full name NER typed ORG next to an e-mail address, and addresses with `lgh`
  or `3 tr` (the sample's street pattern; the TasmanianDevil package's `SeStreetAddressRecognizer` takes `lgh`).
- **Over-masking grows a little** (18 → 21): propagation repeats NER's own mistakes (`Pat:s`, a class name `NA1b`
  tagged ORG).
- The remaining splits are mostly deliberate: a surname alone shared by two people in the text (`Holm` with Ingrid and
  Björn Holm) keeps its own placeholder rather than guess.

Ten texts are a small sample, written by language models rather than taken from real case files; treat the numbers as
direction, not as a recall study.
