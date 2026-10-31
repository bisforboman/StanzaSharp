# Swedish masking (maskning) sample

Suggests what to mask in a Swedish document before it is released: people, places and organisations from
StanzaSharp's Swedish NER (`ner/suc3shuffle`, Stanza's own model, byte for byte), plus patterns for what NER doesn't
tag. A person reviews every suggestion; this is a first pass, not a decision.

```
dotnet run --project samples/StanzaSharp.Masking -- samples/StanzaSharp.Masking/example.txt --json spans.json
```

The first run downloads Swedish `tokenize` and `ner` into `models/stanza/sv` (`--models DIR` to change it). Output: the
text with placeholders, the same one wherever the same text recurs (`[PER-1]` for every "Karin Lindqvist"), and with
`--json` the spans (start, length, type, source, text; UTF-16 offsets) for a review tool.

| type | from |
|---|---|
| PER, LOC, ORG | NER (default). `--types PER,LOC,ORG,TME` adds dates and times; amounts (MSR) are never masked. |
| PNR | personnummer and samordningsnummer, 10 or 12 digits, `-`/`+` optional, Luhn-checked, valid month and day |
| ORGNR | organisationsnummer (third digit 2 or more), Luhn-checked |
| PHONE | a leading `0` or `+46`, then groups of digits |
| EMAIL | e-mail addresses |
| ADDRESS | a street (`…gatan`, `…vägen`, … plus a number) and a postal code with its town |

Where a pattern and an entity overlap, they become one span typed by the pattern (`Storgatan 14 B`, not the LOC
`Storgatan`).

## How well Swedish NER finds names

Measured on two fictional case-file texts (2026-10-31): a decision on social assistance written as running text (not
included), and `example.txt`, written to be hard (an all-caps surname-first header, form fields, genitives, an initial,
non-Nordic names, names at the start of a sentence and alone). Everything in `example.txt` is invented; its
personnummer have years no one can be born in (2098, 2099) and its organisationsnummer is synthetic.

| | running text | example.txt |
|---|---|---|
| person mentions found | 11 of 11 | 15 of 15 (the header `LINDQVIST, KARIN` as two spans; `Mikaels` typed ORG, masked anyway) |
| boundary errors | `Uppsala kommun` split into LOC + ORG | `KARIN Personnummer`: the field's label masked with the name |
| left to the patterns | street numbers, postal codes, phone, e-mail, personnummer | the same; NER tagged a personnummer as an amount |

Two texts are not a recall study: test it on real documents before relying on it. Known limits:

- Indirect identifiers ("hans syster, rektor på X-skolan") are beyond NER and patterns.
- The same person written differently (`Karin Lindqvist`, `K. Lindqvist`, `Karins`) gets different placeholders.
- Public bodies (`Socialförvaltningen`, `Försäkringskassan`) are organisations to NER and are masked too; unmask them in
  review if they may stay.
- The model was trained on general text (SUC 3), not case files.
