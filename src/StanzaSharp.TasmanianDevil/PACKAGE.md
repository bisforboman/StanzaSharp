# StanzaSharp.TasmanianDevil

[StanzaSharp](https://www.nuget.org/packages/StanzaSharp) for [TasmanianDevil](https://github.com/filipw/tasmaniandevil),
a .NET library for detecting and de-identifying PII. It adds two components.

- `StanzaNerRecognizer` finds `PERSON`, `LOCATION`, `ORGANIZATION` and `DATE_TIME`, the entity types that regular
  expressions can't reach.
  - It uses Stanza's supervised NER model for the pipeline's language: English (OntoNotes), Swedish, German and many more.
  - It runs in managed code and gives the same spans on every run.
  - It joins TasmanianDevil's analyzer like any other recognizer, so overlap resolution and anonymization treat its spans
    like the regex matches.
- `StanzaTokenNormalizer` gives TasmanianDevil's context matching (`LemmaContextAwareEnhancer`) real lemmas, in any
  language Stanza has a lemmatizer for, instead of English Porter stems.
- `SePersonnummerRecognizer` and `SeOrganisationsnummerRecognizer` find Swedish identity numbers. TasmanianDevil has no
  Swedish country pack yet.
  - `SE_PERSONNUMMER` covers personnummer and samordningsnummer. A number must have a valid date and Luhn check digit.
  - `SE_ORGANISATIONSNUMMER` must have a third digit of 2 or more and a valid Luhn check digit.
  - Written with its separator, a valid number counts with full confidence. A bare run of digits also needs a context
    word nearby, such as "personnummer".
- `SeStreetAddressRecognizer` and `SePostnummerRecognizer` find Swedish addresses.
  - `ADDRESS` (TasmanianDevil's own type) is a street name with a definite street suffix (`Storgatan`, `Västra
    Ringvägen`, `S:t Eriksgatan`), its number, an entrance letter and an apartment number: `Storgatan 14 B, lgh 1102`.
  - `SE_POSTNUMMER` is a postal code from 100 00 to 984 99. With `SE-` or followed by its town (`753 20 Uppsala`) it
    counts on its own; a bare five-digit number also needs a context word nearby, such as "postnummer" or "adress".
- `SeBankgiroRecognizer`, `SePlusgiroRecognizer` and `SeBankAccountRecognizer` find Swedish payment numbers. Each
  number must pass its check digits, and it also needs a context word nearby: the checks pass for one number in ten.
  - `SE_BANKGIRO` is `NNN-NNNN` or `NNNN-NNNN` (context: "bankgiro", "bg"); `SE_PLUSGIRO` is up to eight digits with a
    hyphen before the last, `12 34 56-6` (context: "plusgiro", "pg").
  - `SE_BANK_ACCOUNT` is a clearing number and an account number, `5000-1234560` or `8327-9, 914 123 456-8`, checked by
    the bank's rules in Bankgirot's table (context: "konto", "kontonummer", "clearingnummer"). TasmanianDevil's own
    `IBAN_CODE` covers the IBAN form.
- `SeVehicleRegistrationRecognizer` finds `SE_VEHICLE_REGISTRATION`, Swedish registration plates (`ABC 123`, `ABC 12D`),
  next to a context word such as "registreringsnummer", "regnr" or "bilen". Currency amounts (`EUR 100`) aren't taken.
- `SeDateOfBirthRecognizer` finds `DATE_OF_BIRTH`: a date right after "född", "föddes", "f.", "födelsedatum" or
  "födelsedag" (`född 1985-03-12`, `föddes den 12 mars 1985`, `f. 12/3-85`). Other dates stay `StanzaNerRecognizer`'s
  `DATE_TIME`, so dates of birth can be masked while other dates are kept.

## Use

```csharp
using StanzaSharp;
using StanzaSharp.TasmanianDevil;
using TasmanianDevil.Analyzer;

// Once: await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Processors = "tokenize,ner" });
using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions { Processors = "tokenize,ner" });
registry.AddRecognizer(new StanzaNerRecognizer(nlp));                    // English
registry.AddRecognizer(new StanzaNerRecognizer(swedishNlp, "sv"));       // a pipeline per language
```

For Swedish:

```csharp
using var sv = Pipeline.Load("models/stanza/sv", new PipelineOptions { Language = "sv", Processors = "tokenize,ner" });
using var engine = new PiiEngine(new PiiOptions { Language = "sv" }, extraRecognizers:
    [new StanzaNerRecognizer(sv, "sv"), new SePersonnummerRecognizer(), new SeOrganisationsnummerRecognizer(),
     new SeStreetAddressRecognizer(), new SePostnummerRecognizer(), new SeBankgiroRecognizer(), new SePlusgiroRecognizer(),
     new SeBankAccountRecognizer(), new SeVehicleRegistrationRecognizer(), new SeDateOfBirthRecognizer()]);
engine.Anonymize(text); // names, places, organisations, dates, id numbers, addresses, postal codes, payment numbers,
                        // plates, dates of birth, phone, e-mail
```

Addresses are matched by their shape: a street name without a definite suffix (`Drottning Kristinas väg`) or a P.O. box
number isn't found.

### One placeholder per person

TasmanianDevil replaces every person with the same `<PERSON>`. With `LinkNames`, `StanzaNerRecognizer` links each
person's mentions across the text, and `PersonPlaceholders` numbers them, so a reader can still follow who is who:

```csharp
using TasmanianDevil.Anonymizer;

var recognizer = new StanzaNerRecognizer(sv, "sv") { LinkNames = true };
using var engine = new PiiEngine(new PiiOptions { Language = "sv" }, extraRecognizers: [recognizer, /* ... */]);
var results = engine.Analyze(text);
var anonymized = new AnonymizerEngine().Anonymize(text, results, StanzaNerRecognizer.PersonPlaceholders(text, results),
    ConflictResolutionStrategy.MergeSimilarOrContained, false);
// "LINDQVIST, KARIN … Karins ansökan … K. Lindqvist" → "<PERSON_1>, <PERSON_1> … <PERSON_1> ansökan … <PERSON_1>"
```

- Every capitalized or all-caps occurrence of a name NER tagged somewhere in the text is found too, with its genitive
  and an initial in front (`Karins`, `K. Lindqvist`); lowercase words aren't (`sten`, `ek` and `rosa` are also words).
- A full name is a person; a shorter mention joins the one full name it fits. A surname alone shared by two people in
  the text (`Holm` next to Ingrid and Björn Holm) gets a placeholder of its own rather than a guess.
- Each `PERSON` result carries its person in `RecognitionMetadata[StanzaNerRecognizer.PersonKey]`; added results carry
  `LinkedKey`. Organisations and places are also found again wherever their exact text recurs.
- Measured on labelled Swedish texts: one placeholder for 92% of the pairs of one person's mentions held-out (35%
  without), about as many spans missed, and a few more masks a reviewer removes
  ([samples/StanzaSharp.Masking/eval](https://github.com/bisforboman/StanzaSharp/tree/main/samples/StanzaSharp.Masking/eval)).

`StanzaNerRecognizer` needs only `tokenize,ner`. `StanzaTokenNormalizer` needs `tokenize,mwt,pos,lemma`; pass it to
`LemmaContextAwareEnhancer`'s `normalizer` parameter.

### Entity types

`StanzaNerRecognizer.DefaultTypeMap` maps Stanza's entity types to TasmanianDevil's.

| Stanza | TasmanianDevil |
|---|---|
| PERSON, PER | PERSON |
| ORG | ORGANIZATION |
| GPE, LOC, FAC | LOCATION |
| DATE, TIME, TME | DATE_TIME |

The map covers both naming conventions of Stanza's models: English's OntoNotes types, and the PER/LOC/ORG types most
other languages use. Pass your own map to change it. Every span gets the same score, 0.85 by default; Stanza gives no
confidence, and 0.85 is what Python Presidio's spaCy recognizer uses.

### Compared with TasmanianDevil.Onnx

TasmanianDevil's own NER add-on uses a zero-shot GLiNER model: one multilingual model of 580 MB (fp16), which you
prompt with labels.

This package uses Stanza's supervised model for each language instead, with a fixed set of types per language. For
English, `tokenize` and `ner` are about 210 MB. It needs no native ONNX runtime, and it gives the same output as Python
Stanza.

## License

Apache 2.0, like Stanza, whose code StanzaSharp ports; see NOTICE. TasmanianDevil is MIT. The models have their own
licenses, which vary with their training data.
