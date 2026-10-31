- The `StanzaSharp.TasmanianDevil` package (#234) plugs StanzaSharp into TasmanianDevil, a .NET library for PII
  detection and de-identification.
  - `StanzaNerRecognizer` adds `PERSON`, `LOCATION`, `ORGANIZATION` and `DATE_TIME` from Stanza's NER model for each
    language, Swedish included.
  - `StanzaTokenNormalizer` gives TasmanianDevil's context matching real lemmas in any Stanza language.
  - `SePersonnummerRecognizer` and `SeOrganisationsnummerRecognizer` find Swedish personnummer, samordningsnummer and
    organisationsnummer, checked by date and Luhn check digit.
