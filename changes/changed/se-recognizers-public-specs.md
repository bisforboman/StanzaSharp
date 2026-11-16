- StanzaSharp.TasmanianDevil's Swedish recognizers follow the public specifications more closely, each rule cited from
  Skatteverket's descriptions, the statutes or Bankgirot's "Bankernas kontonummer". `SePersonnummerRecognizer` accepts
  only real dates (no 30 February; 29 February only in a leap year). `SeBankAccountRecognizer` no longer checks the fifth
  digit of Swedbank's five-digit clearing numbers, and accepts Plusgirot and Swedbank 8xxx accounts of up to ten digits.
