- `stanzasharp benchmark` warms up on the whole text, repeating until each side's stage times settle (2 to 5 runs; the
  block says how many), instead of one run on a tenth of the text. StanzaSharp's numbers now show a warmed service:
  on the same runners about 5% more words/s, with tokenize and lemma 30–45% faster; Python Stanza's moved about 1%.
  Results from earlier versions aren't directly comparable. A run takes about twice as long.
