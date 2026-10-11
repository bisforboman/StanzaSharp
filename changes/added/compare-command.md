- `stanzasharp compare FILE [--package NAME] [--processors LIST] [--models DIR] [--python PATH]` (`StanzaSharp.Tool`):
  runs Python Stanza 1.15.0 and StanzaSharp on your own text with the same, MD5-checked model files and reports
  whether the CoNLL-U output is identical, or the first difference (sentence, both lines), with sentence and word
  counts and both sides' load and processing times. Offsets after characters outside the BMP are converted to code
  points (Stanza's) before comparing. Exit code 0 identical, 1 different, 2 usage or setup error.
