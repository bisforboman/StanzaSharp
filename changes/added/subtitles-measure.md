- `samples/StanzaSharp.Subtitles`: reads tagged text (MuST-Cinema's `<eol>`/`<eob>`) besides SRT, and `--measure`
  compares a file's own line breaks with midpoint breaks; on MuST-Cinema's English Amara test set (`measure.sh`) it
  flags 10.4% of the human breaks and 41.4% of the midpoint ones (#267).
