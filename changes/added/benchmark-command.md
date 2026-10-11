- `stanzasharp benchmark [--package NAME] [--processors LIST] [--models DIR] [--threads N] [--words N] [--python PATH]
  [--quick] [--json FILE]` (`StanzaSharp.Tool`): times StanzaSharp on a built-in English text (load, each processor,
  words/s, one-sentence calls with median and p90, peak memory) and, if Python Stanza 1.15.0 is available, Stanza on
  the same text with the same models, runs alternating, and prints a Markdown block with the machine (CPU model,
  cores, RAM, OS, .NET, SIMD path), both sides and the ratio. A manual workflow (`benchmark.yml`) runs it on every free
  GitHub-hosted runner and combines the results into one table.
