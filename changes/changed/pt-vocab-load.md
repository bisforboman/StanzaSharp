- `Pipeline.Load` from Stanza's own `.pt` files reads the large vocabularies straight from the unpickled checkpoint
  instead of mirroring them into JSON and parsing that back: 35–48% less load time on GitHub's free runners (e.g.
  1,993 → 1,046 ms on ubuntu-26.04), with a load peak mostly lower. Converted models load as before, and the output is
  unchanged, bit for bit.
