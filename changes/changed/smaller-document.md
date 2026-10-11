- Less memory on long texts: the returned `Document` takes about 19% less (482 instead of 594 bytes per word with all
  eight processors: tokens hold room for one word, `SpaceAfter` and equal feats strings are shared, tree children are
  arrays), and NER frees the charlm outputs the tagger kept for it (up to 256 MB) as it goes. One call on 670,000
  words peaked at 3.0 GB instead of 3.6 GB. Outputs are unchanged. `Tree.Children` of a parsed tree is now an array
  (still an `IReadOnlyList<Tree>`).
