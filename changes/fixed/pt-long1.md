- `.pt` loading: an integer pickled as LONG1 that fits in 64 bits (such as a storage of 2^31 elements or more) was kept
  as a big integer, so such a checkpoint failed to load.
