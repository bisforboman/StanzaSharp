- `Pipeline.Load` and the model loaders throw `InvalidDataException` for a converted `.json` checkpoint that repeats a
  property name in one object; before, it loaded, and reading that object later threw `ArgumentException` (found by the
  nightly checkpoint fuzz).
