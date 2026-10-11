- Loading a malformed or hostile model file now always fails with `InvalidDataException` (`NotSupportedException` for
  a big-endian `.pt`), in time and memory bounded by the file's size. Found by a new fuzz test of the `.pt` and
  `.json` + `.safetensors` loaders:
  - A pickle with many nested MARKs took quadratic time (each MARK copied the stack): 400 KB could hang the load.
  - Deeply nested or self-referencing lists threw `InvalidOperationException` or `JsonException` after recursing up
    to 1000 levels (a stack overflow on a small thread stack); a dict key of deeply nested tuples overflowed the stack
    when hashed, crashing the process.
  - An object referenced many times through the pickle memo was copied into the JSON each time (a 264 KB file asked for
    gigabytes).
  - Storage and tensor sizes could overflow: a storage of 2^62 floats passed the "fits in the file" check, a storage
    view at a huge offset passed its bounds check, and huge strides wrapped a tensor's extent negative. A stride-0
    tensor could claim terabytes, and a negative size next to a zero was accepted.
  - A zip `.pt` whose `data.pkl` header claimed 2 GB allocated it before reading; compressed (zip bomb) entries are
    now rejected, since torch.save stores them uncompressed.
  - safetensors headers were not validated: an unknown dtype, a shape that disagrees with the byte size, overlapping
    or missing data, or a malformed header threw `FormatException`, `KeyNotFoundException`, `JsonException` or
    `InvalidOperationException`, or loaded.
