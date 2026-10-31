# Model files and checkpoint loading

How the models are stored and read (`.pt`, converted `.safetensors` + `.json`) and the loader's failure contract. Moved from CLAUDE.md on 2026-10-31.

C# loads either format. `Checkpoint.Load(basePath)` reads `basePath.json` + `basePath.safetensors`
when they exist, else the original `basePath.pt`, so `Pipeline.Load` accepts both `models/converted/en`
and Stanza's own download directory `models/stanza/en` (`<processor>/<name>.pt`). Converting is
optional: it makes the checkpoints easy to inspect, but since round 21 it no longer loads faster than the `.pt`
files (round 34; still slower after round 36). A packed-weight cache was measured and dropped (owner, 2026-10-16: 0.1–0.2 s for 250–420 MB
per package).

Use `setup.ps1 -Models`, or run the steps by hand:

```powershell
tools\.venv\Scripts\python -c "import stanza; stanza.download('en', model_dir=r'models\stanza', processors='tokenize,mwt,pos,constituency')"
tools\.venv\Scripts\python tools\stanza_convert.py inspect models\stanza\en\constituency\ptb3-revised_charlm.pt
tools\.venv\Scripts\python tools\stanza_convert.py convert models\stanza\en --out models\converted\en
```

Converter output for each checkpoint is a `<name>.safetensors` file holding all tensors, plus a
`<name>.json` file mirroring the checkpoint's structure. In the JSON:

- Each tensor is replaced by `{"$tensor": key, "dtype", "shape"}`, keyed into the safetensors file.
- Non-JSON Python types are tagged:
  - `$tuple`, `$set`
  - `$dict` (a list of `[key, value]` pairs, for dicts with non-string keys)
  - `$bytes` (base64)
  - `$float` (`nan`/`inf`)
  - `$repr`/`$type` (unknown objects)
- `optimizer` and `scheduler` are dropped by default.

Safetensors layout: a u64 little-endian header length, a JSON header (`dtype`, `shape`,
`data_offsets` relative to the data section, optional `__metadata__`), padded to 8-byte alignment,
then raw little-endian tensor data. Read it with a small hand-written reader in `StanzaSharp.Core`;
no package is needed.

The fourteen checkpoints of the English default package (the first port) convert with `torch.load(weights_only=True)`; no unsafe pickling is needed.

### Reading .pt files in C#

Those fourteen checkpoints use torch's legacy format (`_use_new_zipfile_serialization=False`), a
sequence of protocol-2 pickles: magic number, protocol version (1001), sys_info (little-endian),
the checkpoint, the list of storage keys; then per key an int64 element count and the raw bytes.
Tensors are `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)`
over persistent ids `('storage', torch.FloatStorage|LongStorage, key, location, numel, None)`; LSTM
weights share one storage at different offsets. The only globals are `collections.OrderedDict`,
`torch._utils._rebuild_tensor_v2`, the storage types, and `_codecs.encode` (protocol 2's encoding of
`bytes`, used by the lemma's `dicts`). The zip format (`archive/data.pkl` +
`archive/data/<key>`) is read too, for checkpoints saved by newer trainers.

Large plain containers (a str→int dict or list of str with ≥ 1,024 items, `PyContainer.MinCount`) skip the JSON
round trip when loading a `.pt` (round 21) and a converted `.json` (round 36: built in the validating reader pass, the
rest parsed with them as `null`): the node is a `JsonValue` holding the unpickled object, not a `JsonArray`/
`JsonObject`, and `ToJsonString()` still equals the converter's JSON. Read vocabularies with `Checkpoint.UnitToId` and
long string lists with `Checkpoint.Strings`; `AsArray()`/`AsObject()` on them throws.

`Unpickler` is restricted like `weights_only=True`: torch's protocol-2 opcodes only, plain data
(None/bool/int/float/str/bytes/list/tuple/dict), and an allowlist of OrderedDict, `_rebuild_tensor_v2`,
`_codecs.encode(str, 'latin1')` and the typed storage classes. Anything else throws `InvalidDataException`; nothing is instantiated by
name. Not supported (none of the models need them): sets, bfloat16, big-endian files,
`$repr` objects. `TorchCheckpoint` then mirrors the converter's `Splitter` exactly, including
Python's float `repr` in the JSON, key collisions (`key#2`) and numpy turning 0-d tensors into
shape `[1]`. `tests/golden/pt/` holds tiny fixtures in both formats with their converter output
(`make_golden.py --pt-only` regenerates them).

Failure contract (model files may come from anywhere): `Checkpoint.Load` and every later tensor lookup or read
either succeed or throw `InvalidDataException` for a malformed, truncated or hostile file; `NotSupportedException`
only for a well-formed big-endian `.pt`; OS errors (missing file) stay `IOException`s. Time, memory and stack are
bounded by the file's size. The limits that make it so:
- Unpickler: MARK records a stack height (O(1)); items below an open MARK can't be popped; dict keys are scalars only
  (str/int/bool/float; a nested-tuple key was hashed recursively); a length is checked against the stream before
  anything is allocated.
- Splitter: JSON depth ≤ 60 (`JsonNode.Parse` reads 64; also stops cyclic lists) and JSON size ≤ 16 × the file + 1 MB
  (the memo lets one object be referenced many times, and the JSON copies it each time).
- Storages and views: sizes compared by division, never `numel * size` (overflow); a storage key listed once; a tensor
  no larger than its storage (bounds stride-0 broadcasts) with non-negative sizes/strides/offset, its extent checked
  step by step before it can overflow. Zip entries must be stored uncompressed (as torch.save writes them) and fit in
  the file, so no claimed size is allocated.
- safetensors: header ≤ 100 MB (the library's limit), known dtype, `numel × size == end - begin`, and the tensors tile
  the data section exactly (no gaps, overlaps or trailing bytes), as the safetensors library checks.
- Converted `.json`: valid UTF-8, no escaped lone surrogate and no property name twice in one object, checked in the
  validating pass (`JsonNode` decodes strings and builds an object's dictionary lazily, so these would throw
  `InvalidOperationException` or `ArgumentException` wherever a model first read them; the duplicate names were found by
  the first nightly fuzz, 2026-10-18, about 10 ms on conll17's 250k-word JSON).
- `CheckpointFuzzTests`: seeded mutations (bit flips, truncations, extreme ints, inserted/deleted spans) of the fixtures
  and three small Stanza checkpoints in CI (their `.pt` files, and their converted `.json` where present; ~10 s), plus
  crafted inputs for each bug. Each load also reads every property name, string and vocabulary (`ReadNodes`). Longer runs:
  `STANZASHARP_FUZZ_ITERATIONS`, `STANZASHARP_FUZZ_SEED`, `STANZASHARP_FUZZ_MODELS=all`, `STANZASHARP_FUZZ_REPORT=FILE`.
  It does not validate the model code's reading of the JSON (a missing config key there is still a model-specific error).
