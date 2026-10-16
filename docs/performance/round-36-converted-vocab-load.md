# Results, round 36: converted models' vocabularies at load

Part of [performance.md](../performance.md)'s measurement history.

Since [round 21](round-21-pt-vocabularies.md) a `.pt` file's large vocabularies skip the JSON round trip. The
converted `.json` path still parsed them into a `JsonNode` tree, and `Checkpoint.UnitToId` then wrote `_unit2id` back
to JSON text and read that. [Round 34](round-34-load-packed-cache.md) found converted models no faster than `.pt` on
the runners, sometimes slower.

## Result

**Kept.** `Checkpoint.Load` reads a converted `.json` like a `.pt` file: one `Utf8JsonReader` pass (the one that
already validated its strings) builds every large flat container (a str → int object or a list of str with
`PyContainer.MinCount` or more items, as `PyContainer.IsLarge` defines them) straight into the unpickler's types, and
`JsonNode.Parse` reads the rest, with those containers written as `null` and then replaced by `PyContainer` nodes.
`ToJsonString()` is unchanged, `AsArray()`/`AsObject()` on those nodes throw as they do for a `.pt` file.

Converted models load 13-24% faster. They still load slower than the `.pt` files locally (English 768 vs 548 ms),
so conversion stays an inspection aid, not a speed-up; CLAUDE.md already says so.

## Where it went

Locally, English `default`, converted, one process (thread ms; a `Stopwatch` around each step, not shipped):

| checkpoint | JSON | before: validate + parse + `UnitToId` | after: one pass + parse the rest + `UnitToId` |
|---|---:|---:|---:|
| pretrain/conll17 | 9.1 MB | 112 + 138 + 120 = 370 | 205 + 25 |
| ner/ontonotes-ww-multi_charlm | 11.4 MB | 130 + 190 + 25 = 345 | 225-305 |

The remaining pass is the strings and the ordered dictionary of 250,000 words, about what unpickling them costs.

## Measurements

Local Ryzen 7 5800X, Windows 11, otherwise idle. `tools/startup.py`, converted models (`models/converted/<lang>`),
default threads, 7 processes per setting alternating, medians; one 13-word sentence per call. "round 35" is this
branch before this round.

| config | setting | load ms (CPU) | load peak MB | load + 2 calls |
|---|---|---:|---:|---:|
| en default, converted | round 35 | 899 (1,609) | 863 | 1,148 |
| en default, converted | round 36 | **768 (1,312)** | 794 | **1,021** |
| de default, converted | round 35 | 1,131 (2,219) | 1,198 | 1,394 |
| de default, converted | round 36 | **988 (1,797)** | 1,181 | **1,266** |

An earlier run of the English pair gave 945 → 718 ms (-24%). For comparison, the `.pt` files load in 535 ms (English)
and 775 ms (German) on this machine (round 35).

## Exactness

- `startup --hash` with the converted models: English `4FD0F4EE1F4DBBED`, German `6C5B75A493E5306E`, the same before
  and after (and the same as from the `.pt` files).
- `TorchCheckpointTests.LoadsStanzaCheckpointLikeTheConverter`: every checkpoint's `ToJsonString()` and tensors equal
  the `.pt` file's. `CheckpointFuzzTests` (fuzzed converted `.json` included) passes;
  `Crafted_LargeContainersLoadLikeTheConverter` now also loads the converter's JSON of its crafted vocabulary and checks
  the same node kinds (long lists inside a `$tuple` and a `$dict` are `PyContainer`s, a short list and one with a
  `null` stay JSON). A `$tuple`'s or `$set`'s own item list is never one: in a `.pt` file it is a tuple, not a list.
- Pos, NER and sentiment golden tests on the converted models (managed) pass.
