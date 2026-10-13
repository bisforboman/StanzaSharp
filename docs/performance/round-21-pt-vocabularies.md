# Results, round 21: loading `.pt` vocabularies

Part of [performance.md](../performance.md)'s measurement history.

`Pipeline.Load` from Stanza's own `.pt` files (`models/stanza/en`) mirrored each checkpoint into JSON
(`TorchCheckpoint.Splitter`), parsed that back into nodes, and `Checkpoint.UnitToId` then wrote the vocabulary's node
back to JSON text and read it into a dictionary. The pretrain's and NER's 250k-word vocabularies made the pretrain and
NER checkpoints the slowest to read, and they are the load's longest chain (round 17). Output unchanged, bit for bit
(below).

## What changed

- **Large flat containers stay out of the JSON.** A dict of str → int or a list of str with at least
  `PyContainer.MinCount` (1,024) items is written as `null`, and after parsing its node is a `JsonValue` holding the
  unpickled object (`PyContainer`). `Checkpoint.UnitToId` builds the dictionary straight from it, and the new
  `Checkpoint.Strings` reads such lists (constituency's `words`, sentiment's `extra_vocab`, NER's word `_id2unit` count).
  In the English models that covers the pretrain and NER vocabularies (250k), NER's delta (48k), constituency's
  `words` (40k), sentiment's `extra_vocab` (31k) and the POS/depparse word and lemma vocabularies (4.6–5.5k).
- **The JSON is unchanged.** Writing such a node (`ToJsonString`) runs the same Splitter over the object, so `Root`'s
  JSON still equals the converter's for all fourteen checkpoints (`LoadsStanzaCheckpointLikeTheConverter`). Only the
  node kind differs: read long lists with `Checkpoint.Strings`, not `AsArray()`. The converted `.json` path is
  unchanged; both readers accept both forms.
- **The limits still hold.** A left-out container counts against the JSON size cap (16 × the file + 1 MB) with a lower
  bound of its JSON size (its strings' lengths plus quotes and separators), so memo amplification still fails, and it
  is refused at the depth where its items would have been, so exactly the same files load. Its ids must fit an int, as
  in the JSON reader.
- **The unpickler reads ahead.** A seekable stream is read in 64 KB blocks instead of one stream call per opcode, and
  its position is set back to just after the STOP, so the legacy format's storages are found as before.

## Numbers

Warm, in a loop, locally (Ryzen 7 5800X, Windows, a machine busy with other work): `Checkpoint.Load` plus the
vocabulary reads the model makes, min of 5 after the first:

| checkpoint (`.pt`) | before | after |
|---|---:|---:|
| pretrain (conll17) | 296 ms | 100 ms |
| NER (ontonotes-ww-multi_charlm) | 233 | 121–150 (noisy) |
| constituency (load only) | 12 | 6 |
| sentiment (load only) | 9 | 3 |

`Pipeline.Load` from `.pt` files, default package, fresh processes (`tools/startup.py`, main and this branch
alternating, 9 processes each, medians): **1,662 → 934 ms** (CPU 2,547 → 1,562 ms), load peak 858 → 860 MB.

The Benchmark workflow's startup jobs (run [38369410782](https://github.com/bisforboman/StanzaSharp/actions/runs/38369410782),
`startup_base=origin/main`, the plain JIT build; each process fresh; one-sentence text, 5 processes, then the
benchmark text, 3 processes; medians):

| runner | load, one-sentence processes | load, benchmark-text processes | load CPU | load peak MB | load + 6 one-sentence calls |
|---|---:|---:|---:|---|---:|
| macos-26 | 1,177 → **769** (-35%) | 1,952 → 1,377 | 1,728 → 1,237 | - | 2,612 → 1,746 |
| ubuntu-26.04 | 1,993 → **1,046** (-48%) | 1,535 → 901 | 3,096 → 1,831 | 916 → 846, 903 → 811 | 3,699 → 2,938 |
| ubuntu-26.04-arm | 1,388 → **738** (-47%) | 1,290 → 733 | 2,113 → 1,363 | 1,036 → 948, 988 → 970 | 3,207 → 1,694 |
| windows-11-arm | 1,282 → **736** (-43%) | 1,306 → 731 | 1,953 → 1,219 | 856 → 808, 858 → 946 | 2,455 → 1,702 |
| windows-2025 | 1,610 → **971** (-40%) | 1,604 → 974 | 2,688 → 1,797 | 919 → 833, 937 → 840 | 3,248 → 2,434 |

The load takes 35–48% less time and CPU on every runner. Its peak is mostly lower (no 250k-node JSON documents to
parse, no second copy of the vocabulary as JSON text): 70–97 MB less on the x64 runners, mixed on the Arm64 ones
(one median 88 MB higher on windows-11-arm, within the run-to-run spread of the other rows). Calls after the load are
unchanged (they share none of this code).

What is left of a `.pt` vocabulary's cost is the unpickling itself: about 0.1 s for a 250k-word vocabulary (500k
strings decoded, a memo entry and a dict entry each).

## Exactness

The CoNLL-U of every `tests/golden/*.txt` (SHA-256, each text processed alone, `StanzaSharp.Benchmark startup
--hash`), main against this branch, from `models/stanza/en` and `models/converted/en`: identical (`4FD0F4EE…`). Every
startup process on every runner gave the same CoNLL-U for both texts as main. TorchCheckpointTests,
CheckpointFuzzTests, PipelineTests, NerTests, SentimentTests and ConstituencyTests pass with `STANZASHARP_MODELS` set
to both directories.

## Fuzzing

- `Fuzz_LargeContainers`: the seeded mutations (300 by default) of a crafted `.pt` holding a vocabulary, long lists
  under a tuple and under a non-string-keyed dict, a short list and a long list with a non-string item.
- The fuzz harness now also reads every vocabulary and long list of a loaded `.pt` (`UnitToId`, `Strings`), which
  must succeed or throw `InvalidDataException`.
- Crafted: the left-out containers' JSON equals the converter's (`Crafted_LargeContainersLoadLikeTheConverter`, also
  checking paths through `$tuple` and `$dict`), ids past an int fail in both readers, nesting at the depth limit
  (56–64 levels) loads exactly when the converter's JSON would, and a 1 MB list referenced 10,000 times through the
  memo still hits the JSON cap.
- Found on the way (not changed): a converted `.json` with invalid UTF-8 in a property name loads, and reading that
  object later throws `InvalidOperationException` (System.Text.Json parses lazily). The contract leaves the model
  code's reading of the JSON out; the harness reads vocabularies only for `.pt` files. Fixed afterwards (#94):
  `Checkpoint.Load` rejects such a file, and the harness reads every load's names, strings and vocabularies.
