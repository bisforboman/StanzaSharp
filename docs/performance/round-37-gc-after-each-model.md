# Results, round 37: the GC after each model at load

Part of [performance.md](../performance.md)'s measurement history.

The managed `Pipeline.Load` runs `GC.Collect()` after each model (`BackendModels.CollectAfterEachModel`, round 17), so
the next model reuses the memory the last one's unpacked weights leave behind. [Round 34](round-34-load-packed-cache.md)
measured those GCs at 130-270 ms of thread time, about 9% of the load, and suggested reading the weights that only feed
packing into rented buffers so the GCs could go without raising the peak.

## Result

**Dropped; nothing changed.** Even with no GC at all, the load is at most 0-12% faster, and the peak rises by 170-220
MB. Cheaper ways to keep fewer GCs gained nothing measurable. Rented buffers were not built: the no-GC upper bound
is what they could win at most, and round 34 showed that most of the peak comes back without the GCs even when no
unpacked tensor is read at all (its cache hits: +50-180 MB), so rented buffers would not keep the peak down either.

## Measurements

Local Ryzen 7 5800X, Windows 11. `tools/startup.py`, `.pt` files, default threads, one 13-word sentence per call;
medians. The settings were a temporary environment switch in `Pipeline.Load` (not merged). The machine was quiet for
the first English run and about 60% busy with other work for the other two, so only differences within a run count.

| setting | what runs after each model |
|---|---|
| all | `GC.Collect()` (main) |
| none | nothing |
| dedupe | `GC.Collect()` unless another lane's GC ran during this model's load |
| optimized | `GC.Collect(2, GCCollectionMode.Optimized)` |
| nocompact | a blocking gen-2 GC without compaction |
| big64 / big150 | `GC.Collect()` only if the model's load allocated more than 64 / 150 MB on its thread |

English `default`, quiet machine, 7 processes per setting:

| setting | load ms (CPU) | load peak MB | load + 2 calls |
|---|---:|---:|---:|
| all (main) | 600 (1,141) | 837 | 846 |
| none | 542 (1,141) | 1,007 | 794 |
| dedupe | 585 (1,156) | 973 | 841 |
| optimized | 612 (1,203) | 949 | 874 |
| nocompact | 717 (1,188) | 834 | 1,015 |

English `default`, busy machine, 9 processes per setting:

| setting | load ms (CPU) | load peak MB | load + 2 calls |
|---|---:|---:|---:|
| all (main) | 604 (1,188) | 840 | 866 |
| none | 597 (1,219) | 1,040 | 907 |
| big64 | 633 (1,266) | 834 | 944 |
| big150 | 703 (1,281) | 862 | 1,034 |

German `default`, busy machine, 7 processes per setting:

| setting | load ms (CPU) | load peak MB | load + 2 calls |
|---|---:|---:|---:|
| all (main) | 1,137 (1,938) | 1,231 | 1,534 |
| none | 1,002 (1,766) | 1,447 | 1,368 |
| big64 | 1,096 (2,125) | 1,217 | 1,476 |

What each model allocates on its thread while loading (English `default`, MB): the two charlms 53 each, tokenizer 2,
mwt 3, pretrain 161, lemma 25, pos 99, depparse 390, ner 180, constituency 272, sentiment 172. big64 skips the six
small ones' GCs, which saved nothing measurable (likely because a gen-2 GC's cost grows with the live heap, not the
garbage, and the models after them then started with that garbage still there).

- **dedupe** keeps the peak as high as no GC (likely because another lane's GC runs before this model's garbage is dead).
- **optimized** lets the runtime decide whether to collect: the peak rises without a gain.
- **nocompact** was slower, not faster.

CoNLL-U: identical under every setting (`E607513C5DDC136F` English, `BC75A8F5D1B1AC0E` German, one sentence per call).
