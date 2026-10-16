# Converted models' vocabularies at load

An open idea ([list](../../performance.md#remaining-ideas)).

Since [round 21](../round-21-pt-vocabularies.md), a `.pt` file's large vocabularies skip the JSON round trip. The
converted `.json` path still parses them into a `JsonNode` tree, and `Checkpoint.UnitToId` then writes `_unit2id` back
to JSON text and reads that. [Round 34](../round-34-load-packed-cache.md) measured the cost on ubuntu-26.04: vocab
160-350 ms of thread time against 25-50 ms from `.pt`, and parse times that varied between about 500 and 1,000 ms from
process to process. Converted English `default` loaded no faster than the `.pt` files, sometimes slower. CLAUDE.md's
"converting saves about a second of load time" predates round 21.

Options:
- Read the `.json` with a `Utf8JsonReader` pass that builds the large `_unit2id`/`_id2unit` containers directly, as
  `PyContainer` nodes, and leaves the rest to `JsonNode`.
- Or stop recommending conversion for speed, and keep it for inspection.
