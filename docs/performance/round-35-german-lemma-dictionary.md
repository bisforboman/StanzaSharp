# Results, round 35: the lemma dictionary at load

Part of [performance.md](../performance.md)'s measurement history.

[Round 34](round-34-load-packed-cache.md) found German `default`'s lemma dictionary to be the largest single item of
the German load: 2.2 MB of gzip, 14.7 MB of JSON, 524,460 entries read by `JsonSerializer` into dictionaries of
strings, about 0.77 s of thread time on ubuntu-26.04. English's has 37,130 entries.

## Result

**Kept.** The lemmatizer keeps the dictionary's JSON as UTF-8 bytes and builds a hash table of (POS, word) → lemma
offsets over them (`LemmaDictionary`); a lookup encodes the word to UTF-8 and decodes the lemma it finds. No string is
made at load. German `default` loads 16% faster (925 → 775 ms, CPU 1,578 → 1,359 ms); English is unchanged (its
dictionary was never on the critical path). Output is identical.

## What the dictionary costs

German has 18 POS keys, but `*` holds 522,699 of the 524,460 entries; 148,295 lemmas equal their word, and there are
163,630 distinct lemmas. Only 14 strings have a JSON escape. Measured locally, one process each, thread ms of the
dictionary step (base64 7, gunzip 23 in every variant):

| variant | build | German load |
|---|---:|---:|
| main: `JsonSerializer` into `Dictionary<string, Dictionary<string, string?>>` | 375-430 | 900-1,000 |
| `Utf8JsonReader` loop (round 34's) | 370-410 | 890-940 |
| + dictionaries presized by a counting pass | 540-610 | 950-1,100 |
| + a lemma equal to its word shares the word's string | 530-600 | 950-1,120 |
| + lemmas interned (`HashSet` alternate lookup) | 560-610 | 980-1,040 |
| bytes + hash table, scanned with `Utf8JsonReader` | 215-265 | 750-860 |
| **bytes + hash table, own scanner (kept)** | **108-136** | **720-780** |
| no dictionary at all (upper bound) | 2 | 680-780 |

- Presizing made it slower, not faster, and sharing or interning lemma strings saved nothing measurable: building
  half a million strings and dictionary entries is what costs, not the parser or the rehashing.
- Keeping the bytes removes the strings. `Utf8JsonReader` then took 160 ms of the 230 to scan 15 MB; a scanner of
  the format `json.dumps` writes (`{"pos": {"word": "lemma" or null, ...}, ...}`, any whitespace between tokens,
  `IndexOfAny('"', '\\')` per string) takes 60. A string with an escape is decoded by `Utf8JsonReader` and its UTF-8
  stored after the JSON. Anything else in the file throws `InvalidDataException`.
- What is left over the upper bound (~60 ms of German's load) is the scan and the table.

## Measurements

Local Ryzen 7 5800X, Windows 11, idle apart from this work (3% CPU load before the runs). `tools/startup.py` (this
round adds `--lang`, `--package` and `--hash` to it and `--lang`/`--package` to `StanzaSharp.Benchmark startup`):
`.pt` files, default threads, 7 processes per setting alternating, medians; one 13-word sentence per call.

| config | setting | load ms (CPU) | load peak MB | call 1 ms | load + 2 calls |
|---|---|---:|---:|---:|---:|
| de default | main | 925 (1,578) | 1,169 | 219 | 1,200 |
| de default | round 35 | **775 (1,359)** | 1,191 | 209 | **1,041** |
| en default | main | 548 (1,078) | 896 | 201 | 787 |
| en default | round 35 | 535 (1,078) | 894 | 206 | 781 |

The load peak is the working set right after the load; its median moves ±25 MB from process to process. What the
dictionary keeps afterwards falls by about a third (computed, not measured: ~58 MB of strings and dictionary entries
for German before, ~37 MB of bytes and table now).

## Exactness

- `startup --hash` (every golden text of the language, each processed alone): English `4FD0F4EE1F4DBBED` (round 17's),
  German `6C5B75A493E5306E`, the same for main and round 35.
- `LemmaTests.Dictionary_EqualsJsonSerializers` looks every entry of the English and German dictionaries up in the new
  table and compares it with `JsonSerializer`'s; `Dictionary_ReadsJsonLikeJsonSerializer` covers escapes, null lemmas,
  repeated keys (the last wins, a repeated POS replaces the earlier one), whitespace, a word with a lone surrogate and
  malformed JSON. `LemmaTests` passes on both backends and languages.
- Lookups keep their semantics: (UPOS, word) first, then `*`; a null lemma counts as missing.

`LemmaJson`, the source-generated `JsonSerializer` context the dictionary needed for Native AOT (docs/aot.md), is gone
with it.
