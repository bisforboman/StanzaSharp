# German lemma dictionary at load

An open idea ([list](../../performance.md#remaining-ideas)).

[Round 34](../round-34-load-packed-cache.md) found that German `default`'s lemmatizer spends about 0.77 s of thread
time (1.0-1.2 s locally) in its dictionaries: 2.2 MB of gzip, 14.7 MB of JSON, 524,460 entries read by
`JsonSerializer.Deserialize` into 31 dictionaries. That is 27% of the German load's thread time, the largest single
item. English's dictionary is 37,130 entries, about 80 ms.

A `Utf8JsonReader` loop (same dictionaries, checked entry by entry) only got it to 554 ms: most of the cost is building
a million strings and their dictionary entries. Untried:
- Share a lemma string with its word when they are equal, and look the word up once for all POS.
- Build the dictionaries lazily on the first call that needs them, or on a background task started at load.
- A cheaper stored form, such as sorted arrays with binary search, or one frozen dictionary of (POS, word) keys.

Lookups must keep the current semantics: (UPOS, word) first, then `*`, and a repeated key keeps the last value.
