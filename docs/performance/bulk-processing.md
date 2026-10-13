# Bulk processing: many short texts

Part of [performance.md](../performance.md)'s measurement history.

`agent/input-modes`: 2,000 one-sentence texts (the `# text` lines of `tests/golden/validation*.conllu`, cycled),
all processors of each package, 8 torch threads. "One by one" calls `Process(text)` per text (Python `nlp(text)`);
"bulk" makes one `Process(texts)` call (Python `nlp.bulk_process(texts)`). One timed run each, after a warm-up
on 50 texts:

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --models models\converted\en --threads 8 --documents 2000 [--package default_fast]
tools\.venv\Scripts\python tools\benchmark.py --models models\stanza --threads 8 --documents 2000 [--package default_fast]
```

| | C# default | Python default | C# default_fast | Python default_fast |
|---|---:|---:|---:|---:|
| one by one | 421.96 s (5 docs/s) | 366.33 s (5 docs/s) | 217.10 s (9 docs/s) | 130.06 s (15 docs/s) |
| bulk | 54.54 s (37 docs/s) | 91.87 s (22 docs/s) | 24.23 s (83 docs/s) | 32.20 s (62 docs/s) |
| speed-up | 7.7× | 4.0× | 9.0× | 4.0× |

- Bulk is 7.7–9× faster in C#, and 1.3–1.7× faster than Python's bulk.
- One by one, C# is *slower* than Python (by 15% and 67%): a single short sentence pays a fixed cost per call
  that bulk spreads over many sentences. Per call on one sentence (default_fast, 8 threads), C# vs Python:
  sentiment 53 vs 26 ms, tokenize 6.4 vs 2.2 ms, depparse 22.5 vs 20.3 ms, pos 11.7 vs 12.0 ms. That was the TorchSharp
  backend; on the managed backend (the default since 0.5) one call takes 5–7× less time than Python's (round 7).
- The output of bulk differs from one by one only in sentiment labels (and sentence ids), exactly as in Stanza:
  see `Pipeline.Process(IEnumerable<string>)`.
