# Results, round 3: memory of a short-lived process

Part of [performance.md](../performance.md)'s measurement history.

Issue #19: a process that loads `tokenize,mwt,pos,constituency` (`default` package) from Stanza's `.pt` files, makes one
`Process` call and exits peaked 230–310 MB above Python Stanza. `--memory N` measures exactly that, one fresh process
per run: the text is `corpus.txt` and `validation*.txt`, repeated and cut after the paragraph that reaches N
whitespace-separated words.

```powershell
dotnet run -c Release --project samples/StanzaSharp.Benchmark -- --memory 3750 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza\en [--verbose]
tools\.venv\Scripts\python tools\benchmark.py --memory 3750 --processors tokenize,mwt,pos,constituency --threads 8 --models models\stanza
```

Peak working set in MB, 8 threads, Windows. "Load" is the peak after `Pipeline.Load`, "after load" the working set
then. Server GC is `DOTNET_gcServer=1`. "Before" is `main` at eb87c77. Python is 1.15.0 with torch 2.14.1.

| | .pt, workstation GC | .pt, Server GC | converted, workstation | converted, Server | Python |
|---|---:|---:|---:|---:|---:|
| load peak, before | 968 | 751–1,090 | 484 | 482 | 733 |
| load peak, after | 536 | 498–537 | 484 | 482 | |
| after load, before | 967 | 750–1,090 | 483 | 482 | 659 |
| after load, after | 536 | 498–537 | 483 | 482 | |
| 499 words, before | 1,066 | 1,209 | 607 | 602 | 746 |
| 499 words, after | **652** | **652** | 604 | 600 | |
| 5,196 words, before | 2,604 | 2,756 | 2,357 | 2,292 | 2,356 |
| 5,196 words, after | **1,613** | **1,586** | 1,567 | 1,580 | |
| 15,151 words, before | 2,826 | 2,821 | 2,519 | 2,565 | 2,460 |
| 15,151 words, after | **1,684** | **1,715** | 1,660 | 1,706 | |

Process times did not change (0.8 s, 6.2 s and 15.8 s; Python 1.3 s, 13.1 s and 33.3 s). The 5k-word output equals
Python's byte for byte.

Where the memory went:
- **Loading `.pt` files** (430–590 MB). `TorchCheckpoint` read each file into a `byte[]` and copied its tensors into a
  second one; the 290 MB of checkpoints this processor set reads became about 580 MB of garbage. The load ended before
  a GC returned it, so it stayed in the working set into `Process`. Server GC collected at different points, hence
  its spread. Now the unpickler reads from a stream, only the storages' positions are noted, and each tensor is read
  from the file straight into its memory, as for safetensors. The `.pt` path now costs 50 MB over the converted one
  (the unpickled object graph, garbage until the next GC).
- **The tagger's padded batches** (800 MB at 5k words, in C# and Python alike). Batches are 250 sentences in
  document order, padded to the longest: with a 139-word sentence, the concatenated input alone is
  [250, 139, 2248] floats, 312 MB, and the padded charlm outputs another 285 MB. Everything stayed in the batch's
  dispose scope: the inputs, their concatenation, and each highway layer's six [250, 139, 400] intermediates. Now
  the inputs are freed once concatenated, the concatenation after the first layer, and each layer's intermediates when
  it ends, with the gate's elementwise steps in place (the same kernels, so the same values). That is why C# now
  peaks 0.8 GB below Python on the longer texts. The reporter's transcript has shorter sentences, so this gains less
  there.
- **Not the cause:** tensors left to the finalizer (a full GC with finalizers frees nothing after `Process`), the
  `CharlmCache` (8 KB per word: 40 MB at 5k words; off, the peak differs by under 50 MB, while the run takes 30–40%
  longer), and the constituency parser, whose 50 states in flight never set the peak.
- After `Process` the working set stays near the peak although every tensor is freed: the native allocator keeps
  the pages for reuse. That raises the steady state of a long-running process, not its peak. Round 5 finds who keeps
  them, and returns them on Linux.

Splitting the input (15,151 words, converted models, workstation GC). The output of the parts equals one call's
except for sentence ids, offsets (each part's own) and the whitespace at each cut (`SpaceAfter` of a part's last
token, `SpacesBefore` of its first):

| calls | peak | time |
|---|---:|---:|
| one `Process` call | 1,664 MB | 15.5 s |
| 12 calls of about 1,000 words | 1,156 MB | 18.0 s |
| 6 calls of about 2,000 words (`.pt`) | 1,836 MB | 16.7 s |
| 629 calls, one paragraph each | 574 MB | 81.0 s |
| one bulk `Process(texts)` call on the 12 parts | 1,705 MB | 15.6 s |

The peak follows the largest padded tagger batch, so parts help only once they hold well under 250 sentences: the
2,000-word parts happened to put the longest sentence in a fuller batch than one call did. Bulk processing batches all
texts together, like one call.
