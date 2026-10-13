# Results, round 10: sentiment memory on the managed backend

Part of [performance.md](../performance.md)'s measurement history.

With all eight processors, sentiment set the managed pipeline's peak: on 6,761 words it added ~1.15 GB (1,150 → 2,300 MB).

**Where it goes** (a temporary probe of the batch shapes, `--memory`, 8 threads): Stanza's batches are 5000 *real* tokens,
but each is padded to its longest sentence, and the classifier's LSTM and convolutions run over the padding. The first
batch of the 6,761-word text is 257 sentences × 139 tokens = **35,723 rows** for 5,000 tokens (20,513 words: 129 × 139 =
17,931 rows). Every per-row buffer was multiplied by that: the input [rows, 2148], the LSTM's GEMM output (both
directions' gates, [rows, 2400]), two layer outputs and the copied output ([rows, 600] each), its batch-major copy, and
each full-width convolution's output ([rows, 1008]): ~7,950 floats = 32 KB per row, 1.14 GB on 35,723 rows.

**What changed** (exact; see below):
- `ManagedLstm.ForwardUnpacked` (new; sentiment's unpacked biLSTM, which ran `ForwardPacked` with every batch size = n):
  per layer and direction, the input projection is computed for a chunk of ~2048 rows (a few time steps) just before
  those steps run, through a panel view of that direction's gate columns, and the steps are driven with
  `PackedLstm.Step` one direction at a time. The layer-0 input rows are built per chunk by a callback (embeddings +
  charlm rows, which stay per real token: from the cache, or one [tokens, 1024] buffer per direction), so the [rows, 2148]
  input never exists. The last layer writes batch-major, which is what the convolutions read, so the copy is gone too.
  What is left per row: layer 0's output and the final output, 2 × 600 floats.
- The full-width convolutions run 2048 rows at a time into one [2048, 1008] buffer, each chunk max-pooled as it comes.
- `Gemm.RunRows(a, r0, r1, m, ...)` computes rows [r0, r1) of an m-row GEMM with the bits the whole call gives them:
  every kernel but the single-row one gives a row the same arithmetic (the 3-row kernel's per-row sum is the 6-row
  one's), and `Run` uses the single-row kernel only for row m − 1 when m % 6 == 1 (MC = 96 is a multiple of 6). So a
  chunk runs `rowInvariant`, and that one row alone. Round 9's "splitting by rows is not exact" was about splitting a
  batch's sequences (fewer rows per LSTM step); splitting a GEMM's rows this way is.
- Proof: `ManagedBackendTests.Gemm_RowRangesGiveTheWholeGemmsBits` (chunks of 1, 5, 7, 13, 96 rows; m % 6 == 1 and not)
  and `ManagedLstm_UnpackedGivesThePackedBits` (sentiment's checkpoint; one-step chunks, multi-step chunks, n·width % 6 == 1)
  on every path. And every managed sentiment logit and label on all golden sentiment data (the 14 files and all of them
  as one document, each without and with the tagger's charlm cache: 10,632 logits), SHA-256 per kernel path, on both
  builds (a temporary test): Vector256 `01356120…`, Vector128 `9763DA4D…`, Scalar `8473EDF3…`, identical before and after.

**Measured** (`--memory N`, converted models, Windows; before = 475211d (PR #65's branch), after = this round,
alternated, two rounds; peaks first, sentiment stage time in brackets. Up to three other agents' test suites kept the CPU
at 100% throughout, so these times are noise and only the peaks count; idle-machine times are below the table):

| | 8 threads | 1 thread |
|---|---:|---:|
| all eight, 6,761 words | 2,306 / 2,304 → **1,314 / 1,311 MB** (71 / 68 → 115 / 86 s) | 2,308 / 2,291 → **1,318 / 1,336 MB** (95 / 77 → 85 / 75 s) |
| all eight, 20,513 words | 1,859 / 1,873 → **1,379 / 1,372 MB** (147 / 76 → 63 / 175 s) | 1,864 / 1,880 → **1,378 / 1,412 MB** (135 / 100 → 24 / 22 s) |
| `default_fast`, 6,761 words | 2,002 / 2,001 → **1,058 / 1,056 MB** (203 / 97 → 241 / 110 s) | 2,015 / 2,011 → **1,060 / 1,062 MB** (135 / 135 → 135 / 115 s) |
| `default_fast`, 20,513 words | 1,476 / 1,478 → **1,071 / 1,073 MB** (272 / 180 → 113 / 109 s) | 1,463 / 1,483 → **1,075 / 1,076 MB** (261 / 96 → 260 / 80 s) |

- The default pipeline's peak drops ~1.0 GB at 6,761 words (−43%) and ~0.48 GB at 20,513 words (−26%; its widest batch
  has half the rows); sentiment now adds ~0.18 GB on 6,761 words (1,126 → 1,303 MB in the per-stage report), which is the
  two [rows, 600] layer outputs (171 MB on 35,723 rows). `default_fast` drops ~0.95 GB (−47%) and ~0.4 GB (−27%).
- Speed: the same multiply-adds; the input rows are built twice (once per direction of layer 0), a copy of 2148 floats
  per row against 2148 · 1200 multiply-adds. On the idle machine afterwards (one run each, all eight): the sentiment
  stage took 3.51 → 3.50 s (6,761 words, 8 threads), 14.6 → 13.7 s (1 thread), 4.26 → 4.10 s (20,513 words, 8 threads),
  20.4 → 20.1 s (1 thread), with the same peaks as in the table (2,286 → 1,329 MB; 1,908 → 1,409 MB).

**What is left:** layer 0's time-major output and the final batch-major output (1,200 floats per padded row). Layer 1's
backward direction frees layer 0's rows as it goes while the output fills, so a shared layout could hold ~900 floats
per row (−43 MB at 6,761 words); not worth the bookkeeping. The padding itself is Stanza's batching and must stay.
