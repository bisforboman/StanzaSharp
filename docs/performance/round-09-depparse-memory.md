# Results, round 9: depparse memory on the managed backend

Part of [performance.md](../performance.md)'s measurement history.

Without sentiment, depparse sets the managed pipeline's peak: `tokenize,mwt,pos,lemma,depparse` on 6,761 words rises
from ~700 MB after lemma to ~900 MB. Round 8 measured where that goes and cut what can be cut exactly.

**Where it goes** (working set at points inside `ManagedDepparseNet.Forward`, first 5000-word batch, 8 threads; a
temporary probe, not committed): the charlm pass +57 MB (the 2423-wide input, 51 MB, and the charlm outputs), the
highway LSTM +107 MB (its GEMM output, 12·400 floats per row = 100 MB, plus two layer outputs), the scorers +34 MB (all
four scorers' W1/W2 in one [rows, 3200] GEMM, 67 MB, plus the arc scorers' T, 26 MB), the label scorer +1 MB.

**What changed** (both exact: a GEMM column depends only on its 16-column panel, so splitting a GEMM by whole panels
gives the same bits; `ManagedBackendTests.Gemm_ColumnPanelsAreIndependent` on every path):
- `ManagedHighwayLstm` (pos and depparse): per layer the GEMM wrote both directions' gates and the gate and highway
  layers (12·hidden per row). Now the 8·hidden gate columns go first, the recurrence runs, and then the gate/highway
  columns are computed into the same buffer. Both GEMMs are panel views of the one packed matrix (`new
  PackedMatrix(source, firstPanel, panels)`, no copy). Depparse's buffer for 5,250 rows: 100 → 67 MB; pos's: 48 → 32 MB.
- `ManagedDepparseNet`: each scorer computes its own W1/W2 [rows, 800] just before it is used, and the arc scorers run
  one after another (each pair's three terms are accumulated in the same order as before, in a [rows, width] double
  buffer), so only one scorer's in1/in2 and T are live.
- Proof beyond the panel test: every managed depparse arc log-prob and label score of every pair (`labelScores: true`)
  and every pos UPOS logit on all golden data (depparse/ and fast/, 8,401 words, both checkpoints), hashed per kernel
  path (Vector256, Vector128, Scalar): identical before and after (a temporary test, run on both builds).

**Measured** (`--memory N`, all `.json`/`.safetensors` models, Windows; before = main at 629634f, after = this round,
alternated, two rounds; another job kept the CPU at 100% throughout, so the times are noise and only the peaks count):

| | peak, 8 threads | peak, 1 thread | depparse, 8 threads | depparse, 1 thread |
|---|---:|---:|---:|---:|
| 5 processors, 6,761 words | 896 / 901 → 826 / 831 MB | 900 / 900 → 830 / 830 MB | 10.4 / 8.3 → 12.1 / 8.1 s | 21.2 / 20.2 → 21.3 / 19.7 s |
| 5 processors, 20,513 words | 994 / 1,018 → 933 / 949 MB | 1,000 / 1,015 → 945 / 947 MB | 21.2 / 16.8 → 18.9 / 16.9 s | 57.8 / 48.5 → 50.2 / 46.1 s |
| all eight, 6,761 words | 2,353 / 2,354 → 2,402 / 2,409 MB | 2,358 / 2,356 → 2,395 / 2,409 MB | 8.4 / 9.3 → 8.9 / 7.5 s | 22.5 / 20.3 → 21.7 / 18.7 s |
| all eight, 20,513 words | 1,976 / 2,021 → 2,012 / 2,005 MB | 2,004 / 2,014 → 2,019 / 2,006 MB | 18.2 / 13.7 → 10.2 / 17.3 s | 51.4 / 53.6 → 57.1 / 53.5 s |

- Without sentiment the peak drops 60–70 MB (7%). Each change alone (6,761 words, 8 threads): the LSTM split −60 MB, the
  per-scorer W1/W2 −10 MB.
- With all eight, the peak is sentiment's, not depparse's: sentiment adds ~1.2 GB on 6,761 words (1,156 → 2,344 MB).
  Its buffers come from the same `ArrayPool`, and with the LSTM split pos's buffer moved from the 16M-float bucket to
  the 8M one, so sentiment no longer finds a touched 16M array to reuse and touches a new 64 MB one: +50–60 MB at 6,761
  words (the per-scorer change alone: −3 MB), nothing at 20,513. `default_fast` (all seven): 2,107 / 2,109 → 2,106 /
  2,107 MB at 6,761 words, 1,632 → 1,647 MB at 20,513.
- Speed (`--processors tokenize,mwt,pos,lemma,depparse`, 26,264 words, 8 threads, one run each, same load): pos 25.4 →
  26.3 s, depparse 20.2 → 21.0 s, tokenize (unchanged code) 5.9 → 9.1 s: within the noise. The work is the same; the
  LSTM reads its input twice per layer.

**Tried and dropped:**
- *Labels only for the chosen heads* (the parser decodes the trees first, then the label scorer runs one head per word,
  bit for bit the full matrix's scores). The managed net only ever kept an int per pair (~5 MB) and T dominates the label
  scorer (the 400 × 401·49 GEMM every word needs), so it saved ~5 MB and no measurable time, for an extra decode
  callback in the network interface.
- *Exact-size arrays instead of `ArrayPool`* for the three biggest buffers (input, LSTM GEMM output, scorer W1/W2): peak
  820 → 970 MB. The pool's power-of-two rounding costs no working set (the tail of a rented array is never touched);
  fresh arrays pile up on the large-object heap between collections.
- *Splitting the batch* for the encoder is not exact: the charlm and the packed LSTM step all of a batch's rows through
  one GEMM, whose single-row kernel sums differently, so a different split moves the last bits (the arc log-softmax is
  per row, but its inputs would change). After the per-scorer change the scorers add ~20 MB, so splitting them alone
  wasn't worth the bookkeeping that keeps their GEMMs' row blocks as they are.

**What is left:** sentiment's ~1.2 GB on a 5000-token batch (its padded [rows, 2148] input, the unpacked biLSTM's GEMM
output, the convolutions' outputs) is now the default pipeline's peak; depparse's input (51 MB) and LSTM buffers stay.
