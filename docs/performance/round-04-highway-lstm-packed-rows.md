# Results, round 4: the highway LSTMs on packed rows

Part of [performance.md](../performance.md)'s measurement history.

Round 3 left the tagger's padded batch as the peak: batches are 250 sentences in document order, so one long sentence
padded the whole [250, longest, 2248] input, both charlm outputs and every highway layer's intermediates. Now nothing
in the tagger is padded:
- `Rnn.PackedOrder` gives the row order pack_padded_sequence (enforce_sorted: false) would produce, and the tagger
  builds its input straight in that order: embeddings looked up in packed order, each charlm's per-sentence outputs
  concatenated and reordered once.
- `Rnn.Pack` makes the `PackedSequence` without a padded input (TorchSharp can't build one from data: it packs a
  one-wide zero input and points the result's data at the packed rows with `set_`, no copy).
- `HighwayLstm.Forward(PackedSequence)` runs the gate and highway layers on the packed rows and adds them in place to
  the LSTM's packed output, which is the next layer's input. The heads read the last layer's rows in sentence order.
- Depparse keeps its padded `Forward(Tensor, lengths)`, now a wrapper (pack, the packed path, pad). Its batches are
  sorted by length, so it had little padding; its code is unchanged.

The LSTMs see exactly the same packed data, batch sizes and order as before. Only the linear layers (`trans_pretrained`,
`trans_char`, gate, highway) now multiply [words, n] instead of [batch, longest, n] matrices, which moves the last
float bits. Measured against the previous implementation over `corpus.txt` and every `validation*.txt`
(tokenize,mwt,pos,lemma,depparse, both packages), max abs difference:

| output | default | default_fast |
|---|---:|---:|
| UPOS logits | 2.3e-5 | 2.3e-5 |
| XPOS scores | 4.6e-5 | 4.6e-5 |
| UFeats scores (21 heads) | 4.6e-5 | 4.6e-5 |
| depparse arc log-probs | 1.5e-5 | 2.3e-5 |
| depparse label log-probs | 1.9e-5 | 2.3e-5 |

That is the size of the existing drift from Python (UPOS logits ≈ 2e-5); the tests' tolerances are unchanged and pass.
Every discrete output is identical: the 26 CoNLL-U files of that comparison byte for byte, the full test suite (all
golden files: pipeline, validation, lemma, depparse, ner, fast, sentiment, bulk, pretokenized) with converted and
with `.pt` models, nothing skipped, and the 26k-word benchmark output.

`--memory N --processors tokenize,mwt,pos,constituency --threads 8`, peak working set in MB, workstation GC; before is
`main` at aaf6e02. Before and after were alternated, two runs each (the peaks agreed within 45 MB). The machine was
shared with other jobs, so the times are noisy: the pos columns give the range over the four runs of each size (both
model formats), leaving out two runs that took 2–3× as long (one after at 5k words, one before at 15k).

| words | converted, before | converted, after | `.pt`, before | `.pt`, after | pos before | pos after |
|---|---:|---:|---:|---:|---:|---:|
| 499 | 596–611 | 611 | 656 | 656 | 0.6–1.0 s | 0.6–0.8 s |
| 5,196 | 1,567 | **1,016** | 1,615 | **1,054** | 6.8–9.1 s | 4.7–6.0 s |
| 14,767 | 1,655 | **1,248** | 1,678–1,722 | **1,270** | 15.7–16.6 s | 12.0–14.1 s |

- At 5k words the peak falls by 550 MB (35%), at 15k by 410 MB. Python Stanza peaked at 2,356 MB on the same
  5,196 words (round 3).
- The tagger no longer sets the peak at 15k words: after pos the process is at about 1,110 MB, and the constituency
  parser takes it to 1,250 MB.
- 499 words has no long sentence in a batch, so nothing changes there.
- pos is about 20–25% faster where a batch has a long sentence: the gate and highway layers no longer multiply padding.

All eight processors, the usual benchmark (26,264 words, converted models, 8 threads, 3 timed runs after a warm-up),
before and after alternated twice; the shared machine made the other stages vary by up to 2.5× between runs:

| | before | after |
|---|---:|---:|
| pos | 19.72 s, 15.71 s | 13.02 s, 11.03 s |
| total | 96.72 s, 71.53 s | 68.86 s, 68.31 s |
| peak memory | 3,007 MB, 3,002 MB | 3,015 MB, 3,002 MB |

pos is 25–35% faster. The 8-processor peak is set by later stages, so it does not change. Both outputs were
byte-identical.
