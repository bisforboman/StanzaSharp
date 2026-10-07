# StanzaSharp

A native C# port of the inference side of [Stanza](https://github.com/stanfordnlp/stanza), Stanford's
PyTorch-based NLP library, running the original pretrained models through TorchSharp.

## Scope

- **Language:** English only, to start.
- **Processors:** all eight of Stanza's English default, in Stanza's order (`PIPELINE_NAMES`): tokenize,
  mwt, pos, lemma, constituency, depparse, sentiment, ner. This is `Pipeline.AllProcessors`, the default
  (user's decision, 2026-10-06: like Stanza; it costs about 36% more time than without ner and sentiment).
  - `Pipeline.Process` runs them in that order whatever order they are listed in. Only the CoNLL-U
    comment order is observable (`# sentiment` after `# constituency`).
  - `pipeline.conllu` and `validation*.conllu` are generated with all eight (`PROCESSORS` in
    make_golden.py).
- **Inference only.** Training stays in Python; we load Stanza's released weights.
- **Package:** Stanza's English *default* package, which needs no transformer:
  - tokenize: `combined_nocharlm`
  - mwt: `combined`
  - pos: `combined_charlm`
  - lemma: `combined_nocharlm`
  - constituency: `ptb3-revised_charlm`
  - depparse: `combined_charlm`
  - ner: `ontonotes-ww-multi_charlm`
  - sentiment: `sstplus_charlm`
- **Package `default_fast`** (user's decision, 2026-10-06: selected by Stanza's package name,
  `PipelineOptions.Package`): tokenize, mwt and lemma as above; pos `combined_nocharlm`, depparse
  `combined_nocharlm`, ner `ontonotes-ww-multi_nocharlm`; sentiment `sstplus_charlm`; no constituency.
  Its default processors are those seven. `Pipeline.Packages` maps package → processor → model.
- **Shared dependencies:** `pos`, `constituency`, `depparse`, `ner` and `sentiment` all depend on the
  pretrained word-vector file (pretrain). Every `_charlm` model also depends on the forward and backward
  character LMs (charlm); the `_nocharlm` ones have their own character LSTM (`Nn.CharacterModel`) instead.
  The charlm is shared infrastructure and lives in `StanzaSharp.Nn`. The pipeline loads only what the
  selected models read (`Pipeline.SharedModels`): in `default_fast` the charlms only with sentiment.

## Status

All 7 steps of the build order are done:
- Models are converted, golden data is in `tests/golden/`, and the checkpoint configs are summarized
  under "Checkpoint findings".
- `Core` has the document model, `Conllu` (round-trips the golden file byte for byte), `Tree`,
  `SafeTensorFile` and `Checkpoint`.
- `Nn.Weights.LoadFrom` fills any TorchSharp module from a checkpoint by parameter-name prefix.
- `Tokenize.Tokenizer` matches Stanza's tokens, offsets, spacing and sentences on the golden corpus
  and the stress set.
- `Mwt.MwtExpander` uses the dictionary, then the character classifier. It matches the golden
  words, word offsets, and `mwt.json`, which covers dictionary and classifier-only output.
- `Nn.Rnn.RunPacked` wraps pack/LSTM/unpack for models that pack by real length (MWT, POS,
  and the parser's word LSTM).
- `Nn.CharLanguageModel.BuildCharRepresentation` gives one vector per word, taken at the space
  after it. It matches the golden charlm tensors within 1e-4. The tagger and the parser both use it.
- `Nn.Pretrain` provides `UnitToId` (spaces stored as U+00A0) and `Embeddings`. Callers decide
  on lowercasing: POS lowercases every word, while the parser tries the word as written, then
  lowercased.
- `Pos.PosTagger` takes the shared `Pretrain` and both charlms, which the caller owns. It matches the
  golden UPOS/XPOS/feats exactly after tokenize → mwt → pos. `Nn` gained `HighwayLstm` and
  `Biaffine`. Like Stanza's `simplify_punct`, the tagger sees runs such as `??` or `!?!` as `?`
  or `!`. Only the tagger does this, not the parser.
- `Constituency.ConstituencyParser` also takes the shared `Pretrain` and charlms. It reproduces the
  golden trees and the per-step transition scores. It schedules like Stanza's `parse_sentences`:
  sentences sorted longest first, 50 states in flight, each finished state replaced by the next.
  Each step is one forward pass plus batched stack pushes. Stack LSTMs are stepped without
  padding, so batching does not change results.
- `Nn.CharlmCache`: the pipeline passes one to `PosTagger.Process` and `ConstituencyParser.Process`,
  so the parser reuses the tagger's charlm outputs (the tagger keeps a sentence only if
  `simplify_punct` left its words unchanged). The charlm is not bitwise batch-invariant, so cached
  values can differ from a fresh computation in the last bits; outputs are still identical.
- Performance: `docs/performance.md` has the benchmark (`samples/StanzaSharp.Benchmark`,
  `tools/benchmark.py`), C# vs Python numbers, what was optimized, and remaining ideas.
  Round 2 (memory): checkpoints are read tensor by tensor straight into TorchSharp memory (no
  whole-file buffers), depparse's biaffine scorers run a few sentences at a time, POS batches are
  cut at 5000 words like Stanza, and `CharlmCache` keeps at most 32k words (`MaxWords`).
  Round 3 (issue #19, `--memory`): `.pt` files are read tensor by tensor too (the unpickler reads a stream; storages
  are read from the file on demand), and `HighwayLstm` frees each layer's padded tensors as soon as they are used.
- `Tree.ToString` prints `(`/`)` in labels and words as `-LRB-`/`-RRB-`, like Stanza. The tree
  itself keeps the raw text.
- Measured float drift vs Python: charlm < 1e-7, tokenizer logits ≈ 4e-6, UPOS logits ≈ 2e-5,
  parser scores ≈ 2e-5. Test tolerances are 1e-4 and 1e-3.
- `StanzaSharp.Pipeline` ties the processors together: `Load(modelDir, "tokenize,mwt,...")`, then
  `Process(text)`. It loads the shared pretrain/charlms once. `Conllu.Write(pipeline.Process(corpus))`
  is byte-identical to `tests/golden/pipeline.conllu`.
  - The `samples/StanzaSharp.Cli` sample writes the same CoNLL-U from a file or stdin.
  - `Conllu.Write` fills a missing HEAD with `id - 1`, like Stanza's writer.
- `Depparse.DependencyParser` (processor `depparse`, requires tokenize, mwt, pos, lemma; runs after
  lemma and constituency, as in Stanza) sets `Word.Head`/`Deprel` with the pipeline's shared pretrain/charlms.
  `Pipeline` with `tokenize,mwt,pos,lemma,depparse` reproduces every `tests/golden/depparse/*.conllu`
  byte for byte (13 files, 8401 words), and so does the parser alone on Stanza's own tags and lemmas.
  Arc/label log-prob drift ≈ 2e-5 (tolerance 1e-4).
  - No `CharlmCache`: the parser feeds the charlms `"\n"` + the words (a ROOT word), so every forward
    state and the backward ROOT state differ from the tagger's.
  - A missing lemma is read as `_`, as in Stanza, so without the lemmatizer the parses differ.
  - `Conllu.Write` needs no change: with heads set it matches Stanza (DEPS stays `_`).
- `Lemma.Lemmatizer` (processor `lemma`, requires tokenize, mwt, pos) reproduces `tests/golden/lemma/`
  byte for byte: 13 files, 8401 words, 868 of them through the seq2seq model. It mirrors
  LemmaProcessor: dictionary skip, DeltaVocab, `batch_size` 50 batches sorted like `sort_all`, greedy
  decoding, edits, `<UNK>` fallback, and the `Word.lemma` setter turning `_` into null.
- `Ner.NerTagger` (processor `ner`, requires only tokenize: it reads token text, like Stanza) sets
  `Token.Ner` (BIOES) and `Sentence.Entities` (`Entity`: Text, Type, StartChar, EndChar, Tokens;
  `Document.Entities` concatenates them). `tests/golden/ner/` is reproduced byte for byte: 13 files,
  845 sentences, 706 entities, with `tokenize,mwt,pos,lemma,depparse,ner`, with the cache off, and with
  `tokenize,ner` alone. Emission drift ≈ 2.4e-6 (tolerance 1e-4).
  - `Entity.Build` is `build_ents`/`decode_from_bioes`; `Conllu.Read` calls it too (text from the
    tokens, since a read document has no `Text`).
  - `Conllu` writes/reads `ner=` after `end_char` in MISC, on the MWT range line and on single-word
    tokens' word lines, never on MWT words. `multi_ner` is not written by Stanza and not ported (with
    one model it is just `(ner,)`).
  - `CharlmCache`: NER's charlm input (`"\n"`, each token + `" "`, same vocab) equals the tagger's
    when a sentence's tokens are its words (no MWT), so it reuses those; others are computed.
- `Checkpoint.Load` also reads Stanza's original `.pt` files (`Core/TorchCheckpoint.cs` +
  `Core/Pickle.cs`), so `Pipeline.Load("models/stanza/en")` works without Python. For all fourteen
  checkpoints the result equals the converter's: identical JSON and byte-identical tensors.
- `Sentiment.SentimentClassifier` (processor `sentiment`, requires tokenize; runs last, as in Stanza)
  sets `Sentence.Sentiment` (0 negative, 1 neutral, 2 positive). `Conllu` reads and writes
  `# sentiment = N` after `# constituency`, Stanza's comment order. `tokenize,mwt,sentiment`
  reproduces `tests/golden/sentiment/` byte for byte (14 files, 886 sentences), with every label and
  logit (drift ≤ 2.2e-5, tolerance 1e-4), also as one document in two 5000-token batches (`all.json`).
  - Batches must match Stanza's: padding runs through the unpacked biLSTM and the convolutions, so a
    sentence's result depends on its batch (one document vs per file: 173 of 886 labels change in Stanza itself).
  - `CharlmCache`: the classifier reads token texts, so it reuses the tagger's outputs for sentences
    without multi-word tokens (641 of 886). Labels stay identical; logits drift up to ~1e-4 (vs
    Stanza), as the network amplifies the cached charlm's last-bit differences; tested at 1e-3.
- `default_fast` (`PipelineOptions.Package`): `PosTagger`, `DependencyParser` and `NerTagger` switch on the
  checkpoint's `charlm` flag and take null charlms for `_nocharlm` checkpoints. `tests/golden/fast/` is
  reproduced byte for byte (13 files, 845 sentences, 8,401 words, all seven processors). Drift: UPOS
  logits 1.5e-5, arc log-probs 7.6e-6, label log-probs 1.5e-5, NER emissions 1.9e-6 (tolerance 1e-4).
  - No `CharlmCache` there: the nocharlm tagger has no charlm outputs to share (`PosTagger.UsesCharlm`).
  - Stanza silently skips a processor its package lacks (`processors='...,constituency'` with
    `default_fast` loads no parser); we throw instead (user's decision), naming the `default` package.
- Input modes (2026-10-06, the user's approved examples): `Process(IEnumerable<string>)` (bulk) and
  `Process(IEnumerable<IEnumerable<string>>)` (pretokenized). Collection expressions pick the right overload;
  `Process([])` and `Process(null)` are ambiguous (compile errors), never a silent wrong pick.
  - Bulk = Stanza's `bulk_process`: `Tokenizer.Process(texts)` tokenizes `"\n\n".Join(texts)` and `Tokenizer.Split`
    deals sentences out like `TokenizeProcessor.bulk_process` (a sentence goes to the current text while its last
    token ends inside it; offsets shifted; the last token's SpaceAfter and first token's SpacesBefore from the text
    itself). The other processors run on one combined `Document` (UDProcessor.bulk_process), then entities are
    rebuilt per document. Sentence ids continue across documents, as in Stanza.
  - Bulk ≠ one by one, in Stanza too: sentiment batches change labels (172 of 854 in the golden; depparse could too, via
    its padded log-softmax, but no head changed), and sent_ids continue. Without those two processors the output equals one call per text apart from
    sent_id (`Bulk_EqualsProcessingAloneExceptBatchedProcessors`).
  - Pretokenized = `process_pre_tokenized_text` with a list: text = tokens joined by single spaces, offsets into
    it, `IsMwtCandidate` false, so mwt changes nothing ("don't" stays one word, as in Stanza). Empty sentences and
    empty/whitespace tokens throw (Stanza fails on the first two; whitespace tokens give broken CoNLL-U there).
    Stanza's string form (whitespace tokens, newline sentences) is not exposed; make_golden asserts it equals the list form.
- 0.4 API (issue #12, items chosen by the owner, 2026-10-06):
  - `PipelineOptions.Threads` (int?): `torch.set_num_threads` at Load, process-wide. Null = min(libtorch's current
    value, `Environment.ProcessorCount`), so a container's CPU quota is respected and a lower count the caller set is
    kept. Inter-op threads are untouched (eager mode never uses that pool; libtorch allows setting it only once).
    libtorch's OpenMP gives each calling thread its own team of `Threads` threads (measured: 8 callers × 8 threads
    = 83 OS threads), and a thread that already ran an op keeps its old count after a later Load.
    Measured (benchmark, default, 6,566 words): 1 thread 57.8 s, 2: 33.2 s, 8: 19.1 s, 16: 19.7 s. One loaded pipeline
    ≈ 950 MB private (default), 700 MB (default_fast). VerifyChecksums ≈ +1.3 s / +0.8 s load.
  - `CancellationToken` overloads of all three `Process` (separate overloads, binary compatible). Checked in
    `Pipeline.Step` before each processor and at the head of each batch loop (tokenizer batches and windows, pos,
    lemma, constituency per step, depparse, sentiment, ner; mwt is one batch). Depparse also checks inside a batch:
    after each charlm pass, between `HighwayLstm` layers (its optional token), before each `DeepBiaffine` chunk and
    per Chu-Liu/Edmonds decode; the longest gap is one charlm pass or LSTM layer (~1/8 of a batch). Every check sits outside the inner
    dispose scopes or inside a try/finally that disposes, so nothing leaks (`ConcurrencyTests`).
  - Thread safety: Process keeps all mutable state per call (the Document, `CharlmCache`, the lemmatizer's
    DeltaVocab copy, the parser states); the models are read-only after Load; TorchSharp's dispose scopes are
    thread-static and libtorch's grad mode thread-local. The stress test (8 threads, mixed modes, both packages)
    needed no code change. Keep it that way: never add mutable fields to a processor.
  - TorchSharp counts the two undefined index tensors of `pack_padded_sequence(enforce_sorted: true)` (lemmatizer)
    as live forever (`DisposeScopeManager.Statistics`); they hold no memory. Leak tests leave lemma out.
  - `SplitSentences = false` = `tokenize_no_ssplit`: `Tokenizer.Decode` ends sentences only at paragraph ends.
    `tests/golden/no_ssplit/` (`make_golden.py --no-ssplit-only`). Pretokenized input ignores it, as in Stanza.
  - `Processor` constants; `VerifyChecksums` (MD5 of each `.pt` Load reads, `ModelDownloader.Verify`; throws for
    converted models, which have no published MD5); `Logger` (`Microsoft.Extensions.Logging.Abstractions` 10.0.0,
    the package's second dependency): load times at Information, per-processor times at Debug, nothing measured
    without a logger.

Tokenizer notes:
- `Tokenizer.Predict` batches like Stanza: sort paragraphs by length, batch by 32, pad to max+1,
  and use 1000-char windows. Both LSTMs are packed at the length of each row's raw units. In
  the normal case that is the row's own length + 1, since collate appends one `<PAD>`. In the
  first long-paragraph window, the length is cut to the window. After `advance_old_batch`, every
  row runs at the full window width, so padding then reaches the backward direction.
- Model units are code points, like Python. Offsets are UTF-16 indices, so they differ from
  Stanza's only after non-BMP characters.
- MWT-flagged tokens (`Token.IsMwtCandidate`) carry a single word until the MWT stage runs.

## Layout

```
src/StanzaSharp.Core           Document model, CoNLL-U I/O, constituency Tree, safetensors reader,
                               checkpoint loading (converted JSON or .pt). No TorchSharp dependency.
src/StanzaSharp.Nn             TorchSharp: charlm, pretrain embeddings, shared layers, weight loading.
src/StanzaSharp.Tokenize       Tokenizer + sentence splitting.
src/StanzaSharp.Mwt            Multi-word token expansion.
src/StanzaSharp.Pos            POS / feature tagger.
src/StanzaSharp.Lemma          Lemmatizer (dictionary + character seq2seq).
src/StanzaSharp.Constituency   Constituency parser.
src/StanzaSharp.Depparse       Dependency parser (biaffine graph parser + Chu-Liu/Edmonds).
src/StanzaSharp.Ner            Named-entity recognizer (biLSTM + CRF Viterbi).
src/StanzaSharp.Sentiment      Sentence sentiment (CNN classifier over biLSTM states).
src/StanzaSharp                Pipeline facade wiring the processors together.
src/StanzaSharp.Tool           The `stanzasharp` .NET tool (model downloads; no native libtorch).
samples/StanzaSharp.Cli        Console runner for quick experiments.
samples/StanzaSharp.Benchmark  Per-stage speed/memory benchmark; tools/benchmark.py is the Python twin.
samples/StanzaSharp.Example    Commented tour of the public API (download, load, every result, CoNLL-U).
tests/StanzaSharp.Tests        xUnit; golden tests against Python Stanza output.
tests/golden/                  Golden data generated from Python Stanza (committed, keep it small).
tools/stanza_convert.py        Checkpoint inspector/converter (.pt -> .safetensors + .json); optional.
tools/.venv/                   Python env with stanza installed (gitignored).
models/stanza/                 Original downloaded models (gitignored).
models/converted/en/           Converted models (gitignored); C# prefers them over the .pt files.
```

Package policy: library projects reference the managed `TorchSharp` package only. Runnable
projects (Cli, Benchmark, Tests) reference `$(TorchSharpNative)` (Directory.Build.props): `TorchSharp-cpu`,
or `TorchSharp-cuda-windows` when built with `STANZASHARP_CUDA=1`, which brings the native libtorch. The
managed and native TorchSharp versions must match. CI and the default build stay on CPU.

GPU: `Pipeline.Load(dir, new PipelineOptions { Device, DisableTf32 })` and every processor's `Load(..., device)`. `Weights.On(device, ...)`
scopes the load; `ToTensor`/`LoadFrom` place weights on `Weights.Device`, and each model keeps
`_device = Weights.Device` for its input tensors (`torch.tensor(..., device: _device)`). Read tensors with
`Weights.ToArray<T>()`, not `data<T>()`. pack_padded_sequence lengths stay on the CPU. A new processor
needs exactly these three things to run on CUDA. GPU tests (`GpuTests.cs`) compile only with
`STANZASHARP_CUDA=1`. Measurements and exactness: docs/gpu.md.

## Reference implementation

The Python Stanza source installed in `tools/.venv/Lib/site-packages/stanza/` is the source of
truth. When porting a model, read its code there (`stanza/models/tokenization`, `mwt`, `pos`,
`constituency`, `common/char_model.py`, `common/pretrain.py`, `pipeline/`) rather than relying on
memory of how Stanza works. Port the code paths the English checkpoints actually use, which the
checkpoint configs tell you; skip unused options.

## Model files

C# loads either format. `Checkpoint.Load(basePath)` reads `basePath.json` + `basePath.safetensors`
when they exist, else the original `basePath.pt`, so `Pipeline.Load` accepts both `models/converted/en`
and Stanza's own download directory `models/stanza/en` (`<processor>/<name>.pt`). Converting is
optional: it saves about a second of load time and makes the checkpoints easy to inspect.

Use `setup.ps1 -Models`, or run the steps by hand:

```powershell
tools\.venv\Scripts\python -c "import stanza; stanza.download('en', model_dir=r'models\stanza', processors='tokenize,mwt,pos,constituency')"
tools\.venv\Scripts\python tools\stanza_convert.py inspect models\stanza\en\constituency\ptb3-revised_charlm.pt
tools\.venv\Scripts\python tools\stanza_convert.py convert models\stanza\en --out models\converted\en
```

Converter output for each checkpoint is a `<name>.safetensors` file holding all tensors, plus a
`<name>.json` file mirroring the checkpoint's structure. In the JSON:

- Each tensor is replaced by `{"$tensor": key, "dtype", "shape"}`, keyed into the safetensors file.
- Non-JSON Python types are tagged:
  - `$tuple`, `$set`
  - `$dict` (a list of `[key, value]` pairs, for dicts with non-string keys)
  - `$bytes` (base64)
  - `$float` (`nan`/`inf`)
  - `$repr`/`$type` (unknown objects)
- `optimizer` and `scheduler` are dropped by default.

Safetensors layout: a u64 little-endian header length, a JSON header (`dtype`, `shape`,
`data_offsets` relative to the data section, optional `__metadata__`), padded to 8-byte alignment,
then raw little-endian tensor data. Read it with a small hand-written reader in `StanzaSharp.Core`;
no package is needed.

All fourteen checkpoints convert with `torch.load(weights_only=True)`; no unsafe pickling is needed.

### Reading .pt files in C#

All fourteen Stanza checkpoints use torch's legacy format (`_use_new_zipfile_serialization=False`), a
sequence of protocol-2 pickles: magic number, protocol version (1001), sys_info (little-endian),
the checkpoint, the list of storage keys; then per key an int64 element count and the raw bytes.
Tensors are `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks)`
over persistent ids `('storage', torch.FloatStorage|LongStorage, key, location, numel, None)`; LSTM
weights share one storage at different offsets. The only globals are `collections.OrderedDict`,
`torch._utils._rebuild_tensor_v2`, the storage types, and `_codecs.encode` (protocol 2's encoding of
`bytes`, used by the lemma's `dicts`). The zip format (`archive/data.pkl` +
`archive/data/<key>`) is read too, for checkpoints saved by newer trainers.

`Unpickler` is restricted like `weights_only=True`: torch's protocol-2 opcodes only, plain data
(None/bool/int/float/str/bytes/list/tuple/dict), and an allowlist of OrderedDict, `_rebuild_tensor_v2`,
`_codecs.encode(str, 'latin1')` and the typed storage classes. Anything else throws `InvalidDataException`; nothing is instantiated by
name. Not supported (none of the models need them): sets, bfloat16, big-endian files,
`$repr` objects. `TorchCheckpoint` then mirrors the converter's `Splitter` exactly, including
Python's float `repr` in the JSON, key collisions (`key#2`) and numpy turning 0-d tensors into
shape `[1]`. `tests/golden/pt/` holds tiny fixtures in both formats with their converter output
(`make_golden.py --pt-only` regenerates them).

## Checkpoint findings

What the English checkpoints actually use (Stanza 1.15.0). Port only these paths.

- **tokenize** (`combined_nocharlm`): `models/tokenization/model.py`.
  - Char embedding 32, plus 9 feature functions (`feat_funcs` in the config).
  - A 1-layer biLSTM with hidden size 64. `conv_res` is null, so there are no conv layers;
    `conv_filters` and `residual` are unused by the model.
  - `hierarchical`: a second biLSTM (`rnn2`) and `*_clf2` heads, with `hier_invtemp` 0.5.
  - `use_mwt`, so the output has 5 classes.
  - No dictionary and no charlm.
- **mwt** (`combined`): a dictionary (`dict`, consulted first because `ensemble_dict`), then
  `models/mwt/character_classifier.py`. This is **not a seq2seq model**: it is a per-character
  binary split classifier with a 2-layer biLSTM (50 per direction) and an MLP.
- **charlm** (forward/backward `1billion`): char embedding 100, a 1-layer LSTM with hidden size 1024,
  and learned `h_init`/`c_init`. The decoder is unused at inference.
- **pretrain** (`conll17`): `emb` [250000, 100] plus a vocabulary.
- **pos** (`combined_charlm`):
  - Inputs: word embedding 75, pretrain → `trans_pretrained` 125, and both charlms (2048).
    There is no own char model and no BERT.
  - A 2-layer highway biLSTM with hidden size 200 and learned `h_init`/`c_init`.
  - Outputs: UPOS through an MLP. XPOS and the 21 feats use biaffine classifiers conditioned on
    UPOS (`tag_columns`).
- **lemma** (`combined_nocharlm`): `models/lemma/trainer.py`, `models/common/seq2seq_model.py`.
  - `dicts` (v3: gzip + JSON) `{upos: {word: lemma}}`, `"*"` POS-independent. With `ensemble_dict`,
    words found by (UPOS, word) then `*` skip the model.
  - Seq2seq: char embedding 50 (vocab 234), UPOS embedding 50 prepended as an extra encoder step,
    1-layer biLSTM 2×100, `LSTMCell` decoder 200 with `soft` (dot) attention, greedy (`beam_size` 1),
    `max_dec_len` 50. No charlm, not caseless, no contextual lemmatizers.
  - `copy`: a copy gate mixes the vocab distribution with attention over the source chars (minus the
    POS step), scattered onto char ids. Characters missing from the vocab get ids past it (DeltaVocab,
    built per document over the model's words in code point order), so they can be copied.
  - `edit`: a 3-way classifier on the final encoder state (`cat(hn[-1], hn[-2])`): 0 use the decoded
    string, 1 the word, 2 the lowercased word. Empty or `<UNK>`-containing output falls back to the word.
  - Stanza's LemmaProcessor only requires tokenize; we require pos too since the model reads UPOS.
- **constituency** (`ptb3-revised_charlm`):
  - `IN_ORDER` transitions, with LSTM transition and constituent stacks.
  - `MAX` composition (`reduce_linear`), ReLU, and 2 output layers. ReLU is applied *before*
    every output layer, including the first.
  - No attention: `use_lattn` is false and `pattn_num_layers` is 0. No BERT.
  - Word input is 2268 = delta 100 + pretrain 100 + tag 20 + charlm 2048. It feeds a 2-layer
    biLSTM with hidden size 512.
  - `combined_dummy_embedding` is on, and `unary_limit` is 4.
  - The parser consumes **XPOS** tags (`retag_method: xpos`) from the POS stage.
  - Stanza prints bracket words as `-LRB-`/`-RRB-` in trees, so `(` and `)` appear that way in
    golden trees.
- **depparse** (`combined_charlm`): `models/depparse/model.py` `GraphParser`.
  - A ROOT word (id 3 in every vocab, `"\n"` for the charlms) is prepended to each sentence.
  - Input 2423 = trans_pretrained 125 + word 75 + lemma 75 + (UPOS+XPOS) 50 *twice* + charlm 2048.
    Stanza appends the tag embedding a second time where it computes the UFeats one, so the
    `ufeats_emb` weights are never used. Words and lemmas are lowercased for their vocabs and
    `simplify_punct`ed, as in the tagger.
  - A 3-layer highway biLSTM, hidden 400, learned `h_init`/`c_init` (the tagger's `HighwayLstm`).
  - Deep biaffine (pairwise) arc and label scorers (hidden 400, 49 labels), plus `linearization`
    and `distance` terms added to the arc scores. No arc embedding.
  - Decoding: log-softmax over heads, then `chuliu_edmonds_one_root` in float64. The log-softmax
    includes the batch's padding columns, so batches must match Stanza's: sorted longest first,
    5000 words (with ROOT) per batch, over 150 alone. `resolve_head_constraints` is false.
- **ner** (`ontonotes-ww-multi_charlm`): `models/ner/model.py` `NERTagger`, `trainer.py` `predict`.
  - Works on **tokens** (`doc.get(TEXT, from_token=True)`), so an MWT is tagged once.
  - Input 2148 = word 100 + charlm 2048, through `input_transform` (Linear 2148→2148). The word
    embedding is the pretrain matrix (the checkpoint's `word` vocab is conll17's) plus a fine-tuned
    `delta_emb` (48309 words); both look up the lowercased token. A word the pretrain knows but delta
    doesn't uses delta's PAD row (zeros). The charlm input is built exactly like `build_char_representation`.
  - A 1-layer biLSTM, hidden 256, `h_init`/`c_init` (zeros).
  - "multi": two tag sets, each a Linear + CRF: OntoNotes (77 BIOES tags, 18 types) and WorldWide
    (41 tags). `predict_tagset` is 0, and Stanza unmaps both but keeps only column 0, so only the
    OntoNotes head is run.
  - Decoding: numpy `viterbi_decode` in float32 (ties to the lowest id), ids < 4 become `O`, then
    `fix_singleton_tags`. `merge_tags` is the identity with one model. Batches of 32 sentences in
    document order; everything is packed, so batching only moves the last float bits.
- **sentiment** (`sstplus_charlm`): `models/classifiers/cnn_classifier.py` `CNNClassifier`
  (`model_type` CNN), in `params` of the checkpoint with `extra_vocab` and `labels` `0,1,2`.
  - Input: the sentence's **token** texts (`extract_sentences`), not words, with no lowercasing or
    `simplify_punct`. Each batch is padded at the end to its longest sentence, at least 5 (the widest filter).
  - Per token 2148 = pretrain 100 + delta embedding 100 summed (`extra_wordvec_method` SUM) + charlm
    2048. Pretrain lookup (`map_word`): the token, else without a trailing `'`, else lowercased;
    still unknown gets the learned `unk` vector (padding keeps row 0). The delta vocab (31014) looks
    the token up as written, else id 1. The charlms run over the tokens like the tagger's.
  - A 2-layer biLSTM (300 per direction) over the **unpacked** padded batch, then Conv2d filters 3, 4
    and 5 tokens × 600 (1000 channels each) and one (5, 5) filter with stride (1, 5) (8 channels ×
    120 = 960), each ReLU then max over all positions (`maxpool_width` 1): 3960 → FC 400 → 100 → 3,
    ReLU between. Dropout is off at inference.
  - `label_sentences`: sentences sorted longest first (stable), `split_into_batches` of 5000 tokens
    (longer alone), argmax.
- **The `_nocharlm` models** (`default_fast`). Each config differs from its `_charlm` twin only in
  `charlm: false`; vocabularies, batching and every other layer are the same. Instead of the charlms they
  carry `charmodel.*`, a `common/char_model.py` `CharacterModel` (`Nn.CharacterModel`):
  - Its own char vocab (`vocab.char`, not lowercased, `char_lowercase` false), code points, unknown → UNK 1.
    Char embedding 100 (padding_idx 0), a 1-layer LSTM with learned `charlstm_h_init`/`c_init`, run over
    every word of the batch as its own packed sequence (sorted by length), so batching only moves float bits.
  - **pos** (`combined_nocharlm`): unidirectional (`char_bidirectional` false), hidden 400, attention
    pooling `sum_t sigmoid(char_attn(h_t)) * h_t`, then `trans_char` (Linear 400→125, no bias). Input
    325 = word 75 + trans_pretrained 125 + char 125, in that order. Same `simplify_punct`ed words.
  - **depparse** (`combined_nocharlm`): the same character model (always unidirectional with attention in
    `GraphParser`) + `trans_char` 400→125. ROOT's characters are the single id `ROOT_ID` (3). Input 500 =
    trans_pretrained 125 + word 75 + lemma 75 + tags 50 twice + char 125.
  - **ner** (`ontonotes-ww-multi_nocharlm`): bidirectional, hidden 100, **no** attention: the final forward
    and backward states (`h[-2:]`), 200. Input 300 = word 100 + char 200 → `input_transform` 300→300.
    Reads tokens, as the charlm model does.

## Validation

- Golden tests: run Python Stanza (same version, same models) on a fixed English corpus and save
  each stage's output under `tests/golden/`:
  - token and sentence boundaries
  - MWT expansions
  - UPOS/XPOS/feats
  - bracketed trees
- Also save selected intermediate tensors so numeric drift can be localized, for example charlm
  hidden states and tagger logits for a few sentences. Compare floats with a tolerance and
  discrete outputs exactly.
- Tests that need model files must skip, not fail, when `models/converted/en` is missing.
- Versions are pinned in `tools/requirements.txt`; golden data depends on them.
- `tools/make_golden.py` regenerates golden data from `tests/golden/corpus.txt`:
  - `pipeline.conllu`: the full pipeline output, including offsets, MWT ranges, tags, and
    `# constituency` comments. Ignore its HEAD column, since there is no depparse.
  - `intermediates.safetensors` + `.json`, covering the first 3 sentences, each run alone:
    - tokenizer logits
    - charlm per-word representations
    - UPOS logits
    - per-step parser transition logits
  - `tokenize_stress.txt` + `.conllu`: tokenizer output for a paragraph over 1000 characters and
    for more paragraphs than fit in one batch.
  - `lemma/`: its own folder, written by `write_lemma_golden` (`--lemma-only`): `<name>.conllu` from
    tokenize,mwt,pos,lemma for corpus.txt and every validation*.txt, plus `words.json` (pipeline lemma,
    raw seq2seq output and edit class for hand-picked words: dictionary hits, all edit types,
    unknown and non-BMP characters, a word past `max_dec_len`).
  - `mwt.json`: expansions of a word list, both through the pipeline and classifier-only.
  - `depparse/` (`make_golden.py --depparse-only`): `tokenize,mwt,pos,lemma,depparse` output for
    `corpus.txt` (as `corpus.conllu`) and each `validation*.txt`; arc/label log-probs for the
    first 3 sentences; `mst.json`, Stanza's trees on tie-heavy integer score matrices.
  - `ner/` (`make_golden.py --ner-only`): `tokenize,mwt,pos,lemma,depparse,ner` output for `corpus.txt`
    (as `corpus.conllu`) and each `validation*.txt`, `<name>.json` with each sentence's entities, and
    the emission scores of the first 3 sentences, each tagged alone.
  - `sentiment/` (`--sentiment-only`): `tokenize,mwt,sentiment` `<name>.conllu` + `<name>.json`
    (label and 3 logits per sentence) for corpus.txt, each validation*.txt and `sentiment/reviews.txt`
    (opinionated sentences); `all.json` for all of them as one document (two batches).
  - `fast/` (`--fast-only`): `package='default_fast'` output (all seven of its processors) for `corpus.txt`
    (as `corpus.conllu`) and each `validation*.txt`; for the first 3 sentences, each run alone, UPOS logits,
    arc/label log-probs (from the run's tags and lemmas) and NER emissions.
  - `pretokenized/` (`--pretokenized-only`): `inputs.json` (cases of token lists, plus `corpus`: corpus.txt's
    Stanza tokens) and `<name>.conllu` / `<name>.fast.conllu` for both packages.
  - `bulk/<package>.json` (`--bulk-only`): one `bulk_process` call over `BULK_TEXTS` (tiny, empty and
    whitespace-only texts; stored whole) + corpus.txt + validation*.txt (stored as the golden file processed alone,
    a sent_id offset and the sentences bulk changes), so it stays small.
  - `validation*.conllu`: full pipeline output for each hand-written `validation*.txt` (news,
    academic, instructions, social, dialogue/poetry, tech, multilingual, contractions, long,
    whitespace, nonbmp; ~830 sentences). They are read as bytes, so CR/CRLF reach Stanza as-is.
    `validation_whitespace.txt` is `-text` in `.gitattributes` to keep its CRLF. The test compares
    each file and reports its first differing sentence. `validation_nonbmp` is compared without
    `start_char`/`end_char`, because offsets differ by design after non-BMP characters.
- Python `str` semantics that .NET lacks live in `Core.PyString` (`Lower` with İ and final sigma,
  `IsUpper`/`IsLower` with Other_Uppercase/Other_Lowercase, and `IsSpace` with U+001C..U+001F).
  Use it wherever Stanza calls `lower()`, `isupper()`, `isspace()` or `split()`. Regexes ported
  from Python need care too: Python's `\b`/`\w` count No/Nl characters such as `²` as word
  characters and combining marks as non-word (see `Tokenizer.PyRegex`).
- CI (`.github/workflows/ci.yml`, github.com/bisforboman/StanzaSharp; pushes to `main`, PRs, and
  `workflow_call` from release.yml; `ubuntu-24.04`, `windows-2025`, `macos-15`, `windows-11-arm` pinned).
  - Branch protection on `main` requires the check names `build-test` and `golden`, so they stay
    plain jobs; a matrix would rename them (`build-test (ubuntu-24.04)`) and block every PR.
  - `build-test` runs the suite without models.
  - Docs-only pull requests (`docs/`, `*.md`, `LICENSE`, `NOTICE`): the `changes` job makes `golden`, `docker` and
    `cross-os` run with their steps skipped, so the required checks pass in about a minute. No job is skipped
    outright: a skipped job would count as passed. Pushes to main and
    release tags always run everything; if `changes` fails, everything runs.
  - `concurrency`: a newer push to the same PR or to main cancels the older run; tag runs are never cancelled.
  - `golden`, on a model-cache miss, downloads the models with the C# `ModelDownloader` (via the
    CLI) and converts them with CPU torch. The models are cached on `ModelDownloader.cs`,
    `tools/requirements.txt` and `tools/stanza_convert.py`. It then fails if any test is skipped
    (`outcome="NotExecuted"` in the trx; the trx `notExecuted` counter stays 0 for skips), and runs
    `tools/verify-package.ps1` against `models/stanza/en`.
  - `cross-os (windows-2025)` / `cross-os (macos-15)` (Apple Silicon; `TorchSharp-cpu` brings
    `libtorch-cpu-osx-arm64`): build, full suite with `STANZASHARP_MODELS` = `models/stanza/en` (the
    `.pt` files, no Python or conversion), the same no-skip check, and `verify-package.ps1`. Models are
    cached per OS on `ModelDownloader.cs` only; no NuGet cache, to stay inside the 10 GB cache budget.
    - macOS first needs `brew install libomp`. `libtorch_cpu.dylib` from `libtorch-cpu-osx-arm64` 2.10.0
      links `/opt/homebrew/opt/libomp/lib/libomp.dylib` by absolute path, not the `libomp.dylib` it
      ships. Without it TorchSharp reports only "doesn't contain a reference to libtorch-cpu-osx-arm64";
      `otool -L` on the dylib shows the real cause. Users need it too (README, PACKAGE.md).
  - `docker`: `tools/verify-docker.sh MODEL_DIR BASE...` packs StanzaSharp + StanzaSharp.Cpu.Linux into
    `samples/docker/feed` (with a generated nuget.config), builds the sample per base image (runtime:10.0 and
    10.0-noble-chiseled) and `cmp`s the container's CoNLL-U for corpus.txt with pipeline.conllu. Models from golden's
    cache via `actions/cache/restore` (no new entry; downloaded on a miss); no Docker layer cache (budget).
  - `cross-os (windows-11-arm)`: Windows on Arm64 (GA runner label for public repos). It skips the plain
    `verify-package.ps1` (TorchSharp-cpu has no Arm64 libtorch) and runs `-Platform WindowsArm64`.
  - `alpine-experiment.yml` (workflow_dispatch only; not required): the docker check on `runtime:10.0-alpine` +
    `gcompat libstdc++ libgcc`.
    - Verdict (2026-10-06, PR #10): it fails. musl's ldd shows glibc-only symbols that gcompat lacks:
      the `__*_chk` fortify functions, `backtrace`/`backtrace_symbols`, `fcntl64`, `__res_init`, and
      `pthread_attr_setaffinity_np` (libgomp).
    - Alpine stays unsupported unless a musl libtorch/TorchSharp build appears. Rerun it then.
  - Keep paths forward-slash and file names case-exact (Linux). The root `.gitattributes` keeps sources
    LF; `tests/golden/.gitattributes` pins golden files (`eol=lf`, `binary`, `validation_whitespace.txt`
    `-text`), so Windows checkouts with `core.autocrlf=true` stay byte-exact. Bash steps run under
    `shell: bash` on every OS (Git Bash on Windows).

## Packaging and releases

- Public API (user's decision for 0.1.0, 2026-10-06: "pipeline only"). It is:
  - the facade: `Pipeline`, `PipelineOptions`, `CharlmCacheOptions`, `ModelDownloader`, `Processor` (0.4);
  - Core's results: `Document`, `Sentence`, `Token`, `Word`, `Entity`, `Tree`, `Conllu`.

  Everything else is `internal`.
  - Directory.Build.props gives every `src/*` assembly `InternalsVisibleTo` for the other StanzaSharp
    assemblies, the tests and the benchmark. Don't add per-project `InternalsVisibleTo`.
  - New types default to `internal`. Making one public is an API decision for the user.
  - `src/*` builds docs with CS1591 as an error, so every public member needs an XML doc.
  - `CHANGELOG.md` gets an entry per user-visible change under "Unreleased". A release moves it under
    the version.
- `Pipeline.Load(dir, PipelineOptions?)` (user's decision, 2026-10-06: an options object rather than more
  optional parameters).
  - `PipelineOptions` holds `Processors`, `Device`, `DisableTf32` and
    `CharlmCache { IsEnabled = true, MaxWords = 32_768 }`.
  - `DisableTf32` (name and opt-in are the user's choices) sets both torch TF32 switches to false
    process-wide and doesn't restore them.
  - `CharlmCache.TryAdd` keeps a sentence only if it fits, detaching its tensors from the caller's
    dispose scope. Otherwise it leaves them with the caller. Before this fix the cache disposed
    rejected tensors that the tagger was still using.
- Platform packages (user's decision, 2026-10-06): `StanzaSharp.Cpu.Linux` / `.Windows` / `.WindowsArm64` / `.MacOS`.
  - Each is a code-free package (`src/StanzaSharp.Cpu.props`, `IncludeBuildOutput=false`) depending on the
    same-version `StanzaSharp` and one `libtorch-cpu-<rid>` 2.10.0, instead of `TorchSharp-cpu`, which
    depends on all three.
  - TorchSharp loads a single platform package fine; it was checked with `libtorch-cpu-win-x64` alone.
  - `verify-package.ps1 -Platform X` tests each one on its own OS in CI (Linux in `golden`, Windows and
    macOS in `cross-os`).
  - `release.yml` packs all six packages (these five and `StanzaSharp.Tool`).
  - `.WindowsArm64` (`libtorch-cpu-win-arm64`): `TorchSharp-cpu` lacks it, so on a Windows Arm64 machine
    Directory.Build.props also gives the runnable projects (tests/, samples/) `libtorch-cpu-win-arm64`.
  - Supported platforms (README, PACKAGE.md): where TorchSharp 0.107 has `runtimes/<rid>` (linux-x64, win-x64,
    win-arm64, osx-arm64) and a `libtorch-cpu-<rid>` 2.10.0 exists (the same four). Not Alpine/musl (glibc
    builds only), linux-arm64 (no libtorch package, no native layer) or osx-x64 (no native layer; libtorch stops at 2.2).
  - The linux-x64 libtorch needs only glibc, libstdc++ and libgcc_s (it bundles libgomp). `samples/docker` is a
    standalone app (its own empty Directory.Build.props) on `mcr.microsoft.com/dotnet/runtime:10.0` (`BASE` arg).
  - Every new package ID needs the nuget.org Trusted Publishing policy to allow it.
- `buildTransitive` (StanzaSharp package; `src/StanzaSharp/buildTransitive/StanzaSharp.targets` + a `StanzaSharp.props`
  generated at pack time). `TorchSharpVersion` (0.107.0) and `LibTorchVersion` (2.10.0) are set only in Directory.Build.props.
  - `STANZA001` (warning): a resolved `TorchSharp`, `TorchSharp-*` or `libtorch-*` package has another version. Versions
    come from `@(PackageDependencies)` + the assets file's `"<id>/<version>"` keys (copy-local items miss the
    asset-less `TorchSharp-cpu`; `TorchSharp-cpu` 0.106.0 resolves the same TorchSharp/libtorch as 0.107.0).
    `<NoWarn>STANZA001</NoWarn>` silences it. `verify-package.ps1` fails on any `STANZA` warning.
  - `StanzaSharpTrimNative=true` (opt-in) removes from `NativeCopyLocalItems`/`RuntimeTargetsCopyLocalItems` (so also
    from the .deps.json): libtorch_python, libshm, libnnapi_backend, libtorchbind_test, libjitbackend_test,
    libbackend_with_compiler, libaoti_custom_ops (.so; .dylib for python/shm). By DT_NEEDED/LC_LOAD_DYLIB nothing that
    LibTorchSharp → libtorch → libtorch_cpu → libc10/libgomp needs them, and TorchSharp loads only `LibTorchSharp`
    and `torch_cpu`/`torch_cuda` by name. Windows libtorch has no such files (its only unlinked ones, uv.dll,
    libiompstubs5md.dll and torch_global_deps.dll, 250 KB, are kept). A marker file in `obj/` is a .deps.json input,
    so switching the property regenerates it. Proven by `verify-package.ps1 -Platform` (Linux in golden, macOS in
    cross-os) and the `docker` job (samples/docker sets it; verify-docker.sh checks the files are absent).
- `StanzaSharp.Tool` (`src/StanzaSharp.Tool`, `PackAsTool`, command `stanzasharp`): `download [DIR] [--package NAME]
  [--processors LIST]`. `DownloadCommand.cs` is compiled into samples/StanzaSharp.Cli too (one implementation). A target
  drops all native package assets (LibTorchSharp, SkiaSharp), so the package is 1.2 MB; the downloader never calls
  into libtorch. `tools/verify-tool.ps1` (golden) packs, `dotnet tool install --tool-path`s it from a local feed,
  checks for natives and downloads `tokenize,mwt`. release.yml packs it.
- One NuGet package, `StanzaSharp`, packed from `src/StanzaSharp` (user's decision, 2026-10-06).
  - It carries all nine assemblies plus their XML docs: the facade's ProjectReferences are
    `PrivateAssets="all"`, and an `IncludeProjectReferences` target adds them.
  - It depends only on managed `TorchSharp`; users add `TorchSharp-cpu`/`-cuda` themselves.
  - Every other project is `IsPackable=false` (Directory.Build.props, which also holds the shared
    package metadata and a `0.1.0-dev` default version).
  - Because of `PrivateAssets`, tests and samples reference the library projects they use directly.
  - `src/StanzaSharp/PACKAGE.md` is the package readme; `NOTICE` ships in the package.
- Models are downloaded only explicitly (user's decision): `ModelDownloader.DownloadAsync(dir)` or
  `StanzaSharp.Cli download [DIR]`.
  - Selective download (user's decision, 2026-10-06): `DownloadAsync(dir, processors)` / `download
    --processors LIST`.
  - Packages (user's decision, 2026-10-06: Stanza's package names as strings, so more can be added without
    API changes): `PipelineOptions.Package` (`"default"` or `"default_fast"`; anything else throws) and
    `DownloadAsync(dir, PipelineOptions)` / `download --package NAME`. The downloader takes the same options
    object as `Load`, so one object downloads and loads the same files; it ignores the other options.
    `Processors` is null by default: all the package's processors. A processor the package lacks throws.
  - `DownloadAsync(dir)` and `DownloadAsync(dir, processors)` (0.1.0 API) mean the `default` package.
  - `Pipeline.SelectModels` (package + processors → processor → model) and `Pipeline.SharedModels`
    (pretrain for pos/depparse/ner/constituency/sentiment, charlms for any `_charlm` model) are what both
    `Pipeline`'s constructor and `ModelDownloader.FilesFor` use, so they cannot drift apart. A new package
    needs an entry in `Pipeline.Packages` and its files' MD5s in `ModelDownloader.Files`;
    `FilesFor_AreEnoughToLoadThePipeline` and `FileTableCoversEveryModelThePipelineLoads` check it. CI
    downloads both packages.
  - It fetches the `.pt` files from `huggingface.co/stanfordnlp/stanza-en/resolve/v1.15.0/models/`
    and checks each against the MD5 from Stanza 1.15.0's resources.json, kept in
    `ModelDownloader.Files`.
  - It keeps files that already match. `Pipeline.Load` never downloads.
  - Changing the Stanza version means new MD5s, new golden data and a new cache key.
- `tools/verify-package.ps1` packs a unique `0.0.0-verify.<time>` version, builds a fresh console app
  against it plus `TorchSharp-cpu`, runs the README example and checks the parse. It removes that
  version from the NuGet cache afterwards.
- Releases mirror StyleBro: pushing a tag `v<semver>` runs `.github/workflows/release.yml`. It runs all
  of ci.yml, packs with the tag's version, then pushes via NuGet Trusted Publishing (`NuGet/login@v1`,
  repo variable `NUGET_USER`) and creates a GitHub release (a prerelease if the version has `-`).
  - Environment `prerelease` is for tags with `-`; `release` is for the rest.
  - One-time setup, done 2026-10-06:
    - nuget.org Trusted Publishing policies for both environments (owner bisforboman, repo
      StanzaSharp, workflow `release.yml`), added by the owner.
    - The repo variable `NUGET_USER` = `bisforboman`.
    - GitHub environment `release`: required reviewer bisforboman, tag rule `v*`.
    - GitHub environment `prerelease`: no reviewers, tag rule `v*-*`.
    - A 401 "No matching trust policy" on push means the nuget.org policy for that environment is
      missing or misnamed.
- `main` is protected like StyleBro since 2026-10-06 (user's decision): pull requests required (0
  approvals), required status checks `build-test` and `golden`, enforced for admins, auto-merge allowed,
  merged branches deleted. Every change goes through a PR. A new required CI job must also be added to the
  protection's required checks.

## Design decisions

- Model the document as Document → Sentence → Token → Word from the start. Tokens come from the
  tokenizer, Words come out of MWT expansion, and POS and the parser operate on Words. Keep this
  split even before MWT is implemented.
- Batch inference per sentence from the start.
- Run inference under no-grad / eval mode, and dispose tensors deterministically (TorchSharp
  `DisposeScope`). A temporary tensor, at load time too, belongs to a scope (or a `using` variable): left to its
  finalizer, it can be freed by a GC during the native call that reads it.
- Never pass TorchSharp a temporary `Scalar` (`Nn.Scalars` explains the finalizer race). No tensor `+` or
  `add(Tensor)` (hidden `alpha = 1`): write `a.add(b, Scalars.One)`. No numbers where a Scalar is expected
  (`t * 2`, `1 + t`, `masked_fill(m, 0)`, `eq(0)`, `pow(2)`, `arange(n)`): use `Scalars.*`, a `static readonly Scalar`,
  or a `using var s = value.ToScalar()` local. `F.softplus` goes through `Scalars.Softplus`. Tensor `-` tensor is
  fine (no alpha). Check a new TorchSharp function's source for internal conversions before using it.
  Reported upstream as dotnet/TorchSharp#1583 (open PRs #1434/#1496 would fix it); when upgrading TorchSharp, check
  whether it is fixed and, if so, drop the workaround, including `Scalars.Softplus`'s private `softplus1` call.

## Build order

1. Generate golden data with Python, and inspect every checkpoint's config (start with constituency;
   its config decides which optional layers the parser needs).
2. `Core`: document model, safetensors reader, checkpoint JSON loading.
3. Tokenizer (character-level labeling + post-processing into tokens and sentences).
4. MWT: dictionary lookup first, then the character split classifier.
5. Charlm + pretrain in `Nn`; validate against saved hidden states.
6. POS tagger.
7. Constituency parser: in-order transition system, parser state, Tree type, plus the neural scoring.

## Licensing

StanzaSharp is Apache 2.0 (`LICENSE`), like Stanza, whose code it ports. `NOTICE` credits Stanza;
keep it in any redistribution. Model licenses vary with the training data, so do not commit or
redistribute model files. `models/` is gitignored.
