# StanzaSharp

A native C# port of the inference side of [Stanza](https://github.com/stanfordnlp/stanza), Stanford's
PyTorch-based NLP library, running the original pretrained models through TorchSharp.

## Scope

- **Language:** English only, to start.
- **Processors:** `tokenize` → `mwt` → `pos` → `constituency`.
- **Inference only.** Training stays in Python; we load Stanza's released weights.
- **Package:** Stanza's English *default* package, which needs no transformer:
  - tokenize: `combined_nocharlm`
  - mwt: `combined`
  - pos: `combined_charlm`
  - constituency: `ptb3-revised_charlm`
- **Shared dependencies:** `pos` and `constituency` both depend on the forward and backward
  character LMs (charlm) and on the pretrained word-vector file (pretrain). That makes six model
  files to port, not four. The charlm is shared infrastructure and lives in `StanzaSharp.Nn`.

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
  `Biaffine`.
- `Constituency.ConstituencyParser` also takes the shared `Pretrain` and charlms. It reproduces the
  golden trees and the per-step transition scores. It runs a batch of sentences in lockstep: each
  step is one forward pass plus batched stack pushes. Stack LSTMs are stepped without padding, so
  batching does not change results.
- `Tree.ToString` prints `(`/`)` in labels and words as `-LRB-`/`-RRB-`, like Stanza. The tree
  itself keeps the raw text.
- Measured float drift vs Python: charlm < 1e-7, tokenizer logits ≈ 4e-6, UPOS logits ≈ 2e-5,
  parser scores ≈ 2e-5. Test tolerances are 1e-4 and 1e-3.
- `StanzaSharp.Pipeline` ties the processors together: `Load(modelDir, "tokenize,mwt,...")`, then
  `Process(text)`. It loads the shared pretrain/charlms once. `Conllu.Write(pipeline.Process(corpus))`
  is byte-identical to `tests/golden/pipeline.conllu`.
  - The `samples/StanzaSharp.Cli` sample writes the same CoNLL-U from a file or stdin.
  - `Conllu.Write` fills a missing HEAD with `id - 1`, like Stanza's writer.

Tokenizer notes:
- Stanza passes each row's *padded* length to the LSTM, so padding reaches the backward
  direction and output depends on batch composition. `Tokenizer.Predict` reproduces it exactly:
  sort paragraphs by length, batch by 32, pad to max+1, and use 1000-char windows.
- Model units are code points, like Python. Offsets are UTF-16 indices, so they differ from
  Stanza's only after non-BMP characters.
- MWT-flagged tokens (`Token.IsMwtCandidate`) carry a single word until the MWT stage runs.

## Layout

```
src/StanzaSharp.Core           Document model, CoNLL-U I/O, constituency Tree, safetensors reader,
                               converted-checkpoint JSON loading. No TorchSharp dependency.
src/StanzaSharp.Nn             TorchSharp: charlm, pretrain embeddings, shared layers, weight loading.
src/StanzaSharp.Tokenize       Tokenizer + sentence splitting.
src/StanzaSharp.Mwt            Multi-word token expansion.
src/StanzaSharp.Pos            POS / feature tagger.
src/StanzaSharp.Constituency   Constituency parser.
src/StanzaSharp                Pipeline facade wiring the processors together.
samples/StanzaSharp.Cli        Console runner for quick experiments.
tests/StanzaSharp.Tests        xUnit; golden tests against Python Stanza output.
tests/golden/                  Golden data generated from Python Stanza (committed, keep it small).
tools/stanza_convert.py        Checkpoint inspector/converter (.pt -> .safetensors + .json).
tools/.venv/                   Python env with stanza installed (gitignored).
models/stanza/                 Original downloaded models (gitignored).
models/converted/en/           Converted models used by C# (gitignored).
```

Package policy: library projects reference the managed `TorchSharp` package only. Runnable
projects (Cli, Tests) reference `TorchSharp-cpu`, or `TorchSharp-cuda-windows` when set up with
`-Cuda`, which brings the native libtorch. The managed and native TorchSharp versions must match.

## Reference implementation

The Python Stanza source installed in `tools/.venv/Lib/site-packages/stanza/` is the source of
truth. When porting a model, read its code there (`stanza/models/tokenization`, `mwt`, `pos`,
`constituency`, `common/char_model.py`, `common/pretrain.py`, `pipeline/`) rather than relying on
memory of how Stanza works. Port the code paths the English checkpoints actually use, which the
checkpoint configs tell you; skip unused options.

## Model files

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

All seven checkpoints convert with `torch.load(weights_only=True)`; no unsafe pickling is needed.

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
  - `mwt.json`: expansions of a word list, both through the pipeline and classifier-only.

## Design decisions

- Model the document as Document → Sentence → Token → Word from the start. Tokens come from the
  tokenizer, Words come out of MWT expansion, and POS and the parser operate on Words. Keep this
  split even before MWT is implemented.
- Batch inference per sentence from the start.
- Run inference under no-grad / eval mode, and dispose tensors deterministically (TorchSharp
  `DisposeScope`).

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

Stanza's code is Apache 2.0. Model licenses vary with the training data, so do not commit or
redistribute model files. `models/` is gitignored.
