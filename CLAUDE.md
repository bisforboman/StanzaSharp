# StanzaSharp

A native C# port of the inference side of [Stanza](https://github.com/stanfordnlp/stanza), Stanford's
PyTorch-based NLP library, running the original pretrained models through TorchSharp.

## Scope

- **Language:** English only, to start.
- **Processors:** `tokenize` → `mwt` → `pos` → `lemma` → `depparse` → `constituency`. All six are
  `Pipeline.AllProcessors`, the default, in that order, like Stanza's English default (2026-10-06).
  `pipeline.conllu` and `validation*.conllu` are generated with all six (`PROCESSORS` in make_golden.py).
- **Inference only.** Training stays in Python; we load Stanza's released weights.
- **Package:** Stanza's English *default* package, which needs no transformer:
  - tokenize: `combined_nocharlm`
  - mwt: `combined`
  - pos: `combined_charlm`
  - lemma: `combined_nocharlm`
  - constituency: `ptb3-revised_charlm`
  - depparse: `combined_charlm`
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
  lemma, before constituency) sets `Word.Head`/`Deprel` with the pipeline's shared pretrain/charlms.
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
- `Checkpoint.Load` also reads Stanza's original `.pt` files (`Core/TorchCheckpoint.cs` +
  `Core/Pickle.cs`), so `Pipeline.Load("models/stanza/en")` works without Python. For all eight
  checkpoints the result equals the converter's: identical JSON and byte-identical tensors.

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
src/StanzaSharp                Pipeline facade wiring the processors together.
samples/StanzaSharp.Cli        Console runner for quick experiments.
samples/StanzaSharp.Benchmark  Per-stage speed/memory benchmark; tools/benchmark.py is the Python twin.
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

GPU: `Pipeline.Load(dir, processors, device)` and every processor's `Load(..., device)`. `Weights.On(device, ...)`
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

All eight checkpoints convert with `torch.load(weights_only=True)`; no unsafe pickling is needed.

### Reading .pt files in C#

All eight Stanza checkpoints use torch's legacy format (`_use_new_zipfile_serialization=False`), a
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
  `workflow_call` from release.yml; `ubuntu-24.04` pinned).
  - `build-test` runs the suite without models.
  - `golden`, on a model-cache miss, downloads the models with the C# `ModelDownloader` (via the
    CLI) and converts them with CPU torch. The models are cached on `ModelDownloader.cs`,
    `tools/requirements.txt` and `tools/stanza_convert.py`. It then fails if any test is skipped
    (`outcome="NotExecuted"` in the trx; the trx `notExecuted` counter stays 0 for skips), and runs
    `tools/verify-package.ps1` against `models/stanza/en`.
  - CI runs on Linux: keep paths forward-slash and file names case-exact. The root `.gitattributes`
    keeps sources LF.

## Packaging and releases

- One NuGet package, `StanzaSharp`, packed from `src/StanzaSharp` (user's decision, 2026-10-06).
  - It carries all eight assemblies plus their XML docs: the facade's ProjectReferences are
    `PrivateAssets="all"`, and an `IncludeProjectReferences` target adds them.
  - It depends only on managed `TorchSharp`; users add `TorchSharp-cpu`/`-cuda` themselves.
  - Every other project is `IsPackable=false` (Directory.Build.props, which also holds the shared
    package metadata and a `0.1.0-dev` default version).
  - Because of `PrivateAssets`, tests and samples reference the library projects they use directly.
  - `src/StanzaSharp/PACKAGE.md` is the package readme; `NOTICE` ships in the package.
- Models are downloaded only explicitly (user's decision): `ModelDownloader.DownloadAsync(dir)` or
  `StanzaSharp.Cli download [DIR]`.
  - It fetches the 8 `.pt` files from `huggingface.co/stanfordnlp/stanza-en/resolve/v1.15.0/models/`
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
  - One-time setup by the owner, not yet done: a nuget.org Trusted Publishing policy per
    environment (owner bisforboman, repo StanzaSharp, workflow `release.yml`), the repo variable
    `NUGET_USER`, and the two GitHub environments.

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

StanzaSharp is Apache 2.0 (`LICENSE`), like Stanza, whose code it ports. `NOTICE` credits Stanza;
keep it in any redistribution. Model licenses vary with the training data, so do not commit or
redistribute model files. `models/` is gitignored.
