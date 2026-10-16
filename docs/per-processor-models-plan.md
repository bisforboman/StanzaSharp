# One model per processor (`processors={'ner': 'i2b2'}`): research and plan

> **Owner's decision (2026-10-16):** support choosing a single alternative model per processor, as Stanza does. This
> changes the API, so this file was phase 1: research and a proposal.

## Owner's decisions (2026-10-16)

1. **API: option A.** `PipelineOptions.Models` (processor → model, `Processor` keys), and `--model ner=i2b2`
   (repeatable) on the tool's download, compare and benchmark and the Cli sample.
2. **Keys add processors**, as in Stanza's dict form. With `Processors` null, the set is the package's processors plus
   the keys of `Models`, and requirements are added as usual. With `Processors` set, a key outside it throws.
   Unknown, transformer and unported names throw `ArgumentException` naming the supported choices, and so do two
   charlm pairs.
3. **First PR:** the API plus the drop-in set, **without** the treebank `_nocharlm` taggers and parsers (CI cache):
   - constituency `ptb3_charlm`
   - ner `ontonotes_charlm`, `ontonotes_nocharlm` (+ fasttextcrawl) and `combined_charlm`
   - German ner `conll03`
   - plus `i2b2`, with `Package = "mimic"` (the delta-vocabulary check fix)

   The other domain NER models and multiple charlm pairs come in a later PR.
4. Byte-exact against Stanza on both backends, with golden data per model (`make_golden.py --models-only`, a
   hand-written clinical text for i2b2), the Linux golden shards' cache, and docs.

Built in the same PR as this plan. The selection lives in `Pipeline.SelectModels`, the table in `Pipeline.Alternatives`,
and the tests in `ModelChoiceTests`. The CI cache grows by 501 MB (English, in `models-en-treebanks`) and 51 MB (German
`conll03`, in `models-de-treebanks`).

The research below (2026-10-16) predates the build. The inventory comes from
`src/StanzaSharp.Tool/stanza_resources_en.json` and `_de.json` (Stanza 1.15.0). Checkpoint sizes come from the Hugging
Face tree API (`stanfordnlp/stanza-<lang>`, revision `v1.15.0`).

Each checkpoint not yet on disk was inspected without downloading its weights. Every Stanza checkpoint uses torch's
legacy format: the pickle (config, vocabularies, tensor names and shapes) comes first and the storages follow. So an
HTTP range request for the first 4 MB (more where needed), unpickled with stub tensors, gives everything
`stanza_convert.py inspect` shows except the values. Each checkpoint was then diffed against the one StanzaSharp
already runs that is closest to it: config keys that differ, and parameters added, removed or reshaped. Stanza's code
is in `tools/.venv/Lib/site-packages/stanza/`.

**In short:**

- 56 non-transformer models are not yet supported (45 English, 11 German), plus 4 new shared files.
- 24 of them load **as they are**. These are the `_nocharlm` taggers and parsers of the treebanks,
  `constituency/ptb3_charlm`, NER `ontonotes_*` and `combined_charlm`, and German NER `conll03`. They need only
  table entries: the model name, its dependencies and its MD5.
- The 11 domain NER models (biomedical, clinical and `conll03_charlm`) use the **old NER checkpoint format with a
  delta vocabulary**. Each needs one or two small changes in `NerTagger`: cased word lookups, and a check that
  is too strict.
- The `_charlm` **tokenizers and lemmatizers** have new code paths: the charlm feeds per-character states into the
  tokenizer's LSTM and into the lemmatizer's encoder. Stanza uses them only in `default_accurate`.
- **One structural blocker: a pipeline loads exactly one pair of charlms.** The biomedical NER models read the `pubmed`
  charlms and the clinical ones read `mimic`. So `default` + `ner: i2b2` needs two pairs, and today that fails in
  `Pipeline` (`SingleOrDefault`). The natural pairings work with one pair: `craft`/`genia` + a biomedical model,
  `mimic` + `i2b2`/`radiology`.
- **Recommended API:** `PipelineOptions.Models`, a processor → model dictionary with Stanza's dict semantics, and
  `--model ner=i2b2` in the tool. The first PR adds only the models that load as they are, plus `i2b2`.

## Inventory

All sizes are in MB. "Deps" are the shared files listed in resources.json (pretrain, then charlms). An entry with
nothing under Deps reads none.

### English

| processor | model | MB | deps | status |
|---|---|---:|---|---|
| constituency | `ptb3_charlm` | 114.0 | conll17, 1billion | **as is**: same config as `ptb3-revised_charlm`, labels 28 vs 27, transitions 30 vs 29 |
| ner | `ontonotes_charlm` | 90.4 | **fasttextcrawl** (300-d), 1billion | **as is**: new format, one tag set (OntoNotes, 18 types), word 300 |
| ner | `ontonotes_nocharlm` | 54.9 | fasttextcrawl | **as is**: `_nocharlm` like `ontonotes-ww-multi_nocharlm`, input 500 |
| ner | `combined_charlm` | 69.7 | conll17, 1billion | **as is**: new format, **3 tag sets** (77, 41, 21), `predict_tagset` 0 |
| ner | `conll03_charlm` | 47.5 | **glove** (100-d), 1billion | old format + delta: delta vocab `lower: false` (check relaxed) |
| ner | `i2b2` | 49.8 | mimic, mimic | old format + delta, delta `lower: false`; 3 types (PROBLEM, TEST, TREATMENT) |
| ner | `radiology` | 45.2 | mimic, mimic | as `i2b2`, plus **`char_lowercase: true`** (the charlm reads lowercased text); 5 types |
| ner | `anatem` `bc4chemd` `bc5cdr` `bionlp13cg` `jnlpba` `linnaeus` `ncbi_disease` `s800` | 49–66 each, 429 together | biomed (200-d), **pubmed** | old format + delta, **`lowercase: false`** (cased pretrain and delta lookups) |
| pos | `{atis,childes,eslspok,ewt,gum,lines,partut}_nocharlm` | 16–23 each, 137 together | conll17 | **as is**: the `_charlm` twin's legacy heads + `combined_nocharlm`'s character model |
| depparse | `{atis,childes,eslspok,ewt,gum,lines,partut}_nocharlm`, `mimic_nocharlm` | 98–111 each, 829 together | conll17 (mimic: mimic) | **as is**: the twin with `charmodel.*` + `trans_char` (as `combined_nocharlm`) |
| tokenize | `{atis,childes,combined,eslspok,ewt,gum,lines,partut}_charlm` | 2.7–2.8 each, 22 together | forward 1billion only | **new path**: `rnn.weight_ih_l0` is [256, 1065] = 41 features + 1024 forward-charlm units per character |
| lemma | `{atis,childes,combined,ewt,gum,lines,partut}_charlm` | 8.5–8.9 each (combined 31.0), 83 together | 1billion pair (combined: also conll17) | **new path**: the encoder input is [400, 2098] = char 50 + 2048 charlm per character |
| (shared) | `pretrain/fasttextcrawl` 122.6, `pretrain/glove` 42.6, `forward_charlm/pubmed` 19.7, `backward_charlm/pubmed` 19.7 | | | new files. Same formats as conll17 and 1billion (pretrains [100000, 300] and [100000, 100]; pubmed is char 100, LSTM 1024, vocab 279) |

Skipped:

- the `_electra-large` variants of pos, depparse, constituency, sentiment and NER (transformers);
- `coref` (docs/coref-plan.md);
- the hub's legacy aliases (`tokenize/ewt.pt` and others), which resources.json doesn't list.

`sentiment` and `mwt` have no non-transformer alternatives beyond the packages already supported.

### German

| processor | model | MB | deps | status |
|---|---|---:|---|---|
| ner | `conll03` | 50.8 | fasttextwiki, newswiki | **as is**: same config and shapes as `germeval2014` (old format, no delta, 21 tags, PER/LOC/ORG/MISC) |
| pos | `gsd_nocharlm`, `hdt_nocharlm` | 23.3, 30.1 | conll17 | **as is**: as English |
| depparse | `gsd_nocharlm`, `hdt_nocharlm` | 103.3, 111.9 | conll17 | **as is** |
| tokenize | `{combined,gsd,hdt}_charlm` | 2.7–2.8 each | forward newswiki | new path (as English) |
| lemma | `{combined,gsd,hdt}_charlm` | 9.3–12.1 each | newswiki pair | new path (as English) |

"As is" means: no config key that `CheckSupported` rejects, and a parameter set StanzaSharp already reads. This was
checked against the head of each checkpoint, not by loading it. The first PR loads each one and runs it against
Stanza.

## What the checkpoints need

- **NER, old format with a delta vocabulary** (`conll03_charlm`, `i2b2`, `radiology`, the 8 biomedical models).
  - The heads are `tag_clf.*`/`crit._transitions`. These are German `germeval2014`'s names, and `NerTagger.HeadPrefix`
    already handles them. The tag vocabulary is a plain `TagVocab` (one tag set).
  - Unlike germeval, these models have a fine-tuned `delta_emb` (6.7k–27k words), stored as a `PretrainedWordVocab` with
    `lower: false`. `NerTagger.CheckSupported` rejects them ("a cased delta vocab"). The check is stricter than Stanza:
    `extract_static_embeddings` applies the model's `lowercase` flag to the word and then looks it up as is (a
    `PretrainedWordVocab` also maps `' '` to U+00A0). So the fix is to drop that check and apply the space mapping.
  - The 8 biomedical models have `lowercase: false`: both the pretrain and the delta lookups use the token as written.
    `NerTagger` lowercases unconditionally today. This is a one-flag change, plus the check that rejects it.
  - `radiology` has `char_lowercase: true`. `DataLoader.preprocess` lowercases each character before the charlm reads
    it, so the charlm input differs from the tagger's. That needs lowercased charlm input and **no `CharlmCache`
    reuse** for that model.
  - All of them have word vocabularies of 100,000. These equal their pretrains (biomed, mimic and glove are all
    100,000 rows), so `CheckSupported`'s size check holds.
  - Input sizes: 2248 = word 200 + charlm 2048 (biomed and mimic), and 2148 for conll03 (glove 100). BIOES throughout.
    Tag counts: 9, 9, 13, 66, 25, 9, 9 and 9 for the biomedical models, 17 for i2b2, 25 for radiology, 21 for conll03.
- **NER `combined_charlm`**: a new-format model with three tag sets. Only `predict_tagset` 0 is run, as for
  `ontonotes-ww-multi_charlm`. The delta has 57k words.
- **New shared files**: pretrains `fasttextcrawl` and `glove`, and charlms `pubmed`. They use the same formats and
  sizes as what we load (all pretrains are `{vocab, emb}`, all charlms char 100 → LSTM 1024).
- **Tokenizer `_charlm`** (`models/tokenization/model.py`): the forward charlm's hidden state per character is
  concatenated to the 41 feature units before `rnn`. This is new code in `ITokenizerNet` on both backends. The batches
  and windows stay as they are, but the charlm has to run over the 1000-character windows. Estimate: **M**.
- **Lemmatizer `_charlm`** (`lemma/trainer.py` + `common/seq2seq_model.py`): both charlms' per-character states are
  concatenated to the char embedding at each encoder step (2098). This is new code in `ISeq2SeqNet`. Estimate:
  **S–M**.
- **Constituency `ptb3_charlm`, the `_nocharlm` taggers and parsers, German `conll03`**: nothing new. Only `Dependencies`
  and `ModelDownloader.Files` entries.

### Blocker: one charlm pair per pipeline

`Pipeline` picks the forward and backward charlm with `SharedModels(...).SingleOrDefault(...)`. `BackendModels`
stores one `_forward`/`_backward`, and `CharlmCache` assumes that every reader shares the tagger's charlms. Today
every package reads at most one pair. With per-processor models that no longer holds:

| pipeline | charlms | today |
|---|---|---|
| `default` + `ner: i2b2` | 1billion (pos, constituency, depparse, sentiment) + mimic (ner) | throws `InvalidOperationException` in Load |
| `default_fast` + `ner: anatem` | 1billion (sentiment) + pubmed (ner) | throws |
| `mimic` + `ner: i2b2` / `radiology` | mimic | works (radiology without the cache) |
| `craft` or `genia` + any biomedical NER | pubmed (only NER reads a charlm; craft and genia are `_nocharlm`) | works |
| `Processors = "tokenize,ner"` + any NER model | that model's pair | works |
| `default` + `ner: conll03_charlm` / `ontonotes_charlm` | 1billion | works (pretrains glove or fasttextcrawl next to conll17: several pretrains already work) |

Making it general is a contained change, estimate **S–M**:

- `BackendModels.Charlm` keeps one charlm per path.
- Each processor gets its own pair from `Dependencies`, as it already does for its pretrain (`PretrainOf`).
- `CharlmCache` is shared only between processors that read the same pair. Simplest: the pipeline creates the cache
  only for the tagger's pair, and a reader on another pair passes null.

**Proposal:** the first PR throws a clear `ArgumentException` at selection time ("ner `i2b2` reads the mimic charlms,
pos `combined_charlm` reads 1billion; one pair per pipeline for now: use Package = "mimic", or leave out pos..."). The
PR that adds the biomedical and clinical NER models then lifts the limit, since that is where it matters.

## Stanza's semantics

From `stanza/resources/common.py` (`process_pipeline_parameters`, `maintain_processor_list`, `add_mwt`,
`add_dependencies`) and `stanza/pipeline/core.py`:

1. **`processors` as a string** (`"tokenize,ner"`) and `package` as a string or None: each listed processor gets
   `package` (default `'default'`) as its value, and **only the listed processors** load (`package` becomes None
   afterwards).
2. **`package` as a dict** (only together with a `processors` string): each listed processor takes its value from the
   dict, and `'default'` for the processors the dict doesn't name. Example: `processors='tokenize,ner',
   package={'ner': 'i2b2'}`.
3. **`processors` as a dict** (`{'ner': 'i2b2'}`): each value is resolved, then **every processor of `package`**
   (default `'default'`) that the dict doesn't name is added. So `Pipeline('en', processors={'ner': 'i2b2'})` loads all
   eight default processors with NER replaced. `package=None` loads only the dict's keys.
4. **Resolving a value** for processor `key`, in order:
   1. A model name in `resources[lang][key]` (`'i2b2'`, `'ptb3_charlm'`). Names are exact. There is no `_charlm`
      suffix fallback: `'ptb3'` or `'ontonotes'` matches nothing.
   2. A package name containing `key`: `{'pos': 'ewt'}` → `ewt_charlm`, `{'ner': 'default'}` → `ontonotes-ww-multi_charlm`.
   3. A package's `optional` entry (only coref).
   4. A processor variant (spaCy and others: not ported), or `lemma: 'identity'`.
   5. Otherwise: **a warning and the value is ignored**. In case 3 that silently falls back to the package's model,
      in case 1 the processor silently disappears.
   - A value can also be a list (several NER models whose tags are merged, earlier ones first). Only NER supports that.
5. **Unknown processor keys** raise `UnknownProcessorError`. Keys and values are stripped and lowercased.
6. **MWT follows tokenize** (`add_mwt`): when `tokenize` is given without `mwt`, mwt is added from tokenize's value. The
   value can be a package with mwt, an mwt model of the same name, or the name without `_charlm`/`_nocharlm`.
   `{'tokenize': 'ewt_nocharlm'}` → `mwt: ewt`, not the package's `combined`. This is the only place Stanza treats the
   suffixes specially.
7. **Dependencies** (`add_dependencies`) come per model from resources.json's `dependencies`, never from the package.
   Each model loads its own pretrain and charlms (`build_default_config`: `<processor>_pretrain_path`, ...). The
   `FoundationCache` loads a file once per path, so mixed charlms just work in Stanza.
8. **A package name that is not a package** (`package='i2b2'`) is treated as a model name for every processor that has
   it. That gives `ner: i2b2` only, with a warning if none has it.

## API proposals

All three share the same resolution. Each produces the processor → model map that `Pipeline.SelectModels` returns
today. So `Load`, `ModelDownloader.DownloadAsync(dir, options)`, `FilesFor`, `VerifyChecksums` and the tool all read
it unchanged.

### A. `PipelineOptions.Models` (recommended)

```csharp
// Stanza: stanza.Pipeline('en', processors={'ner': 'i2b2'}, package='mimic')
using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions
{
    Package = "mimic",
    Models = new Dictionary<string, string> { [Processor.Ner] = "i2b2" },
});

// Stanza: stanza.Pipeline('en', processors={'constituency': 'ptb3_charlm'})
var options = new PipelineOptions { Models = new Dictionary<string, string> { [Processor.Constituency] = "ptb3_charlm" } };
await ModelDownloader.DownloadAsync("models/stanza/en", options); // the default package + ptb3_charlm and its deps
using var ptb3 = Pipeline.Load("models/stanza/en", options);

// Stanza: stanza.Pipeline('en', processors='tokenize,ner', package={'ner': 'anatem'})
using var bio = Pipeline.Load("models/stanza/en", new PipelineOptions
{
    Processors = "tokenize,ner",
    Models = new Dictionary<string, string> { [Processor.Ner] = "anatem" },
});
```

- Type: `IReadOnlyDictionary<string, string>?`, init-only, null by default. It stays string-keyed and string-valued,
  like `Package`, so new models need no API change (the owner's 2026-10-06 rule for packages). Keys are `Processor`
  constants.
- Semantics:
  - With `Processors` null, the set is the package's processors plus the dictionary's keys (Stanza's dict form, rule
    3). So `Package = "ewt", Models = { ner: … }` gives ewt's tokenize to depparse + NER.
  - With `Processors` set, it is that set (Stanza's string + package dict, rule 2), and a key outside it throws.
- Tool: `--model ner=i2b2`, repeatable or comma-separated (`--model ner=i2b2,constituency=ptb3_charlm`). This works on
  `download`, `compare` and `benchmark`. `compare` passes the resolved map to `compare.py` as a JSON processors dict
  with `package=None`, so Stanza loads exactly the same models. The existing check (Stanza's `.pt` set ==
  `ModelDownloader.FilesFor`) still proves it.
- Name: `--models DIR` already exists in compare and benchmark, hence the singular `--model`. The property could also
  be called `ProcessorModels` if `Models` reads too much like a directory. **That is an owner's choice.**
- For: one obvious place, mirrors Stanza's dict, and no parsing in the library. Several NER models per processor
  (Stanza's list value) would need another type or a separator later (Open questions, 4).
- Against: a dictionary initializer is wordier than a string.

### B. Inline in `Processors`: `"tokenize,mwt,pos,ner:i2b2"`

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Package = "mimic", Processors = "tokenize,mwt,pos,lemma,depparse,ner:i2b2" });
```

- No new member, and the CLI is identical (`--processors tokenize,ner:i2b2`). `ParseProcessors` gains `name[:model]`.
- Against:
  - It is a mini-language Stanza doesn't have.
  - It always needs the full processor list. "All of the package's processors but NER from i2b2" has no spelling,
    unless `Processors = "ner:i2b2"` meant "add". That would contradict what `Processors` means today.
  - A list value (several NER models) would need another separator.

### C. A fluent method

```csharp
var options = new PipelineOptions { Package = "mimic" }.WithModel(Processor.Ner, "i2b2");
```

- Reads well, and `PipelineOptions` stays immutable (it returns a copy).
- Against:
  - It adds a method and still needs the storage of A underneath (a property for the downloader and the tool).
  - It is the only `With…` in the API.
  - Collection expressions and object initializers can't use it.

**Recommendation: A**, with `--model PROCESSOR=NAME` in the tool. B could be accepted later by the tool as sugar
without touching the library.

### Validation (any shape)

All errors are `ArgumentException`s with `paramName` `options`, thrown by `SelectModels`, so `Load` and `DownloadAsync`
fail the same way before any I/O:

- An unknown key → "Unknown processor 'nerr'. Available: …" (as today).
- An unknown value → "Unknown ner model 'ontonotes'. Available: ontonotes-ww-multi_charlm, ontonotes_charlm, …".
  **We throw where Stanza warns and ignores**, as we already do for a processor missing from a package (the owner's
  rule). Exact names only, like Stanza: no suffix guessing.
- A transformer name (`*_electra-large`, `*bert*`) → the same message `default_accurate` gets.
- A known but unported model (the tokenizer and lemmatizer `_charlm` until they are ported) → "not supported yet".
  This comes from a table, not from a load failure.
- A value that is a package name resolves to that package's model for the key (`{pos: "ewt"}` → `ewt_charlm`), as in
  Stanza. `lemma: "identity"` is allowed.
- With `Processors` set, a key it doesn't list throws ("Models names ner, which Processors doesn't list").
- MWT follows an overridden tokenizer as in `add_mwt` (`tokenize: ewt_nocharlm` → `mwt: ewt`), unless `Models` names
  mwt.
- Requirements are unchanged (`Requires`). A processor the package lacks but `Models` supplies is fine (`ewt` + `ner`).
- Mixing treebanks (pos from gum, depparse from ewt) is allowed, as in Stanza. The tag sets match in English UD, but
  e.g. `atis` has no XPOS, and constituency reads XPOS. This is documented, not blocked.
- Until the charlm blocker is lifted: two charlm pairs → the message above.
- `Package` keeps its meaning: it gives the defaults for every processor `Models` doesn't name. The log line "Loaded
  the {Package} pipeline" also lists the overridden models.

### Code touched (for the build phase)

- `Pipeline`: a per-language table of the supported models per processor. The packages' models are added
  automatically, the alternatives are listed explicitly. Plus `SelectModels(options)` with the rules above,
  `Dependencies` entries for each new model, and (later) the charlm pairs.
- `ModelDownloader.Files`: MD5s for each new file. `CompareCommandTests.EmbeddedResources_MatchModelDownloader`
  already checks them against `stanza_resources_*.json`, and `FilesFor_AreEnoughToLoadThePipeline` checks the sets.
- `PipelineOptions.Models` + XML docs, `PACKAGE.md`/README, a `changes/added/` entry.
- `DownloadCommand`, `CompareCommand`, `BenchmarkCommand`: `--model`. `compare.py`/`benchmark.py`: a JSON processors
  dict.
- `NerTagger` (second PR): the `lowercase` flag, the delta check, the U+00A0 mapping, `char_lowercase` with the cache
  off.

## Which models first

| PR | models | model code | new cache (MB) | golden (est.) |
|---|---|---|---:|---:|
| **1: API + "as is"** | constituency `ptb3_charlm`; ner `ontonotes_charlm`, `ontonotes_nocharlm`, `combined_charlm`; de ner `conll03` | none (tables, API, tool) | 114 + 345 (incl. fasttextcrawl 123) + 51 = **510** | ~0.4 MB |
| 2: domain NER + several charlm pairs | ner `i2b2`, `radiology`, `conll03_charlm` (+ glove), the 8 biomedical (+ pubmed) | `NerTagger` flags; several charlms in `Pipeline`/`BackendModels`/`CharlmCache` | 95 + 90 + 469 = **654** | ~0.4 MB + two hand-written domain texts |
| 3: treebank `_nocharlm` taggers and parsers | en pos ×7, depparse ×8; de pos ×2, depparse ×2 | none | 966 + 269 = **1,235** | ~1–1.5 MB (corpus + validation only) |
| 4 (on demand) | tokenizer and lemmatizer `_charlm` (en 15, de 6) | new paths in `ITokenizerNet` and `ISeq2SeqNet` on both backends | 105 + 41 = 146 | ~1 MB |

The owner's examples were `ner: i2b2` and `constituency: ptb3`. `ptb3` is `ptb3_charlm` (exact names). `i2b2` belongs to
PR 2, but can move into PR 1 without the multi-charlm work, if the owner wants it there: it needs only the
delta-check change, and it works with `Package = "mimic"` or `Processors = "tokenize,ner"`.

### Golden data and CI cost

- **Golden** (`make_golden.py --models-only`, `tests/golden/models/<lang>/<processor>/<model>/`): the generic corpus
  shows few biomedical or clinical entities, so PR 2 adds two short hand-written texts (`biomedical.txt`,
  `clinical.txt`, ~3 KB each).
  - NER models: the entities and per-token BIOES tags as JSON (as `ner/<name>.json`), for corpus.txt, validation.txt
    and the domain text. The tokens are already checked by the default golden. That is ~20–40 KB per model, about
    0.3 MB for all 14, plus the first 3 sentences' emissions for one model per checkpoint format.
  - Constituency: the trees (one line per sentence), ~30 KB.
  - Taggers and parsers: CoNLL-U for corpus.txt + validation.txt (~110 KB per package).
- **Cache.** The repository's caches already hold about 7 GB of the 10 GB budget (2026-10-16):
  - models-en 1.35 GB
  - models-en-treebanks 1.73 GB
  - models-de-treebanks 1.27 GB
  - models-pt-{Linux, Windows, macOS} 0.68 GB each
  - nuget-Linux 0.6 GB

  Every key hashes `ModelDownloader.cs`, so the PR that adds the MD5s rotates them all, and old and new keys coexist
  until eviction.
  - Proposal: one new Linux-only cache, `models-en-models-<hash>` (and `models-de-…` for German `conll03`). Golden
    shard 1 writes it, shards 2–3 restore it, cross-os and linux-arm64 skip those rows. This is the pattern of
    `models-en-treebanks`.
  - PR 1 + PR 2 add ~1.2 GB, which fits.
  - **PR 3's 1.2 GB would push the budget to ~9.4 GB**, too tight for key rotation. PR 3 needs the owner's call:
    test those models only on the light golden texts with the cache downloaded on demand (no cache: ~1.2 GB from
    Hugging Face per golden run, about 30–60 s), or not at all.
- **Test time** (an estimate, not measured): a NER model is the size of the default one, which loads and tags
  corpus + validation in a few seconds, so 14 models add about a minute per backend on one shard.

## Open questions for the owner

1. API shape: **A** (`PipelineOptions.Models`, recommended), B (inline `ner:i2b2`) or C (fluent)?
2. Name: `Models`, `ProcessorModels` or another? Tool option: `--model ner=i2b2`?
3. With `Processors` null, does `Models` add its keys to the package's processors (Stanza's dict form; recommended), or
   must they be in the package?
4. Several NER models per pipeline (Stanza's list value, merged tags): not now. Should the value type leave room for it
   (a `;`-separated string as in Stanza's `ner_model_path`), or add it later as a new overload?
5. Scope of PR 1: the "as is" set alone, or with `i2b2`? And PR 3's cache trade-off.
