# Pretagged input (Stanza's `*_pretagged`): research and plan

> **Owner's choice (2026-10-17):** look at porting Stanza's `*_pretagged` options, which CLAUDE.md lists as not ported
> ("depparse without pos in the pipeline"). This file is phase 1: research and a proposal. Nothing is built yet; the
> public API is the owner's decision.

## Questions for the owner

1. **API:** alternative **A** (one `PipelineOptions.Pretagged` bool, recommended), **B** (`CheckRequirements = false`)
   or **C** (`Process(Document, processors)`)? See "Alternatives".
2. **Tokenizer:** with A, does a pretagged pipeline still load tokenize (and the package's mwt), as recommended (1.2 MB,
   and the text overloads keep working), or does tokenize become optional, so text overloads throw without it?
3. **Constituency on a word without its tag:** keep throwing `InvalidOperationException` (recommended), or port
   Stanza, which parses with an empty tag and prints trees like `(NP ( The) ( quick))`?

## What Stanza 1.15.0 does

Read from `stanza/pipeline/` (core.py, processor.py, depparse_processor.py, lemma_processor.py,
constituency_processor.py) and checked by running Stanza on `models/stanza/en`.

### Two mechanisms

- **`pretagged`.** Only three processors read it: lemma, depparse and constituency. Each one's `_set_up_requires`
  sets `self._requires = set()` when it is on, so it needs no processor at all, not even tokenize. It changes nothing
  else: `process` is the same code, and the flag is read only at load time.
  - It can be given per processor (`depparse_pretagged=True`, through `filter_config`'s `<processor>_` prefix) or once
    for the whole pipeline (`pretagged=True`). core.py line 301 ("TODO: this is obviously a hack") copies the global
    value into every processor's config, so it reaches all three.
  - pos, ner, sentiment, mwt and tokenize have no pretagged form. pos, ner and sentiment require tokenize, and the
    flag doesn't change that: `processors='pos,depparse', pretagged=True` fails because pos lacks tokenize.
- **`check_requirements`** (`Processor._check_requirements`). `<processor>_check_requirements=False` skips that
  processor's check entirely. It is the general escape hatch: `processors='pos', pos_check_requirements=False` loads.
  It is not documented as a user option, but it works for any processor.

The check itself runs once per processor at load. `provided_reqs` is the union of what the processors loaded so far
provide. Each processor is loaded in `PIPELINE_NAMES` order, so the check compares against the processors before it.
Failures are collected and thrown together as `PipelineRequirementsException`.

### Load results (measured)

| `stanza.Pipeline('en', ...)` | result |
|---|---|
| `processors='depparse'` | PipelineRequirementsException |
| `processors='depparse', depparse_pretagged=True` | loads; `load_list` is `['depparse']` (plus its pretrain and charlms) |
| `processors='depparse', pretagged=True` | loads |
| `processors='depparse', depparse_check_requirements=False` | loads |
| `processors='tokenize,depparse', depparse_pretagged=True` | loads `tokenize, mwt, depparse` (add_mwt) |
| `processors='lemma'` | PipelineRequirementsException (needs tokenize) |
| `processors='lemma', lemma_pretagged=True` | loads |
| `processors='constituency'` | PipelineRequirementsException |
| `processors='constituency', constituency_pretagged=True` | loads |
| `processors='pos'` / `'pos', pretagged=True` | PipelineRequirementsException (both) |
| `processors='pos', pos_check_requirements=False` | loads |
| `processors='ner'` / `'ner', pretagged=True` / `'sentiment'` | PipelineRequirementsException |

Lemma without pretagged requires only tokenize in Stanza (`lemma_pos` is never set in practice), so `tokenize,lemma`
loads and lemmatizes with no UPOS. StanzaSharp requires pos for lemma too (an earlier decision: the model reads UPOS).

### Processing

`Pipeline.process(doc)` runs each loaded processor in `PIPELINE_NAMES` order. A pretagged pipeline without tokenize
has to get a `Document`, such as one from `CoNLL.conll2doc`. Given a string, depparse fails with `AttributeError: 'str'
object has no attribute 'sentences'` (lemma: `... no attribute 'get'`).

The output doesn't depend on which pipeline runs it. A `processors='depparse', depparse_pretagged=True` pipeline gives
byte-identical CoNLL-U to a full pipeline's `process(doc, processors='depparse')` on the same document (measured). This
is also what StanzaSharp's `Process(Document)` already does on a full pipeline, except that there pos and lemma run
first and overwrite the given tags.

What each processor does when the annotations it reads are missing (measured on corpus sentences with fields removed):

| processor | reads | missing | Stanza | StanzaSharp today |
|---|---|---|---|---|
| depparse | UPOS, XPOS, lemma (feats are read but unused: its `ufeats_emb` is never used) | UPOS **and** XPOS on any word | `ValueError: POS not run before depparse!` | `InvalidOperationException` ("Run the POS tagger before the dependency parser"), same condition |
| depparse | | XPOS only | parses (XPOS `_`) | parses (`w.Xpos ?? "_"`) |
| depparse | | lemma | parses with lemma `_`; heads differ from the tagged run | the same (`w.Lemma ?? "_"`) |
| depparse | | feats | parses; heads unchanged | the same |
| lemma | text, UPOS | UPOS | lemmatizes (dictionary and seq2seq with no UPOS) | `Upos ?? "_"` in the seq2seq; the dictionary lookup with null UPOS is unverified |
| constituency | text, XPOS (en) / UPOS (de) | that tag | parses; the tree has empty tags: `(NP ( The) ( quick) ( brown) ( fox))` | `InvalidOperationException` ("Run the POS tagger before the parser") |

Today none of these cases can be reached in StanzaSharp, because Load requires pos before lemma, depparse and
constituency, and pos sets every word's tags. Pretagged makes them reachable, so they become part of the contract.

`tokenize_pretokenized` and `tokenize_no_ssplit` are already covered: `Process(IEnumerable<IEnumerable<string>>)` and
`SplitSentences = false`. `process_conllu` is `Process(Document)`.

## Alternatives

All three run lemma, depparse and constituency exactly as now. Only what Load accepts (A, B) or which loaded processors a
call runs (C) changes.

### A. `PipelineOptions.Pretagged` (recommended)

One bool, Stanza's global `pretagged=True`: lemma, depparse and constituency drop their requirements at Load.

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Processors = "depparse", Pretagged = true });
var doc = nlp.Process(Conllu.Read(File.ReadAllText("tagged.conllu"))); // your UPOS/XPOS/lemmas, Stanza's heads
```

- Load: in `SelectModels`, when `Pretagged` is set, lemma, depparse and constituency need no processor. pos, ner and
  sentiment keep theirs, as in Stanza. So `Processors = "depparse"` loads depparse (plus its pretrain and charlms), not
  the 140 MB tagger or the lemmatizer.
- Tokenizer (question 2), recommended: tokenize and the package's mwt are always added. They are 0.66 and 0.5 MB, and
  `_tokenizer` stays non-null. This is like Stanza's `tokenize,depparse` pretagged pipeline, which loads mwt too.
  `Process(string)`, bulk and pretokenized then run as they do now. Without pos, depparse throws the existing
  `InvalidOperationException` for untagged words, as Stanza's does in the same pipeline. The other choice is to
  leave tokenize out unless it is listed, make `_tokenizer` nullable, and have the three text overloads throw
  `InvalidOperationException` ("this pipeline has no tokenizer: use Process(Document)"). That is closer to Stanza's
  load list, but it is more code for 1.2 MB.
- Downloads: `ModelDownloader.DownloadAsync(dir, options)` and `FilesFor` take the same options. With `Pretagged` they
  fetch only what Load reads, through `SelectModels`, as now. A `--pretagged` flag on `stanzasharp download` is
  optional and can wait.
- With `Pretagged` and pos loaded, nothing changes: pos still runs and overwrites the tags, as in Stanza.
- Why one bool and not a set (`Pretagged = "depparse"`, Stanza's `depparse_pretagged`): the flag only relaxes a
  load-time check, so any pipeline a per-processor set allows, the bool allows too, with the same output. The set would
  only turn away more pipelines. Stanza's own pretagged pipelines can't contain pos anyway (pos needs tokenize, and
  pretagged doesn't relax that), so there is no realistic mixed case. It also follows the `SplitSentences` bool.
- Trade-off: it covers only the three processors Stanza covers. Running pos, ner or sentiment on a document still
  needs the tokenizer loaded, which A always adds anyway, so in practice that already works: `Processors =
  "tokenize,ner"` plus `Process(Document)` runs NER on given tokens today.

### B. `PipelineOptions.CheckRequirements` (Stanza's `check_requirements=False`)

One bool that turns the requirement check off for every processor: any subset loads.

```csharp
using var nlp = Pipeline.Load(dir, new PipelineOptions { Processors = "depparse,constituency", CheckRequirements = false });
nlp.Process(doc);
```

- More general: pos, ner and sentiment load alone too. That gains nothing over A, since A's tokenizer is always there
  and costs 1.2 MB.
- It ports an undocumented Stanza switch rather than the documented `pretagged`, and the name says what is skipped,
  not what the user has. Every pipeline that is wrong today ("lemma without pos") would load silently and fail or
  degrade only at Process time.
- Same Load and tokenizer questions as A.

### C. `Process(Document, string processors)` (Stanza's `process(doc, processors=...)`)

No load option: a full pipeline runs only the listed processors on a document.

```csharp
using var nlp = Pipeline.Load(dir);                        // all eight, as now
var doc = nlp.Process(Conllu.Read(text), "depparse");      // skips pos and lemma: the given tags are kept
```

- No new Load semantics, and one pipeline serves both uses (annotate text, or parse given tags).
- It loads every model, about 950 MB for `default`, to run one of them. That is what pretagged exists to avoid.
- It needs its own rules: an unloaded processor in the list (throw?), requirements inside the list (Stanza checks none
  and adds mwt when tokenize is listed), and cancellation overloads (two more public methods).
- It could be added later on top of A if someone needs it.

**Recommendation: A with tokenize kept.** It is Stanza's documented option, and the change is one condition in
`SelectModels`, a property, its XML doc and a `changes/added/` file. The runtime contract is the one the processors
already enforce, and it matches Stanza except for constituency (question 3).

## Missing annotations (contract for A and B)

- **depparse:** throws `InvalidOperationException` if any word has neither UPOS nor XPOS, which is Stanza's
  `ValueError` condition. A missing XPOS, lemma or feats is read as `_`, byte for byte like Stanza. A missing lemma
  degrades the parse, in Stanza too, so the XML doc should say "give lemmas for Stanza-quality parses".
- **lemma:** a missing UPOS is read as `_`, like Stanza's `resolve_none`. Still to verify while implementing: that
  `Lookup(text, null)` matches Stanza's `skip_seq2seq`/`ensemble`, which see `None`. That needs a golden case.
- **constituency (question 3):** recommended is to keep the throw. Stanza's output with an empty tag is a tree with
  blank labels that `Tree.ToString`/`Conllu` would have to print as `( The)`. The precedent is "a processor the
  package lacks": Stanza skips it silently, and StanzaSharp throws. Porting Stanza instead means finding the tag id
  that `None` maps to in `parse_tagged_words`, and printing empty labels.
- **Missing words/sentences:** a `Document` with sentences but no words is skipped, as now.

## Golden data and tests

- `make_golden.py --pretagged-only` writes `tests/golden/pretagged/`. Its input is
  `tests/golden/process_document/input.conllu` (corpus.txt tokenized, every annotation wrong), so the output depends on
  the given tags. That is unlike the gold tags, where pretagged depparse just reproduces `pipeline.conllu`. The output:
  - `depparse.conllu`: `stanza.Pipeline(processors='depparse', depparse_pretagged=True)` on the input as it is;
  - `lemma.conllu`, `constituency.conllu`: the same for lemma and constituency;
  - `all.conllu`: `processors='lemma,depparse,constituency', pretagged=True` (lemma runs first, so depparse reads the
    new lemmas);
  - `depparse_nolemma.conllu` / `depparse_noxpos.conllu` / `lemma_noupos.conllu`: the input with lemmas, XPOS or UPOS
    removed, for the "read as `_`" rows above;
  - `.fast.conllu` twins for `default_fast` (no constituency).
  - English only at first. German rows can follow the usual `[German]` pattern, which also covers constituency on UPOS
    (`retag_method: upos`).
- `PretaggedTests`, theories over `bool managed`:
  - each file byte for byte through `Pipeline.Load(dir, new() { Processors = ..., Pretagged = true }).Process(Conllu.Read(input))`;
  - with gold tags, `Processors = "depparse"` reproduces `pipeline.conllu`'s heads (no new file);
  - `Processors = "depparse"` without `Pretagged` still throws `ArgumentException`;
  - untagged words throw `InvalidOperationException` (depparse, constituency);
  - `ModelDownloader.FilesFor(options)` with `Pretagged` lists depparse's files plus tokenize and mwt, and not pos's;
  - `Process(string)` on a pretagged depparse-only pipeline throws the depparse `InvalidOperationException`.
- Docs: CLAUDE.md's "not ported" sentence, docs/verification.md (new golden folder and counts), README/PACKAGE.md one
  example. The CI cache needs nothing new: the default package's models are already there.
