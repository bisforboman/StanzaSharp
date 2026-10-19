# Verification

StanzaSharp claims that it gives the same output as Python Stanza 1.15.0 with the same English and German models. This page
shows how that is checked, on which platforms, and how you can check it on your own text.

## Same output as Python Stanza

The reference output in [`tests/golden/`](../tests/golden) was made by Python Stanza 1.15.0 (torch 2.14.1, CPU;
versions pinned in `tools/requirements.txt`) with `tools/make_golden.py`, from the same model files that
`stanzasharp download` fetches. The downloader checks each file against the MD5 in Stanza 1.15.0's `resources.json`.

The tests compare StanzaSharp's output with it **byte for byte**, with no tolerance:

| What | Input | Compared |
|---|---|---|
| Full pipeline, `default` package (all eight processors) | `corpus.txt` and 12 hand-written `validation*.txt` texts: 13 files, 845 sentences, 8,401 words | the whole CoNLL-U: tokens, sentences, offsets, multi-word tokens, lemmas, UPOS/XPOS/features, heads and relations, constituency trees, sentiment, NER tags |
| Full pipeline, `default_fast` package (seven processors) | the same 13 files | the whole CoNLL-U |
| Each processor on its own | the same 13 files; sentiment also on `sentiment/reviews.txt` (14 files, 886 sentences) | lemmas (868 words through the seq2seq model), dependency parses, entities (706), sentiment labels |
| Input modes, both packages | bulk processing (Stanza's `bulk_process`), pretokenized text, one sentence per paragraph (`tokenize_no_ssplit`), very long tokens | the CoNLL-U |
| JSON output (`Json.Write`), both packages and `tokenize,mwt,pos` | a short text with every JSON field (multi-word tokens with and without spacing, leading spaces, tabs, CRLF, NBSP, a control character, quotes, backslashes, entities): 4 sentences, 35 words | Stanza's `str(doc)`; `Json.Read` round-trips it |
| Annotating a document (`Process(Document)`, Stanza's `process_conllu`), both packages | `corpus.txt` after tokenize and mwt, read from CoNLL-U with every annotation wrong (tags, lemmas, features, heads, relations, NER, a tree): 10 sentences | the CoNLL-U |
| Pretagged documents (`Pretagged`, Stanza's `pretagged=True`): lemma, depparse and constituency loaded without the tagger, both packages | `default_fast`'s tagged `corpus.txt` and `validation.txt` (114 sentences), the wrong-annotation document above, and copies without lemmas, XPOS or UPOS: 9 cases | the CoNLL-U |
| Stanza's options `ner_predict_tagset`, `lemma_dict_only`, `mwt_dict_only` and the `{processor}_batch_size` ones (`NerTagSet`, `LemmaDictionaryOnly`, `MwtDictionaryOnly`, `PosBatchSize`, ...), English and German | `corpus.txt` and some validation texts per option (the WorldWide and CoNLL tag sets, dictionary-only lemmas also with `lemma=combined_charlm`, dictionary-only multi-word tokens, `sentiment_batch_size=20` and every batch size small on `default`, `default_fast` and German `default`, and `{processor}_model_path`: a nocharlm tagger and NER model in the default package, a constituency parser added to `default_fast`, and `lemma_beam_size` on English, `lemma=combined_charlm` and German): 17 cases (`OptionTests`, `tests/golden/options/`); `TokenizeBatchSize` (no Stanza twin) against the default tokens | the CoNLL-U |
| The language identifier (`LanguageIdentifier`, Stanza's `langid`) | 47 texts in 36 languages and many scripts (whitespace, one character, digits, mixed scripts, emoji, a long text), alone and in one call, unrestricted and with the `en,de` subset (`LanguageIdentifierTests`, `tests/golden/langid/`) | every language exactly; the summed scores within 1e-4 of Stanza's float32 or float64 sums (relative above \|100\|) |
| The multilingual pipeline (`MultilingualPipeline`, Stanza's `MultilingualPipeline(restrict=True)`), both backends | 9 English, German and French/Dutch/mixed texts, each language's texts in one bulk call through every processor of its default package (`MultilingualPipelineTests`, `tests/golden/langid/multilingual.json`) | each document's language and CoNLL-U |
| Tokenizer edge cases | a paragraph over 1,000 characters, more paragraphs than one batch | tokens, sentences and offsets |
| Parts | MWT expansions of a word list (dictionary and classifier); the dependency parser's tree decoding on tie-heavy score matrices | the expansions and trees |
| Model loading | the 14 `.pt` checkpoints of the English `default` and `default_fast` packages, read by StanzaSharp's own unpickler | every tensor byte-identical to torch's |

The validation texts are news, academic writing, instructions, social media, dialogue and poetry, technical text,
multilingual names and quotes, contractions, long sentences, unusual whitespace (CRLF, tabs) and characters outside
the BMP (emoji). One known difference is by design: after a character outside the BMP, `start_char`/`end_char` count
UTF-16 code units in StanzaSharp and code points in Stanza, so `validation_nonbmp` is compared without offsets.

**German.** [`tests/golden/de/`](../tests/golden/de) holds the same kinds of reference output for 13 hand-written
German texts (181 sentences, 2,227 words: news, dialogue, instructions, social media, poetry, umlauts and ß,
compounds, every quote style, contractions, capitals, CRLF whitespace, characters outside the BMP, a paragraph over
1,000 characters), made with `make_golden.py --lang de`, and every row of the table above is checked on it the same
way (JSON on the same kind of short text, 4 sentences, 35 words; `Process(Document)` on its `corpus.txt`, 12 sentences): both packages and the treebank packages `combined` (whose models are the default package's), `gsd` and `hdt`
(their own folders, with their MWT word lists), each processor
(171 words through the lemmatizer's seq2seq model, 58 entities, 200 sentiment sentences), the input modes, the
tokenizer edge cases, and the MWT expansions of a German word list with and without the dictionary (German MWT is a
seq2seq model; 68 multi-word tokens in the texts). German texts with characters outside the BMP
(`validation_nonbmp`, `validation_social`) are compared without offsets, as in English. The sentiment golden also
classifies all 200 sentences as one document, a single batch, which Stanza labels almost all positive: a stress test
of the batch padding.

**English treebank packages.** Stanza's English treebank packages (tokenize, mwt, pos, lemma, depparse) run on the same
English texts (`corpus.txt` and every `validation*.txt`), each against its own folder of `tests/golden/<package>/`
(`make_golden.py --treebank-only`): `ewt` (858 sentences, 8,393 words), `gum` (819, 8,395), `lines` (767, 8,111),
`partut` (942, 8,454), `atis` (472, 6,587), `childes` (1,231, 7,905), `eslspok` (855, 7,131; no MWT, and Stanza's
identity lemmatizer) and the biomedical `craft` (833, 8,237), `genia` (830, 8,075) and `mimic` (829, 8,296; no MWT,
their own word vectors), plus `combined`, whose models are the default package's, against `depparse/`. The packages with
MWT also have their MWT word list (`mwt.json`, with and without the dictionary). They exercise what the default package
doesn't: a tokenizer without MWT heads, composite XPOS tags (`lines`, `partut`: a scorer and an embedding per part),
a caseless lemmatizer (`atis`), an MWT seq2seq model (`partut`), which Stanza decodes with a beam search of 5, a
lemmatizer without a copy gate, and the parser's repair of heads with two subjects or objects (the biomedical ones).

**Fuzzing.** `tools/text_fuzz.py` generates seeded adversarial texts (control and zero-width characters, every
newline and space, emoji sequences, bidi marks, combining and decomposed forms, scripts outside the BMP, URLs and
e-mails, tokens around `max_seqlen`, paragraphs past the 1,000-unit window, punctuation runs, contractions) and
compares Stanza's CoNLL-U with StanzaSharp's on each, byte for byte, offsets included. Its findings, shrunk to small
texts, are kept in `tests/golden/fuzz.json` (and `de/fuzz.json`) and checked for both packages:
[Differential text fuzzing](text-fuzzing.md) has the method, the runs and the results.

**One model per processor.** Each model `PipelineOptions.Models` can choose beyond the packages' runs against its own
folder of `tests/golden/models/<processor>=<model>/` (`make_golden.py --models-only`; Stanza with the same
processors dict). The texts are `corpus.txt`, `validation.txt` and a hand-written clinical note (`models/clinical.txt`,
31 sentences). The models:

- English constituency `ptb3_charlm` (tokenize, mwt, pos, constituency);
- English ner `ontonotes_charlm`, `ontonotes_nocharlm` and `combined_charlm` (tokenize, mwt, ner);
- English ner `i2b2` on the `mimic` package (tokenize, pos, lemma, depparse, ner; also with the charlm cache off),
  and on the default package (all eight processors), where the pipeline loads two pairs of character language models
  (1billion, and mimic for ner): also with the cache off, and one-sentence calls against the uncached pipeline;
- English ner `radiology` on the `mimic` package (its charlm reads lowercased characters, so it shares nothing with the
  tagger) on a hand-written radiology report, `conll03_charlm` (GloVe vectors), and the eight biomedical models
  (`anatem`, `bc4chemd`, `bc5cdr`, `bionlp13cg`, `jnlpba`, `linnaeus`, `ncbi_disease`, `s800`: cased word vectors,
  PubMed charlms) with the `craft` package's tokenizer on a hand-written abstract (`models/biomedical.txt`), and
  `bc5cdr` on the default package (two charlm pairs);
- German ner `conll03` (on German's `corpus.txt` and `validation.txt`).
- the treebanks' taggers and parsers without character language models, on their package's tokenizer, mwt and lemmatizer
  (tokenize to depparse, `corpus.txt` only): English pos and depparse `ewt_nocharlm`, `gum_nocharlm`, `lines_nocharlm`,
  `partut_nocharlm`, `atis_nocharlm`, `childes_nocharlm`, `eslspok_nocharlm`, depparse `mimic_nocharlm` (with mimic's
  tagger), and German pos and depparse `gsd_nocharlm` and `hdt_nocharlm`;
- the treebanks' tokenizers and lemmatizers with character language models (per-character charlm states), on their
  package (tokenize to lemma, `corpus.txt`; `combined_charlm`'s tokenizer also on `tokenize_stress.txt`, whose long
  paragraph runs in windows): English tokenize `combined`, `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`, `eslspok`
  and lemma `ewt`, `gum`, `lines`, `partut`, `atis`, `childes` (each `_charlm`), German tokenize and lemma `combined`,
  `gsd`, `hdt`. Each lemmatizer's seq2seq model also runs alone on every word of `corpus.txt` and the lemma word list
  (`seq2seq.json`: Stanza's decoded strings and edit classes), on both backends and, for ewt's, on every SIMD path;
- English lemma `combined_charlm` (default package, tokenize to lemma) on `corpus.txt`, `validation.txt` and
  `validation_contractions.txt`, where its contextual classifiers decide 33 words (`'s` be/have, `her` she/her; 4 of
  them differ from the dictionary and seq2seq lemma): the lemmas byte for byte, and each decision's label and logits
  within 1e-4 of Stanza's (`contextual.json`).

**Transformer input (work in progress: `default_accurate`, [transformer-plan.md](transformer-plan.md)).** The
tokenizer of `google/electra-large-discriminator`, the transformer of Stanza's `default_accurate` package, gives
transformers' ids and word ids, and Stanza's blank-word fix and word positions, on 1,086 golden cases
(`transformer/tokenizer.json`, `make_golden.py --transformer-only`): every golden sentence's words, its
`simplify_punct`ed words (what the tagger and parser send) and its token texts (what ner and sentiment send), and a list
of edge cases (accents composed and decomposed, soft hyphens and zero-width characters, controls, emoji, CJK, Hangul,
words of 100 and 101 characters, the special tokens' texts). The encoder, on both backends, gives the word vectors
Stanza's `extract_bert_embeddings` gives (`transformer/hidden.safetensors`): the first three golden sentences within
2e-5 (managed) and 5e-6 (TorchSharp) of Stanza's float32 or float64 values, under the 1e-4 tolerance, with the legacy
layer mix and with the dependency parser's learned one; a 180-word sentence over the model's 512 positions, in Stanza's
overlapping windows, within 4e-6; and three short sentences on which ELECTRA is badly conditioned in float32 (Stanza's
own float32 is up to 7.6e-3 from float64 there), listed in the golden data with a bound of 1e-2, within 1.5e-3. On the
managed backend a sentence gives the same bits alone, in a batch and on any thread count, and every SIMD path meets the
tolerance. `roberta-base` (#172, for ner `conll03_electra-large`), a byte-level BPE tokenizer and a RoBERTa encoder, does
the same: transformers' ids on 1,093 cases (`transformer/tokenizer_roberta-base.json`: the golden words plus the BPE's
edges, such as prefix spaces, contractions, whitespace kinds, `<mask>` and non-BMP characters), and Stanza's word vectors on
the same sentences and the long one (`transformer/hidden_roberta-base.safetensors`) within 1.2e-5 on both backends.

**Transformer models.** `default_accurate`'s tagger, parser and NER model (pos and depparse `combined_electra-large`, ner
`ontonotes-ww-multi_electra-large`, chosen with `PipelineOptions.Models`) reproduce Stanza's output byte for byte on both
backends: `models/pos+depparse+ner=electra-large/` (`make_golden.py --models-only`), corpus.txt, the validation,
contractions, social and tech texts and `models/transformer_long.txt` (a 633-word sentence of 706 pieces, over the
transformer's 512 positions, so in Stanza's windows), 413 sentences with tags, lemmas, heads, relations and entities
(`TransformerModelTests`). The pipeline runs the transformer once per sentence for all three (Stanza once per processor):
the parser reads the tagger's states, ner those of sentences whose tokens are the words (`TransformerCacheTests`).
The constituency parser `ptb3-revised_electra-large` (frozen, its own `bert_layer_mix`, [CLS] and [SEP] next to the
learned sentence boundary vectors) and the sentiment classifier `sstplus_electra-large` (LoRA rank 64 on the
transformer, applied at run time) match too: `models/constituency=ptb3-revised_electra-large/` (274 sentences, trees),
`models/sentiment=sstplus_electra-large/` (315 sentences, with sentiment/reviews.txt, labels) and `models/electra-large/`
(all five transformer models in one pipeline, 274 sentences). With adapters a sentence's states keep the same bits alone,
in a batch and on any thread count on every SIMD path (`TransformerKernelTests`). The `default_accurate` package itself
(its `combined_charlm` tokenizer and lemmatizer with the five transformer models) matches on corpus.txt, the validation,
contractions, social and tech texts and the long sentence: `models/default_accurate/`, 419 sentences.
The treebanks' transformer taggers and parsers (pos and depparse `ewt`, `gum`, `lines`, `partut`, `atis`, `childes`,
`eslspok` + `_electra-large`, each on its package, corpus.txt) and ner `ontonotes_electra-large` (corpus.txt and
validation.txt, 131 entities) match too: `models/pos+depparse=<treebank>_electra-large/`, `models/ner=ontonotes_electra-large/`.
So does the LoRA-adapted parser `ptb3_electra-large`, the one with TOP_DOWN transitions (its trees differ from the IN_ORDER
parser's in 45 of validation.txt's 104 sentences): `models/constituency=ptb3_electra-large/`, 274 sentences.
On the managed backend a short call runs the frozen transformer and the LoRA adapters' nets in one sweep over the weights
([round 48](performance/round-48-joint-encoder-pass.md)); every row above runs that path, and
`TransformerKernelTests.JointPass_GivesEachGroupItsOwnNetsBits` checks that each sentence keeps the bits its own net gives
it alone (every native path, 1 and 4 threads, either order), with `TransformerCacheTests`' fake-net `Prefill` tests in
build-test.

German's `default_accurate` (`german-nlp-group/electra-base-german-uncased`, a base ELECTRA) matches the same way, in
`tests/golden/de/models/`: pos and depparse `combined_german-nlp-electra` sharing one frozen pass (79 sentences, with
the umlauts, social and contractions texts and `de/models/transformer_long.txt`, one sentence of 558 pieces), the
constituency parser `spmrl_german-nlp-electra` and the sentiment classifier `sb10k_german-nlp-electra`, each with its own
LoRA adapters (46 and 47 sentences), and the package itself (80 sentences): three passes per sentence.

They exercise what the packages don't: a second pretrain next to conll17 (fastText crawl, 300-dimensional), a tag
vocabulary with three tag sets, and the old NER checkpoint format with a fine-tuned delta embedding (`i2b2`).

Every test above runs on both backends (the JSON comparison on the managed one: the writer does not depend on the backend): the managed backend (the default) and the TorchSharp backend
(`StanzaSharp.Cuda`). The managed backend's kernels are also tested on each SIMD path a CPU can take (Vector512 with
AVX-512, Vector256 with AVX2, Vector128, Scalar), so a machine without AVX2 or an Arm64 machine runs code that was
checked too. Vector512 runs only on a CPU with AVX-512, which the x64 runners get only sometimes (an AVX-512 EPYC or
Xeon instead of an EPYC 7763): the golden and cross-os jobs then run it as the detected path, and the Benchmark
workflow's `avx512` jobs test and time it against Vector256 when they get one. Vector512 gives Vector256's results bit
for bit (`Vector512_GivesVector256sBits`), so which CPU a run gets can't change a test's outcome.

## Scores and tolerances

The discrete output (every token, tag, lemma, head, tree and label) must match exactly. The scores behind it are
float32 numbers summed in a different order than in PyTorch (StanzaSharp's SIMD kernels, or libtorch's), so their last
bits differ. The tests compare the scores Stanza saved (for the first three corpus sentences; for sentiment, every
sentence) with a tolerance. Largest differences from Python Stanza measured on the managed backend:

| Scores | Largest difference, English | German | Tolerance |
|---|---:|---:|---:|
| tokenizer log-probabilities | 1.1e-5 | 1.1e-5 | 1e-4 |
| UPOS logits | 2.9e-5 (6.1e-5 from float32) | 6.3e-5 (1.5e-4 from float32) | 1e-4, relative above \|100\| (of Stanza's float32 or float64 logits) |
| dependency arc / label log-probabilities | 2.3e-5 / 3.4e-5 | 1.5e-5 / 2.3e-5 | 1e-4 |
| NER emissions | 1.1e-5 | 1.3e-5 | 1e-4 |
| sentiment logits | 8.2e-5 | 4.6e-6 | 1e-4 (of Stanza's float32 or float64 logits) |
| constituency transition scores | 3.4e-5 (at most 4.6e-5 on any SIMD path) | 3.8e-5 | 1e-3 |

- **Large values.** A 1e-4 tolerance grows with the reference value above |100|: 1e-4 × max(1, |expected| / 100)
  (owner's decision, 2026-10-14). Float32 can't hold 1e-4 above |x| ≈ 300: from 256 up one float32 step is 3.05e-5,
  so 1e-4 would allow only about three roundings. Only UPOS logits go above |100| in the golden data (German up to
  |380|, English |139|), so nothing else changes; the 1e-3 tolerances stay absolute.
- **UPOS logits** are compared with Stanza's float64 logits as well, both languages and both packages (owner's
  decision, 2026-10-13, as for sentiment): a logit passes within 1e-4 of either. The German tagger's logits reach |140|,
  and Stanza's own float32 logits are up to 1.01e-4 from float64; the managed backend is 1.5e-4 from float32 but within
  6.3e-5 of the nearer value, and TorchSharp, one sentence at a time, 1.22e-4 from float32. Summing `upos_hid` and
  `upos_clf` in double changed nothing. Every tag is identical.
- **Sentiment** is compared with Stanza's float64 logits as well: the classifier is ill-conditioned, and on one
  sentence Stanza's own float32 logits are 1.49e-4 from float64. With the tagger's character-model outputs reused
  (`CharlmCache`, on by default), logits move by up to about 1e-4 more, so those tests use 1e-3; the labels stay
  identical.
- **The lemmatizer**'s log-probabilities are reported, not asserted: no backend comes within 1e-4 of float64 there,
  Stanza's float32 included. The lemmas, the decoding and the edit decisions are compared exactly, and the smallest
  gap between the best and second-best character is 5.6e-3, far above the drift.
- **Near-ties.** A small drift could still flip a decision whose two best options are almost equal. The constituency
  parser's smallest decision margin on the golden sentences is 3.6e-4 over 22,569 steps, against 4.0e-5 between the
  two backends; `ConstituencyTests.NearTies_StayClearOfBackendDrift` fails if a decision differs or the margin drops
  under 3× the drift. German: smallest margin 1.2e-2 over 4,575 steps, drift 7.6e-5. German MWT's seq2seq: smallest
  margin 0.14 between the best and second-best character, drift 6.0e-5 (`MwtTests.Seq2Seq_German_ManagedMatchesTorchSharp`,
  which also fails under 3×). On UD English EWT (254,589 words, 684,407 parser decisions; not in the repository) no tree and no
  lemma differed between Python Stanza, the TorchSharp backend and the managed backend. On UD German GSD and HDT test
  (652,046 words, both packages, every processor) no output differed among the three either; the smallest ratio of a
  parser margin to the backends' margin difference is 11 (EWT: 15) ([german-divergence.md](german-divergence.md)).

Per-processor details: [backends.md](backends.md#phase-2-progress). On a GPU the output is identical with TF32 turned
off (`disableTf32: true`); with TF32 on, 3 of 845 golden sentences differed on an RTX 3080 ([gpu.md](gpu.md)).

## Where it is checked

[CI](../.github/workflows/ci.yml) runs on free GitHub-hosted runners. All eight jobs are required checks on `main`.
Test counts are passed xUnit test cases (each case of a theory counts, summed over a job's shards) in `main`'s run
[39135862100](https://github.com/bisforboman/StanzaSharp/actions/runs/39135862100) of 2026-10-18:

| Job | Runner | What it checks |
|---|---|---|
| `build-test` | ubuntu-26.04 (x64) | builds with warnings as errors, checks the code style, runs the 317 tests that need no models |
| `golden` (4 shards) | ubuntu-26.04 (x64) | the whole suite on both backends against downloaded models, original and converted, English and German, every package, the transformer models included (872 tests); then packs the packages and builds fresh apps from them: `StanzaSharp` alone must reproduce the golden CoNLL-U with nothing of TorchSharp loaded, and `StanzaSharp.Cuda` with `TorchSharp-cpu` must too; then installs the `stanzasharp` tool and downloads models with it; then publishes the [managed check](../tests/StanzaSharp.ManagedCheck/Program.cs) with Native AOT (`x86-64-v3`) and trimmed, with no trim/AOT warning and no torch file allowed, and both binaries must reproduce the golden CoNLL-U ([tools/verify-aot.sh](../tools/verify-aot.sh)) |
| `cross-os (windows-2025)` | Windows x64 | the whole suite on both backends, reading Stanza's `.pt` files directly (527 tests; the English default packages), both package checks, and the Native AOT and trimming check |
| `cross-os (macos-26)` | macOS, Apple Silicon | the same (501 tests: Arm64 has no Vector256 path) |
| `cross-os (windows-11-arm)` | Windows on Arm64 | the same suite (501 tests) and only the managed package check (`TorchSharp-cpu` has no Arm64 libtorch) |
| `linux-arm64` | ubuntu-26.04-arm | every managed-backend test (287; there is no libtorch for Linux Arm64) and the managed package check |
| `docker` | ubuntu-26.04 | the [Docker sample](../samples/docker/Dockerfile) on `runtime:10.0` and `runtime:10.0-noble-chiseled`: no libtorch in the image, and the container turns `corpus.txt` into the golden CoNLL-U byte for byte |
| `alpine` | ubuntu-26.04 | the same on `runtime:10.0-alpine` (musl) |

Every job that runs the model tests fails if any of them was skipped. The only exceptions are the tests that run
`stanzasharp compare` and `benchmark` against Python Stanza (`Run_PythonStanza_*`, English and German), which only
golden shard 2 installs and runs, and the German tests, the English treebank packages' and the chosen models' tests outside `golden`: only
the golden shards have those models (Stanza's `.pt` files, cached), so `cross-os` and `linux-arm64` run the English
default packages only. The transformer tests (`Transformer*`) run in golden shard 4, their own, which downloads the transformer's
files from Hugging Face and the transformer models from Stanza's repository on every run (no cache: the model caches are
near GitHub's 10 GB limit). A pull request runs that shard only when it changes what those tests read (the shared
kernels, the TorchSharp backend, the downloader, CI, the transformer's golden data or tests); every push to main runs it.

The [nightly workflow](../.github/workflows/nightly.yml) runs on main every night: golden shard 4, the checkpoint fuzz
over every Stanza checkpoint with ten times CI's iterations and a new seed, [text fuzzing](text-fuzzing.md) against
Python Stanza (1,000 texts through tokenize and mwt, 60 through each default package), and the Benchmark workflow's
quick run. A failed night comments on the open issue "Nightly run failed".

On x64 runners the test steps set `ATEN_CPU_CAPABILITY=default` and `MKL_CBWR=COMPATIBLE,STRICT`, so libtorch picks
the same kernels on every CPU the runner draws (Intel or AMD) and the TorchSharp backend gives the same bits on all of
them ([round 26](performance/round-26-torchsharp-intel-drift.md)). Users keep libtorch's defaults.

A pull request runs a subset on the other operating systems; every push to `main` and every release runs everything
above. Releases are built from a commit that passed all of it.

## Check it yourself

`stanzasharp compare` runs Python Stanza 1.15.0 and StanzaSharp on your own text with the same model files and says
whether the CoNLL-U is identical, or shows the first difference:

```
dotnet tool install -g StanzaSharp.Tool
pip install stanza==1.15.0
stanzasharp download models/stanza/en
stanzasharp compare my-text.txt          # --package default_fast, --processors LIST
stanzasharp download --lang de           # German, into models/stanza/de
stanzasharp compare my-text.txt --lang de
```

It checks the model files' MD5s on both sides and converts the offsets after non-BMP characters before comparing.
Exit code 0 means identical. More: [the tool's README](../src/StanzaSharp.Tool/README.md#check-it-yourself).

Before 1.9.0 (2026-10-16, main d5deefe, managed backend), `compare` ran on five short texts written for it, none in
the golden data ([`docs/spot-check/`](spot-check)). Every one gave identical CoNLL-U, byte for byte:

| Text | Package | Sentences, words | What it tries |
|---|---|---:|---|
| [`ewt.txt`](spot-check/ewt.txt) | `ewt` | 7, 109 | everyday English: chores, brunch, a flight at 6:15 |
| [`partut.txt`](spot-check/partut.txt) | `partut` | 7, 120 | contractions, `'90s`, `'98`, 15%, 4:30 p.m., quotes: the seq2seq MWT with beam search |
| [`mimic.txt`](spot-check/mimic.txt) | `mimic` | 8, 130 | clinical abbreviations (`h/o`, `s/p`, `SpO2`, `N/V`, doses): its own charlms and word vectors |
| [`default.txt`](spot-check/default.txt) | `default` | 5, 80 | news with money, a hyphenated compound and a quote, all eight processors |
| [`de.txt`](spot-check/de.txt) | German `default` | 5, 82 | German news with „…“ quotes and umlauts, all eight processors |

## Speed on the free runners

`stanzasharp benchmark` times StanzaSharp and Python Stanza on the same text with the same models, on your machine. The
[Benchmark workflow](../.github/workflows/benchmark.yml) (manual, and quick every night) runs it on every free GitHub-hosted runner. Run
[38735518300](https://github.com/bisforboman/StanzaSharp/actions/runs/38735518300) (2026-10-15, StanzaSharp
1.7.0-12-g873f8c9, English `default`, all eight processors on 2,009 words, each side's default threads) gave identical
CoNLL-U on every runner, and:

| Runner | StanzaSharp words/s | Python Stanza words/s | Speed-up | Per call (one sentence), median | Peak memory |
|---|---:|---:|---:|---:|---:|
| ubuntu-26.04 (AMD EPYC 7763, 2 cores) | 274 | 68 | 4.1× | 93 / 276 ms | 1,031 / 3,549 MB |
| ubuntu-26.04-arm (4 cores) | 257 | 79 | 3.2× | 89 / 661 ms | 1,160 / 3,704 MB |
| windows-2025 (AMD EPYC 7763, 2 cores) | 273 | 59 | 4.7× | 122 / 352 ms | 981 / 3,606 MB |
| windows-11-arm (4 cores) | 257 | 29 | 8.9× | 87 / 553 ms | 925 / 3,509 MB |
| macos-26 (Apple M1, 3 vCPUs) | 214 | 53 | 4.0× | 150 / 520 ms | 806 / 2,785 MB |

German `default` is 2.4–7.4× faster than Python Stanza on the same runners, `default_fast` 1.8–4.4×, with identical
CoNLL-U ([round 33](performance/round-33-german-benchmark.md)). The runners are shared virtual machines whose CPU model
can change between runs (ubuntu-26.04 was an Intel Xeon in an earlier run), so compare numbers within one run. More, including how to run it yourself:
[performance.md](performance.md#run-it-yourself).
