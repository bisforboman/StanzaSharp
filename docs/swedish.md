# Swedish

StanzaSharp runs Stanza's Swedish models in plain .NET: tokenization, part-of-speech tags and morphological features,
lemmas, dependency parsing and named entities, with no Python and no native libraries. On Swedish text the output is the
same as Python Stanza 1.15.0's, byte for byte (`tests/golden/languages/sv`, checked by CI's nightly `languages` job).

**Before you build a product on it, read [Licences](#licences):** the Swedish tagger, parser and NER read word vectors
whose licence is CC BY-NC-SA 4.0 (non-commercial).

## Quick start

```
dotnet add package StanzaSharp
dotnet tool install -g StanzaSharp.Tool
stanzasharp download models/stanza/sv --lang sv
```

```csharp
using StanzaSharp;

var swedish = new PipelineOptions { Language = "sv" };
using var nlp = Pipeline.Load("models/stanza/sv", swedish);
var doc = nlp.Process("Karin Larsson flyttade till Malmö förra året och började arbeta på Skanska.");

foreach (var word in doc.Sentences[0].Words)
    Console.WriteLine($"{word.Id}\t{word.Text}\t{word.Lemma}\t{word.Upos}\t{word.Head}\t{word.Deprel}");
foreach (var entity in doc.Entities)
    Console.WriteLine($"{entity.Text}/{entity.Type}");
```

```
1	Karin	Karin	PROPN	3	nsubj
2	Larsson	Larsson	PROPN	1	flat:name
3	flyttade	flytta	VERB	0	root
4	till	till	ADP	5	case
5	Malmö	Malmö	PROPN	3	obl
6	förra	förra	ADJ	7	amod
7	året	år	NOUN	3	obl
8	och	och	CCONJ	9	cc
9	började	börja	VERB	3	conj
10	arbeta	arbeta	VERB	9	xcomp
11	på	på	ADP	12	case
12	Skanska	Skanska	PROPN	10	obl
13	.	.	PUNCT	3	punct
Karin Larsson/PER
Malmö/LOC
förra året/TME
Skanska/ORG
```

`ModelDownloader.DownloadAsync("models/stanza/sv", swedish)` downloads the same files from code. `word.Feats` holds the
morphological features (`Case=Nom|Definite=Def|Gender=Neut|Number=Sing` for `året`), `word.Xpos` the SUC tag (`NN|NEU|SIN|DEF|NOM`), and
`sentence.Dependencies` the parse as (head, relation, dependent). The rest of the API is the same as for every language
([README](../README.md#usage)).

## Models

StanzaSharp runs Stanza's `default` Swedish package. Swedish has no multi-word tokens, sentiment or constituency models
in Stanza.

| Processor | Model | Trained on | Reads |
|---|---|---|---|
| tokenize | `talbanken_nocharlm` | UD Swedish-Talbanken | |
| pos | `talbanken_charlm` | UD Swedish-Talbanken | `pretrain/conll17`, `forward_charlm/conll17`, `backward_charlm/conll17` |
| lemma | `talbanken_nocharlm` | UD Swedish-Talbanken | (needs pos) |
| depparse | `talbanken_charlm` | UD Swedish-Talbanken | `pretrain/conll17`, both charlms |
| ner | `suc3shuffle` | SUC 3.0 (the shuffled release) | `pretrain/conll17`, both charlms |

The download is 386 MB: the parser 135 MB, the word vectors (`pretrain/conll17`) 107 MB, NER 66 MB, the tagger 34 MB and
the two character language models 21 MB each. `--processors tokenize,pos,lemma` (or `PipelineOptions.Processors`)
downloads and loads only those models (186 MB).

NER has the eight SUC types: `PER`, `LOC`, `ORG`, `TME` (time expressions), `MSR` (measures), `WRK` (works),
`EVN` (events) and `OBJ` (objects).

Stanza also has Swedish models trained on UD Swedish-LinES, models on the KB-BERT transformer and an NER model on the
officially licensed SUC 3.0 (`suc3licensed`). StanzaSharp doesn't run these yet. `PipelineOptions.Models` and other
packages throw for Swedish.

For masking personal data in Swedish documents (personnummer, organisationsnummer, addresses, bank and plate numbers
next to NER), see the `StanzaSharp.TasmanianDevil` package and [samples/StanzaSharp.Masking](../samples/StanzaSharp.Masking/README.md).

## Licences

StanzaSharp's code is Apache 2.0. The models are not part of it: you download them from Stanford's Hugging Face
repository (`stanfordnlp/stanza-sv`), whose model card says `license: apache-2.0`. The data behind the models carries
its own terms:

| Data | Licence | Used by |
|---|---|---|
| CoNLL 2017 shared task word embeddings and raw texts ([LINDAT 11234/1-1989](http://hdl.handle.net/11234/1-1989)) | **CC BY-NC-SA 4.0** | `pretrain/conll17` (the word vectors) and the `conll17` character language models; through them pos, depparse and ner |
| [UD Swedish-Talbanken](https://github.com/UniversalDependencies/UD_Swedish-Talbanken) | CC BY-SA 4.0 | tokenize, pos, lemma, depparse |
| [SUC 3.0](https://spraakbanken.gu.se/en/resources/suc3), the shuffled release | CC BY 4.0 on the SUC 3.0 page; CC BY-SA 4.0 on the SUCX 3.0 page for the same file | ner |

What this means in practice:

- Every Swedish pipeline beyond tokenization loads `pretrain/conll17`. The tagger reads it, and the lemmatizer, parser
  and NER all need the tagger or the vectors directly. There is no Swedish package without it.
- `pretrain/conll17` is built from the CoNLL 2017 word vectors, so it falls under CC BY-NC-SA's
  non-commercial and share-alike terms more directly than a model trained on such data does.
- If you plan commercial use, have the licences checked for your case before you ship. This page states what the
  sources say; it is not legal advice.

Other languages differ: [docs/languages.md](languages.md) lists each language's models, and Stanza's resources say
which word vectors each one reads.

## StanzaSharp and Catalyst

[Catalyst](https://github.com/curiosity-ai/catalyst) is the other pure-C# NLP library with Swedish models
(`Catalyst.Models.Swedish`, MIT). The two make different trade-offs: Catalyst is a fast tagger with a lemma lookup,
StanzaSharp runs Stanza's neural models.

| | StanzaSharp 1.16.0 | Catalyst 26.10.6514 (Swedish models 26.9.5316) |
|---|---|---|
| Tokenization, sentences | yes | yes |
| UPOS tags | yes, plus SUC tags and features | yes |
| Lemmas | neural, in context | dictionary lookup, without context |
| Dependency parsing | yes | no |
| Named entities | yes (8 SUC types) | no Swedish model |
| Matches Python Stanza | byte for byte | — |
| Package licence | Apache 2.0 | MIT |
| Model licence | see [Licences](#licences) (CC BY-NC-SA 4.0 word vectors) | MIT package; training data not stated |
| Install size (`dotnet publish`, framework-dependent) | 1.7 MB, plus 386 MB of models (186 MB for tokenize, pos, lemma) | 28 MB, models included |

Measured on the UD Swedish-Talbanken test set (r2.15: 1,219 sentences, 20,377 tokens), given to both as raw text, one
paragraph per sentence. Accuracy is against the treebank's own tokens, tags and lemmas; a token counts only where the
library found the same token. Speed is the median of five runs after a warm-up, on a 4-core Intel Xeon (2.1 GHz) cloud VM,
.NET 10, Linux, default settings (the managed backend, all cores):

| | Tokens found | UPOS | Lemma | Time | Words/s | Peak memory |
|---|---|---|---|---|---|---|
| Catalyst (tokenizer, sentences, tagger, lemmas) | 99.53% | 93.66% | 76.58% | 0.09 s | 236,000 | 140 MB |
| StanzaSharp `tokenize,pos,lemma` | 99.97% | 97.99% | 97.03% | 11.0 s | 1,860 | 760 MB |
| StanzaSharp, all five processors | | | | 21.9 s | 930 | 1.4 GB |

Two caveats about the accuracy numbers. Stanza's models are trained on Talbanken's training set and evaluated here on its
held-out test set; Catalyst doesn't say which data its Swedish tagger was trained on. Some of Catalyst's lemma errors are
a different convention (`det` → `en` rather than UD's `den`), but most come from the lookup ignoring context: `har` →
`hare`, `med` → `mede`, `man` → `mana`, `andra` → `två`.

So: if you need speed on large volumes and UPOS tags are enough, Catalyst is two orders of magnitude faster. If you need
lemmas you can rely on, dependency parses, named entities or Python Stanza's exact output, StanzaSharp has them. Check
the licences either way.
