# German divergence study (UD GSD and HDT)

Do Stanza, StanzaSharp's TorchSharp backend and its managed backend ever disagree on real German text? The English
studies (docs/backends.md, "constituency" and "lemma": UD EWT) found no difference; this repeats them for German, for
every processor's output, after #129 (German MWT decodes with Stanza's beam search of 5). Measured 2026-10-15/16 on main
1.8.0-2-gec4f30f, Stanza 1.15.0 (torch 2.14.1), Ryzen 7 5800X.

## Verdict

**No output differs.** Over 652,046 words of UD German GSD and HDT, both packages, every token, sentence break, MWT
expansion, UPOS/XPOS/feats, lemma, tree, head, deprel, entity tag and sentiment label is the same in Stanza float32,
TorchSharp and managed (the CoNLL-U documents compare equal as strings). The closest decisions stay clear of the
backends' drift: the smallest ratio of a parser margin to the margin difference between two backends is 11 (EWT: 15),
and the smallest lemmatizer margin, 1.8e-3, is 18x the largest distance of any run's margins from Stanza's float64 ones.

## Data

UD German GSD (all of train/dev/test) and HDT (test only: all of HDT is 3.4M tokens; its test split is a subset of the
same heise news text as the rest). CC BY-SA 4.0; downloaded from github.com/UniversalDependencies, not in the repository.
Neither treebank has `# newdoc`, so `lemma_divergence.py text` makes a document of every 20 `# text` lines in file
order, in paragraphs of 5; each document is one Process call.

| corpus | documents | sentences (Stanza's) | words (after MWT) | MWT tokens | entity starts |
|---|---:|---:|---:|---:|---:|
| GSD train+dev+test | 780 | 15,535 | 292,731 | 5,048 | 19,662 |
| HDT test | 923 | 17,052 | 359,315 | 5,383 | 23,445 |

## Whole-pipeline outputs

`constituency_divergence.py stanza --lang de --package P --processors ...` and `StanzaSharp.Benchmark
constituency-divergence --lang de --package P --processors ... --backend torch|managed`, each document's CoNLL-U
compared field by field (`compare`).

| package (processors) | corpus | TorchSharp vs Stanza | managed vs Stanza |
|---|---|---|---|
| `default` (all eight) | GSD | 780 of 780 documents identical | 780 of 780 identical |
| `default` (all eight) | HDT test | 923 of 923 identical | 923 of 923 identical |
| `default_fast` (seven, no constituency) | GSD | 780 of 780 identical | 780 of 780 identical |
| `default_fast` (seven, no constituency) | HDT test | 923 of 923 identical | 923 of 923 identical |

Two outputs that look odd are Stanza's own and are reproduced: the seq2seq lemma of `KURZNACHRICHTEN` is
`KurmrkeißeißTEN`, and `KOnten` lemmatizes to `KOntos`.

## Constituency decisions (`default`)

Per parser step: the best and second legal transition and their scores (German's parser reads UPOS, `retag_method:
upos`). Steps with one legal transition are left out of the margins.

| corpus | decisions | < 1e-3 | < 1e-4 | smallest margin (Stanza / TorchSharp / managed) |
|---|---:|---:|---:|---|
| GSD | 558,595 | 24 | 1 | 1.93e-5 / 1.88e-5 / 2.05e-5 |
| HDT test | 671,327 | 34 | 2 | 4.28e-5 / 4.27e-5 / 4.27e-5 |

Every decision of every sentence is the same in all three runs. |Margin difference| per decision:

| pair | corpus | median | 99th pct. | 99.99th pct. | max | smallest margin / difference |
|---|---|---:|---:|---:|---:|---:|
| Stanza vs TorchSharp | GSD | 4.8e-7 | 3.8e-6 | 9.5e-6 | 1.8e-5 | 40 |
| Stanza vs managed | GSD | 3.3e-6 | 1.7e-5 | 3.6e-5 | 6.6e-5 | 16 |
| TorchSharp vs managed | GSD | 3.3e-6 | 1.7e-5 | 3.6e-5 | 6.8e-5 | 11 |
| Stanza vs TorchSharp | HDT | 4.8e-7 | 3.8e-6 | 8.6e-6 | 2.0e-5 | 296 |
| Stanza vs managed | HDT | 3.2e-6 | 1.7e-5 | 3.4e-5 | 8.9e-5 | 34 |
| TorchSharp vs managed | HDT | 3.2e-6 | 1.7e-5 | 3.4e-5 | 8.8e-5 | 33 |

The managed backend's margins differ from the others' about 7x more than TorchSharp's from Stanza's, as on EWT (median
4.4e-6 there); docs/backends.md traces that to the float32 charlm's summation order amplified by the word encoder.

The closest call, GSD (train) "Die Straße 29A verläuft über Vârfu Câmpului nach Suceava ( 39 km ) , …", step 32:
Shift over Close, margins 1.93e-5 (Stanza) / 1.88e-5 (TorchSharp) / 2.05e-5 (managed) / 2.37e-5 (Stanza float64). It
is in the float64 subset: GSD documents 520–619 (40,468 words, 76,906 decisions), parsed again by a float64 copy of Stanza's parser from its own
float64 states. There no tree differs either, and the largest |margin − float64 margin| is Stanza 5.2e-5, TorchSharp
4.9e-5, managed 4.5e-5: managed is no further from the exact value than Stanza's own float32.

## Lemmatizer decisions (`default`, GSD)

`lemma_divergence.py stanza --lang de` (with its float64 copy) and `lemma-divergence --lang de`. The German dictionary
covers GSD's training text: only 1,847 of 292,731 words go through the seq2seq model (19,032 decoder steps), 71 of
them in train, 699 in dev and 1,077 in test. No lemma differs among Stanza float32, float64, TorchSharp
and managed. Smallest top-2 margin per decoder step: Stanza 1.77e-3, TorchSharp 1.76e-3, managed 1.78e-3, float64
1.77e-3; largest |margin − float64 margin|: Stanza 6.9e-5, TorchSharp 9.7e-5, managed 5.0e-5. Smallest edit-class
margin 1.47e-2 in every run. HDT's lemmas were compared in the whole-pipeline runs only (all identical).

## Not measured

The tokenizer, tagger, depparse (MST over log-probs), NER (Viterbi) and sentiment margins, and the MWT beam's scores:
their outputs are compared above, not their margins. The golden tests check their scores against Stanza within 1e-4.

## Rerun

```
python tools/lemma_divergence.py text UD_German-GSD gsd.json                    # 780 documents
python tools/lemma_divergence.py text UD_German-HDT hdt-test.json --splits test # 923 documents
# split each into shards; per shard, 1 thread (parallel multi-thread libtorch processes crawl):
python tools/constituency_divergence.py stanza SHARD.json OUT.json --lang de --package default \
    --processors tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner --threads 1 --no-f64
StanzaSharp.Benchmark constituency-divergence --docs SHARD.json --out OUT.json --lang de --package default \
    --processors tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner --backend torch|managed --threads 1
# merge the shards' "docs", then
python tools/constituency_divergence.py compare stanza.json torch.json managed.json
# lemma: lemma_divergence.py stanza --lang de / lemma-divergence --lang de, then lemma_divergence.py compare
```

Summed shard times, 1 thread per shard and up to 17 processes on 8 cores (so much slower than alone): Stanza `default`
18.6 h for both corpora, TorchSharp 15.1 h, managed 5.7 h.
