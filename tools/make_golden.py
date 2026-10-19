#!/usr/bin/env python3
"""
Generate golden test data from Python Stanza for the C# port.

  python tools/make_golden.py [--lang en|de] [--models models/stanza] [--out tests/golden]

--lang de writes the same files (except pt/ and depparse/mst.json, which don't depend on the language) for the
hand-written German texts in tests/golden/de, with German word lists (GERMAN).

Reads <out>/corpus.txt and writes:
  pipeline.conllu       full pipeline output: token offsets (misc start_char/end_char), MWT ranges,
                        UPOS/XPOS/feats, and a "# constituency = ..." comment per sentence
  intermediates.safetensors + intermediates.json
                        tensors for the first INTERMEDIATE_SENTENCES sentences, each stage run
                        on one sentence at a time (no batching) so they are easy to reproduce:
                          s{i}.tokenize.pred        tokenizer output logits for the sentence's paragraph text
                          s{i}.charlm_forward       forward charlm per-word representation [n_words, 1024]
                          s{i}.charlm_backward      backward charlm per-word representation [n_words, 1024]
                          s{i}.pos.upos_logits      tagger UPOS logits [n_words, n_upos]
                          s{i}.pos.upos_logits64    the same in float64 (fast/ too; `--upos64-only` adds them)
                          s{i}.constituency.scores  parser transition logits per step [n_steps, n_transitions]
                        intermediates.json lists the words/text each tensor was computed from.
  tokenize_stress.txt + tokenize_stress.conllu
                        tokenizer-only output on text that exercises long-paragraph windowing
                        and multi-batch padding (see write_tokenize_stress)
  mwt.json              MWT expansions of MWT_WORDS, with and without the dictionary
  validation*.conllu    full pipeline output for each validation*.txt, varied hand-written corpora
                        (validation_nonbmp has text outside the BMP; C# reports its offsets in
                        UTF-16 units, so its test ignores start_char/end_char)
  depparse/<name>.conllu
                        tokenize,mwt,pos,lemma,depparse output for corpus.txt (as corpus.conllu) and
                        each validation*.txt; `--depparse-only` regenerates just depparse/
  depparse/intermediates.safetensors + .json
                        for the first INTERMEDIATE_SENTENCES sentences, each parsed alone from its
                        golden words/tags/lemmas: s{i}.unlabeled (arc log-probs [n+1, n+1], -inf on
                        the diagonal) and s{i}.deprel (label log-probs [n+1, n+1, n_deprels])
  lemma/<name>.conllu   tokenize,mwt,pos,lemma output for corpus.txt and each validation*.txt
  lemma/words.json      lemmatizer on LEMMA_WORDS: pipeline lemma, raw seq2seq output and edit class
                        (`--lemma-only` regenerates just lemma/)
  ner/<name>.conllu     tokenize,mwt,pos,lemma,depparse,ner output for corpus.txt (as corpus.conllu) and
                        each validation*.txt (`ner=` in MISC)
  ner/<name>.json       doc.ents of the same run: per sentence, text/type/start_char/end_char
  ner/intermediates.safetensors + .json
                        for the first INTERMEDIATE_SENTENCES sentences, each tagged alone from its golden
                        tokens: s{i}.emissions (tag_clfs[0] logits [n_tokens, n_tags]); `--ner-only`
                        regenerates just ner/
  sentiment/<name>.conllu + <name>.json
                        tokenize,mwt,sentiment output for corpus.txt and each validation*.txt, and per
                        sentence the label and the 3 class logits, float32 and float64 (`--sentiment-only`)
  fast/<name>.conllu    Stanza's package='default_fast' (tokenize,mwt,pos,lemma,depparse,sentiment,ner with
                        the nocharlm pos/depparse/ner) for corpus.txt (as corpus.conllu) and each
                        validation*.txt
  fast/intermediates.safetensors + .json
                        for the first INTERMEDIATE_SENTENCES sentences, each run alone: s{i}.pos.upos_logits,
                        s{i}.depparse.unlabeled/.deprel (from the run's tags and lemmas) and
                        s{i}.ner.emissions; `--fast-only` regenerates just fast/
  pretokenized/inputs.json + <name>.conllu + <name>.fast.conllu
                        tokenize_pretokenized=True on PRETOKENIZED_CASES and corpus.txt's tokens, with both
                        packages (`--pretokenized-only`; see write_pretokenized_golden)
  bulk/<package>.json   Pipeline.bulk_process on BULK_TEXTS + corpus.txt + validation*.txt, stored as the
                        differences from processing each text alone (`--bulk-only`; see write_bulk_golden)
  no_ssplit/<name>.conllu + <name>.fast.conllu + bulk<suffix>.json
                        tokenize_no_ssplit=True (one sentence per paragraph) on NO_SSPLIT_FILES and, in bulk,
                        BULK_TEXTS, with both packages (`--no-ssplit-only`; see write_no_ssplit_golden)
  long_token.txt + long_token.conllu + long_token.fast.conllu
                        both packages on tokens around max_seqlen (200), which Stanza replaces with <UNK>
                        (`--long-token-only`; see write_long_token_golden)
  json/inputs.json + json/<package>.json + json/pos.json
                        str(doc) (Document.__repr__: to_dict as indented JSON) of JSON_TEXT, a short text with
                        every field Stanza's JSON has (MWTs with and without spacing, SpacesBefore, escaped spaces,
                        quotes and backslashes, entities), for both packages and for tokenize,mwt,pos alone
                        (`--json-only`; see write_json_golden)
  process_document/input.conllu + corpus.conllu + corpus.fast.conllu
                        Pipeline.process on CoNLL.conll2doc of tokenize,mwt's output for corpus.txt with wrong
                        annotations, running every processor but tokenize and mwt (`--process-document-only`; see
                        write_process_document_golden)
  pretagged/<name>.conllu
                        pretagged=True pipelines (only lemma, depparse and/or constituency loaded) on golden CoNLL-U
                        files that already carry tags, some with fields removed (PRETAGGED_CASES, `--pretagged-only`)
  <package>/            the treebank packages (TREEBANK_PACKAGES: English ewt/, gum/, ..., German de/gsd/, de/hdt/):
                        package=<package> (all its processors, tokenize to depparse) on corpus.txt (as corpus.conllu)
                        and each validation*.txt, plus mwt.json where the package has mwt (`--treebank-only`, some of
                        them with `--packages`)
  fuzz.json             both packages on FUZZ_TEXTS, each processed alone: minimized texts on which tools/text_fuzz.py
                        found StanzaSharp's output differing from Stanza's (`--fuzz-only`; see write_fuzz_golden)
  models/<name>/<text>.conllu + models/models.json
                        single models chosen per processor (MODEL_CHOICES, Stanza's processors dict; StanzaSharp's
                        PipelineOptions.Models), e.g. models/ner=i2b2/, on corpus.txt (as corpus.conllu), validation.txt
                        and each choice's hand-written texts in models/ (clinical.txt, biomedical.txt, radiology.txt) (`--models-only`);
                        a chosen lemmatizer's folder also has seq2seq.json (its model alone on corpus.txt's words)
  transformer/tokenizer.json
                        the transformer input Stanza builds (bert_embedding.py: the tokenizer with is_split_into_words,
                        fix_blank_tokens, convert_to_position_list) for google/electra-large-discriminator, on every golden
                        sentence's words, simplify_punct'ed words and token texts, and on TRANSFORMER_TOKENIZER_CASES;
                        "positions" only where not 0..n+1, "fixed" (the words after fix_blank_tokens) only where it changed any
                        (`--transformer-only`; needs tools/requirements-transformer.txt and the transformer's files in
                        <models>/en/transformer/google/electra-large-discriminator, and depparse/combined_electra-large.pt)
  transformer/hidden.safetensors + hidden.json
                        Stanza's extract_bert_embeddings on the first INTERMEDIATE_SENTENCES golden sentences and
                        TRANSFORMER_ILL_CONDITIONED's, in one batch with a sentence of over 512 pieces, float32 and float64
                        (.f64, rounded to float32): s{i}.legacy (the legacy mix, no endpoints), s{i}.mix (depparse's
                        bert_layer_mix, with endpoints), long.legacy (every TRANSFORMER_LONG_STRIDE-th word)
  pt/tiny_{legacy,zip}.pt + their stanza_convert.py output (.json/.safetensors)
                        small checkpoints in both torch.save formats for the C# .pt loader tests;
                        `python tools/make_golden.py --pt-only` regenerates just these
"""
import argparse
import copy
import json
import tempfile
import sys
from pathlib import Path

import stanza
from stanza.utils.conll import CoNLL
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from stanza_convert import NP_TO_ST, convert_object, drop_skipped, load_checkpoint, write_safetensors  # noqa: E402

INTERMEDIATE_SENTENCES = 3
LANG = "en"  # --lang; GERMAN replaces the English word lists below for "de"
PROCESSORS = "tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner"  # the default Pipeline (Stanza's order)


class Capture:
    """Records the outputs of a module (or a wrapped method) while active."""

    def __init__(self):
        self.outputs = []

    def hook(self, _module, _inputs, output):
        self.outputs.append(output.detach().cpu().clone())


def capture_module(module):
    cap = Capture()
    handle = module.register_forward_hook(cap.hook)
    return cap, handle


def write_tokenize_stress(models, corpus, out):
    """
    tokenize_stress.txt/.conllu: tokenizer-only output for text that hits the batching paths the
    main corpus doesn't: one paragraph over the 1000-character window, and more paragraphs than
    one batch (32), with differing lengths so padding varies across rows.
    """
    lines = [line for line in corpus.splitlines() if line.strip()]
    long_para = " ".join(lines * 3)
    short_paras = [lines[i % len(lines)] * (1 + i % 3) for i in range(40)]
    text = "\n\n".join([short_paras[0], long_para] + short_paras[1:]) + "\n"
    (out / "tokenize_stress.txt").write_text(text, encoding="utf-8", newline="\n")

    nlp = stanza.Pipeline(LANG, dir=models, processors="tokenize,mwt", download_method=None,
                          use_gpu=False, logging_level="WARN")
    with torch.no_grad():
        doc = nlp(text)
    (out / "tokenize_stress.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
    print(f"tokenize_stress: {len(long_para)}-char paragraph, {len(short_paras) + 1} paragraphs, "
          f"{len(doc.sentences)} sentences")


# Words to expand with the MWT stage: dictionary hits and misses, case variants, characters
# outside the MWT vocabulary, and words that shouldn't split.
MWT_WORDS = [
    "don't", "Don't", "DON'T", "can't", "Can't", "won't", "it's", "It's", "we'll", "she'd", "I'd've",
    "wouldn't've", "y'all", "gonna", "wanna", "dunno", "cannot", "ain't", "o'clock", "Mary's",
    "children's", "Smiths'", "Kowalczyk's", "Zürich's", "naïve's", "東京's", "rock'n'roll",
    "they’re", "CEO's", "1990's", "U.S.'s", "x", "it'll've", "Jean-Luc's",
]


def write_mwt_golden(models, out, package="default"):
    """
    mwt.json: for each word in MWT_WORDS, the words the MWT stage expands it to, both with the
    pipeline's dictionary+model ensemble and with the model alone (English: the character classifier; German: the seq2seq).
    """
    nlp = stanza.Pipeline(LANG, dir=models, package=package, processors="tokenize,mwt", download_method=None,
                          use_gpu=False, logging_level="WARN")
    mwt = nlp.processors["mwt"]

    def expand(ensemble):
        doc = stanza.Document([[{"id": 1, "text": w, "misc": "MWT=Yes"}] for w in MWT_WORDS])
        old = mwt.config["ensemble_dict"]
        mwt.config["ensemble_dict"] = ensemble
        try:
            with torch.no_grad():
                mwt.process(doc)
        finally:
            mwt.config["ensemble_dict"] = old
        return [[w.text for w in s.words] for s in doc.sentences]

    result = [{"word": w, "pipeline": p, "model": m}
              for w, p, m in zip(MWT_WORDS, expand(True), expand(False))]
    with open(out / "mwt.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"mwt.json: {len(result)} words, {sum(r['pipeline'] != r['model'] for r in result)} differ by dictionary")


DEPPARSE_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"

# The treebank packages (tokenize to depparse) per language; "combined" uses default's models, so depparse/ covers it.
TREEBANK_PACKAGES = {
    "en": ["ewt", "gum", "lines", "partut", "atis", "childes", "eslspok", "craft", "genia", "mimic"],
    "de": ["gsd", "hdt"],
}


def write_treebank_golden(models, out, packages=None):
    """
    <package>/ for each of TREEBANK_PACKAGES (or `packages`): package=<package> output (all its processors, tokenize to
    depparse; eslspok and the biomedical ones have no mwt) for corpus.txt (as corpus.conllu) and each validation*.txt, and
    mwt.json (MWT_WORDS with and without the dictionary) where the package has mwt.
    """
    with open(Path(models) / "resources.json", encoding="utf-8") as f:
        resources = json.load(f)[LANG]["packages"]
    for package in packages or TREEBANK_PACKAGES[LANG]:
        processors = [p for p in DEPPARSE_PROCESSORS.split(",") if p in resources[package]]
        nlp = stanza.Pipeline(LANG, dir=models, package=package, processors=",".join(processors), download_method=None,
                              use_gpu=False, logging_level="WARN")
        folder = out / package
        folder.mkdir(exist_ok=True)
        for path in [out / "corpus.txt"] + sorted(out.glob("validation*.txt")):
            with torch.no_grad():
                doc = nlp(path.read_bytes().decode("utf-8"))
            name = "corpus" if path.name == "corpus.txt" else path.stem
            (folder / f"{name}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
            print(f"{package}/{name}.conllu: {len(doc.sentences)} sentences, {doc.num_words} words")
        if "mwt" in processors:
            write_mwt_golden(models, folder, package)


# Single models chosen per processor (PipelineOptions.Models, Stanza's processors dict): name, package, processors,
# processor -> model, and its texts besides corpus.txt (validation.txt, hand-written domain texts in models/; the
# biomedical and radiology models find almost nothing in validation.txt, so they skip it).
MODEL_CHOICES = {
    "en": [
        ("constituency=ptb3_charlm", "default", "tokenize,mwt,pos,constituency", {"constituency": "ptb3_charlm"}, ["validation.txt", "models/clinical.txt"]),
        ("ner=ontonotes_charlm", "default", "tokenize,mwt,ner", {"ner": "ontonotes_charlm"}, ["validation.txt", "models/clinical.txt"]),
        ("ner=ontonotes_nocharlm", "default", "tokenize,mwt,ner", {"ner": "ontonotes_nocharlm"}, ["validation.txt", "models/clinical.txt"]),
        ("ner=combined_charlm", "default", "tokenize,mwt,ner", {"ner": "combined_charlm"}, ["validation.txt", "models/clinical.txt"]),
        ("ner=conll03_charlm", "default", "tokenize,mwt,ner", {"ner": "conll03_charlm"}, ["validation.txt"]),
        # with the mimic package, whose tagger reads the same (mimic) charlms as i2b2 and radiology (which lowercases
        # its charlm input, so it can't reuse the tagger's)
        ("ner=i2b2", "mimic", "tokenize,pos,lemma,depparse,ner", {"ner": "i2b2"}, ["validation.txt", "models/clinical.txt"]),
        ("ner=radiology", "mimic", "tokenize,pos,lemma,depparse,ner", {"ner": "radiology"}, ["models/radiology.txt"]),
        # with the default package: two charlm pairs in one pipeline (1billion, and mimic or pubmed for ner)
        ("default+ner=i2b2", "default", PROCESSORS, {"ner": "i2b2"}, ["validation.txt", "models/clinical.txt"]),
        ("default+ner=bc5cdr", "default", PROCESSORS, {"ner": "bc5cdr"}, ["models/biomedical.txt"]),
        # the biomedical models (cased word vectors, pubmed's charlms) on the craft package's tokenizer
        ("ner=anatem", "craft", "tokenize,ner", {"ner": "anatem"}, ["models/biomedical.txt"]),
        ("ner=bc4chemd", "craft", "tokenize,ner", {"ner": "bc4chemd"}, ["models/biomedical.txt"]),
        ("ner=bc5cdr", "craft", "tokenize,ner", {"ner": "bc5cdr"}, ["models/biomedical.txt"]),
        ("ner=bionlp13cg", "craft", "tokenize,ner", {"ner": "bionlp13cg"}, ["models/biomedical.txt"]),
        ("ner=jnlpba", "craft", "tokenize,ner", {"ner": "jnlpba"}, ["models/biomedical.txt"]),
        ("ner=linnaeus", "craft", "tokenize,ner", {"ner": "linnaeus"}, ["models/biomedical.txt"]),
        ("ner=ncbi_disease", "craft", "tokenize,ner", {"ner": "ncbi_disease"}, ["models/biomedical.txt"]),
        ("ner=s800", "craft", "tokenize,ner", {"ner": "s800"}, ["models/biomedical.txt"]),
        # the treebanks' _nocharlm taggers and parsers, on their package (corpus.txt only: owner's decision)
        ("pos+depparse=ewt_nocharlm", "ewt", "tokenize,mwt,pos,lemma,depparse", {"pos": "ewt_nocharlm", "depparse": "ewt_nocharlm"}, []),
        ("pos+depparse=gum_nocharlm", "gum", "tokenize,mwt,pos,lemma,depparse", {"pos": "gum_nocharlm", "depparse": "gum_nocharlm"}, []),
        ("pos+depparse=lines_nocharlm", "lines", "tokenize,mwt,pos,lemma,depparse", {"pos": "lines_nocharlm", "depparse": "lines_nocharlm"}, []),
        ("pos+depparse=partut_nocharlm", "partut", "tokenize,mwt,pos,lemma,depparse", {"pos": "partut_nocharlm", "depparse": "partut_nocharlm"}, []),
        ("pos+depparse=atis_nocharlm", "atis", "tokenize,mwt,pos,lemma,depparse", {"pos": "atis_nocharlm", "depparse": "atis_nocharlm"}, []),
        ("pos+depparse=childes_nocharlm", "childes", "tokenize,mwt,pos,lemma,depparse", {"pos": "childes_nocharlm", "depparse": "childes_nocharlm"}, []),
        ("pos+depparse=eslspok_nocharlm", "eslspok", "tokenize,pos,lemma,depparse", {"pos": "eslspok_nocharlm", "depparse": "eslspok_nocharlm"}, []),
        ("depparse=mimic_nocharlm", "mimic", "tokenize,pos,lemma,depparse", {"depparse": "mimic_nocharlm"}, []),
        # the _charlm tokenizers and lemmatizers (per-character charlm states), on their package; combined's tokenizer
        # also on tokenize_stress.txt (windows); lemma combined_charlm's contextual classifiers ('s: be/have, her: she/her)
        # decide words of validation.txt and validation_contractions.txt
        ("lemma=combined_charlm", "default", "tokenize,mwt,pos,lemma", {"lemma": "combined_charlm"}, ["validation.txt", "validation_contractions.txt"]),
        ("tokenize+lemma=ewt_charlm", "ewt", "tokenize,mwt,pos,lemma", {"tokenize": "ewt_charlm", "lemma": "ewt_charlm"}, []),
        ("tokenize+lemma=gum_charlm", "gum", "tokenize,mwt,pos,lemma", {"tokenize": "gum_charlm", "lemma": "gum_charlm"}, []),
        ("tokenize+lemma=lines_charlm", "lines", "tokenize,mwt,pos,lemma", {"tokenize": "lines_charlm", "lemma": "lines_charlm"}, []),
        ("tokenize+lemma=partut_charlm", "partut", "tokenize,mwt,pos,lemma", {"tokenize": "partut_charlm", "lemma": "partut_charlm"}, []),
        ("tokenize+lemma=atis_charlm", "atis", "tokenize,mwt,pos,lemma", {"tokenize": "atis_charlm", "lemma": "atis_charlm"}, []),
        ("tokenize+lemma=childes_charlm", "childes", "tokenize,mwt,pos,lemma", {"tokenize": "childes_charlm", "lemma": "childes_charlm"}, []),
        ("tokenize=eslspok_charlm", "eslspok", "tokenize", {"tokenize": "eslspok_charlm"}, []),
        ("tokenize=combined_charlm", "default", "tokenize,mwt", {"tokenize": "combined_charlm"}, ["tokenize_stress.txt"]),
        # default_accurate's frozen-transformer tagger, parser and NER model (google/electra-large-discriminator, read from
        # <models>/en/transformer/, electra_locally); ner reads token texts, so contractions' sentences miss the tagger's states;
        # transformer_long.txt has a sentence of 706 pieces (two 512-piece windows)
        ("pos+depparse+ner=electra-large", "default", "tokenize,mwt,pos,lemma,depparse,ner",
         {"pos": "combined_electra-large", "depparse": "combined_electra-large", "ner": "ontonotes-ww-multi_electra-large"},
         ["validation.txt", "validation_contractions.txt", "validation_social.txt", "validation_tech.txt", "models/transformer_long.txt"]),
        # constituency's frozen transformer (its bert_layer_mix is not zero) with the boundary vectors: [CLS] and [SEP] next
        # to word_start and word_end
        ("constituency=ptb3-revised_electra-large", "default", "tokenize,mwt,pos,constituency", {"constituency": "ptb3-revised_electra-large"},
         ["validation.txt", "validation_contractions.txt", "models/transformer_long.txt"]),
        # the LoRA-adapted TOP_DOWN parser (#171): Open(ROOT) first, Close builds from the children above the open marker
        ("constituency=ptb3_electra-large", "default", "tokenize,mwt,pos,constituency", {"constituency": "ptb3_electra-large"},
         ["validation.txt", "validation_contractions.txt", "models/transformer_long.txt"]),
        # sentiment's LoRA-adapted transformer (no charlms, filters 3, 4 and 5 only): a pass of its own
        ("sentiment=sstplus_electra-large", "default", "tokenize,mwt,sentiment", {"sentiment": "sstplus_electra-large"},
         ["validation.txt", "validation_contractions.txt", "sentiment/reviews.txt", "models/transformer_long.txt"]),
        # all five transformer models in one pipeline: the frozen states shared four ways (raw words for constituency,
        # simplify_punct'ed for pos and depparse, tokens for ner), sentiment's LoRA pass apart
        ("electra-large", "default", PROCESSORS,
         {"pos": "combined_electra-large", "constituency": "ptb3-revised_electra-large", "depparse": "combined_electra-large",
          "sentiment": "sstplus_electra-large", "ner": "ontonotes-ww-multi_electra-large"},
         ["validation.txt", "validation_contractions.txt", "models/transformer_long.txt"]),
        # the treebanks' frozen-transformer taggers and parsers (#168) on their package (corpus.txt only, like the _nocharlm
        # rows), and ontonotes' NER model (fasttextcrawl vectors | transformer | charlms)
        ("pos+depparse=ewt_electra-large", "ewt", "tokenize,mwt,pos,lemma,depparse", {"pos": "ewt_electra-large", "depparse": "ewt_electra-large"}, []),
        ("pos+depparse=gum_electra-large", "gum", "tokenize,mwt,pos,lemma,depparse", {"pos": "gum_electra-large", "depparse": "gum_electra-large"}, []),
        ("pos+depparse=lines_electra-large", "lines", "tokenize,mwt,pos,lemma,depparse", {"pos": "lines_electra-large", "depparse": "lines_electra-large"}, []),
        ("pos+depparse=partut_electra-large", "partut", "tokenize,mwt,pos,lemma,depparse", {"pos": "partut_electra-large", "depparse": "partut_electra-large"}, []),
        ("pos+depparse=atis_electra-large", "atis", "tokenize,mwt,pos,lemma,depparse", {"pos": "atis_electra-large", "depparse": "atis_electra-large"}, []),
        ("pos+depparse=childes_electra-large", "childes", "tokenize,mwt,pos,lemma,depparse", {"pos": "childes_electra-large", "depparse": "childes_electra-large"}, []),
        ("pos+depparse=eslspok_electra-large", "eslspok", "tokenize,pos,lemma,depparse", {"pos": "eslspok_electra-large", "depparse": "eslspok_electra-large"}, []),
        ("ner=ontonotes_electra-large", "default", "tokenize,mwt,ner", {"ner": "ontonotes_electra-large"}, ["validation.txt"]),
        # conll03's NER model reads roberta-base despite its name (#172: glove vectors | roberta-base | charlms), and with
        # electra-large's other four models a pipeline holds two transformers
        ("ner=conll03_electra-large", "default", "tokenize,mwt,ner", {"ner": "conll03_electra-large"}, ["validation.txt", "validation_social.txt"]),
        ("electra-large+ner=conll03_electra-large", "default", PROCESSORS,
         {"pos": "combined_electra-large", "constituency": "ptb3-revised_electra-large", "depparse": "combined_electra-large",
          "sentiment": "sstplus_electra-large", "ner": "conll03_electra-large"},
         ["validation.txt"]),
        # the default_accurate package itself (no models chosen: package=default_accurate), all eight processors
        ("default_accurate", "default_accurate", PROCESSORS, {},
         ["validation.txt", "validation_contractions.txt", "validation_social.txt", "validation_tech.txt", "models/transformer_long.txt"]),
    ],
    "de": [
        ("ner=conll03", "default", "tokenize,mwt,ner", {"ner": "conll03"}, ["validation.txt"]),
        ("pos+depparse=gsd_nocharlm", "gsd", "tokenize,mwt,pos,lemma,depparse", {"pos": "gsd_nocharlm", "depparse": "gsd_nocharlm"}, []),
        ("pos+depparse=hdt_nocharlm", "hdt", "tokenize,mwt,pos,lemma,depparse", {"pos": "hdt_nocharlm", "depparse": "hdt_nocharlm"}, []),
        ("tokenize+lemma=combined_charlm", "combined", "tokenize,mwt,pos,lemma", {"tokenize": "combined_charlm", "lemma": "combined_charlm"}, []),
        ("tokenize+lemma=gsd_charlm", "gsd", "tokenize,mwt,pos,lemma", {"tokenize": "gsd_charlm", "lemma": "gsd_charlm"}, []),
        ("tokenize+lemma=hdt_charlm", "hdt", "tokenize,mwt,pos,lemma", {"tokenize": "hdt_charlm", "lemma": "hdt_charlm"}, []),
        # default_accurate's frozen-transformer tagger and parser (german-nlp-group/electra-base-german-uncased, read from
        # <models>/de/transformer/, electra_locally): one shared pass; transformer_long.txt is one sentence of 558 pieces
        ("pos+depparse=german-nlp-electra", "default", "tokenize,mwt,pos,lemma,depparse",
         {"pos": "combined_german-nlp-electra", "depparse": "combined_german-nlp-electra"},
         ["validation.txt", "validation_contractions.txt", "validation_umlauts.txt", "validation_social.txt", "models/transformer_long.txt"]),
        # the constituency parser's LoRA-adapted transformer (its top-level bert_lora): a pass of its own
        ("constituency=spmrl_german-nlp-electra", "default", "tokenize,mwt,pos,constituency", {"constituency": "spmrl_german-nlp-electra"},
         ["validation.txt", "validation_umlauts.txt", "models/transformer_long.txt"]),
        # sentiment's LoRA-adapted transformer: another pass of its own
        ("sentiment=sb10k_german-nlp-electra", "default", "tokenize,mwt,sentiment", {"sentiment": "sb10k_german-nlp-electra"},
         ["validation.txt", "validation_social.txt", "models/transformer_long.txt"]),
        # the default_accurate package itself, all eight processors: three passes (frozen, constituency's, sentiment's)
        ("default_accurate", "default_accurate", PROCESSORS, {},
         ["validation.txt", "validation_contractions.txt", "validation_umlauts.txt", "validation_social.txt", "models/transformer_long.txt"]),
        # the treebanks' frozen-transformer taggers and parsers on their package (corpus.txt only, like English's), and
        # germeval2014's NER model on the same transformer (fasttextwiki vectors | transformer | charlms)
        ("pos+depparse=gsd_german-nlp-electra", "gsd", "tokenize,mwt,pos,lemma,depparse",
         {"pos": "gsd_german-nlp-electra", "depparse": "gsd_german-nlp-electra"}, []),
        ("pos+depparse=hdt_german-nlp-electra", "hdt", "tokenize,mwt,pos,lemma,depparse",
         {"pos": "hdt_german-nlp-electra", "depparse": "hdt_german-nlp-electra"}, []),
        ("ner=germeval2014_german-nlp-electra", "default", "tokenize,mwt,ner", {"ner": "germeval2014_german-nlp-electra"},
         ["validation.txt", "models/transformer_long.txt"]),
        # dbmdz/bert-base-german-cased's: ner (conll17's vectors, PRETRAIN_FIXES | BERT | charlms) and sentiment (LoRA, no charlms)
        ("ner=germeval2014_dbmdz-bert-german-cased", "default", "tokenize,mwt,ner", {"ner": "germeval2014_dbmdz-bert-german-cased"},
         ["validation.txt", "models/transformer_long.txt"]),
        ("sentiment=sb10k_dbmdz-bert-german-cased", "default", "tokenize,mwt,sentiment", {"sentiment": "sb10k_dbmdz-bert-german-cased"},
         ["validation.txt", "validation_social.txt", "models/transformer_long.txt"]),
    ],
}


# Every {processor}_batch_size Stanza honours (it ignores tokenize_batch_size), small: many batches on short texts.
SMALL_BATCHES = {"mwt_batch_size": 1, "pos_batch_size": 1, "pos_batch_maximum_tokens": 50, "lemma_batch_size": 3,
                 "constituency_batch_size": 2, "depparse_batch_size": 100, "depparse_min_length_to_batch_separately": 5,
                 "sentiment_batch_size": 20, "ner_batch_size": 1}

# Stanza options with a PipelineOptions property (owner's choice, 2026-10-18): (name, package, processors, models,
# Stanza's keyword arguments, texts besides corpus.txt). The C# tests map the arguments to PipelineOptions (options.json).
OPTION_CASES = {
    "en": [
        ("ner_predict_tagset=1", "default", "tokenize,mwt,ner", {}, {"ner_predict_tagset": 1},
         ["validation.txt", "validation_tech.txt", "validation_social.txt"]),
        ("ner=combined_charlm+ner_predict_tagset=2", "default", "tokenize,mwt,ner", {"ner": "combined_charlm"}, {"ner_predict_tagset": 2},
         ["validation.txt"]),
        ("lemma_dict_only", "default", "tokenize,mwt,pos,lemma", {}, {"lemma_dict_only": True},
         ["validation.txt", "validation_contractions.txt", "validation_nonbmp.txt"]),
        ("lemma=combined_charlm+lemma_dict_only", "default", "tokenize,mwt,pos,lemma", {"lemma": "combined_charlm"}, {"lemma_dict_only": True},
         ["validation_contractions.txt"]),
        ("mwt_dict_only", "default", "tokenize,mwt,pos", {}, {"mwt_dict_only": True},
         ["validation.txt", "validation_contractions.txt"]),
        # The batch sizes (#174): only sentiment's changes these texts' results (depparse's could, through its padded
        # log-softmax, but no head moved at any size tried).
        ("sentiment_batch_size=20", "default", "tokenize,mwt,sentiment", {}, {"sentiment_batch_size": 20},
         ["validation.txt", "validation_social.txt"]),
        ("batch_sizes", "default", PROCESSORS, {}, SMALL_BATCHES, ["validation.txt"]),
        ("default_fast+batch_sizes", "default_fast", "tokenize,mwt,pos,lemma,depparse,sentiment,ner", {},
         {k: v for k, v in SMALL_BATCHES.items() if not k.startswith("constituency")}, ["validation.txt"]),

        # The lemmatizer's beam (#174, lemma_beam_size), on the texts sending the most words through its model.
        ("lemma_beam_size=5", "default", "tokenize,mwt,pos,lemma", {}, {"lemma_beam_size": 5},
         ["validation.txt", "validation_contractions.txt", "validation_nonbmp.txt", "validation_tech.txt", "validation_multilingual.txt",
          "options/lemma_rare.txt"]),
        ("lemma=combined_charlm+lemma_beam_size=3", "default", "tokenize,mwt,pos,lemma", {"lemma": "combined_charlm"},
         {"lemma_beam_size": 3}, ["validation_contractions.txt", "validation_multilingual.txt", "options/lemma_rare.txt"]),
        # Model files of one's own (#174, {processor}_model_path): "{lang_dir}" is the language's model folder. A nocharlm
        # tagger and NER model in place of the package's charlm ones (the charlms they are given go unread), and a processor
        # default_fast lacks, with the shared files it reads.
        ("pos_model_path", "default", "tokenize,mwt,pos", {}, {"pos_model_path": "{lang_dir}/pos/combined_nocharlm.pt"},
         ["validation.txt"]),
        ("ner_model_path", "default", "tokenize,mwt,ner", {}, {"ner_model_path": "{lang_dir}/ner/ontonotes-ww-multi_nocharlm.pt"},
         ["validation.txt"]),
        ("default_fast+constituency_model_path", "default_fast", "tokenize,mwt,pos,constituency", {},
         {"constituency_model_path": "{lang_dir}/constituency/ptb3-revised_charlm.pt",
          "constituency_pretrain_path": "{lang_dir}/pretrain/conll17.pt",
          "constituency_forward_charlm_path": "{lang_dir}/forward_charlm/1billion.pt",
          "constituency_backward_charlm_path": "{lang_dir}/backward_charlm/1billion.pt"}, ["validation.txt"]),
    ],
    "de": [
        ("mwt_dict_only", "default", "tokenize,mwt,pos", {}, {"mwt_dict_only": True}, ["validation.txt"]),
        ("lemma_dict_only", "default", "tokenize,mwt,pos,lemma", {}, {"lemma_dict_only": True}, ["validation.txt"]),
        ("batch_sizes", "default", PROCESSORS, {}, SMALL_BATCHES, ["validation.txt"]),
        ("lemma_beam_size=5", "default", "tokenize,mwt,pos,lemma", {}, {"lemma_beam_size": 5}, ["validation.txt"]),
    ],
}


def write_options_golden(models, out):
    """
    options/<name>/<text>.conllu + options/options.json: OPTION_CASES, Stanza with the option on corpus.txt (as
    corpus.conllu) and the case's texts.
    """
    folder = out / "options"
    folder.mkdir(exist_ok=True)
    index = []
    for name, package, processors, chosen, kwargs, extra in OPTION_CASES[LANG]:
        texts = [out / "corpus.txt"] + [out / t for t in extra]
        index.append({"name": name, "package": package, "processors": processors, "models": chosen, "options": kwargs,
                      "texts": [path.relative_to(out).as_posix() for path in texts]})
        kwargs = {k: v.replace("{lang_dir}", str(Path(models) / LANG)) if isinstance(v, str) else v for k, v in kwargs.items()}
        if chosen:
            spec = {p: chosen.get(p, package) for p in processors.split(",")}
            nlp = stanza.Pipeline(LANG, dir=models, package=None, processors=spec, download_method=None, use_gpu=False,
                                  logging_level="WARN", **kwargs)
        else:
            nlp = stanza.Pipeline(LANG, dir=models, package=package, processors=processors, download_method=None,
                                  use_gpu=False, logging_level="WARN", **kwargs)
        target = folder / name
        target.mkdir(exist_ok=True)
        for path in texts:
            stem = "corpus" if path.name == "corpus.txt" else path.stem
            with torch.no_grad():
                doc = nlp(path.read_bytes().decode("utf-8"))
            (target / f"{stem}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
            print(f"options/{name}/{stem}.conllu: {len(doc.sentences)} sentences")
    with open(folder / "options.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
        f.write("\n")


def write_seq2seq_golden(nlp, doc, path):
    """
    seq2seq.json: the lemmatizer's seq2seq model alone (no dictionary), as in words.json, on every word of `doc` (with
    Stanza's UPOS) and LEMMA_WORDS: the decoded string and the edit class. A chosen lemmatizer's corpus.conllu sends few
    words through the model.
    """
    from stanza.models.lemma.data import DataLoader

    proc = nlp.processors["lemma"]
    pairs = [(w.text, w.upos) for s in doc.sentences for w in s.words] + list(LEMMA_WORDS)
    words = stanza.Document([[{"id": 1, "text": w, "upos": u}] for w, u in pairs])
    batch = DataLoader(words, proc.config["batch_size"], proc.config, vocab=proc.vocab, evaluation=True, expand_unk_vocab=True)
    raw, edits = [], []
    with torch.no_grad():
        for b in batch:
            r, e = proc.trainer.predict(b, proc.config["beam_size"], batch.vocab)
            raw += r
            edits += e
    result = [{"word": w, "upos": u, "seq2seq": r, "edit": e} for (w, u), r, e in zip(pairs, raw, edits)]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"{path.parent.name}/seq2seq.json: {len(result)} words")


class RecordContextual:
    """
    Records each contextual lemma classifier decision into `rows`, in call order (LemmaClassifier.predict, called once per
    classifier and document with every target word, the last classifier first): the word, its index in its sentence, the
    label and the logits.
    """
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        from stanza.models.lemma_classifier.base_model import LemmaClassifier
        self.original = original = LemmaClassifier.predict
        rows = self.rows

        def predict(model, position_indices, sentences, upos_tags=[]):
            logits = model.forward(position_indices, sentences, model.convert_tags(upos_tags))
            labels = original(model, position_indices, sentences, upos_tags)
            rows.extend({"word": s[i], "position": i, "label": label, "logits": l.tolist()}
                        for i, s, label, l in zip(position_indices, sentences, labels, logits))
            return labels
        LemmaClassifier.predict = predict

    def __exit__(self, *exc):
        from stanza.models.lemma_classifier.base_model import LemmaClassifier
        LemmaClassifier.predict = self.original


def electra_locally(models):
    """
    Has Stanza's load_bert read TRANSFORMER and ROBERTA (English) and GERMAN["TRANSFORMER"] and DBMDZ from <models>/<lang>/transformer/
    (the files ModelDownloader fetches at the pinned revision) instead of Hugging Face's cache.
    """
    from stanza.models.common import bert_embedding
    load = bert_embedding.load_bert
    local = {TRANSFORMER: Path(models) / "en" / "transformer" / TRANSFORMER,
             ROBERTA: Path(models) / "en" / "transformer" / ROBERTA,
             GERMAN["TRANSFORMER"]: Path(models) / "de" / "transformer" / GERMAN["TRANSFORMER"],
             DBMDZ: Path(models) / "de" / "transformer" / DBMDZ}

    def load_bert(model_name, *args, **kwargs):
        if model_name in local:
            return load(str(local[model_name]), *args, **{**kwargs, "local_files_only": True})
        return load(model_name, *args, **kwargs)

    bert_embedding.load_bert = load_bert


# Models trained on another pretrain than resources.json lists (Pipeline.ListedPretrains,
# https://github.com/stanfordnlp/stanza/issues/1698): Stanza gets this one as {processor}_pretrain_path.
PRETRAIN_FIXES = {("en", "ner", "conll03_electra-large"): "conll17", ("de", "ner", "germeval2014_dbmdz-bert-german-cased"): "conll17"}


def write_models_golden(models, out, only=None):
    """
    models/<name>/ for each of MODEL_CHOICES (or only those named in `only`): Stanza with processors={p: model, other p:
    package} (package None) on corpus.txt (as corpus.conllu), validation.txt and its texts in models/; models/models.json lists
    every choice for the tests.
    """
    folder = out / "models"
    folder.mkdir(exist_ok=True)
    index = []
    electra_locally(models)
    for name, package, processors, chosen, extra in MODEL_CHOICES[LANG]:
        texts = [out / "corpus.txt"] + [out / t for t in extra]
        index.append({"name": name, "package": package, "processors": processors, "models": chosen,
                      "texts": [path.relative_to(out).as_posix() for path in texts]})
        if only and name not in only:
            continue
        if chosen:
            spec = {p: chosen.get(p, package) for p in processors.split(",")}
            # a model trained on another pretrain than resources.json lists (Pipeline.ListedPretrains, compare.py's PRETRAIN_FIXES)
            fixes = {f"{p}_pretrain_path": str(Path(models) / LANG / "pretrain" / f"{PRETRAIN_FIXES[(LANG, p, m)]}.pt")
                     for p, m in chosen.items() if (LANG, p, m) in PRETRAIN_FIXES}
            nlp = stanza.Pipeline(LANG, dir=models, package=None, processors=spec, download_method=None, use_gpu=False,
                                  logging_level="WARN", **fixes)
        else:  # a package as it is
            nlp = stanza.Pipeline(LANG, dir=models, package=package, processors=processors, download_method=None, use_gpu=False,
                                  logging_level="WARN")
        def model_of(processor):
            path = nlp.processors[processor].config["model_path"]
            path = path[0] if isinstance(path, (list, tuple)) else path.split(";")[0]
            return Path(path).stem
        loaded = {p: model_of(p) for p in chosen}
        assert loaded == chosen, (name, loaded)
        target = folder / name
        target.mkdir(exist_ok=True)
        contextual = {}
        for path in texts:
            stem = "corpus" if path.name == "corpus.txt" else path.stem
            with torch.no_grad(), RecordContextual(contextual.setdefault(stem, [])):
                doc = nlp(path.read_bytes().decode("utf-8"))
            (target / f"{stem}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
            print(f"models/{name}/{stem}.conllu: {len(doc.sentences)} sentences, {len(doc.ents)} entities")
            if "lemma" in chosen and path.name == "corpus.txt":
                write_seq2seq_golden(nlp, doc, target / "seq2seq.json")
        if any(contextual.values()):
            with open(target / "contextual.json", "w", encoding="utf-8", newline="\n") as f:
                json.dump(contextual, f, indent=1, ensure_ascii=False)
            print(f"models/{name}/contextual.json: {sum(map(len, contextual.values()))} words")
    with open(folder / "models.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1, ensure_ascii=False)
        f.write("\n")


def write_depparse_golden(models, out):
    """
    depparse/: full output with heads and deprels, kept apart from the files above. The C# parser
    gets each file's own words, tags and lemmas, since the lemmatizer is ported separately.
    """
    nlp = stanza.Pipeline(LANG, dir=models, processors=DEPPARSE_PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    dep = out / "depparse"
    dep.mkdir(exist_ok=True)
    if LANG == "en":
        write_mst_golden(out)
    sources = [out / "corpus.txt"] + sorted(out.glob("validation*.txt"))
    corpus_doc = None
    for path in sources:
        with torch.no_grad():
            doc = nlp(path.read_bytes().decode("utf-8"))
        name = "corpus" if path.name == "corpus.txt" else path.stem
        corpus_doc = corpus_doc or doc
        (dep / f"{name}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        print(f"depparse/{name}.conllu: {len(doc.sentences)} sentences, {doc.num_words} words")

    # Intermediates: parse each sentence alone, capturing GraphParser.forward's predictions.
    processor = nlp.processors["depparse"]
    model = processor.trainer.model
    captured = []
    original = model.forward

    def forward(*a, **k):
        loss, preds = original(*a, **k)
        captured.append(preds)
        return loss, preds

    model.forward = forward
    tensors, index = {}, []
    try:
        for i, sent in enumerate(corpus_doc.sentences[:INTERMEDIATE_SENTENCES]):
            words = [{"id": w.id, "text": w.text, "lemma": w.lemma, "upos": w.upos, "xpos": w.xpos,
                      "feats": w.feats} for w in sent.words]
            single = stanza.Document([words])
            captured.clear()
            with torch.no_grad():
                processor.process(single)
            tensors[f"s{i}.unlabeled"] = captured[0][0][0]
            tensors[f"s{i}.deprel"] = captured[0][2][0]
            index.append({"sentence": i, "words": [w["text"] for w in words],
                          "heads": [w.head for w in single.sentences[0].words],
                          "deprels": [w.deprel for w in single.sentences[0].words]})
    finally:
        model.forward = original
    write_safetensors(tensors, dep / "intermediates.safetensors", {"stanza": stanza.__version__})
    with open(dep / "intermediates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"stanza": stanza.__version__, "sentences": index}, f, indent=1, ensure_ascii=False)
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


def mst_scores(n, seed):
    """Integer-valued scores with many ties, cheap to rebuild in C# (DepparseTests.MstScores)."""
    return np.array([[float(((i * 31 + j * 17 + seed * 7) * (i + 2 * j + seed + 1)) % 23 - 11)
                      for j in range(n)] for i in range(n)])


def write_mst_golden(out):
    """depparse/mst.json: chuliu_edmonds_one_root on mst_scores(n, seed); null where Stanza asserts."""
    from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root
    cases = []
    for n in (2, 4, 7, 12, 30, 60):
        for seed in range(40):
            try:
                tree = [int(h) for h in chuliu_edmonds_one_root(mst_scores(n, seed))]
            except AssertionError:
                tree = None
            cases.append({"n": n, "seed": seed, "tree": tree})
    with open(out / "depparse" / "mst.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(cases, f, separators=(",", ":"))
# (word, UPOS) pairs for the lemmatizer: dictionary hits by POS and POS-independent, and seq2seq
# words for every edit type, unknown characters (copied through the delta vocab), non-BMP text,
# a word longer than max_dec_len, and odd inputs.
LEMMA_WORDS = [
    ("saw", "VERB"), ("saw", "NOUN"), ("was", "AUX"), ("children", "NOUN"), ("better", "ADJ"),
    ("Running", "VERB"), ("flibbertigibbets", "NOUN"), ("unfriended", "VERB"), ("GOOGLING", "VERB"),
    ("Zoomers", "PROPN"), ("ãntennae", "NOUN"), ("Zürichers", "PROPN"), ("naïvely", "ADV"),
    ("東京", "PROPN"), ("𝒳s", "NOUN"), ("😀😀s", "SYM"), ("blorptastically", "ADV"),
    ("supercalifragilisticexpialidociousnessesqwertyuiopasdfgh", "NOUN"), ("_", "PUNCT"),
    ("<UNK>", "X"), ("quizzeroos", "NOUN"), ("Wugs", "NOUN"), ("ÉCOLES", "NOUN"), ("ﬁnalized", "VERB"),
]


def write_lemma_golden(models, out):
    """
    lemma/: lemmatizer golden data, in its own folder.
      <name>.conllu  tokenize,mwt,pos,lemma output for corpus.txt and every validation*.txt
      words.json     for each LEMMA_WORDS pair: the pipeline lemma (dictionary + seq2seq ensemble),
                     and the seq2seq model alone: raw decoded string, edit class and final lemma
    """
    from stanza.models.lemma.data import DataLoader

    lemma_out = out / "lemma"
    lemma_out.mkdir(exist_ok=True)
    nlp = stanza.Pipeline(LANG, dir=models, processors="tokenize,mwt,pos,lemma", download_method=None,
                          use_gpu=False, logging_level="WARN")
    for path in [out / "corpus.txt"] + sorted(out.glob("validation*.txt")):
        with torch.no_grad():
            doc = nlp(path.read_bytes().decode("utf-8"))
        conllu = lemma_out / path.with_suffix(".conllu").name
        conllu.write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        print(f"lemma/{conllu.name}: {len(doc.sentences)} sentences, {doc.num_words} words")

    proc = nlp.processors["lemma"]
    trainer = proc.trainer
    doc = stanza.Document([[{"id": 1, "text": w, "upos": u}] for w, u in LEMMA_WORDS])
    with torch.no_grad():
        proc.process(doc)
        batch = DataLoader(doc, proc.config["batch_size"], proc.config, vocab=proc.vocab,
                           evaluation=True, expand_unk_vocab=True)
        raw, edits = [], []
        for b in batch:
            r, e = trainer.predict(b, proc.config["beam_size"], batch.vocab)
            raw += r
            edits += e
    words = [w for w, _ in LEMMA_WORDS]
    model = trainer.postprocess(words, raw, edits=edits)
    result = [{"word": w, "upos": u, "lemma": s.words[0].lemma, "seq2seq": r, "edit": e, "model": m}
              for (w, u), s, r, e, m in zip(LEMMA_WORDS, doc.sentences, raw, edits, model)]
    with open(lemma_out / "words.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"lemma/words.json: {len(result)} words")


SENTIMENT_PROCESSORS = "tokenize,mwt,sentiment"


def write_sentiment_golden(models, out):
    """
    sentiment/: tokenize,mwt,sentiment output, kept apart from the files above.
      <name>.conllu + <name>.json   for corpus.txt, every validation*.txt and sentiment/reviews.txt
                                    (opinionated sentences): the CoNLL-U, and per sentence the label
                                    and the 3 class logits as computed in the pipeline's batches
                                    ("logits"), and the same batches run in float64 ("logits64")
      all.json                      the same for all of those texts as one document (joined by blank
                                    lines), which takes several 5000-word batches
    """
    from stanza.models.common.utils import sort_with_indices

    sent_out = out / "sentiment"
    sent_out.mkdir(exist_ok=True)
    nlp = stanza.Pipeline(LANG, dir=models, processors=SENTIMENT_PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    model = nlp.processors["sentiment"]._model
    captured = []
    original = model.forward

    def forward(*a, **k):
        result = original(*a, **k)
        captured.append(result)
        return result

    def run(text, name, conllu):
        """The document, each sentence's logits in document order, and the number of batches; writes the CoNLL-U if asked."""
        captured.clear()
        with torch.no_grad():
            doc = nlp(text)
        if conllu:
            (sent_out / f"{name}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        # label_sentences runs the batches in length-sorted order; put the logits back in document order.
        _, orig_idx = sort_with_indices(model.extract_sentences(doc), key=len, reverse=True)
        sorted_logits = torch.cat(captured).tolist()
        logits = [None] * len(sorted_logits)
        for k, i in enumerate(orig_idx):
            logits[i] = sorted_logits[k]
        return doc, logits, len(captured)

    model.forward = forward
    try:
        texts = []
        for path in [out / "corpus.txt"] + sorted(out.glob("validation*.txt")) + [sent_out / "reviews.txt"]:
            texts.append(("corpus" if path.name == "corpus.txt" else path.stem, path.read_bytes().decode("utf-8")))
        texts.append(("all", "\n\n".join(t for _, t in texts)))
        results = {name: run(text, name, name != "all") for name, text in texts}

        # The same batches in float64: the exact logits. The classifier is ill-conditioned on some sentences, where
        # Stanza's own float32 result is over 1e-4 from them (validation.txt sentence 41: 1.49e-4), so the tests accept
        # a logit within tolerance of either.
        model.double()
        for m in model.modules():  # tensors held as plain attributes, not parameters or buffers
            for k, v in list(vars(m).items()):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    setattr(m, k, v.double())
        processor = nlp.processors["sentiment"]
        process = processor.process

        def process64(doc):
            # build_char_reps' torch.zeros; only here, as German's seq2seq MWT makes float32-only tensors the same way.
            torch.set_default_dtype(torch.float64)
            try:
                return process(doc)
            finally:
                torch.set_default_dtype(torch.float32)

        processor.process = process64
        exact = {name: run(text, name, False)[1] for name, text in texts}
    finally:
        torch.set_default_dtype(torch.float32)
        model.forward = original

    for name, (doc, logits, batches) in results.items():
        result = [{"sentiment": s.sentiment, "logits": l, "logits64": e} for s, l, e in zip(doc.sentences, logits, exact[name])]
        with open(sent_out / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"stanza": stanza.__version__, "batches": batches, "sentences": result}, f,
                      indent=None if name == "all" else 1)
        worst = max(abs(x - y) for r in result for x, y in zip(r["logits"], r["logits64"]))
        print(f"sentiment/{name}: {len(doc.sentences)} sentences, {batches} batches, labels "
              f"{[sum(r['sentiment'] == c for r in result) for c in range(3)]}, float32 vs float64 {worst:.2e}")


def write_pt_fixtures(out):
    """
    pt/: a tiny checkpoint saved in the legacy format (as Stanza's models are) and in the zip format
    (torch.save's default), each next to what stanza_convert.py makes of it. It covers what the real
    models don't: non-contiguous views, non-float dtypes, big ints, nan/inf, non-string dict keys and
    colliding tensor keys.
    """
    from collections import OrderedDict

    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    ckpt = {
        "model": OrderedDict(
            weight=base, weight_t=base.t(), rows=base[1:3], col=base[:, 2], step=base[::2, 1::3],
            ids=torch.tensor([3, -1, 2**40]), half=torch.tensor([1.5, -2.0], dtype=torch.float16),
            flag=torch.tensor([True, False]), scalar=torch.tensor(7.0), empty=torch.zeros(0, 3)),
        "config": {"lr": 1e-05, "eps": 0.0001, "big": 1e16, "whole": 2.0, "third": 1 / 3,
                   "nan": float("nan"), "ninf": float("-inf"), "huge": 2**70, "neg": -5, "flag": True,
                   "none": None, "name": "naïve ✓", "shape": (3, (4, 5)), "list": [1, 2.5, "x"]},
        "by_id": {1: "one", 2: torch.ones(2)},
        "a.b": torch.zeros(1),
        "a": {"b": torch.ones(1)},
        "twice": [base[0], base[0]],
        "optimizer": {"state": {}},
    }
    (out / "pt").mkdir(exist_ok=True)
    for name, kwargs in [("tiny_legacy", {"_use_new_zipfile_serialization": False}),
                         ("tiny_zip", {})]:
        path = out / "pt" / f"{name}.pt"
        torch.save(ckpt, path, **kwargs)
        convert_object(drop_skipped(load_checkpoint(path, False), {"optimizer", "scheduler"}),
                       out / "pt" / name, source_name=path.name)


NER_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,ner"


def write_ner_golden(models, out):
    """ner/: NER golden data, in its own folder (see the module docstring)."""
    nlp = stanza.Pipeline(LANG, dir=models, processors=NER_PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    ner = out / "ner"
    ner.mkdir(exist_ok=True)
    corpus_doc = None
    for path in [out / "corpus.txt"] + sorted(out.glob("validation*.txt")):
        with torch.no_grad():
            doc = nlp(path.read_bytes().decode("utf-8"))
        name = "corpus" if path.name == "corpus.txt" else path.stem
        corpus_doc = corpus_doc or doc
        (ner / f"{name}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        ents = [[{"text": e.text, "type": e.type, "start_char": e.start_char, "end_char": e.end_char}
                 for e in s.ents] for s in doc.sentences]
        with open(ner / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(ents, f, indent=1, ensure_ascii=False)
        print(f"ner/{name}.conllu: {len(doc.sentences)} sentences, {len(doc.ents)} entities")

    # Intermediates: tag each sentence alone, capturing the emission scores of the predicted tag set.
    processor = nlp.processors["ner"]
    trainer = processor.trainers[0]
    tensors, index = {}, []
    for i, sent in enumerate(corpus_doc.sentences[:INTERMEDIATE_SENTENCES]):
        tokens = [t.text for t in sent.tokens]
        single = stanza.Document([[{"id": j + 1, "text": t} for j, t in enumerate(tokens)]])
        cap, h = capture_module(trainer.model.tag_clfs[trainer.args.get("predict_tagset", 0)])  # German's old checkpoint lacks it
        with torch.no_grad():
            processor.process(single)
        h.remove()
        tensors[f"s{i}.emissions"] = cap.outputs[0].numpy()
        index.append({"sentence": i, "tokens": tokens, "ner": [t.ner for t in single.sentences[0].tokens]})
    write_safetensors(tensors, ner / "intermediates.safetensors", {"stanza": stanza.__version__})
    with open(ner / "intermediates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"stanza": stanza.__version__, "sentences": index}, f, indent=1, ensure_ascii=False)
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


def upos_logits64(models, sentences, package="default"):
    """
    The tagger's UPOS logits for each sentence (a list of words) alone, in float64: the model and its tensors in
    float64, as for sentiment's logits64. German's float32 logits (up to |140|) are up to 1.01e-4 from these, so the
    tests accept a logit within tolerance of either (owner's decision, 2026-10-13). A fresh pipeline, so the float32
    models of the other writers are untouched.
    """
    nlp = stanza.Pipeline(LANG, dir=models, package=package, processors="tokenize,mwt,pos", download_method=None,
                          use_gpu=False, logging_level="WARN")
    pos = nlp.processors["pos"]
    model = pos.trainer.model
    model.double()
    for m in model.modules():  # tensors held as plain attributes, not parameters or buffers
        for k, v in list(vars(m).items()):
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                setattr(m, k, v.double())
    result = []
    torch.set_default_dtype(torch.float64)
    try:
        for words in sentences:
            single = stanza.Document([[{"id": j + 1, "text": w} for j, w in enumerate(words)]])
            cap, h = capture_module(model.upos_clf)
            with torch.no_grad():
                pos.process(single)
            h.remove()
            result.append(cap.outputs[0].numpy())
    finally:
        torch.set_default_dtype(torch.float32)
    return result


def read_safetensors(path):
    """The tensors of a safetensors file, in file order, and its metadata."""
    st_to_np = {v: k for k, v in NP_TO_ST.items()}
    data = path.read_bytes()
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + n])
    metadata = header.pop("__metadata__", None)
    tensors = {}
    for key, entry in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        begin, end = entry["data_offsets"]
        tensors[key] = np.frombuffer(data[8 + n + begin:8 + n + end], dtype=st_to_np[entry["dtype"]]).reshape(entry["shape"])
    return tensors, metadata


def write_upos64_golden(models, out):
    """
    --upos64-only: adds s{i}.pos.upos_logits64 to intermediates.safetensors and fast/intermediates.safetensors, after
    the tensors already there, which stay byte-identical (the full run writes the same files).
    """
    for folder, package in [(out, "default"), (out / "fast", "default_fast")]:
        path = folder / "intermediates.safetensors"
        tensors, metadata = read_safetensors(path)
        index = json.load(open(folder / "intermediates.json", encoding="utf-8"))["sentences"]
        for i, logits in enumerate(upos_logits64(models, [e["words"] for e in index], package)):
            tensors[f"s{i}.pos.upos_logits64"] = logits
            diff = np.abs(tensors[f"s{i}.pos.upos_logits"].astype(np.float64) - logits).max()
            print(f"{path.relative_to(out)} s{i}: UPOS float32 vs float64 {diff:.2e}")
        write_safetensors(tensors, path, metadata)


def write_fast_golden(models, out):
    """fast/: Stanza's default_fast package (nocharlm pos, depparse and ner; see the module docstring)."""
    nlp = stanza.Pipeline(LANG, dir=models, package="default_fast", download_method=None,
                          use_gpu=False, logging_level="WARN")
    assert list(nlp.processors) == FAST_PROCESSORS.split(","), list(nlp.processors)
    fast = out / "fast"
    fast.mkdir(exist_ok=True)
    corpus_doc = None
    for path in [out / "corpus.txt"] + sorted(out.glob("validation*.txt")):
        with torch.no_grad():
            doc = nlp(path.read_bytes().decode("utf-8"))
        name = "corpus" if path.name == "corpus.txt" else path.stem
        corpus_doc = corpus_doc or doc
        (fast / f"{name}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        print(f"fast/{name}.conllu: {len(doc.sentences)} sentences, {doc.num_words} words")

    # Intermediates: each sentence alone, from the run's own words (pos), tags and lemmas (depparse)
    # and tokens (ner).
    pos, dep, ner = (nlp.processors[n] for n in ("pos", "depparse", "ner"))
    dep_model = dep.trainer.model
    captured = []
    original = dep_model.forward

    def forward(*a, **k):
        loss, preds = original(*a, **k)
        captured.append(preds)
        return loss, preds

    tensors, index = {}, []
    dep_model.forward = forward
    try:
        for i, sent in enumerate(corpus_doc.sentences[:INTERMEDIATE_SENTENCES]):
            words = [w.text for w in sent.words]
            with torch.no_grad():
                single = stanza.Document([[{"id": j + 1, "text": w} for j, w in enumerate(words)]])
                cap, h = capture_module(pos.trainer.model.upos_clf)
                pos.process(single)
                h.remove()
                tensors[f"s{i}.pos.upos_logits"] = cap.outputs[0].numpy()

                tagged = stanza.Document([[{"id": w.id, "text": w.text, "lemma": w.lemma, "upos": w.upos,
                                            "xpos": w.xpos, "feats": w.feats} for w in sent.words]])
                captured.clear()
                dep.process(tagged)
                tensors[f"s{i}.depparse.unlabeled"] = captured[0][0][0]
                tensors[f"s{i}.depparse.deprel"] = captured[0][2][0]

                tokens = [t.text for t in sent.tokens]
                single = stanza.Document([[{"id": j + 1, "text": t} for j, t in enumerate(tokens)]])
                cap, h = capture_module(ner.trainers[0].model.tag_clfs[ner.trainers[0].args.get("predict_tagset", 0)])
                ner.process(single)
                h.remove()
                tensors[f"s{i}.ner.emissions"] = cap.outputs[0].numpy()
            index.append({"sentence": i, "words": words, "tokens": tokens,
                          "xpos": [w.xpos for w in sent.words],
                          "heads": [w.head for w in tagged.sentences[0].words],
                          "ner": [t.ner for t in single.sentences[0].tokens]})
    finally:
        dep_model.forward = original
    for i, logits in enumerate(upos_logits64(models, [e["words"] for e in index], "default_fast")):
        tensors[f"s{i}.pos.upos_logits64"] = logits
    write_safetensors(tensors, fast / "intermediates.safetensors", {"stanza": stanza.__version__})
    with open(fast / "intermediates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"stanza": stanza.__version__, "sentences": index}, f, indent=1, ensure_ascii=False)
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


FAST_PROCESSORS = "tokenize,mwt,pos,lemma,depparse,sentiment,ner"  # what package='default_fast' loads

PACKAGES = {"default": "", "default_fast": ".fast"}  # package -> golden file suffix

PRETOKENIZED_CASES = {
    "simple": [["Hello", "world", "."], ["Bye", "."]],
    "single": [["Stop"]],
    "contractions": [["I", "don't", "know", "."], ["We", "ca", "n't", "go", "!"], ["It's", "John's", "book", "."]],
    "punctuation": [["(", "Hello", ")", "--", '"', "she", "said", "...", '"'], ["Really", "?!"], ["A", ":", "b", ";", "c", "/", "d"]],
    "spaces": [["New York", "is", "big", "."], ["I", "live", "in", "Los Angeles", "."]],
    "nonbmp": [["I", "love", "😀", "!"], ["𝒳", "marks", "the", "spot", "."]],
}

BULK_TEXTS = ["Hi there.", "", "   ", "\n\n", "Short one. Two sentences!", " padded text \n", "ok",
              "I don't know, can't say.", "Barack Obama was born in Hawaii.\n\nHe was elected in 2008.", "?"]


def write_pretokenized_golden(models, out):
    """
    pretokenized/: tokenize_pretokenized=True on lists of token lists. inputs.json maps each case name to its
    sentences; <name>.conllu is the default package's output and <name>.fast.conllu default_fast's. The "corpus"
    case is corpus.txt as Stanza's own tokenizer splits it.
    """
    pre = out / "pretokenized"
    pre.mkdir(exist_ok=True)
    tok = stanza.Pipeline(LANG, dir=models, processors="tokenize", download_method=None, use_gpu=False, logging_level="WARN")
    cases = dict(PRETOKENIZED_CASES)
    cases["corpus"] = [[t.text for t in s.tokens] for s in tok((out / "corpus.txt").read_text(encoding="utf-8")).sentences]
    with open(pre / "inputs.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(cases, f, indent=1, ensure_ascii=False)
    for package, suffix in PACKAGES.items():
        nlp = stanza.Pipeline(LANG, dir=models, package=package, tokenize_pretokenized=True, download_method=None,
                              use_gpu=False, logging_level="WARN")
        for name, sentences in cases.items():
            with torch.no_grad():
                doc = nlp(sentences)
                # The string form (whitespace-separated tokens, one sentence per line) gives the same document.
                if all(" " not in t for s in sentences for t in s):
                    assert "{:C}".format(nlp("\n".join(" ".join(s) for s in sentences))) == "{:C}".format(doc)
            (pre / f"{name}{suffix}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
            print(f"pretokenized/{name}{suffix}.conllu: {len(doc.sentences)} sentences, {doc.num_words} words")


def write_bulk_golden(models, out):
    """
    bulk/: Pipeline.bulk_process on one list of texts per package: BULK_TEXTS, then corpus.txt and every
    validation*.txt. bulk/<package>.json lists the documents; each has either its full CoNLL-U ("conllu", for the
    BULK_TEXTS, as "{:C}" without the trailing newline the other golden files have) or the name of the golden file with its output when processed alone ("alone": pipeline.conllu or
    validation*.conllu for default, fast/<name>.conllu for default_fast) plus what bulk changes in it: "sent_id_offset"
    (sentence ids continue across documents) and "sentences", the CoNLL-U of each sentence that differs once the ids
    are shifted (sentiment and depparse batch sentences across documents).
    """
    bulk = out / "bulk"
    bulk.mkdir(exist_ok=True)
    files = [out / "corpus.txt"] + sorted(out.glob("validation*.txt"))
    texts = BULK_TEXTS + [p.read_bytes().decode("utf-8") for p in files]
    for package in PACKAGES:
        nlp = stanza.Pipeline(LANG, dir=models, package=package, download_method=None, use_gpu=False, logging_level="WARN")
        with torch.no_grad():
            docs = nlp.bulk_process(texts)
        entries, offset, changed = [], 0, 0
        for i, doc in enumerate(docs):
            conllu = "{:C}\n".format(doc)
            if i < len(BULK_TEXTS):
                # "{:C}" alone: an empty document is "", not "\n".
                entries.append({"text": texts[i], "conllu": "{:C}".format(doc)})
            else:
                path = files[i - len(BULK_TEXTS)]
                name = "corpus" if path.name == "corpus.txt" else path.stem
                alone = ("pipeline.conllu" if name == "corpus" else f"{name}.conllu") if package == "default" else f"fast/{name}.conllu"
                expected = (out / alone).read_text(encoding="utf-8").split("\n\n")
                got = conllu.split("\n\n")
                assert len(expected) == len(got), (alone, len(expected), len(got))
                shifted = [e.replace(f"# sent_id = {j}\n", f"# sent_id = {j + offset}\n", 1) for j, e in enumerate(expected)]
                diffs = {str(j): g for j, (e, g) in enumerate(zip(shifted, got)) if e != g}
                changed += len(diffs)
                entries.append({"file": path.name, "alone": alone, "sent_id_offset": offset, "sentences": diffs})
            offset += len(doc.sentences)
        with open(bulk / f"{package}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"stanza": stanza.__version__, "documents": entries}, f, indent=1, ensure_ascii=False)
        print(f"bulk/{package}.json: {len(docs)} documents, {offset} sentences, {changed} differ from processing alone")


# A manual expansion ("gonna") comes before every automatic one in each document, bulk included: Stanza shifts a manual
# expansion's word ids by the automatic expansions before it in the whole document (set_mwt_expansions' SKIP branch adds
# idx_e), which breaks its trees ("Dependency tree is incorrectly constructed"); StanzaSharp numbers them as written.
POSTPROCESS_TEXTS = [
    "It is gonna rain. I wanna see New York, but I cannot go. I can't say. Bye now.",
    "Short one.",
    "We cannot stay in New York City.\n\nShe's here. They don't know it.",
]


def postprocess(sentences):
    """
    The tokenize_postprocessor of postprocess/ (PostprocessorTests has the same in C#): merges "New" "York" into
    "New York", splits "cannot" into "can" "not", expands "gonna" by hand into "gon" "na", flags "wanna" for the MWT model,
    keeps the tokenizer's MWT flags, and merges each text's last two sentences.
    """
    result = []
    for sentence in sentences:
        tokens, i = [], 0
        while i < len(sentence):
            token = sentence[i]
            text = token[0] if isinstance(token, tuple) else token
            nxt = sentence[i + 1] if i + 1 < len(sentence) else None
            if text == "New" and nxt == "York":
                tokens.append("New York")
                i += 2
                continue
            if text == "cannot":
                tokens += ["can", "not"]
            elif text == "gonna":
                tokens.append(("gonna", ["gon", "na"]))
            elif text == "wanna":
                tokens.append(("wanna", True))
            else:
                tokens.append(token)
            i += 1
        result.append(tokens)
    if len(result) >= 2:
        result[-2:] = [result[-2] + result[-1]]
    return result


def write_postprocess_golden(models, out):
    """
    postprocess/: the default package with tokenize_postprocessor=postprocess (and once with tokenize_no_ssplit):
    each of POSTPROCESS_TEXTS alone, all of them in one bulk_process call, and the first with no_ssplit; "{:C}" each.
    """
    folder = out / "postprocess"
    folder.mkdir(exist_ok=True)
    result = {"texts": POSTPROCESS_TEXTS}
    for name, extra in [("default", {}), ("no_ssplit", {"tokenize_no_ssplit": True})]:
        nlp = stanza.Pipeline(LANG, dir=models, download_method=None, use_gpu=False, logging_level="WARN",
                              tokenize_postprocessor=postprocess, **extra)
        with torch.no_grad():
            if name == "default":
                result["alone"] = ["{:C}".format(nlp(t)) for t in POSTPROCESS_TEXTS]
                result["bulk"] = ["{:C}".format(d) for d in nlp.bulk_process(POSTPROCESS_TEXTS)]
            else:
                result["no_ssplit"] = "{:C}".format(nlp(POSTPROCESS_TEXTS[0]))
    with open(folder / "postprocess.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"postprocess/postprocess.json: {len(POSTPROCESS_TEXTS)} texts")


STREAM_BATCH_SIZE = 4


def write_stream_golden(models, out):
    """
    stream/<package>.json: Pipeline.stream over BULK_TEXTS and validation.txt's paragraphs with batch_size
    STREAM_BATCH_SIZE (each batch one bulk_process call, sentence ids counted across the stream), each document's
    "{:C}" CoNLL-U.
    """
    folder = out / "stream"
    folder.mkdir(exist_ok=True)
    paragraphs = [p for p in (out / "validation.txt").read_bytes().decode("utf-8").split("\n\n") if p.strip()][:20]
    texts = BULK_TEXTS + paragraphs
    for package in PACKAGES:
        nlp = stanza.Pipeline(LANG, dir=models, package=package, download_method=None, use_gpu=False, logging_level="WARN")
        with torch.no_grad():
            docs = [{"text": t, "conllu": "{:C}".format(d)} for t, d in zip(texts, nlp.stream(texts, batch_size=STREAM_BATCH_SIZE))]
        with open(folder / f"{package}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"batch_size": STREAM_BATCH_SIZE, "documents": docs}, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"stream/{package}.json: {len(docs)} documents")


NO_SSPLIT_FILES = ["corpus.txt", "validation.txt", "validation_contractions.txt", "validation_dialogue.txt",
                   "validation_long.txt", "validation_nonbmp.txt", "validation_whitespace.txt"]


def write_no_ssplit_golden(models, out):
    """
    no_ssplit/: tokenize_no_ssplit=True, where each paragraph (blank-line separated) is one sentence. <name>.conllu is
    the default package's output for NO_SSPLIT_FILES (corpus.txt as corpus.conllu), <name>.fast.conllu default_fast's.
    bulk<suffix>.json: bulk_process on BULK_TEXTS with no_ssplit, each document's "{:C}". Pretokenized input ignores
    no_ssplit in Stanza (asserted here), so it needs no golden data of its own.
    """
    d = out / "no_ssplit"
    d.mkdir(exist_ok=True)
    for package, suffix in PACKAGES.items():
        nlp = stanza.Pipeline(LANG, dir=models, package=package, tokenize_no_ssplit=True, download_method=None,
                              use_gpu=False, logging_level="WARN")
        for file in NO_SSPLIT_FILES:
            name = "corpus" if file == "corpus.txt" else Path(file).stem
            with torch.no_grad():
                doc = nlp((out / file).read_bytes().decode("utf-8"))
            (d / f"{name}{suffix}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
            print(f"no_ssplit/{name}{suffix}.conllu: {len(doc.sentences)} sentences, {doc.num_words} words")
        with torch.no_grad():
            docs = nlp.bulk_process(BULK_TEXTS)
        with open(d / f"bulk{suffix}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump([{"text": t, "conllu": "{:C}".format(doc)} for t, doc in zip(BULK_TEXTS, docs)], f, indent=1, ensure_ascii=False)
        print(f"no_ssplit/bulk{suffix}.json: {len(docs)} documents")
        pre = stanza.Pipeline(LANG, dir=models, package=package, tokenize_pretokenized=True, download_method=None,
                              use_gpu=False, logging_level="WARN")
        pre_no_ssplit = stanza.Pipeline(LANG, dir=models, package=package, tokenize_pretokenized=True,
                                        tokenize_no_ssplit=True, download_method=None, use_gpu=False, logging_level="WARN")
        with torch.no_grad():
            for sentences in PRETOKENIZED_CASES.values():
                assert "{:C}".format(pre(sentences)) == "{:C}".format(pre_no_ssplit(sentences))


def long_url(n):
    """A URL of exactly n characters; URL_RAW_RE forces it into one token."""
    return ("https://example.com/" + "abcdefghij" * 200)[:n]


# Tokens around the tokenizer config's max_seqlen (200), one far over the 1000-unit window, and a long non-URL word.
MAX_SEQLEN = 200
LONG_TOKEN_TEXT = "\n\n".join([
    f"The study cited {long_url(300)} as its source. It was later removed.",
    f"Exactly {long_url(199)} and {long_url(200)} and {long_url(201)} were tested.",
    f"A long word: {'pneumono' * 30} is not a URL.",
    f"The longest one, {long_url(1200)} , ends here.",
    "Short sentence after them all.",
]) + "\n"


def write_long_token_golden(models, out):
    """
    long_token.txt + long_token.conllu (default package, PROCESSORS) + long_token.fast.conllu (default_fast):
    TokenizeProcessor.process replaces tokens longer than the tokenizer config's max_seqlen with "<UNK>" (text only;
    offsets stay). Pretokenized input skips that code, so long tokens stay (asserted here).
    """
    (out / "long_token.txt").write_text(LONG_TOKEN_TEXT, encoding="utf-8", newline="\n")
    for package, suffix in PACKAGES.items():
        nlp = stanza.Pipeline(LANG, dir=models, package=package, download_method=None, use_gpu=False,
                              logging_level="WARN")
        assert nlp.processors["tokenize"].config["max_seqlen"] == MAX_SEQLEN
        with torch.no_grad():
            doc = nlp(LONG_TOKEN_TEXT)
        (out / f"long_token{suffix}.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
        print(f"long_token{suffix}.conllu: {len(doc.sentences)} sentences, "
              f"{sum(t.text == '<UNK>' for s in doc.sentences for t in s.tokens)} <UNK> tokens")
    pre = stanza.Pipeline(LANG, dir=models, processors="tokenize", tokenize_pretokenized=True, download_method=None,
                          use_gpu=False, logging_level="WARN")
    assert pre([["see", long_url(MAX_SEQLEN + 100)]]).sentences[0].tokens[1].text == long_url(MAX_SEQLEN + 100)


# Minimized differences found by tools/text_fuzz.py (docs/text-fuzzing.md): characters outside the BMP in the regexes
# Stanza runs over code points, a CJK Extension B letter before "C:" (labeled_field's \b) and an emoji as a URL's
# [^\s"]{2,} tail (MASK_RE).
FUZZ_TEXTS = [
    "\U00020001C:",
    "Go to example.com.\U0001F600 now",
    "Open https://a.\U0001F600 please",
    "the end.\U0001F600 Next www.a.\U0001F600 Then",
]


def write_fuzz_golden(models, out):
    """fuzz.json: {"texts": FUZZ_TEXTS, <package>: [the CoNLL-U of each text processed alone]} for both packages."""
    result = {"texts": FUZZ_TEXTS}
    for package in PACKAGES:
        nlp = stanza.Pipeline(LANG, dir=models, package=package, download_method=None, use_gpu=False,
                              logging_level="WARN")
        with torch.no_grad():
            result[package] = ["{:C}\n".format(nlp(t)) for t in FUZZ_TEXTS]
    (out / "fuzz.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                   newline="\n")
    print(f"fuzz.json: {len(FUZZ_TEXTS)} texts")


JSON_TEXT = ('  "Hello," she said. I don\'t.\tIt\'s Barack Obama\'s café in New York\u00a0City, see C:\\temp\\x!\r\n\r\n'
             'We can\'t\x1cgo (won\'t)  ')
JSON_OUTPUTS = {"default": None, "default_fast": None, "pos": "tokenize,mwt,pos"}  # file name -> processors


def write_json_golden(models, out):
    """
    json/: Stanza's str(doc), i.e. json.dumps(doc.to_dict(), indent=2, ensure_ascii=False), of JSON_TEXT (stored in
    json/inputs.json, as it holds CRs) for both packages (<package>.json) and for the default package's
    tokenize,mwt,pos (pos.json, where the later fields are missing). No trailing newline, as str(doc) has none.
    """
    folder = out / "json"
    folder.mkdir(exist_ok=True)
    with open(folder / "inputs.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"text": JSON_TEXT}, f, indent=1, ensure_ascii=False)
    for name, processors in JSON_OUTPUTS.items():
        package = "default" if processors else name
        nlp = stanza.Pipeline(LANG, dir=models, package=package, processors=processors, download_method=None,
                              use_gpu=False, logging_level="WARN")
        with torch.no_grad():
            doc = nlp(JSON_TEXT)
        (folder / f"{name}.json").write_text(str(doc), encoding="utf-8", newline="\n")
        print(f"json/{name}.json: {len(doc.sentences)} sentences, {doc.num_words} words")


HELPER_FILES = ["pipeline.conllu", "validation_contractions.conllu", "validation_multilingual.conllu", "validation_nonbmp.conllu"]

# Word texts whose repr() is not just the text in single quotes: quotes of both kinds, a backslash, a no-break space and a
# zero-width space (not printable: \xa0, \u200b), a soft hyphen (\xad), an emoji and a combining accent (printable).
HELPER_WORDS = ["it's", 'say "hi"', "both '\"", "back\\slash", "no\u00a0break", "zero\u200bwidth", "soft\u00adhyphen", "\U0001F600",
                "e\u0301", "plain"]


# Language identification (#174, docs/langid-plan.md): short texts in many languages and scripts, and the edges Stanza
# handles oddly (whitespace, one character, digits, mixed scripts, a long text). The empty text crashes Stanza.
LANGID_TEXTS = [
    "This is an English sentence about the weather.",
    "Das ist ein deutscher Satz über das Wetter.",
    "Ceci est une phrase française sur la météo.",
    "Esta es una frase en español sobre el tiempo.",
    "Questa è una frase italiana sul tempo.",
    "Dit is een Nederlandse zin over het weer.",
    "Det här är en svensk mening om vädret.",
    "Dette er en dansk sætning om vejret.",
    "Tämä on suomenkielinen lause säästä.",
    "Ez egy magyar mondat az időjárásról.",
    "To jest polskie zdanie o pogodzie.",
    "Toto je česká věta o počasí.",
    "Aceasta este o propoziție românească despre vreme.",
    "Esta é uma frase em português sobre o tempo.",
    "Això és una frase catalana sobre el temps.",
    "Hau euskarazko esaldi bat da eguraldiari buruz.",
    "Is abairt Ghaeilge í seo faoin aimsir.",
    "Bu hava durumu hakkında Türkçe bir cümledir.",
    "Это русское предложение о погоде.",
    "Це українське речення про погоду.",
    "Това е българско изречение за времето.",
    "Ово је српска реченица о времену.",
    "Αυτή είναι μια ελληνική πρόταση για τον καιρό.",
    "זהו משפט בעברית על מזג האוויר.",
    "هذه جملة عربية عن الطقس.",
    "این یک جمله فارسی درباره آب و هوا است.",
    "यह मौसम के बारे में एक हिंदी वाक्य है।",
    "இது வானிலை பற்றிய ஒரு தமிழ் வாக்கியம்.",
    "これは天気についての日本語の文です。",
    "这是一个关于天气的中文句子。",
    "這是一個關於天氣的中文句子。",
    "이것은 날씨에 관한 한국어 문장입니다.",
    "Đây là một câu tiếng Việt về thời tiết.",
    "Ini adalah kalimat bahasa Indonesia tentang cuaca.",
    "Gallia est omnis divisa in partes tres.",
    "ἐν ἀρχῇ ἦν ὁ λόγος",
    "Hello",
    "Hallo Welt",
    "   ",
    "a",
    "12345",
    "Hello мир, wie geht's?",
    "I love 🍕 and 🍣!",
    "𝕳𝖊𝖑𝖑𝖔 𝖜𝖔𝖗𝖑𝖉",
    "Hello there.", "Hallo du da.",  # the same length: one batch (as "Hello" and "12345")
]


def langid_resources():
    """Stanza 1.15.0's resources.json entry for the langid model, so no resources.json with it has to be downloaded."""
    resources = Path(tempfile.mkdtemp()) / "resources.json"
    resources.write_text(json.dumps({"multilingual": {"langid": {"ud": {"md5": "84cd508b9702b31cc236ba2b6d94e41a"}},
                                                      "default_processors": {"langid": "ud"},
                                                      "packages": {"default": {"langid": "ud"}}}}), encoding="utf-8")
    return resources


def write_langid_golden(models, out):
    """
    langid/langid.json: LANGID_TEXTS plus validation.txt's first paragraph (a long text) through Stanza's langid
    processor, alone and as one bulk call, unrestricted and with langid_lang_subset en,de, and each text's summed scores
    (LangIDBiLSTM.forward) in float32 and float64.
    """
    from stanza.models.common.doc import Document
    folder = out / "langid"
    folder.mkdir(exist_ok=True)
    long_text = (out / "validation.txt").read_bytes().decode("utf-8").split("\n\n")[0].strip()
    texts = LANGID_TEXTS + [long_text]
    resources = langid_resources()
    def pipeline(**kwargs):
        return stanza.Pipeline(lang="multilingual", dir=models, processors="langid", download_method=None, use_gpu=False,
                               logging_level="WARN", resources_filepath=str(resources), **kwargs)
    plain, subset = pipeline(), pipeline(langid_lang_subset=["en", "de"])
    model = plain.processors["langid"]._model
    def ids(text):
        index = model.char_to_idx
        return torch.tensor([[index.get(c, index["UNK"]) for c in text]], dtype=torch.long)
    cases = []
    with torch.no_grad():
        model64 = copy.deepcopy(model).double()
        for text in texts:
            alone = plain.process(Document([], text=text)).lang
            cases.append({"text": text, "lang": alone, "lang_en_de": subset.process(Document([], text=text)).lang,
                          "scores": model(ids(text))[0].tolist(), "scores64": model64(ids(text))[0].tolist()})
        bulk = plain.bulk_process([Document([], text=t) for t in texts])
        bulk_subset = subset.bulk_process([Document([], text=t) for t in texts])
    assert [d.lang for d in bulk] == [c["lang"] for c in cases], "bulk changed a language"
    assert [d.lang for d in bulk_subset] == [c["lang_en_de"] for c in cases], "bulk changed a language"
    with open(folder / "langid.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"languages": list(model.idx_to_tag), "cases": cases}, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"langid/langid.json: {len(cases)} texts, {len({c['lang'] for c in cases})} languages")


# MultilingualPipeline (#174): English and German texts, mixed in order, plus texts the en,de subset decides (French,
# Dutch, a mixed one), so each language's documents go through one bulk call.
MULTILINGUAL_TEXTS = [
    "The weather in Berlin was lovely yesterday. We walked along the river.",
    "Das Wetter in Berlin war gestern schön. Wir sind am Fluss entlang gegangen.",
    "I don't think she's coming to the meeting.",
    "Ich glaube nicht, dass sie zum Treffen kommt. Sie ist im Urlaub.",
    "Apple is looking at buying a U.K. startup for $1 billion.",
    "Die Bundesregierung hat am Mittwoch neue Regeln beschlossen.",
    "Ceci est une phrase française sur la météo.",
    "Dit is een Nederlandse zin over het weer.",
    "Hello, wie geht's?",
]


def write_multilingual_golden(models, out):
    """
    langid/multilingual.json: MULTILINGUAL_TEXTS through Stanza's MultilingualPipeline(restrict=True) with English and
    German (each language's default package, all processors): each document's language and "{:C}" CoNLL-U.
    """
    nlp = stanza.MultilingualPipeline(model_dir=str(models), lang_configs={"en": {"logging_level": "WARN"}, "de": {"logging_level": "WARN"}},
                                      restrict=True, download_method=None, use_gpu=False,
                                      lang_id_config={"logging_level": "WARN", "resources_filepath": str(langid_resources())})
    with torch.no_grad():
        docs = nlp.process(list(MULTILINGUAL_TEXTS))
    result = [{"text": t, "lang": d.lang, "conllu": "{:C}".format(d)} for t, d in zip(MULTILINGUAL_TEXTS, docs)]
    with open(out / "langid" / "multilingual.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"documents": result}, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"langid/multilingual.json: {len(result)} documents, " + ", ".join(f"{d['lang']}" for d in result))


def write_helpers_golden(out):
    """
    helpers/: Stanza's document helpers on documents read with CoNLL.conll2doc (no models): num_tokens, num_words and,
    per sentence, dependencies_string(), tokens_string() and words_string(), for HELPER_FILES and crafted.conllu, a
    sentence of HELPER_WORDS (with heads and relations) plus one without heads.
    """
    from stanza.utils.conll import CoNLL
    folder = out / "helpers"
    folder.mkdir(exist_ok=True)
    lines = ["# sent_id = 0"]
    for i, word in enumerate(HELPER_WORDS, 1):
        lines.append(f"{i}\t{word}\t{word}\tX\t_\t_\t{0 if i == 1 else 1}\t{'root' if i == 1 else 'dep'}\t_\t_")
    lines += ["", "# sent_id = 1", "1-2\tdon't\t_\t_\t_\t_\t_\t_\t_\t_", "1\tdo\tdo\tAUX\t_\t_\t_\t_\t_\t_",
              "2\tn't\tnot\tPART\t_\t_\t_\t_\t_\t_", "", ""]
    (folder / "crafted.conllu").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    result = {}
    for path in [out / name for name in HELPER_FILES] + [folder / "crafted.conllu"]:
        doc = CoNLL.conll2doc(input_file=str(path))
        result[path.relative_to(out).as_posix()] = {
            "num_tokens": doc.num_tokens, "num_words": doc.num_words,
            "sentences": [{"dependencies": s.dependencies_string(), "tokens": s.tokens_string(), "words": s.words_string()}
                          for s in doc.sentences]}
    with open(folder / "helpers.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"helpers/helpers.json: {len(result)} documents")


WRONG_TREE = "(ROOT (X wrong))"


def write_process_document_golden(models, out):
    """
    process_document/: Pipeline.process(doc, processors=...) on a document read with CoNLL.conll2doc, as
    Pipeline.process_conllu does, running every loaded processor but tokenize and mwt. input.conllu is tokenize,mwt's
    output for corpus.txt with every annotation wrong (UPOS, XPOS, lemma, feats, head, deprel, NER, and a
    constituency comment), so the output shows what the processors overwrite: corpus.conllu for the default package,
    corpus.fast.conllu for default_fast, which has no parser and keeps the input's tree.
    """
    folder = out / "process_document"
    folder.mkdir(exist_ok=True)
    tok = stanza.Pipeline(LANG, dir=models, processors="tokenize,mwt", download_method=None, use_gpu=False, logging_level="WARN")
    doc = tok((out / "corpus.txt").read_text(encoding="utf-8"))
    for sentence in doc.sentences:
        for token in sentence.tokens:
            token.ner = "S-WRONG"
        for word in sentence.words:
            word.upos, word.xpos, word.lemma, word.feats, word.head, word.deprel = "X", "XX", "wrong", "Foo=Bar", 0, "dep"
    # The tree goes after sent_id, where a parser's comment goes (a sentiment comment set before would stay first).
    text = "{:C}\n".format(doc)
    text = "\n".join(line + (f"\n# constituency = {WRONG_TREE}" if line.startswith("# sent_id = ") else "")
                     for line in text.split("\n"))
    (folder / "input.conllu").write_text(text, encoding="utf-8", newline="\n")
    for package, suffix in PACKAGES.items():
        nlp = stanza.Pipeline(LANG, dir=models, package=package, download_method=None, use_gpu=False, logging_level="WARN")
        processors = [p for p in nlp.processors if p not in ("tokenize", "mwt")]
        with torch.no_grad():
            result = nlp.process(CoNLL.conll2doc(input_str=text), processors=processors)
        (folder / f"corpus{suffix}.conllu").write_text("{:C}\n".format(result), encoding="utf-8", newline="\n")
        print(f"process_document/corpus{suffix}.conllu: {','.join(processors)}, {len(result.sentences)} sentences")


# pretagged/<name>.conllu: (input, processors, package, fields removed from every word). Inputs are golden files:
# fast/ has default_fast's tags and lemmas (another tagger's, as a user's would be), process_document/input.conllu
# every annotation wrong. Run with Pipeline(processors=..., pretagged=True), which loads only those processors.
# "sentiment" drops the `# sentiment` comments: Stanza keeps comments in input order, so a new `# constituency` would
# follow them, while Conllu.Write always writes it first (CLAUDE.md, Process(Document)).
PRETAGGED_CASES = {
    "depparse": ("fast/corpus.conllu", "depparse", "default", ()),
    "constituency": ("fast/corpus.conllu", "constituency", "default", ("sentiment",)),
    "lemma": ("process_document/input.conllu", "lemma", "default", ()),
    "all": ("process_document/input.conllu", "lemma,depparse,constituency", "default", ()),
    "all.fast": ("process_document/input.conllu", "lemma,depparse", "default_fast", ()),
    "validation": ("fast/validation.conllu", "lemma,depparse,constituency", "default", ("sentiment",)),
    "depparse_nolemma": ("fast/corpus.conllu", "depparse", "default", ("lemma",)),
    "depparse_noxpos": ("fast/corpus.conllu", "depparse", "default", ("xpos",)),
    "lemma_noupos": ("fast/corpus.conllu", "lemma", "default", ("upos", "lemma")),
}


def write_pretagged_golden(models, out):
    """
    pretagged/: Stanza's pretagged=True pipelines (lemma, depparse, constituency without the processors they require)
    on documents read with CoNLL.conll2doc that already carry tags, run by nlp.process(doc). PRETAGGED_CASES lists them.
    """
    folder = out / "pretagged"
    folder.mkdir(exist_ok=True)
    pipelines = {}
    for name, (source, processors, package, strip) in PRETAGGED_CASES.items():
        key = (processors, package)
        if key not in pipelines:
            pipelines[key] = stanza.Pipeline(LANG, dir=models, package=package, processors=processors, pretagged=True,
                                             download_method=None, use_gpu=False, logging_level="WARN")
        nlp = pipelines[key]
        assert set(nlp.processors) == set(processors.split(",")), (name, list(nlp.processors))
        text = (out / source).read_text(encoding="utf-8")
        if "sentiment" in strip:
            text = "".join(line for line in text.splitlines(keepends=True) if not line.startswith("# sentiment = "))
        doc = CoNLL.conll2doc(input_str=text)
        for sentence in doc.sentences:
            for word in sentence.words:
                for field in strip:
                    if field != "sentiment":
                        setattr(word, field, None)
        with torch.no_grad():
            result = nlp.process(doc)
        (folder / f"{name}.conllu").write_text("{:C}\n".format(result), encoding="utf-8", newline="\n")
        print(f"pretagged/{name}.conllu: {processors} ({package}) on {source}, {len(result.sentences)} sentences")


# --lang de: the German word lists, replacing the English ones above. The German texts are hand-written for
# tests/golden/de (no copyrighted text); the MST cases and the pt/ fixtures are language-independent, so English only.
GERMAN = {
    # German default_accurate's transformer (docs/german-accurate-plan.md): a vocab.txt tokenizer, lowercase, accents kept.
    "TRANSFORMER": "german-nlp-group/electra-base-german-uncased",
    # Dictionary contractions, their capitals, forms the dictionary lacks (the seq2seq decides them), unknown characters.
    "MWT_WORDS": [
        "zum", "Zum", "ZUM", "zur", "Zur", "ZUR", "im", "Im", "IM", "am", "Am", "AM", "ins", "Ins", "beim", "Beim", "vom",
        "Vom", "übers", "Übers", "aufs", "ans", "ums", "unterm", "hinters", "fürs", "Fürs", "durchs", "vorm", "überm",
        "hinterm", "untern", "zu", "Haus", "Straße", "zum.", "çum", "zům", "東京zum", "😀ins", "x",
    ],
    "LEMMA_WORDS": [
        ("ging", "VERB"), ("ging", "NOUN"), ("war", "AUX"), ("Kinder", "NOUN"), ("besser", "ADJ"), ("Häuser", "NOUN"),
        ("Straßen", "NOUN"), ("STRASSEN", "NOUN"), ("Fußballweltmeisterschaften", "NOUN"), ("googelte", "VERB"),
        ("gefrühstückt", "VERB"), ("zerfledderten", "ADJ"), ("Müllers", "PROPN"), ("Zürichs", "PROPN"),
        ("naïven", "ADJ"), ("東京", "PROPN"), ("𝒳e", "NOUN"), ("😀😀s", "SYM"), ("schönsten", "ADJ"),
        ("Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetzes", "NOUN"), ("_", "PUNCT"), ("<UNK>", "X"),
        ("Schmetterlingen", "NOUN"), ("ÄRZTE", "NOUN"), ("ﬁnanzierte", "VERB"), ("Bäume", "NOUN"), ("GROẞE", "ADJ"),
    ],
    "PRETOKENIZED_CASES": {
        "simple": [["Hallo", "Welt", "."], ["Tschüss", "."]],
        "single": [["Halt"]],
        "contractions": [["Ich", "gehe", "zum", "Bahnhof", "."], ["Wir", "waren", "im", "Kino", "!"], ["Er", "geht", "zu", "dem", "Haus", "."]],
        "punctuation": [["„", "Hallo", "“", ",", "sagte", "sie", "..."], ["Wirklich", "?!"], ["A", ":", "b", ";", "c", "/", "d"]],
        "spaces": [["New York", "ist", "groß", "."], ["Ich", "wohne", "in", "Bad Homburg", "."]],
        "nonbmp": [["Ich", "liebe", "😀", "!"], ["𝒳", "markiert", "die", "Stelle", "."]],
    },
    "BULK_TEXTS": ["Hallo zusammen.", "", "   ", "\n\n", "Kurzer Satz. Zwei Sätze!", " gepolsterter Text \n", "ok",
                   "Ich gehe zum Arzt, ins Kino und beim Bäcker vorbei.",
                   "Angela Merkel wurde in Hamburg geboren.\n\nSie war ab 2005 Bundeskanzlerin.", "?"],
    # German's tokenizer has max_seqlen 400.
    "JSON_TEXT": ('  „Hallo“, sagte sie. Wir gehen zum.\tAngela Merkel wohnt im Haus in Berlin\u00a0Mitte, siehe C:\\temp\\x!\r\n\r\n'
                  'Das "ist"\x1cgut (am Ende)  '),
    "FUZZ_TEXTS": [
        "Go to example.com.\U0001F600 now",
        "Open https://a.\U0001F600 please",
        "the end.\U0001F600 Next www.a.\U0001F600 Then",
    ],
    "MAX_SEQLEN": 400,
    "LONG_TOKEN_TEXT": "\n\n".join([
        f"Die Studie nannte {long_url(500)} als Quelle. Später wurde sie gelöscht.",
        f"Genau {long_url(399)} und {long_url(400)} und {long_url(401)} wurden getestet.",
        f"Ein langes Wort: {'Donaudampfschiff' * 30} ist keine URL.",
        f"Die längste, {long_url(1200)} , endet hier.",
        "Danach kommt ein kurzer Satz.",
    ]) + "\n",
}


TRANSFORMER = "google/electra-large-discriminator"

# Words that exercise the tokenizer's normalizer, pre-tokenizer and WordPiece (docs/transformer-plan.md): accents
# composed and decomposed, İ, soft hyphens and zero-width characters (alone they vanish: fix_blank_tokens), controls,
# private use and unassigned characters, non-BMP letters and emoji, CJK, Hangul, other scripts, Unicode punctuation,
# words of 100 and 101 characters (max_input_chars_per_word), words with spaces inside, the special tokens' texts, a
# sentence of blank words only, and one of over 512 pieces.
TRANSFORMER_TOKENIZER_CASES = [
    ["İstanbul", "naïve", "café", "café", "Ångström", "ﬁnance", "Straße", "ΣΟΦΙΑΣ", "Ὀδυσσεύς", "Москва", "ǅungla"],
    ["co­op", "­", "a​b", "​", "‍", "﻿", "x\u0007y", "\u0085", "", "\U000e0001", "͸", "ok"],
    ["𝒳", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "👍🏽", "🇸🇪", "東京", "日本語テキスト", "한국어", "ﾊﾝｶｸ", "العربية", "עברית", "हिन्दी", "ไทย"],
    ["—", "…", "«", "»", "¿", "¡", "§", "¶", "$", "^", "`", "~", "°", "™", "€", "‰", "‽", "⁂", "「", "」", "_", "'s", "n't"],
    ["a" * 100, "b" * 101, "x" * 99 + "é", "supercalifragilisticexpialidocious", "antidisestablishmentarianism"],
    ["New York", "x y", "a\tb", "tab\t", "　", "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]", "##ing"],
    ["???", "?!?", "!!!", "!?", "??", "...", "--", "'''", '""', "(", ")", "[", "]", "{", "}"],
    ["\u00ad", "\u200b", "\ufe0f"],  # every word blank: all become "-"
    # Over the model's 512 positions (the tokenizer doesn't truncate; Stanza's build_cloned_features windows it).
    ("Unbelievably, antidisestablishmentarianism reappeared " * 60).split(),
]


# roberta-base's byte-level BPE (#172; ner conll03_electra-large reads it): Stanza's add_prefix_space=True before every
# piece of a word that doesn't start with U+0020, GPT-2's split pattern (contractions in lowercase only, letter, number and
# other runs, whitespace runs giving up their last character before a non-space), the added tokens (<mask> takes the
# whitespace before it), and the whitespace classes: tab, NBSP, NEL, ideographic space, line separator, the C0 separators.
ROBERTA_TOKENIZER_CASES = [
    ["don't", "it's", "IT'S", "we're", "they've", "I'm", "you'll", "he'd", "''s", "?'s", "x's", "'"],
    ["\ttab", "a\tb", "\t", "x  y", "   leading", "trailing  ", " ", "a\u00a0b", "\u0085x", "\u3000", "a\u2028b", "a\u001cb", "x\r\ny"],
    ["<s>", "</s>", "<pad>", "<unk>", "<mask>", "a <mask> b", "x<mask>", "<mask>y", " <mask>", "<mask> ", "<s><s>", "<<s>>"],
    ["abc123def", "3.14", "123456", "٣٤", "Ⅻ", "½", "x²", "e\u0301", "é", "Ö", "ﬁ", "Ⅸvii"],
    ["𝒳", "👍🏽", "東京", "a👍b", "🇸🇪", "\U000e0001", "\u200b", "\u00ad", "\ufeff", "", "a" * 300],
]


# German's transformer keeps accents (strip_accents false), so umlauts, ß, decomposed and precomposed letters and İ reach
# its lowercasing as written; on top of TRANSFORMER_TOKENIZER_CASES and the German golden words.
GERMAN_TRANSFORMER_TOKENIZER_CASES = [
    ["Größe", "GRÖSSE", "größe", "Straße", "STRASSE", "Maß", "ẞ", "Äpfel", "Öl", "Übung", "ÄÖÜ", "äöü"],
    ["Mädchen", "Ma\u0308dchen", "Fu\u0308ße", "Cafe\u0301", "Café", "naïve", "İstanbul", "IJssel", "Œuvre", "Ærø"],
    ["z.B.", "u.a.", "d.h.", "usw.", "Nr.", "§", "€", "„Hallo“", "‚ja‘", "»Buch«", "Donaudampfschifffahrtsgesellschaft"],
]


# Sentences whose hidden states the golden data keeps (TRANSFORMER_HIDDEN): the first golden sentences, and short ones on
# which ELECTRA-large is badly conditioned in float32 (torch's float32 is up to 7.6e-3 from float64 there,
# docs/transformer-plan.md), each with the bound its values must meet; the others meet 1e-4 of float32 or float64.
TRANSFORMER_ILL_CONDITIONED = {"GO TEAM GO": 1e-2, "So does this one .": 1e-2, "Proof .": 1e-2}
TRANSFORMER_LONG_STRIDE = 10  # the long sentence keeps every 10th word's vector


def write_transformer_hidden_golden(models, out, tokenizer, transformer=TRANSFORMER, file_suffix=""):
    """
    transformer/hidden<file_suffix>.safetensors + .json: Stanza's extract_bert_embeddings on a few sentences, float32 and
    float64. ELECTRA-large's mix is depparse/combined_electra-large's bert_layer_mix; roberta-base's a fixed one of 4 layers
    (no roberta-base model has a mix of its own), and its hard sentences are ELECTRA's ill-conditioned ones too.
    """
    from stanza.models.common.bert_embedding import extract_bert_embeddings, load_bert

    golden = Path(out)
    model, _ = load_bert(str(Path(models) / "en" / "transformer" / transformer), local_files_only=True)
    model.eval()
    words = []
    for path in [golden / "pipeline.conllu"] + sorted(golden.glob("validation*.conllu")):
        for sent in CoNLL.conll2doc(input_file=str(path)).sentences:
            text = [w.text for w in sent.words]
            if (path.name == "pipeline.conllu" and len(words) < INTERMEDIATE_SENTENCES) or " ".join(text) in TRANSFORMER_ILL_CONDITIONED:
                if text not in words:
                    words.append(text)
    assert len(words) == INTERMEDIATE_SENTENCES + len(TRANSFORMER_ILL_CONDITIONED), [" ".join(w) for w in words]
    long_words = TRANSFORMER_TOKENIZER_CASES[-1]
    if transformer == TRANSFORMER:
        depparse = torch.load(str(Path(models) / "en" / "depparse" / "combined_electra-large.pt"), map_location="cpu", weights_only=True)
        mix_weight, mix_name = depparse["model"]["bert_layer_mix.weight"], "depparse/combined_electra-large bert_layer_mix.weight"  # [1, 4]
    else:
        mix_weight, mix_name = torch.tensor([[0.4, -0.3, 0.2, 0.1]]), "fixed"
    bounds = TRANSFORMER_ILL_CONDITIONED if transformer == TRANSFORMER else {}

    tensors, index = {}, []
    with torch.no_grad():
        for dtype, suffix in ((torch.float32, ""), (torch.float64, ".f64")):
            m = model.to(dtype)
            mix = torch.nn.Linear(4, 1, bias=False).to(dtype)
            mix.weight.copy_(mix_weight.to(dtype))
            # One batch, as a processor would send it: padded to the longest, the long sentence's windows included.
            data = words + [long_words]
            legacy = extract_bert_embeddings(transformer, tokenizer, m, data, "cpu", keep_endpoints=False)
            mixed = extract_bert_embeddings(transformer, tokenizer, m, data, "cpu", keep_endpoints=True, bert_layer_mix=mix)
            for i, w in enumerate(words):  # float64 stored rounded to float32: 6e-8 relative, far under the tolerances
                tensors[f"s{i}.legacy{suffix}"] = legacy[i].float().numpy()
                tensors[f"s{i}.mix{suffix}"] = mixed[i].float().numpy()
            tensors[f"long.legacy{suffix}"] = legacy[-1][::TRANSFORMER_LONG_STRIDE].float().numpy()
        model.float()
    for i, w in enumerate(words):
        index.append({"sentence": i, "words": w, "bound": bounds.get(" ".join(w))})
    write_safetensors(tensors, golden / "transformer" / f"hidden{file_suffix}.safetensors", {"stanza": stanza.__version__})
    with open(golden / "transformer" / f"hidden{file_suffix}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"transformer": transformer, "stanza": stanza.__version__, "sentences": index, "long": long_words,
                   "long_stride": TRANSFORMER_LONG_STRIDE, "mix": mix_name,
                   "mix_weight": mix_weight[0].tolist()}, f, indent=1, ensure_ascii=False)
    print(f"transformer/hidden{file_suffix}.safetensors: {', '.join(f'{k} {list(v.shape)}' for k, v in tensors.items() if not k.endswith('.f64'))}")


ROBERTA = "roberta-base"

# German's dbmdz/bert-base-german-cased (ner germeval2014_dbmdz-bert-german-cased, sentiment sb10k_dbmdz-bert-german-cased): a
# cased vocab.txt WordPiece (do_lower_case false, so strip_accents follows it: accents kept), on top of the German cases
DBMDZ = "dbmdz/bert-base-german-cased"
DBMDZ_TOKENIZER_CASES = [
    ["Über", "ÜBER", "über", "Ärger", "ÄRGER", "Öl", "Straße", "STRASSE", "ẞ"],
    ["café", "Café", "naïve", "Noël", "Ångström", "İstanbul", "ǅ"],
    ["Ue\u0308bel", "Cafe\u0301", "ﬁnden", "Ⅻ", "½"],
    ["McDonald's", "iPhone", "E-Mail", "GmbH", "BMW-Werk", "3.500,00", "€", "§§"],
]


def write_transformer_golden(models, out, transformer=None):
    """
    transformer/tokenizer.json (ELECTRA-large) or tokenizer_roberta-base.json: Stanza's transformer input stage
    (extract_base_embeddings up to the model call); then the hidden states.
    """
    from stanza.models.common.bert_embedding import convert_to_position_list, fix_blank_tokens, load_tokenizer
    from stanza.models.common.utils import EXCLAM_RE, QUESTION_RE
    import tokenizers
    import transformers

    transformer = transformer or TRANSFORMER  # the language's (GERMAN replaces the global), not the one bound at definition
    tokenizer = load_tokenizer(str(Path(models) / LANG / "transformer" / transformer), local_files_only=True)
    suffix = "" if transformer == TRANSFORMER else "_" + transformer.replace("/", "_")
    golden = Path(out)
    cases, seen = [], set()

    def add(words, source):
        key = tuple(words)
        if not words or key in seen:
            return
        seen.add(key)
        data = [list(words)]
        tok = tokenizer(data, is_split_into_words=True)
        positions = convert_to_position_list(data[0], tok.word_ids(batch_index=0))
        fixed = None
        if any(x is None for x in positions):
            data = fix_blank_tokens(tokenizer, data)
            fixed = data[0]
            tok = tokenizer(data, is_split_into_words=True)
            positions = convert_to_position_list(data[0], tok.word_ids(batch_index=0))
        case = {"words": list(words), "ids": tok["input_ids"][0], "word_ids": [-1 if w is None else w for w in tok.word_ids(batch_index=0)]}
        if positions != list(range(len(words) + 2)):  # the usual case: one piece per word
            case["positions"] = positions
        if fixed is not None:
            case["fixed"] = fixed
        if source:
            case["source"] = source
        cases.append(case)

    for path in [golden / "pipeline.conllu"] + sorted(golden.glob("validation*.conllu")):
        doc = CoNLL.conll2doc(input_file=str(path))
        for sent in doc.sentences:
            words = [w.text for w in sent.words]
            add(words, None)  # the golden words, then pos/depparse's simplify_punct'ed words, then ner/sentiment's tokens
            add([EXCLAM_RE.sub("!", QUESTION_RE.sub("?", w)) for w in words], None)
            add([t.text for t in sent.tokens], None)
    for words in TRANSFORMER_TOKENIZER_CASES:
        add(words, "TRANSFORMER_TOKENIZER_CASES")
    for words in GERMAN_TRANSFORMER_TOKENIZER_CASES if LANG == "de" else []:
        add(words, "GERMAN_TRANSFORMER_TOKENIZER_CASES")
    for words in ROBERTA_TOKENIZER_CASES if transformer == ROBERTA else []:
        add(words, "ROBERTA_TOKENIZER_CASES")
    for words in DBMDZ_TOKENIZER_CASES if transformer == DBMDZ else []:
        add(words, "DBMDZ_TOKENIZER_CASES")

    (golden / "transformer").mkdir(exist_ok=True)
    with open(golden / "transformer" / f"tokenizer{suffix}.json", "w", encoding="utf-8", newline="\n") as f:
        meta = {"transformer": transformer, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                "stanza": stanza.__version__}
        f.write(json.dumps(meta, ensure_ascii=False)[:-1] + ', "cases": [\n')
        f.write(",\n".join(json.dumps(c, ensure_ascii=False, separators=(",", ":")) for c in cases))
        f.write("\n]}\n")
    fixed = sum("fixed" in c for c in cases)
    print(f"transformer/tokenizer{suffix}.json: {len(cases)} cases, {sum(len(c['ids']) for c in cases)} ids, {fixed} with fix_blank_tokens")
    if LANG == "en":  # German's hidden states come with its models (#170's second PR)
        write_transformer_hidden_golden(models, out, tokenizer, transformer, suffix)


def main():
    global LANG
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="en", choices=["en", "de"], help="de writes the German golden data (default --out tests/golden/de)")
    p.add_argument("--models", default="models/stanza")
    p.add_argument("--out", default=None, help="default tests/golden (en) or tests/golden/<lang>")
    p.add_argument("--pt-only", action="store_true", help="only regenerate the pt/ loader fixtures")
    p.add_argument("--depparse-only", action="store_true", help="only regenerate depparse/")
    p.add_argument("--lemma-only", action="store_true", help="only regenerate the lemma/ golden data")
    p.add_argument("--ner-only", action="store_true", help="only regenerate ner/")
    p.add_argument("--sentiment-only", action="store_true", help="only regenerate sentiment/")
    p.add_argument("--fast-only", action="store_true", help="only regenerate fast/ (package default_fast)")
    p.add_argument("--pretokenized-only", action="store_true", help="only regenerate pretokenized/")
    p.add_argument("--bulk-only", action="store_true", help="only regenerate bulk/")
    p.add_argument("--no-ssplit-only", action="store_true", help="only regenerate no_ssplit/ (tokenize_no_ssplit)")
    p.add_argument("--fuzz-only", action="store_true", help="only regenerate fuzz.json")
    p.add_argument("--long-token-only", action="store_true", help="only regenerate long_token.*")
    p.add_argument("--upos64-only", action="store_true", help="only add the float64 UPOS logits to the intermediates")
    p.add_argument("--json-only", action="store_true", help="only regenerate json/ (str(doc))")
    p.add_argument("--langid-only", action="store_true", help="only regenerate langid/ (the language identifier and MultilingualPipeline; needs <models>/multilingual, en and de)")
    p.add_argument("--helpers-only", action="store_true", help="only regenerate helpers/ (document helpers, no models)")
    p.add_argument("--stream-only", action="store_true", help="only regenerate stream/ (Pipeline.stream)")
    p.add_argument("--postprocess-only", action="store_true", help="only regenerate postprocess/ (tokenize_postprocessor)")
    p.add_argument("--process-document-only", action="store_true", help="only regenerate process_document/")
    p.add_argument("--pretagged-only", action="store_true", help="only regenerate pretagged/ (pretagged=True pipelines)")
    p.add_argument("--treebank-only", action="store_true", help="only regenerate the treebank packages' folders (ewt/, ..., de/gsd/, de/hdt/)")
    p.add_argument("--packages", default=None, help="with --treebank-only: only these treebank packages (comma-separated)")
    p.add_argument("--models-only", action="store_true", help="only regenerate models/ (MODEL_CHOICES: single models chosen per processor)")
    p.add_argument("--transformer-only", action="store_true", help="only regenerate transformer/ (needs tools/requirements-transformer.txt)")
    p.add_argument("--options-only", action="store_true", help="only regenerate options/ (OPTION_CASES: Stanza options with a PipelineOptions property)")
    p.add_argument("--choices", default=None, help="with --models-only: only these MODEL_CHOICES names (comma-separated)")
    args = p.parse_args()
    LANG = args.lang
    out = Path(args.out or ("tests/golden" if LANG == "en" else f"tests/golden/{LANG}"))
    if LANG == "de":
        globals().update(GERMAN)
    if args.upos64_only:
        write_upos64_golden(args.models, out)
        return
    if args.transformer_only:
        write_transformer_golden(args.models, out)
        write_transformer_golden(args.models, out, ROBERTA if LANG == "en" else DBMDZ)
        return
    if args.long_token_only:
        write_long_token_golden(args.models, out)
        return
    if args.fuzz_only:
        write_fuzz_golden(args.models, out)
        return
    if args.treebank_only:
        write_treebank_golden(args.models, out, args.packages and args.packages.split(","))
        return
    if args.options_only:
        write_options_golden(args.models, out)
        return
    if args.models_only:
        write_models_golden(args.models, out, args.choices and args.choices.split(","))
        return
    if args.postprocess_only:
        write_postprocess_golden(args.models, out)
        return
    if args.stream_only:
        write_stream_golden(args.models, out)
        return
    if args.langid_only:
        write_langid_golden(args.models, out)
        write_multilingual_golden(args.models, out)
        return
    if args.helpers_only:
        write_helpers_golden(out)
        return
    if args.json_only:
        write_json_golden(args.models, out)
        return
    if args.process_document_only:
        write_process_document_golden(args.models, out)
        return
    if args.pretagged_only:
        write_pretagged_golden(args.models, out)
        return
    if args.pretokenized_only:
        write_pretokenized_golden(args.models, out)
        return
    if args.bulk_only:
        write_bulk_golden(args.models, out)
        return
    if args.no_ssplit_only:
        write_no_ssplit_golden(args.models, out)
        return
    if args.fast_only:
        write_fast_golden(args.models, out)
        return
    if args.depparse_only:
        torch.manual_seed(0)
        write_depparse_golden(args.models, out)
        return
    if args.lemma_only:
        write_lemma_golden(args.models, out)
        return
    if args.ner_only:
        write_ner_golden(args.models, out)
        return
    if args.sentiment_only:
        write_sentiment_golden(args.models, out)
        return
    if LANG == "en":
        write_pt_fixtures(out)
    if args.pt_only:
        return

    torch.manual_seed(0)
    nlp = stanza.Pipeline(LANG, dir=args.models, processors=PROCESSORS, download_method=None,
                          use_gpu=False, logging_level="WARN")
    text = (out / "corpus.txt").read_text(encoding="utf-8")

    with torch.no_grad():
        doc = nlp(text)
    (out / "pipeline.conllu").write_text("{:C}\n".format(doc), encoding="utf-8", newline="\n")
    print(f"pipeline.conllu: {len(doc.sentences)} sentences")

    # Read as bytes so CR/CRLF reach Stanza unchanged, as File.ReadAllText passes them in C#.
    for path in sorted(out.glob("validation*.txt")):
        with torch.no_grad():
            vdoc = nlp(path.read_bytes().decode("utf-8"))
        conllu = path.with_suffix(".conllu")
        conllu.write_text("{:C}\n".format(vdoc), encoding="utf-8", newline="\n")
        print(f"{conllu.name}: {len(vdoc.sentences)} sentences, {vdoc.num_words} words")

    tok_model = nlp.processors["tokenize"].trainer.model
    pos_model = nlp.processors["pos"].trainer.model
    con_model = nlp.processors["constituency"]._model

    tensors, index = {}, []
    for i, sent in enumerate(doc.sentences[:INTERMEDIATE_SENTENCES]):
        words = [w.text for w in sent.words]
        entry = {"sentence": i, "text": sent.text, "words": words}

        with torch.no_grad():
            # Tokenizer: run on the sentence text alone so the input is exactly `sent.text`.
            cap, h = capture_module(tok_model)
            nlp.processors["tokenize"].process(sent.text)
            h.remove()
            tensors[f"s{i}.tokenize.pred"] = torch.cat([o[0] for o in cap.outputs]).numpy()

            fwd = pos_model.charmodel_forward.build_char_representation([words])[0]
            bwd = pos_model.charmodel_backward.build_char_representation([words])[0]
            tensors[f"s{i}.charlm_forward"] = fwd.cpu().numpy()
            tensors[f"s{i}.charlm_backward"] = bwd.cpu().numpy()

            # Tagger and parser: feed a one-sentence document built from the golden words.
            single = stanza.Document([[{"id": j + 1, "text": w} for j, w in enumerate(words)]])
            cap, h = capture_module(pos_model.upos_clf)
            nlp.processors["pos"].process(single)
            h.remove()
            tensors[f"s{i}.pos.upos_logits"] = cap.outputs[0].numpy()
            entry["xpos"] = [w.xpos for w in single.sentences[0].words]

            cap, h = capture_module(con_model.output_layers[-1])
            nlp.processors["constituency"].process(single)
            h.remove()
            tensors[f"s{i}.constituency.scores"] = torch.cat(cap.outputs).numpy()
            entry["tree"] = "{}".format(single.sentences[0].constituency)
        index.append(entry)

    write_depparse_golden(args.models, out)
    write_tokenize_stress(args.models, text, out)
    write_mwt_golden(args.models, out)
    write_lemma_golden(args.models, out)
    write_ner_golden(args.models, out)
    write_sentiment_golden(args.models, out)
    write_fast_golden(args.models, out)
    write_pretokenized_golden(args.models, out)
    write_bulk_golden(args.models, out)
    write_no_ssplit_golden(args.models, out)
    write_long_token_golden(args.models, out)
    write_fuzz_golden(args.models, out)
    write_json_golden(args.models, out)
    write_process_document_golden(args.models, out)
    if LANG == "en":
        write_pretagged_golden(args.models, out)
    write_treebank_golden(args.models, out)
    write_models_golden(args.models, out)

    for i, logits in enumerate(upos_logits64(args.models, [e["words"] for e in index])):
        tensors[f"s{i}.pos.upos_logits64"] = logits
    write_safetensors(tensors, out / "intermediates.safetensors", {"stanza": stanza.__version__})
    with open(out / "intermediates.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"stanza": stanza.__version__, "sentences": index}, f, indent=1, ensure_ascii=False)
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)}")


if __name__ == "__main__":
    main()
