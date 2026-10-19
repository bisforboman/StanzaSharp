#!/usr/bin/env bash
# The model sets CI caches (ci.yml), each defined once here as lines of `download` arguments, so a cache's key, its
# paths and its download on a miss can't drift apart (owner's decision, 2026-10-16):
#
#   tools/ci-models.sh download GROUP   download the group's models (on a cache miss)
#   tools/ci-models.sh files GROUP      the group's files and MD5s (`download --list`), sorted
#   tools/ci-models.sh transformer      downloads the transformer models (Pipeline.Transformers: default_accurate's, PR 2 of
#                                       docs/transformer-plan.md) and google/electra-large-discriminator's files (into
#                                       models/stanza/en/transformer/), and German default_accurate's (its four transformer models
#                                       and german-nlp-group/electra-base-german-uncased, models/stanza/de/transformer/), with
#                                       retries; never cached (owner's decisions, 2026-10-17 and 2026-10-18)
#   tools/ci-models.sh langid           downloads the language identifier (models/stanza/multilingual/langid/ud.pt, 9 MB), with
#                                       retries; never cached
#   tools/ci-models.sh keys             writes en, en_treebanks, de, treebank_choices, en_treebanks_paths and treebank_choices_paths
#                                       to $GITHUB_OUTPUT
#
# Groups: en (default + default_fast: models-en and models-pt-<os>), en-treebanks (the English treebank packages and
# Pipeline.Alternatives' models, without en's files: models-en-treebanks), de (everything German: models-de-treebanks),
# en-treebank-choices and de-treebank-choices (Pipeline.Alternatives' treebank models: the _nocharlm taggers and parsers,
# the _charlm tokenizers and lemmatizers; without the other groups' files: together models-treebank-choices, owner's
# decision, 2026-10-16, so their 1.3 GB rotate no other cache).
# A cache's key is a hash of its group's file list, so a new or changed MD5 rotates only the caches holding that file.
# STANZASHARP_CLI is the command that runs samples/StanzaSharp.Cli (default: the Release build, without building).
set -euo pipefail

read -r -a cli <<< "${STANZASHARP_CLI:-dotnet run --project samples/StanzaSharp.Cli -c Release --no-build --}"

group() {
  case $1 in
    en)
      echo "models/stanza/en"
      echo "models/stanza/en --package default_fast"
      ;;
    en-treebanks)
      for package in ewt gum lines partut atis childes eslspok craft genia mimic; do
        echo "models/stanza/en --package $package"
      done
      # Pipeline.Alternatives (PipelineOptions.Models, ModelChoiceTests)
      echo "models/stanza/en --processors tokenize,mwt,pos,constituency --model constituency=ptb3_charlm"
      for ner in ontonotes_charlm ontonotes_nocharlm combined_charlm conll03_charlm radiology \
        anatem bc4chemd bc5cdr bionlp13cg jnlpba linnaeus ncbi_disease s800; do
        echo "models/stanza/en --processors tokenize,ner --model ner=$ner"
      done
      echo "models/stanza/en --package mimic --model ner=i2b2"
      ;;
    en-treebank-choices)
      for package in ewt gum lines partut atis childes eslspok; do
        echo "models/stanza/en --package $package --model pos=${package}_nocharlm,depparse=${package}_nocharlm"
      done
      echo "models/stanza/en --package mimic --model depparse=mimic_nocharlm"
      for package in ewt gum lines partut atis childes; do
        echo "models/stanza/en --package $package --model tokenize=${package}_charlm,lemma=${package}_charlm"
      done
      echo "models/stanza/en --package eslspok --model tokenize=eslspok_charlm"
      echo "models/stanza/en --processors tokenize,mwt --model tokenize=combined_charlm"
      echo "models/stanza/en --processors tokenize,mwt,pos,lemma --model lemma=combined_charlm"
      ;;
    de-treebank-choices)
      for package in gsd hdt; do
        echo "models/stanza/de --lang de --package $package --model pos=${package}_nocharlm,depparse=${package}_nocharlm"
      done
      for package in combined gsd hdt; do
        echo "models/stanza/de --lang de --package $package --model tokenize=${package}_charlm,lemma=${package}_charlm"
      done
      ;;
    de)
      for package in default default_fast gsd hdt; do
        echo "models/stanza/de --lang de --package $package"
      done
      echo "models/stanza/de --lang de --processors tokenize,ner --model ner=conll03"
      ;;
    *)
      echo "unknown group: $1" >&2
      exit 2
      ;;
  esac
}

listed() {
  for g in "$@"; do group "$g"; done | while read -r -a args; do "${cli[@]}" download "${args[@]}" --list < /dev/null; done | tr -d '\r' | LC_ALL=C sort -u
}

files() {
  case $1 in
    en-treebanks) LC_ALL=C comm -23 <(listed en-treebanks) <(listed en) ;; # their own files only: en's are in models-en
    en-treebank-choices) LC_ALL=C comm -23 <(listed en-treebank-choices) <(listed en en-treebanks | LC_ALL=C sort -u) ;;
    de-treebank-choices) LC_ALL=C comm -23 <(listed de-treebank-choices) <(listed de) ;;
    *) listed "$1" ;;
  esac
}

hash() {
  if command -v sha256sum > /dev/null; then sha256sum; else shasum -a 256; fi | cut -c1-20
}

case ${1:-} in
  download)
    group "$2" | while read -r -a args; do "${cli[@]}" download "${args[@]}" < /dev/null; done
    ;;
  files)
    files "$2"
    ;;
  transformer)
    # the default_accurate package: its five transformer models, the transformer, and its _charlm tokenizer and lemmatizer
    # (also in the treebank-choices cache; files that match are kept); the treebanks' and ontonotes' (#168: their packages'
    # other files are in the en-treebanks cache), constituency ptb3_electra-large (#171), roberta-base (#172)
    # and coref's FacebookAI/xlm-roberta-large (2.24 GB; docs/coref-plan.md PR a: no model reads it yet)
    # and German's (#170: four transformer models, german-nlp-group's ELECTRA; the rest is in the de caches; gsd's, hdt's and
    # germeval2014's on it, their packages' other files from the de cache; dbmdz/bert-base-german-cased's ner and sentiment)
    transformers() {
      "${cli[@]}" download models/stanza/en --package default_accurate < /dev/null || return 1
      for package in ewt gum lines partut atis childes eslspok; do
        "${cli[@]}" download models/stanza/en --package "$package" \
          --model "pos=${package}_electra-large,depparse=${package}_electra-large" < /dev/null || return 1
      done
      "${cli[@]}" download models/stanza/en --processors tokenize,mwt,pos,constituency --model constituency=ptb3_electra-large < /dev/null \
        || return 1
      "${cli[@]}" download models/stanza/en --processors tokenize,ner --model ner=conll03_electra-large < /dev/null || return 1
      "${cli[@]}" download-transformer models/stanza/en FacebookAI/xlm-roberta-large < /dev/null || return 1
      "${cli[@]}" download models/stanza/en --processors tokenize,ner --model ner=ontonotes_electra-large < /dev/null || return 1
      "${cli[@]}" download models/stanza/de --lang de --package default_accurate < /dev/null || return 1
      for package in gsd hdt; do
        "${cli[@]}" download models/stanza/de --lang de --package "$package" \
          --model "pos=${package}_german-nlp-electra,depparse=${package}_german-nlp-electra" < /dev/null || return 1
      done
      "${cli[@]}" download models/stanza/de --lang de --processors tokenize,ner --model ner=germeval2014_german-nlp-electra < /dev/null || return 1
      "${cli[@]}" download models/stanza/de --lang de --processors tokenize,ner,sentiment \
        --model ner=germeval2014_dbmdz-bert-german-cased,sentiment=sb10k_dbmdz-bert-german-cased < /dev/null
    }
    for attempt in 1 2 3; do
      transformers && exit 0
      echo "download failed (attempt $attempt)" >&2
      sleep $((attempt * 15))
    done
    exit 1
    ;;
  langid)
    for attempt in 1 2 3; do
      "${cli[@]}" download models/stanza/multilingual --lang multilingual < /dev/null && exit 0
      echo "download failed (attempt $attempt)" >&2
      sleep $((attempt * 15))
    done
    exit 1
    ;;
  keys)
    out=${GITHUB_OUTPUT:-/dev/stdout}
    for g in en en-treebanks de; do
      echo "${g/-/_}=$(files "$g" | hash)" >> "$out"
    done
    echo "treebank_choices=$({ files en-treebank-choices; files de-treebank-choices; } | hash)" >> "$out"
    {
      echo "en_treebanks_paths<<EOF"
      files en-treebanks | cut -d' ' -f1 | sed 's|^|models/stanza/en/|'
      echo "EOF"
      echo "treebank_choices_paths<<EOF"
      files en-treebank-choices | cut -d' ' -f1 | sed 's|^|models/stanza/en/|'
      files de-treebank-choices | cut -d' ' -f1 | sed 's|^|models/stanza/de/|'
      echo "EOF"
    } >> "$out"
    ;;
  *)
    echo "usage: $0 download GROUP | files GROUP | transformer | langid | keys" >&2
    exit 2
    ;;
esac
