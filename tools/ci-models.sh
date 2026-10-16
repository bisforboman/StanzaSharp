#!/usr/bin/env bash
# The model sets CI caches (ci.yml), each defined once here as lines of `download` arguments, so a cache's key, its
# paths and its download on a miss can't drift apart (owner's decision, 2026-10-16):
#
#   tools/ci-models.sh download GROUP   download the group's models (on a cache miss)
#   tools/ci-models.sh files GROUP      the group's files and MD5s (`download --list`), sorted
#   tools/ci-models.sh keys             writes en, en_treebanks, de, nocharlm, en_treebanks_paths and nocharlm_paths
#                                       to $GITHUB_OUTPUT
#
# Groups: en (default + default_fast: models-en and models-pt-<os>), en-treebanks (the English treebank packages and
# Pipeline.Alternatives' models, without en's files: models-en-treebanks), de (everything German: models-de-treebanks),
# en-nocharlm and de-nocharlm (the treebanks' _nocharlm taggers and parsers, Pipeline.Alternatives, without the other
# groups' files: together models-nocharlm, owner's decision, 2026-10-16, so their 1.2 GB rotate no other cache).
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
    en-nocharlm)
      for package in ewt gum lines partut atis childes eslspok; do
        echo "models/stanza/en --package $package --model pos=${package}_nocharlm,depparse=${package}_nocharlm"
      done
      echo "models/stanza/en --package mimic --model depparse=mimic_nocharlm"
      ;;
    de-nocharlm)
      for package in gsd hdt; do
        echo "models/stanza/de --lang de --package $package --model pos=${package}_nocharlm,depparse=${package}_nocharlm"
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
    en-nocharlm) LC_ALL=C comm -23 <(listed en-nocharlm) <(listed en en-treebanks | LC_ALL=C sort -u) ;;
    de-nocharlm) LC_ALL=C comm -23 <(listed de-nocharlm) <(listed de) ;;
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
  keys)
    out=${GITHUB_OUTPUT:-/dev/stdout}
    for g in en en-treebanks de; do
      echo "${g/-/_}=$(files "$g" | hash)" >> "$out"
    done
    echo "nocharlm=$({ files en-nocharlm; files de-nocharlm; } | hash)" >> "$out"
    {
      echo "en_treebanks_paths<<EOF"
      files en-treebanks | cut -d' ' -f1 | sed 's|^|models/stanza/en/|'
      echo "EOF"
      echo "nocharlm_paths<<EOF"
      files en-nocharlm | cut -d' ' -f1 | sed 's|^|models/stanza/en/|'
      files de-nocharlm | cut -d' ' -f1 | sed 's|^|models/stanza/de/|'
      echo "EOF"
    } >> "$out"
    ;;
  *)
    echo "usage: $0 download GROUP | files GROUP | keys" >&2
    exit 2
    ;;
esac
