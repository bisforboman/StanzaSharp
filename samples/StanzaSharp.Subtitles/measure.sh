#!/usr/bin/env bash
# Measures the line-break check on human-made subtitles (#267; README, "Measurement"): downloads MuST-Cinema's English
# Amara test set as EvalSubtitle ships it (tagged text, CC BY-NC-SA 4.0: fetched, never committed) into
# models/subtitles/, checks its SHA-256, and runs the sample's --measure on it.
#
#   samples/StanzaSharp.Subtitles/measure.sh [--models DIR] [--max-chars N]
#
# Run from the repository root; the arguments go to the sample (models default to models/stanza/en, downloaded there
# on the first run).
set -euo pipefail

commit=5431972aa471a4f93404337bf0daf77c6cab6e38 # fyvo/EvalSubtitle, 2025-03-28
sha256=5076ffb039290e9dfff89dc3a7a79ab002ea5ce0a011510c7d499f0c8d50d088
file=models/subtitles/amara.en

if [[ ! -f "$file" ]] || ! echo "$sha256  $file" | sha256sum --check --status; then
  mkdir -p "$(dirname "$file")"
  curl -fsSL --retry 4 -o "$file.tmp" "https://raw.githubusercontent.com/fyvo/EvalSubtitle/$commit/data/amara.en"
  echo "$sha256  $file.tmp" | sha256sum --check --quiet
  mv "$file.tmp" "$file"
fi

dotnet run -c Release --project samples/StanzaSharp.Subtitles -- --measure "$file" "$@"
