#!/usr/bin/env bash
# Measures the line-break check on professionally made subtitles (#283; README, "Measurement"): downloads the IWSLT 2023
# subtitling track's ITV dev set (7 episodes of 3 series from ITV Studios, English SRT; research use only, no
# redistribution: fetched, never committed) into models/subtitles/itv/, checks its SHA-256, keeps the English SRT files,
# and runs the sample's --measure on the 7 episodes as one file.
#
#   samples/StanzaSharp.Subtitles/measure-itv.sh [--models DIR] [--max-chars N]
#
# Run from the repository root; the arguments go to the sample. The archive (476 MB, mostly video) is deleted once
# the subtitles are out. The download host is fbk.sharepoint.com (the link on iwslt.org/2023/subtitling).
set -euo pipefail

url='https://fbk.sharepoint.com/:u:/s/MTUnit/EZXtQE00-Z5VP5Stmbhw1Y4BI3k6WyppWs0_cSCCBbwjyQ?e=nxIqkf&download=1'
sha256=4b91a111d912ccf83f4fe515ec6bb47ac04105a6e1074d63f13661c3f13bdded
dir=models/subtitles/itv
file=$dir/itv.en.srt

if [[ ! -f "$file" ]]; then
  mkdir -p "$dir"
  curl -fsSL --retry 4 -c "$dir/cookies" -b "$dir/cookies" -o "$dir/itv_dev.tgz" "$url"
  echo "$sha256  $dir/itv_dev.tgz" | sha256sum --check --quiet
  tar -xzf "$dir/itv_dev.tgz" -C "$dir" itv/README itv/dev/srt/en
  rm "$dir/itv_dev.tgz" "$dir/cookies"
  for f in "$dir"/itv/dev/srt/en/0[1-7].srt; do
    tr -d '\r' < "$f"
    printf '\n\n'
  done > "$file"
fi

dotnet run -c Release --project samples/StanzaSharp.Subtitles -- --measure "$file" "$@"
