#!/usr/bin/env bash
# The other languages' golden tests (LanguageGoldenTests) a batch at a time, as nightly.yml's languages job runs them:
# download a batch's models (every package with a tests/golden/languages/<lang>/<package>.conllu), run the tests on just
# those languages (STANZASHARP_LANGUAGES), fail on a skipped test, delete the models, next batch. The languages' models
# come to about 25 GB, more than a runner holds at once (owner's decision, 2026-10-30: all of them every night).
#
#   tools/test-languages.sh [LANG...]   (default: every folder of tests/golden/languages)
#
# Needs the Release builds of the solution and samples/StanzaSharp.Cli. BATCH (default 8) languages per test run.
set -uo pipefail

batch_size=${BATCH:-8}
read -r -a cli <<< "${STANZASHARP_CLI:-dotnet run --project samples/StanzaSharp.Cli -c Release --no-build --}"
langs=("$@")
[ ${#langs[@]} -gt 0 ] || mapfile -t langs < <(ls tests/golden/languages)
failed=()

download() {
  local lang=$1
  for golden in tests/golden/languages/"$lang"/*.conllu; do
    local package
    package=$(basename "$golden" .conllu)
    for attempt in 1 2 3; do
      "${cli[@]}" download "models/stanza/$lang" --lang "$lang" --package "$package" < /dev/null > /dev/null && continue 2
      echo "$lang $package: download failed (attempt $attempt)" >&2
      sleep $((attempt * 15))
    done
    return 1
  done
}

run() {
  local -a batch=("$@")
  local results="TestResults/languages-${batch[0]}"
  echo "::group::${batch[*]}"
  STANZASHARP_LANGUAGES=$(IFS=,; echo "${batch[*]}") dotnet test tests/StanzaSharp.Tests -c Release --no-build \
    --blame-hang-timeout 30m --blame-hang-dump-type none --logger trx --results-directory "$results" \
    --filter "FullyQualifiedName~LanguageGoldenTests.Process_MatchesGolden" || failed+=("${batch[@]}")
  if grep -q 'outcome="NotExecuted"' "$results"/*.trx || ! grep -q '<UnitTestResult ' "$results"/*.trx; then
    echo "::error::${batch[*]}: tests were skipped or none ran"
    failed+=("${batch[@]}")
  fi
  echo "::endgroup::"
  for lang in "${batch[@]}"; do rm -rf "models/stanza/$lang"; done
}

batch=()
for lang in "${langs[@]}"; do
  if download "$lang"; then batch+=("$lang"); else echo "::error::$lang: download failed"; failed+=("$lang"); rm -rf "models/stanza/$lang"; fi
  if [ ${#batch[@]} -ge "$batch_size" ]; then run "${batch[@]}"; batch=(); fi
done
[ ${#batch[@]} -eq 0 ] || run "${batch[@]}"

if [ ${#failed[@]} -gt 0 ]; then
  echo "::error::Failed: ${failed[*]}"
  exit 1
fi
echo "All ${#langs[@]} languages passed."
