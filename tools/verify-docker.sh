#!/usr/bin/env bash
# Packs StanzaSharp from this checkout, builds the Docker sample (samples/docker) on each base image given, and
# checks that the image holds no libtorch and that the container reproduces tests/golden/pipeline.conllu byte for
# byte from tests/golden/corpus.txt (all eight processors, the default package, the default managed backend).
# A base image whose name contains "alpine" is built for linux-musl-x64, the others for linux-x64.
#
#   tools/verify-docker.sh MODEL_DIR BASE_IMAGE...
#
# MODEL_DIR holds the default package's models (Stanza's .pt files or converted); it is mounted read-only.
# Prints each image's size, and appends a table to $GITHUB_STEP_SUMMARY when set. Fails if any base fails.
set -uo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
models=$(cd "$1" && pwd)
shift
context="$root/samples/docker"
version="0.0.0-docker.$(date -u +%Y%m%d%H%M%S)"

# A local feed inside the build context, with a nuget.config that adds it to nuget.org (the SDK image's default).
rm -rf "$context/feed"
dotnet pack "$root/src/StanzaSharp" -c Release -p:Version="$version" -o "$context/feed" --nologo || exit 1
cat > "$context/nuget.config" <<'EOF'
<configuration>
  <packageSources>
    <add key="local" value="feed" />
  </packageSources>
</configuration>
EOF
trap 'rm -rf "$context/feed" "$context/nuget.config"' EXIT

summary="| Base image | Image size | pipeline.conllu |\n|---|---:|---|\n"
failed=0
for base in "$@"; do
  tag="stanzasharp-sample:$(echo "$base" | tr -c 'a-zA-Z0-9.\n' '-')"
  echo "::group::$base"
  case "$base" in *alpine*) rid=linux-musl-x64 ;; *) rid=linux-x64 ;; esac
  if ! docker build "$context" --build-arg BASE="$base" --build-arg RID="$rid" --build-arg STANZASHARP_VERSION="$version" -t "$tag"; then
    echo "::error::docker build failed on $base"; failed=1; summary+="| \`$base\` | | build failed |\n"; echo "::endgroup::"; continue
  fi
  size=$(docker image inspect "$tag" --format '{{.Size}}' | numfmt --to=si --suffix=B)
  echo "$tag: $size"
  # The managed backend needs no libtorch, so none may be in the image.
  container=$(docker create "$tag")
  libtorch=$(docker export "$container" | tar -t | grep -E '^app/(.*/)?lib(torch|c10|gomp)[^/]*\.so')
  docker rm "$container" > /dev/null
  if [ -n "$libtorch" ]; then
    echo "::error::libtorch is in the image on $base: $libtorch"; failed=1
  fi
  out=$(mktemp)
  if ! docker run -i --rm -v "$models:/models:ro" "$tag" < "$root/tests/golden/corpus.txt" > "$out"; then
    echo "::error::The container failed on $base"; failed=1; result="run failed"
  elif ! cmp -s "$out" "$root/tests/golden/pipeline.conllu"; then
    diff "$root/tests/golden/pipeline.conllu" "$out" | head -40
    echo "::error::Output on $base differs from tests/golden/pipeline.conllu"; failed=1; result="differs"
  else
    echo "Identical to tests/golden/pipeline.conllu on $base."; result="identical"
  fi
  summary+="| \`$base\` | $size | $result |\n"
  echo "::endgroup::"
done

printf "$summary"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then printf "$summary" >> "$GITHUB_STEP_SUMMARY"; fi
exit $failed
