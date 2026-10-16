#!/usr/bin/env bash
# Publishes tests/StanzaSharp.ManagedCheck with Native AOT (x86-64-v3) and trimmed (self-contained), then runs each:
#   tools/verify-aot.sh MODEL_DIR RID          e.g. tools/verify-aot.sh models/stanza/en linux-x64
# Fails on any trim/AOT warning (IL2xxx/IL3xxx), on any torch file in the output, or when ManagedCheck fails (it compares
# corpus.txt's CoNLL-U for default and default_fast with the golden files byte for byte, and fails if torch code loads).
# AOT needs the platform's native toolchain: clang + zlib1g-dev on Linux, Visual Studio's C++ workload on Windows.
# docs/aot.md has the background.
set -euo pipefail
models=$(cd "$1" && pwd)
rid=$2
root=$(cd "$(dirname "$0")/.." && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT

publish() { # name, extra properties...
  local name=$1
  shift
  echo "== Publishing ManagedCheck ($name, $rid)"
  local start=$SECONDS
  dotnet publish "$root/tests/StanzaSharp.ManagedCheck" -c Release -r "$rid" -o "$out/$name" "$@" 2>&1 | tee "$out/$name.log"
  echo "Published in $((SECONDS - start)) s"
  if grep -E 'warning IL[23][0-9]{3}' "$out/$name.log"; then echo "::error::$name: trim/AOT warnings (above)"; exit 1; fi
  local torch
  torch=$(find "$out/$name" -iname '*torch*' -o -iname 'c10*' -o -iname 'libc10*' -o -iname 'libgomp*')
  if [ -n "$torch" ]; then echo "::error::$name: torch files in the output: $torch"; exit 1; fi
  echo "== Running ManagedCheck ($name)"
  start=$SECONDS
  "$out/$name/StanzaSharp.ManagedCheck" "$models" "$root/tests/golden"
  echo "Ran in $((SECONDS - start)) s"
}

publish aot -p:PublishAot=true -p:IlcInstructionSet=x86-64-v3
publish trimmed --self-contained -p:PublishTrimmed=true
echo "OK"
