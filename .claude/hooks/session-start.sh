#!/bin/bash
# Cloud sessions only: the .NET 10 SDK, the restored solution, stylebro-migrate (the formatter CI's build-test runs) and
# the English default + default_fast .pt models (tools/ci-models.sh en). Idempotent: the container is cached after it
# runs, so a later start finds everything in place and only checks. STANZASHARP_MODELS points the tests at the .pt files,
# as on cross-os (no conversion, no Python).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
tools="$HOME/.dotnet/tools"

# Ubuntu's own package: the cloud network policy blocks builds.dotnet.microsoft.com (dotnet-install.sh) but not the
# Ubuntu archive.
if ! dotnet --list-sdks 2>/dev/null | grep -q '^10\.'; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dotnet-sdk-10.0
fi
export PATH="$tools:$PATH" DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  {
    echo "export PATH=\"$tools:\$PATH\""
    echo "export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1"
    echo "export STANZASHARP_MODELS=\"$CLAUDE_PROJECT_DIR/models/stanza/en\""
  } >> "$CLAUDE_ENV_FILE"
fi

# The version matches StyleBro.Analyzers in Directory.Build.props (ci.yml's build-test).
if ! dotnet tool list --global | grep -q 'stylebro.migrate *0\.7\.0-alpha\.1'; then
  dotnet tool update --global StyleBro.Migrate --version 0.7.0-alpha.1
fi

dotnet restore StanzaSharp.slnx
dotnet build samples/StanzaSharp.Cli -c Release --no-restore -v q

# The download keeps files whose MD5 already matches, so this is a check after the first run.
tools/ci-models.sh download en
