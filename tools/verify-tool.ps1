<#
Packs the StanzaSharp.Tool package, installs it from that local feed into a fresh tool path the way a user would
(dotnet tool install), checks that it carries no native files, and downloads a small set of models with it.

  pwsh tools/verify-tool.ps1 [-Processors tokenize,mwt]
#>
param([string]$Processors = 'tokenize,mwt')

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
# A version of its own each run, so a package from an earlier run can't come from the NuGet cache.
$version = "0.0.0-verify.$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
$work = Join-Path ([IO.Path]::GetTempPath()) "stanzasharp-tool-$version"
$feed = Join-Path $work 'feed'
$toolPath = Join-Path $work 'tools'
$models = Join-Path $work 'models'

try {
    dotnet pack (Join-Path $root 'src/StanzaSharp.Tool') -c Release -p:Version=$version -o $feed --nologo
    if ($LASTEXITCODE) { throw 'dotnet pack src/StanzaSharp.Tool failed' }
    $package = Get-Item (Join-Path $feed "StanzaSharp.Tool.$version.nupkg")
    Write-Host "StanzaSharp.Tool.$version.nupkg: $([Math]::Round($package.Length / 1MB, 2)) MB"

    dotnet tool install StanzaSharp.Tool --version $version --tool-path $toolPath --add-source $feed
    if ($LASTEXITCODE) { throw 'dotnet tool install failed' }

    # No libtorch, LibTorchSharp or SkiaSharp natives: the tool must work on a machine without them.
    $natives = Get-ChildItem $toolPath -Recurse -File -Include *.so, *.so.*, *.dylib, LibTorchSharp.dll, libSkiaSharp.dll, torch.dll, torch_cpu.dll, c10.dll
    if ($natives) { throw "The tool carries native files: $($natives.Name -join ', ')" }

    $command = Join-Path $toolPath 'stanzasharp'
    & $command download $models --processors $Processors
    if ($LASTEXITCODE) { throw "stanzasharp download failed ($LASTEXITCODE)" }
    $files = Get-ChildItem $models -Recurse -File -Filter *.pt
    if (-not $files) { throw "stanzasharp download wrote no models into $models" }
    $files | ForEach-Object { Write-Host "  $($_.FullName.Substring($models.Length + 1)): $([Math]::Round($_.Length / 1MB, 1)) MB" }

    # Bad arguments exit with 2.
    & $command download $models --nonsense 2>$null
    if ($LASTEXITCODE -ne 2) { throw "Expected exit code 2 for a bad argument, got $LASTEXITCODE" }
    $global:LASTEXITCODE = 0 # the expected 2 would otherwise become the script's exit code
    Write-Host "StanzaSharp.Tool $version installs and downloads models."
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    $packages = $env:NUGET_PACKAGES ?? (Join-Path $HOME '.nuget/packages')
    Remove-Item -Recurse -Force (Join-Path $packages "stanzasharp.tool/$version") -ErrorAction SilentlyContinue
}
