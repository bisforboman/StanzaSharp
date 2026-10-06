#Requires -Version 5.1
<#
.SYNOPSIS
  Scaffolds the StanzaSharp workspace. Safe to re-run: existing projects are left alone.

.PARAMETER Python
  Create tools\.venv and install the Python reference implementation (stanza) + converter deps.

.PARAMETER Models
  Download the English tokenize/mwt/pos/constituency models into models\stanza and convert them
  into models\converted\en. Implies -Python.

.PARAMETER Cuda
  Reference TorchSharp-cuda-windows instead of TorchSharp-cpu in the runnable projects.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Models
#>
[CmdletBinding()]
param(
    [switch]$Python,
    [switch]$Models,
    [switch]$Cuda
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if ($Models) { $Python = $true }

function Invoke-Dotnet {
    & dotnet @args
    if ($LASTEXITCODE -ne 0) { throw "dotnet $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

# ---------------------------------------------------------------- .NET SDK / target framework
$sdk = & dotnet --version
if ($LASTEXITCODE -ne 0) { throw '.NET SDK not found on PATH.' }
$major = [int]($sdk.Split('.')[0])
if ($major -lt 8) { throw "Need .NET SDK 8 or later, found $sdk." }
$tfm = "net$major.0"
Write-Host "Using .NET SDK $sdk -> $tfm" -ForegroundColor Cyan

# ---------------------------------------------------------------- solution
if (-not (Get-ChildItem -Path . -Filter 'StanzaSharp.sln*' -File)) {
    Invoke-Dotnet new sln -n StanzaSharp
}

# name -> (template, folder)
$projects = [ordered]@{
    'StanzaSharp.Core'          = @('classlib', 'src')
    'StanzaSharp.Nn'            = @('classlib', 'src')
    'StanzaSharp.Tokenize'      = @('classlib', 'src')
    'StanzaSharp.Mwt'           = @('classlib', 'src')
    'StanzaSharp.Pos'           = @('classlib', 'src')
    'StanzaSharp.Constituency'  = @('classlib', 'src')
    'StanzaSharp'               = @('classlib', 'src')
    'StanzaSharp.Cli'           = @('console',  'samples')
    'StanzaSharp.Tests'         = @('xunit',    'tests')
}

function Get-ProjPath([string]$Name) {
    $folder = $projects[$Name][1]
    return (Join-Path $PSScriptRoot "$folder\$Name\$Name.csproj")
}

$created = @()
foreach ($name in $projects.Keys) {
    $template = $projects[$name][0]
    $csproj = Get-ProjPath $name
    if (Test-Path $csproj) { Write-Host "  exists: $name"; continue }
    $dir = Split-Path $csproj -Parent
    Invoke-Dotnet new $template -o $dir -n $name -f $tfm
    foreach ($stub in 'Class1.cs', 'UnitTest1.cs') {
        $p = Join-Path $dir $stub
        if (Test-Path $p) { Remove-Item $p }
    }
    Invoke-Dotnet sln add $csproj
    $created += $name
}

# ---------------------------------------------------------------- references (only for newly created projects)
$refs = @{
    'StanzaSharp.Nn'           = @('StanzaSharp.Core')
    'StanzaSharp.Tokenize'     = @('StanzaSharp.Core', 'StanzaSharp.Nn')
    'StanzaSharp.Mwt'          = @('StanzaSharp.Core', 'StanzaSharp.Nn')
    'StanzaSharp.Pos'          = @('StanzaSharp.Core', 'StanzaSharp.Nn')
    'StanzaSharp.Constituency' = @('StanzaSharp.Core', 'StanzaSharp.Nn')
    'StanzaSharp'              = @('StanzaSharp.Core', 'StanzaSharp.Tokenize', 'StanzaSharp.Mwt',
                                   'StanzaSharp.Pos', 'StanzaSharp.Constituency')
    'StanzaSharp.Cli'          = @('StanzaSharp')
    'StanzaSharp.Tests'        = @('StanzaSharp')
}
foreach ($name in $created) {
    if (-not $refs.ContainsKey($name)) { continue }
    foreach ($target in $refs[$name]) {
        Invoke-Dotnet add (Get-ProjPath $name) reference (Get-ProjPath $target)
    }
}

# ---------------------------------------------------------------- packages
# Libraries reference the managed TorchSharp package only; runnable projects pull in native libtorch.
$native = if ($Cuda) { 'TorchSharp-cuda-windows' } else { 'TorchSharp-cpu' }
if ($created -contains 'StanzaSharp.Nn') {
    Invoke-Dotnet add (Get-ProjPath 'StanzaSharp.Nn') package TorchSharp
}
foreach ($name in 'StanzaSharp.Cli', 'StanzaSharp.Tests') {
    if ($created -contains $name) { Invoke-Dotnet add (Get-ProjPath $name) package $native }
}

Invoke-Dotnet build

# ---------------------------------------------------------------- Python reference environment
$venv = Join-Path $PSScriptRoot 'tools\.venv'
$venvPy = Join-Path $venv 'Scripts\python.exe'
if ($Python) {
    if (-not (Test-Path $venvPy)) {
        Write-Host 'Creating tools\.venv' -ForegroundColor Cyan
        if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv $venv }
        else { & python -m venv $venv }
        if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed.' }
    }
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r (Join-Path $PSScriptRoot 'tools\requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed.' }
}

# ---------------------------------------------------------------- models
if ($Models) {
    $stanzaDir = Join-Path $PSScriptRoot 'models\stanza'
    $convertedDir = Join-Path $PSScriptRoot 'models\converted\en'
    New-Item -ItemType Directory -Force -Path $stanzaDir | Out-Null

    Write-Host 'Downloading English models' -ForegroundColor Cyan
    & $venvPy -c "import stanza; stanza.download('en', model_dir=r'$stanzaDir', processors='tokenize,mwt,pos,constituency')"
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }

    Write-Host 'Converting models' -ForegroundColor Cyan
    & $venvPy (Join-Path $PSScriptRoot 'tools\stanza_convert.py') convert (Join-Path $stanzaDir 'en') --out $convertedDir
    if ($LASTEXITCODE -ne 0) { throw 'Model conversion failed. Pipeline.Load can also use the .pt files in models\stanza\en directly.' }
}

# ---------------------------------------------------------------- git
if ((Get-Command git -ErrorAction SilentlyContinue) -and -not (Test-Path (Join-Path $PSScriptRoot '.git'))) {
    & git init -q
    Write-Host 'Initialized git repository (nothing committed yet).'
}

Write-Host "`nDone." -ForegroundColor Green
