<#
Packs StanzaSharp, then builds and runs a fresh console app that references the package (plus
TorchSharp-cpu) the way a user would, and checks that it parses a sentence.

  pwsh tools/verify-package.ps1 [-ModelDir models/converted/en]

ModelDir is any directory Pipeline.Load accepts (converted models or Stanza's .pt files).
#>
param([string]$ModelDir = 'models/converted/en')

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
# Relative to the repository root, else to the current directory; absolute paths as given.
# (Not $modelDir: PowerShell variable names ignore case, so that would overwrite the parameter.)
$models = [IO.Path]::Combine($root, $ModelDir)
if (-not (Test-Path $models)) { $models = Resolve-Path $ModelDir }

# A version of its own each run, so a package from an earlier run can't come from the NuGet cache.
$version = "0.0.0-verify.$([DateTime]::UtcNow.ToString('yyyyMMddHHmmss'))"
$work = Join-Path ([IO.Path]::GetTempPath()) "stanzasharp-verify-$version"
$feed = Join-Path $work 'feed'
$app = Join-Path $work 'app'
New-Item -ItemType Directory -Force $feed, $app | Out-Null

try {
    dotnet pack (Join-Path $root 'src/StanzaSharp') -c Release -p:Version=$version -o $feed --nologo
    if ($LASTEXITCODE) { throw 'dotnet pack failed' }

    Set-Content (Join-Path $app 'nuget.config') @"
<configuration>
  <packageSources>
    <add key="local" value="$feed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
</configuration>
"@
    Set-Content (Join-Path $app 'app.csproj') @"
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="StanzaSharp" Version="$version" />
    <PackageReference Include="TorchSharp-cpu" Version="0.107.0" />
  </ItemGroup>
</Project>
"@
    # The README example, plus a reference to ModelDownloader so the public download API compiles too.
    Set-Content (Join-Path $app 'Program.cs') @'
using StanzaSharp;

Console.WriteLine($"Models for Stanza {ModelDownloader.StanzaVersion}");
using var nlp = Pipeline.Load(args[0]);
var doc = nlp.Process("Barack Obama was born in Hawaii. He was elected president in 2008.");
foreach (var sentence in doc.Sentences)
    Console.WriteLine(sentence.Constituency);
Console.Write(Conllu.Write(doc));
'@
    $output = dotnet run --project $app -- $models 2>&1 | Out-String
    if ($LASTEXITCODE) { throw "The consumer app failed:`n$output" }
    Write-Host $output
    $expected = '(ROOT (S (NP (NNP Barack) (NNP Obama)) (VP (VBD was) (VP (VBN born) (PP (IN in) (NP (NNP Hawaii))))) (. .)))'
    if (-not $output.Contains($expected)) { throw "Expected the parse $expected" }
    Write-Host "Package $version works as a consumer uses it."
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    $cached = Join-Path ($env:NUGET_PACKAGES ?? (Join-Path $HOME '.nuget/packages')) "stanzasharp/$version"
    Remove-Item -Recurse -Force $cached -ErrorAction SilentlyContinue
}
