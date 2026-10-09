<#
Packs StanzaSharp, then builds and runs a fresh console app that references the package the way a user
would, and checks that it parses a sentence.

  pwsh tools/verify-package.ps1 [-ModelDir models/converted/en] [-Platform Linux|Windows|WindowsArm64|MacOS]
  pwsh tools/verify-package.ps1 [-ModelDir models/converted/en] -Managed

ModelDir is any directory Pipeline.Load accepts (converted models or Stanza's .pt files).
Without -Platform the app references StanzaSharp + TorchSharp-cpu. With it, the app references only the
platform package StanzaSharp.Cpu.<Platform> (which brings StanzaSharp and that platform's libtorch), so
nothing else can supply the native libraries, and sets StanzaSharpTrimNative, so the parse also proves that the
trimmed native files are not needed. Either way the build must give no STANZA warning (STANZA001: TorchSharp versions).

-Managed (issue #29, the 0.5 requirement): the app references only StanzaSharp, no TorchSharp-cpu and no platform
package, so no libtorch reaches its output. It is tests/StanzaSharp.ManagedCheck/Program.cs compiled against the
package: it switches the pipeline to the managed backend through the internal PipelineOptions.Backend (by reflection,
until the public PipelineBackend option exists), runs both packages on tests/golden/corpus.txt, compares the golden
CoNLL-U byte for byte and fails if any native torch module was loaded. It runs with an empty NUGET_PACKAGES, so
TorchSharp's fallback (copying libtorch from the NuGet cache next to the app) cannot find libtorch either.
#>
param([string]$ModelDir = 'models/converted/en', [ValidateSet('', 'Linux', 'Windows', 'WindowsArm64', 'MacOS')][string]$Platform = '', [switch]$Managed)
if ($Managed -and $Platform) { throw '-Managed takes no -Platform: the app references only StanzaSharp' }

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
    $projects = @('src/StanzaSharp') + $(if ($Platform) { "src/StanzaSharp.Cpu.$Platform" } else { @() })
    foreach ($project in $projects) {
        dotnet pack (Join-Path $root $project) -c Release -p:Version=$version -o $feed --nologo
        if ($LASTEXITCODE) { throw "dotnet pack $project failed" }
    }
    $references = if ($Managed) {
        "<PackageReference Include=`"StanzaSharp`" Version=`"$version`" />"
    } elseif ($Platform) {
        "<PackageReference Include=`"StanzaSharp.Cpu.$Platform`" Version=`"$version`" />"
    } else {
        "<PackageReference Include=`"StanzaSharp`" Version=`"$version`" />`n    <PackageReference Include=`"TorchSharp-cpu`" Version=`"0.107.0`" />"
    }

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
    <StanzaSharpTrimNative>$(if ($Platform) { 'true' } else { 'false' })</StanzaSharpTrimNative>
  </PropertyGroup>
  <ItemGroup>
    $references
  </ItemGroup>
</Project>
"@
    if ($Managed) {
        Copy-Item (Join-Path $root 'tests/StanzaSharp.ManagedCheck/Program.cs') (Join-Path $app 'Program.cs')
    } else {
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
    }
    $build = dotnet build $app -o (Join-Path $work 'out') --nologo 2>&1 | Out-String
    if ($LASTEXITCODE) { throw "The consumer app failed to build:`n$build" }
    if ($build -match 'warning STANZA') { throw "The build gave a STANZA warning:`n$build" }
    if ($Platform) {
        # StanzaSharpTrimNative: none of these may reach the output (see buildTransitive/StanzaSharp.targets).
        $trimmed = Get-ChildItem (Join-Path $work 'out') -Recurse -Include libtorch_python.*, libshm.*, libnnapi_backend.*, libtorchbind_test.*, libjitbackend_test.*, libbackend_with_compiler.*, libaoti_custom_ops.*
        if ($trimmed) { throw "StanzaSharpTrimNative left $($trimmed.Name -join ', ')" }
    }
    if ($Managed) {
        $libtorch = Get-ChildItem (Join-Path $work 'out') -Recurse -Include torch_cpu*, torch.*, libtorch_cpu*, libtorch.*, c10.*, libc10.*
        if ($libtorch) { throw "libtorch reached the output: $($libtorch.Name -join ', ')" }
        $emptyNuget = Join-Path $work 'empty-nuget'
        New-Item -ItemType Directory -Force $emptyNuget | Out-Null
        $saved = $env:NUGET_PACKAGES
        $env:NUGET_PACKAGES = $emptyNuget
        try { $output = dotnet (Join-Path $work 'out/app.dll') $models (Join-Path $root 'tests/golden') 2>&1 | Out-String }
        finally { $env:NUGET_PACKAGES = $saved }
        if ($LASTEXITCODE) { throw "The managed consumer app failed:`n$output" }
        Write-Host $output
        if (-not $output.Contains('Native torch modules loaded: none')) { throw 'Native torch code was loaded' }
        Write-Host "Package $version runs the managed pipeline without libtorch."
        return
    }
    $output = dotnet (Join-Path $work 'out/app.dll') $models 2>&1 | Out-String
    if ($LASTEXITCODE) { throw "The consumer app failed:`n$output" }
    Write-Host $output
    $expected = '(ROOT (S (NP (NNP Barack) (NNP Obama)) (VP (VBD was) (VP (VBN born) (PP (IN in) (NP (NNP Hawaii))))) (. .)))'
    if (-not $output.Contains($expected)) { throw "Expected the parse $expected" }
    Write-Host "Package $version$(if ($Platform) { " (StanzaSharp.Cpu.$Platform)" }) works as a consumer uses it."
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    $packages = $env:NUGET_PACKAGES ?? (Join-Path $HOME '.nuget/packages')
    foreach ($id in @('stanzasharp') + $(if ($Platform) { "stanzasharp.cpu.$($Platform.ToLowerInvariant())" } else { @() })) {
        Remove-Item -Recurse -Force (Join-Path $packages "$id/$version") -ErrorAction SilentlyContinue
    }
}
