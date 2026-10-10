// Proves that a managed pipeline needs no native libtorch: run it in a process of its own, with no libtorch on disk.
//   StanzaSharp.ManagedCheck MODEL_DIR GOLDEN_DIR
// For each package it loads a pipeline on the default backend (managed; with Threads, a Logger, VerifyChecksums for .pt
// models), checks that a canceled Process throws, compares corpus.txt's CoNLL-U with the golden file byte for byte, and
// at the end fails if any torch module (TorchSharp, LibTorchSharp, torch_cpu, c10, ...) is loaded. Public API only, so that
// tools/verify-package.ps1 -Managed compiles this same file against the packed package.
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using StanzaSharp;

var (models, golden) = (args[0], args[1]);
var corpus = File.ReadAllText(Path.Combine(golden, "corpus.txt"));
// Only Stanza's .pt files have published checksums.
bool pt = File.Exists(Path.Combine(models, "pretrain", "conll17.pt")) && !File.Exists(Path.Combine(models, "pretrain", "conll17.json"));
int failures = 0;

foreach (var (package, expected) in new[] { ("default", "pipeline.conllu"), ("default_fast", Path.Combine("fast", "corpus.conllu")) })
{
    var logger = new CountingLogger();
    // The default backend: what `dotnet add package StanzaSharp` alone must run.
    var options = new PipelineOptions { Package = package, Threads = 2, Logger = logger, VerifyChecksums = pt };
    var start = Stopwatch.GetTimestamp();
    using var nlp = Pipeline.Load(models, options);
    Console.WriteLine($"{package}: loaded in {Stopwatch.GetElapsedTime(start).TotalSeconds:F1} s ({logger.Messages} log messages, checksums {(pt ? "verified" : "not verified")})");
    try
    {
        nlp.Process(corpus, new CancellationToken(canceled: true));
        failures += Fail($"{package}: a canceled Process did not throw");
    }
    catch (OperationCanceledException) { }
    start = Stopwatch.GetTimestamp();
    var conllu = Conllu.Write(nlp.Process(corpus));
    bool same = conllu == File.ReadAllText(Path.Combine(golden, expected));
    Console.WriteLine($"{package}: processed in {Stopwatch.GetElapsedTime(start).TotalSeconds:F1} s, {(same ? "identical to" : "DIFFERENT from")} {expected}");
    if (!same)
        failures += Fail($"{package}: the output differs from {expected}");
    // The other input modes and no_ssplit run too (their output is tested in StanzaSharp.Tests).
    nlp.Process([corpus, "A second text."]);
    nlp.Process([["Hello", "world", "."], ["Bye", "."]]);
    using (var noSsplit = Pipeline.Load(models, new PipelineOptions { Package = package, Processors = "tokenize,mwt", SplitSentences = false, Backend = PipelineBackend.Managed }))
        noSsplit.Process(corpus);
}

// Since 1.0 the StanzaSharp package has no TorchSharp at all (it is in StanzaSharp.Cuda), so no torch module of any kind,
// managed or native, may be loaded. Every check above already proves it where libtorch is absent (TorchSharp throws when
// it can't load it); the module list also catches a load where it is present.
var natives = Modules()
    .Where(m => m.ModuleName.Contains("torch", StringComparison.OrdinalIgnoreCase) || m.ModuleName.StartsWith("c10", StringComparison.OrdinalIgnoreCase)
        || m.ModuleName.StartsWith("libc10", StringComparison.OrdinalIgnoreCase))
    .Select(m => m.FileName)
    .Distinct(StringComparer.OrdinalIgnoreCase)
    .ToList();
Console.WriteLine($"Torch modules loaded: {(natives.Count == 0 ? "none" : string.Join(", ", natives))}");
if (natives.Count > 0)
    failures += Fail("torch code was loaded");
if (AppDomain.CurrentDomain.GetAssemblies().Any(a => a.GetName().Name is "TorchSharp" or "StanzaSharp.Cuda"))
    failures += Fail("a TorchSharp assembly was loaded");
Console.WriteLine(failures == 0 ? "OK" : $"{failures} failure(s)");
return failures == 0 ? 0 : 1;

static IEnumerable<ProcessModule> Modules()
{
    try
    {
        return Process.GetCurrentProcess().Modules.Cast<ProcessModule>().ToList();
    }
    catch (PlatformNotSupportedException)
    {
        Console.WriteLine("(this platform can't list the process's modules)");
        return [];
    }
}

static int Fail(string message)
{
    Console.Error.WriteLine("FAIL: " + message);
    return 1;
}

sealed class CountingLogger : ILogger
{
    public int Messages;
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(LogLevel logLevel) => true;
    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        formatter(state, exception);
        Interlocked.Increment(ref Messages);
    }
}
