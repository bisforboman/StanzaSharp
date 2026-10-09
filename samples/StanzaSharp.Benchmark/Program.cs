using System.Diagnostics;
using StanzaSharp;
using StanzaSharp.Constituency;
using StanzaSharp.Depparse;
using StanzaSharp.Lemma;
using StanzaSharp.Mwt;
using StanzaSharp.Ner;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using StanzaSharp.Pos;
using StanzaSharp.Sentiment;
using StanzaSharp.Tokenize;
using TorchSharp;
using static TorchSharp.torch;

// Times each pipeline stage on a large text built from the golden corpora. The Python counterpart is
// tools/benchmark.py; both build the same text and print the same report. See docs/performance.md.
const string Usage = """
    Usage: StanzaSharp.Benchmark [--models DIR] [--copies N] [--runs N] [--threads N] [--out FILE]
                                 [--device cpu|cuda] [--no-tf32] [--package NAME] [--documents N]

      --models DIR    converted models (default: models/converted/en)
      --copies N      copies of the golden texts in the input (default: 8, ~20k words)
      --runs N        timed runs after a warm-up run on one copy; medians are reported (default: 3)
      --threads N     threads: the managed pool's size, or torch's intra-op threads (default: all cores)
      --out FILE      write the last run's CoNLL-U here, to compare with Python's
      --device D      torch backend: cpu (default) or cuda; cuda needs a build with STANZASHARP_CUDA=1 (docs/gpu.md)
      --no-tf32       cuda: turn off TF32 in cuDNN and cuBLAS (process-wide torch settings)
      --package NAME  Stanza's English package: default (all eight processors) or default_fast
      --documents N   instead: N one-sentence texts, Process per text vs one bulk Process call
      --memory N      instead: Pipeline.Load, then one Process call on a text of about N words, reporting
                      the load peak and the peak memory; run it in a fresh process each time
      --processors P  the stages to time (default: all of the package's); --memory: the processors to load
      --backend B     managed (default) or torch; --device and --no-tf32 select torch
      --chunk-words K --memory: split the text at paragraphs into parts of about K words and call Process
                      on each part in turn (--bulk: one Process(IEnumerable<string>) call on the parts)
      --calls N       --memory: repeat the Process call(s) N times, reporting the memory after each (default: 1)
      --no-trim       on Linux (glibc), keep the free native heap after each Process call (no malloc_trim)
      --idle-gc       --memory: then an aggressive GC at once, and another after 61 s idle, reporting the memory
    """;

if (args is ["lemma-divergence", ..])
    return LemmaDivergence.Run(args[1..]); // tools/lemma_divergence.py; docs/backends.md, "lemma"
if (args is ["managed-spike", ..])
    return ManagedSpike.Run(args[1..], FindRepoRoot(), BuildText); // issue #29: docs/managed-backend-spike.md

string modelDir = Path.Combine("models", "converted", "en");
int copies = 8, runs = 3, threads = 0, documents = 0, memoryWords = 0, chunkWords = 0, calls = 1, cacheWords = CharlmCache.DefaultMaxWords;
string? outFile = null, processors = null;
bool bulkCall = false, verbose = false, noTrim = false, idleGc = false;
torch.Device? device = null; // CPU; not torch.CPU, which would load libtorch in a managed run
bool noTf32 = false;
string package = Pipeline.DefaultPackage;
Backend? backendArg = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models" when i + 1 < args.Length: modelDir = args[++i]; break;
        case "--copies" when i + 1 < args.Length: copies = int.Parse(args[++i]); break;
        case "--runs" when i + 1 < args.Length: runs = int.Parse(args[++i]); break;
        case "--threads" when i + 1 < args.Length: threads = int.Parse(args[++i]); break;
        case "--out" when i + 1 < args.Length: outFile = args[++i]; break;
        case "--device" when i + 1 < args.Length: device = torch.device(args[++i]); break;
        case "--no-tf32": noTf32 = true; break;
        case "--package" when i + 1 < args.Length: package = args[++i]; break;
        case "--documents" when i + 1 < args.Length: documents = int.Parse(args[++i]); break;
        case "--memory" when i + 1 < args.Length: memoryWords = int.Parse(args[++i]); break;
        case "--processors" when i + 1 < args.Length: processors = args[++i]; break;
        case "--backend" when i + 1 < args.Length && args[i + 1] is "managed" or "torch":
            backendArg = args[++i] == "managed" ? Backend.Managed : Backend.TorchSharp; break;
        case "--chunk-words" when i + 1 < args.Length: chunkWords = int.Parse(args[++i]); break;
        case "--bulk": bulkCall = true; break;
        case "--charlm-cache" when i + 1 < args.Length: cacheWords = int.Parse(args[++i]); break;
        case "--verbose": verbose = true; break;
        case "--idle-gc": idleGc = true; break;
        case "--calls" when i + 1 < args.Length: calls = int.Parse(args[++i]); break;
        case "--no-trim": noTrim = true; break;
        default:
            Console.Error.WriteLine(Usage);
            return 2;
    }
}
// Like PipelineOptions: a device or --no-tf32 means the torch backend.
var backend = backendArg ?? (device != null || noTf32 ? Backend.TorchSharp : Backend.Managed);
if (backend == Backend.Managed && (device != null || noTf32))
{
    Console.Error.WriteLine("--device and --no-tf32 need --backend torch");
    return 2;
}
var pipelineBackend = backend == Backend.Managed ? PipelineBackend.Managed : PipelineBackend.TorchSharp;
if (threads > 0 && backend == Backend.TorchSharp)
    torch.set_num_threads(threads);
if (threads > 0 && backend == Backend.Managed)
    ManagedThreads.Count = threads;
if (noTf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = false;

if (documents > 0)
{
    // Many short texts: one Process call per text vs one bulk call (Pipeline.Process(IEnumerable<string>)).
#pragma warning disable CS0618 // Device: the TorchSharp backend's device until StanzaSharp.Cuda
    using var nlp = Pipeline.Load(modelDir, new PipelineOptions { Package = package, Backend = pipelineBackend, Device = device, TrimNativeHeap = !noTrim });
#pragma warning restore CS0618
    var texts = BuildDocuments(documents);
    nlp.Process(texts.Take(50)); // warm-up
    var watch = Stopwatch.StartNew();
    foreach (var t in texts)
        nlp.Process(t);
    double alone = watch.Elapsed.TotalSeconds;
    watch.Restart();
    nlp.Process(texts);
    double bulk = watch.Elapsed.TotalSeconds;
    Console.WriteLine($"C# StanzaSharp ({package}, {backend} backend) on {device?.ToString() ?? "cpu"}, " +
                      (backend == Backend.Managed ? $"managed threads {ManagedThreads.Count}" : $"torch threads {torch.get_num_threads()}") +
                      $", {texts.Count} documents of one sentence");
    Console.WriteLine($"{"one by one",-14}{alone,9:F2} s {texts.Count / alone,10:F0} docs/s");
    Console.WriteLine($"{"bulk",-14}{bulk,9:F2} s {texts.Count / bulk,10:F0} docs/s");
    return 0;
}

if (memoryWords > 0)
{
    // What a short-lived process pays (issue #19): load, one Process call (or one per part), exit.
    var paragraphs = BuildParagraphs(memoryWords);
    var clockLoad = Stopwatch.StartNew();
    using var nlp = Pipeline.Load(modelDir, new PipelineOptions { Package = package, Processors = processors, Backend = pipelineBackend, Threads = threads > 0 ? threads : null, Logger = verbose ? new MemoryLogger() : null, TrimNativeHeap = !noTrim,
        CharlmCache = new CharlmCacheOptions { IsEnabled = cacheWords > 0, MaxWords = Math.Max(cacheWords, 1) } });
    double loadSeconds = clockLoad.Elapsed.TotalSeconds;
    double loadPeak = PeakMB(), afterLoad = WorkingSetMB();
    var gcAfterLoad = GC.GetGCMemoryInfo();
    var after = new List<string>();
    List<Document> docs = null!;
    double processSeconds = 0;
    GCMemoryInfo gc = default;
    for (int call = 1; call <= calls; call++)
    {
        clockLoad.Restart();
        if (chunkWords <= 0)
            docs = [nlp.Process(string.Join("\n\n", paragraphs))];
        else
        {
            var parts = Chunk(paragraphs, chunkWords);
            docs = bulkCall ? nlp.Process((IEnumerable<string>)parts) : parts.Select(nlp.Process).ToList();
        }
        double seconds = clockLoad.Elapsed.TotalSeconds;
        if (call == 1)
        {
            processSeconds = seconds;
            gc = GC.GetGCMemoryInfo();
        }
        after.Add($"after call {call}  {seconds,7:F2} s  peak {PeakMB(),6:F0} MB  working set {WorkingSetMB(),6:F0} MB");
        if (verbose)
            after.Add("  " + HeapStats.Describe());
    }
    if (outFile != null)
        File.WriteAllText(outFile, string.Concat(docs.Select(Conllu.Write)));
    int wordCount = docs.Sum(d => d.Sentences.Sum(s => s.Words.Count()));
    string mode = chunkWords <= 0 ? "one Process call" : $"{docs.Count} parts of ~{chunkWords} words, " + (bulkCall ? "one bulk call" : "one call each");
    Console.WriteLine($"C# StanzaSharp ({package}: {processors ?? "all"}) from {modelDir}, {(System.Runtime.GCSettings.IsServerGC ? "Server" : "workstation")} GC, " +
                      // A managed pipeline never loads libtorch, so don't ask it.
                      (backend == Backend.Managed ? $"managed threads {ManagedThreads.Count}" : $"torch threads {torch.get_num_threads()}") +
                      $": {wordCount} words, {mode}, {(NativeTorchLoaded() ? "native libtorch loaded" : "no native libtorch")}");
    Console.WriteLine($"load          {loadSeconds,7:F2} s  load peak {loadPeak,6:F0} MB  after load {afterLoad,6:F0} MB  " +
                      $"(GC heap {gcAfterLoad.HeapSizeBytes / 1048576.0:F0} MB, committed {gcAfterLoad.TotalCommittedBytes / 1048576.0:F0} MB)");
    Console.WriteLine($"process       {processSeconds,7:F2} s  peak      {PeakMB(),6:F0} MB  at the end {WorkingSetMB(),6:F0} MB  " +
                      $"(GC heap {gc.HeapSizeBytes / 1048576.0:F0} MB, committed {gc.TotalCommittedBytes / 1048576.0:F0} MB, {GC.CollectionCount(2)} gen2 GCs)");
    foreach (var line in after)
        Console.WriteLine(line);
    if (verbose)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        Console.WriteLine($"after a full GC and finalizers: working set {WorkingSetMB(),6:F0} MB");
        Console.WriteLine("  " + HeapStats.Describe());
    }
    if (idleGc)
    {
        // What a service can do to get the memory back (docs/performance.md, round 6): an aggressive GC decommits the
        // GC's free memory, but ArrayPool<T>.Shared (the managed backend's scratch buffers) gives an array up only at a
        // gen2 GC at least a minute after its last use.
        void Report(string what)
        {
            var info = GC.GetGCMemoryInfo();
            Console.WriteLine($"{what,-36} working set {WorkingSetMB(),6:F0} MB  (GC heap {info.HeapSizeBytes / 1048576.0:F0} MB, committed {info.TotalCommittedBytes / 1048576.0:F0} MB)");
        }
        void AggressiveGc(string when)
        {
            var clock = Stopwatch.StartNew();
            GC.Collect(2, GCCollectionMode.Aggressive, blocking: true, compacting: true);
            Report($"{when}: aggressive GC, {clock.Elapsed.TotalMilliseconds:F0} ms");
        }
        AggressiveGc("at once");
        Thread.Sleep(61_000);
        GC.Collect();
        Report("61 s later, a full GC");
        AggressiveGc("then");
    }
    return 0;
}

string text = BuildText(copies);
// The package's models, loaded as Pipeline does: the charlms only if a model reads them.
var models = Pipeline.SelectModels(package, null, addRequired: false, "--package");
string Model(string processor) => Path.Combine(modelDir, processor, models[processor]);

var clock = Stopwatch.StartNew();
using var tokenizer = Tokenizer.Load(Model("tokenize"), device, backend);
using var mwt = MwtExpander.Load(Model("mwt"), device, backend);
using var pretrain = backend == Backend.Managed // every processor is managed then (Pipeline.ManagedProcessors)
    ? Pretrain.LoadManaged(Path.Combine(modelDir, Pipeline.PretrainPath))
    : Pretrain.Load(Path.Combine(modelDir, Pipeline.PretrainPath), device);
// As Pipeline does: managed processors get the managed charlms (their _nocharlm models none); each backend's charlms
// are loaded only if a processor on it reads them.
bool Managed(string processor) => backend == Backend.Managed && Pipeline.ManagedProcessors.Contains(processor);
bool ManagedCharlm(string processor) => Managed(processor) && models[processor].EndsWith("_charlm");
bool torchCharlms = models.Any(kv => kv.Value.EndsWith("_charlm") && !Managed(kv.Key));
using var charlmForward = torchCharlms ? CharLanguageModel.Load(Path.Combine(modelDir, Pipeline.ForwardCharlmPath), device) : null;
using var charlmBackward = torchCharlms ? CharLanguageModel.Load(Path.Combine(modelDir, Pipeline.BackwardCharlmPath), device) : null;
using var lemma = Lemmatizer.Load(Model("lemma"), device, backend);
var managedForward = models.Keys.Any(ManagedCharlm) ? ManagedCharLanguageModel.Load(Path.Combine(modelDir, Pipeline.ForwardCharlmPath)) : null;
var managedBackward = managedForward != null ? ManagedCharLanguageModel.Load(Path.Combine(modelDir, Pipeline.BackwardCharlmPath)) : null;
using var parser = !models.ContainsKey("constituency") ? null : Managed("constituency")
    ? ConstituencyParser.LoadManaged(Model("constituency"), pretrain, managedForward!, managedBackward!)
    : ConstituencyParser.Load(Model("constituency"), pretrain, charlmForward!, charlmBackward!, device);
using var pos = Managed("pos")
    ? PosTagger.LoadManaged(Model("pos"), pretrain, ManagedCharlm("pos") ? managedForward : null, ManagedCharlm("pos") ? managedBackward : null)
    : PosTagger.Load(Model("pos"), pretrain, charlmForward, charlmBackward, device);
using var depparse = Managed("depparse")
    ? DependencyParser.LoadManaged(Model("depparse"), pretrain, ManagedCharlm("depparse") ? managedForward : null, ManagedCharlm("depparse") ? managedBackward : null)
    : DependencyParser.Load(Model("depparse"), pretrain, charlmForward, charlmBackward, device);
using var ner = Managed("ner")
    ? NerTagger.LoadManaged(Model("ner"), pretrain, ManagedCharlm("ner") ? managedForward : null, ManagedCharlm("ner") ? managedBackward : null)
    : NerTagger.Load(Model("ner"), pretrain, charlmForward, charlmBackward, device);
using var sentiment = Managed("sentiment")
    ? SentimentClassifier.LoadManaged(Model("sentiment"), pretrain, managedForward!, managedBackward!)
    : SentimentClassifier.Load(Model("sentiment"), pretrain, charlmForward!, charlmBackward!, device);
if (device?.type == DeviceType.CUDA)
    torch.cuda.synchronize();
double load = clock.Elapsed.TotalSeconds;

// Stanza's order (Pipeline.AllProcessors), without the stages the package lacks.
string[] stages = Pipeline.AllProcessors.Split(',').Where(models.ContainsKey).Where(p => processors == null || processors.Split(',').Contains(p)).ToArray();
var times = stages.ToDictionary(s => s, _ => new List<double>());
var peaks = new Dictionary<string, double> { ["load"] = PeakMB() }; // peak working set after each stage of the warm-up run
Document doc = null!;
for (int run = 0; run <= runs; run++)
{
    var input = run == 0 ? BuildText(1) : text; // run 0 warms up
    var timed = new Dictionary<string, double>();
    double Time(string stage, Action action)
    {
        clock.Restart();
        action();
        double seconds = clock.Elapsed.TotalSeconds;
        if (run == 0)
            peaks[stage] = PeakMB();
        return seconds;
    }
    // The same steps as Pipeline.Process.
    using var charlms = pos.UsesCharlm ? new CharlmCache() : null;
    // Stages left out with --processors are skipped; each stage needs only those before it (tokenize, mwt, pos, lemma).
    void Stage(string stage, Action action)
    {
        if (stages.Contains(stage))
            timed[stage] = Time(stage, action);
    }
    Stage("tokenize", () => doc = tokenizer.Process(input));
    Stage("mwt", () => mwt.Process(doc));
    Stage("pos", () => pos.Process(doc, charlms));
    Stage("lemma", () => lemma.Process(doc));
    if (parser != null)
        Stage("constituency", () => parser.Process(doc, charlms));
    Stage("depparse", () => depparse.Process(doc));
    Stage("sentiment", () => sentiment.Process(doc, charlms));
    Stage("ner", () => ner.Process(doc, charlms));
    if (run > 0)
        foreach (var s in stages)
            times[s].Add(timed[s]);
}
if (outFile != null)
    File.WriteAllText(outFile, Conllu.Write(doc));

int words = doc.Sentences.Sum(s => s.Words.Count());
Console.WriteLine($"C# StanzaSharp ({package}, {backend} backend) on {device?.ToString() ?? "cpu"}, " +
                  (backend == Backend.Managed ? $"managed threads {ManagedThreads.Count}" : $"torch threads {torch.get_num_threads()}") + $", {copies} copies: " +
                  $"{text.Length} chars, {doc.Sentences.Count} sentences, {words} words, {runs} runs");
Console.WriteLine($"{"load",-14}{load,9:F2} s {"",16} {peaks["load"],8:F0} MB peak");
double total = 0;
foreach (var s in stages)
{
    double median = Median(times[s]);
    total += median;
    Console.WriteLine($"{s,-14}{median,9:F2} s {words / median,10:F0} words/s {peaks[s],8:F0} MB peak (warm-up)");
}
Console.WriteLine($"{"total",-14}{total,9:F2} s {words / total,10:F0} words/s");
Console.WriteLine($"{"peak memory",-14}{PeakMB(),9:F0} MB (peak working set)");
return 0;

static string BuildText(int copies)
{
    // The same text as tools/benchmark.py: the golden texts, repeated, separated by blank lines.
    string golden = Path.Combine(FindRepoRoot(), "tests", "golden");
    string unit = string.Join("\n\n", new[] { "validation.txt", "corpus.txt", "tokenize_stress.txt" }.Select(f => File.ReadAllText(Path.Combine(golden, f))));
    return string.Join("\n\n", Enumerable.Repeat(unit, copies));
}

static List<string> BuildDocuments(int count)
{
    // The same texts as tools/benchmark.py --documents: the golden validation sentences ("# text = "), cycled.
    string golden = Path.Combine(FindRepoRoot(), "tests", "golden");
    var sentences = Directory.GetFiles(golden, "validation*.conllu").Order(StringComparer.Ordinal)
        .SelectMany(File.ReadLines).Where(l => l.StartsWith("# text = ")).Select(l => l["# text = ".Length..]).ToList();
    return Enumerable.Range(0, count).Select(i => sentences[i % sentences.Count]).ToList();
}

static string FindRepoRoot()
{
    foreach (var start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "StanzaSharp.slnx")))
                return dir.FullName;
    throw new DirectoryNotFoundException("Run from inside the repository: StanzaSharp.slnx not found");
}

static List<string> BuildParagraphs(int words)
{
    // The same text as tools/benchmark.py --memory: corpus.txt and validation*.txt, repeated, cut after the paragraph
    // (blank-line separated) that reaches the word count. Words are counted by whitespace, before tokenizing.
    string golden = Path.Combine(FindRepoRoot(), "tests", "golden");
    var files = new[] { "corpus.txt" }.Concat(Directory.GetFiles(golden, "validation*.txt").Select(Path.GetFileName).Order(StringComparer.Ordinal));
    var unit = files.SelectMany(f => File.ReadAllText(Path.Combine(golden, f!)).Split("\n\n"))
        .Select(p => p.Trim('\n')).Where(p => p.Trim().Length > 0).ToList();
    var result = new List<string>();
    for (int count = 0; count < words; )
        foreach (var p in unit)
        {
            if (count >= words)
                break;
            result.Add(p);
            count += WordCount(p);
        }
    return result;
}

static int WordCount(string s) => s.Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries).Length;

static List<string> Chunk(List<string> paragraphs, int words)
{
    var parts = new List<string>();
    var current = new List<string>();
    int count = 0;
    foreach (var p in paragraphs)
    {
        current.Add(p);
        count += WordCount(p);
        if (count >= words)
        {
            parts.Add(string.Join("\n\n", current));
            current.Clear();
            count = 0;
        }
    }
    if (current.Count > 0)
        parts.Add(string.Join("\n\n", current));
    return parts;
}

static bool NativeTorchLoaded() => Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Any(m => m.ModuleName.StartsWith("torch_cpu", StringComparison.OrdinalIgnoreCase) || m.ModuleName.StartsWith("libtorch_cpu", StringComparison.OrdinalIgnoreCase));

static double PeakMB() =>Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0;

static double WorkingSetMB() => Process.GetCurrentProcess().WorkingSet64 / 1048576.0;

static double Median(List<double> xs)
{
    var sorted = xs.Order().ToList();
    int n = sorted.Count;
    return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}


/// <summary>Prints the pipeline's log messages (each model load, each processor) with the working set and its peak.</summary>
sealed class MemoryLogger : Microsoft.Extensions.Logging.ILogger
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
    public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
    public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, Microsoft.Extensions.Logging.EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        using var p = Process.GetCurrentProcess();
        Console.WriteLine($"  {formatter(state, exception),-70} working set {p.WorkingSet64 / 1048576.0,6:F0} MB  peak {p.PeakWorkingSet64 / 1048576.0,6:F0} MB  GC heap {GC.GetGCMemoryInfo().HeapSizeBytes / 1048576.0,5:F0} MB  {TorchSharp.DisposeScopeManager.Statistics.TensorStatistics.ThreadTotalLiveCount} live tensors, {TorchSharp.DisposeScopeManager.Statistics.TensorStatistics.CreatedOutsideScopeCount - TorchSharp.DisposeScopeManager.Statistics.TensorStatistics.DisposedOutsideScopeCount} outside scopes");
    }
}

/// <summary>Diagnostics of the native heap on Linux (glibc): mallinfo2 and /proc/self/smaps_rollup.</summary>
static class HeapStats
{
    [System.Runtime.InteropServices.DllImport("libc.so.6", EntryPoint = "mallinfo2")]
    static extern MallInfo2 MallInfo();

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    struct MallInfo2 { public nuint Arena, Ordblks, Smblks, Hblks, Hblkhd, Usmblks, Fsmblks, Uordblks, Fordblks, Keepcost; }

    public static string Describe()
    {
        var gc = GC.GetGCMemoryInfo();
        string text = $"GC heap {gc.HeapSizeBytes / 1048576.0:F0} MB, committed {gc.TotalCommittedBytes / 1048576.0:F0} MB";
        if (!NativeHeap.CanTrim) // glibc
            return text;
        var m = MallInfo();
        double mb(nuint x) => x / 1048576.0;
        text += $"; glibc: arenas {mb(m.Arena):F0} MB (in use {mb(m.Uordblks):F0}, free {mb(m.Fordblks):F0}, trimmable top {mb(m.Keepcost):F0}), mmapped {mb(m.Hblkhd):F0} MB in {m.Hblks}";
        var rollup = File.ReadAllLines("/proc/self/smaps_rollup").Where(l => l.StartsWith("Rss:") || l.StartsWith("Anonymous:") || l.StartsWith("Private_Dirty:"));
        return text + "; smaps_rollup " + string.Join(", ", rollup.Select(l => System.Text.RegularExpressions.Regex.Replace(l, @"\s+", " ")));
    }
}
