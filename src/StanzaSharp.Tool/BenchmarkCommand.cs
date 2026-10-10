using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Tool;

/// <summary>
/// <c>benchmark [--package] [--processors] [--models DIR] [--threads N] [--words N] [--python PATH] [--quick]
/// [--json FILE]</c>: times StanzaSharp (managed backend) and, when Python Stanza is available, Stanza on the same text
/// with the same models and threads, alternating their runs, and prints a Markdown block with the machine, both sides'
/// times and the ratio. Exit code 0, or 2 for bad arguments or setup.
/// </summary>
internal static class BenchmarkCommand
{
    /// <summary>One side's medians: <see cref="Stages"/> in seconds per run, per-call times in ms, peak memory in MB.</summary>
    internal sealed record Side(int Threads, double Load, Dictionary<string, double> Stages, double CallMedian, double CallP90, double PeakMB)
    {
        public double Total => Stages.Values.Sum();
    }

    internal sealed record Machine(string Cpu, int? Cores, int LogicalProcessors, double RamGB, string Os, string DotNet, string Simd, string StanzaSharp);

    /// <summary>
    /// <paramref name="Stanza"/> is null when Python Stanza didn't run (<paramref name="Python"/> says why); otherwise
    /// <paramref name="Python"/> names its versions and <paramref name="Output"/> says whether the CoNLL-U matched.
    /// </summary>
    internal sealed record Result(Machine Machine, string Package, string Processors, int Words, int Sentences,
        int Runs, int Calls, Side StanzaSharp, Side? Stanza, string Python, string? Output);

    internal static int Run(string[] args, string usage, TextWriter output)
    {
        string? processors = null, json = null;
        string package = Pipeline.DefaultPackage, models = DownloadCommand.DefaultDir, python = "python";
        int threads = 0, words = 1500, runs = 3, calls = 50;
        bool quick = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--package" && i + 1 < args.Length)
                package = args[++i];
            else if (args[i] == "--processors" && i + 1 < args.Length)
                processors = args[++i];
            else if (args[i] == "--models" && i + 1 < args.Length)
                models = args[++i];
            else if (args[i] == "--python" && i + 1 < args.Length)
                python = args[++i];
            else if (args[i] == "--json" && i + 1 < args.Length)
                json = args[++i];
            else if (args[i] == "--threads" && i + 1 < args.Length && int.TryParse(args[i + 1], out threads) && threads > 0)
                i++;
            else if (args[i] == "--words" && i + 1 < args.Length && int.TryParse(args[i + 1], out words) && words > 0)
                i++;
            else if (args[i] == "--quick")
                quick = true;
            else
                return Fail($"Unexpected argument: {args[i]}\n\n{usage}");
        }
        if (quick)
            (words, runs, calls) = (Math.Min(words, 300), 1, 10);

        string list;
        try
        {
            list = CompareCommand.ProcessorList(package, processors);
        }
        catch (ArgumentException e)
        {
            return Fail(e.Message);
        }
        if (!Directory.Exists(models))
            return Fail($"Model directory not found: {models}. Download the models with: stanzasharp download {models} --package {package}");

        var paragraphs = Paragraphs(words);
        var text = string.Join("\n\n", paragraphs);
        var warmUp = string.Join("\n\n", Paragraphs(Math.Max(words / 10, 50)));

        var stageLog = new StageLog();
        double load;
        Pipeline nlp;
        try
        {
            Console.Error.WriteLine($"Loading StanzaSharp ({package}: {list}) from {models}...");
            var watch = Stopwatch.StartNew();
            nlp = Pipeline.Load(models, new PipelineOptions { Package = package, Processors = list, Threads = threads > 0 ? threads : null, Logger = stageLog });
            load = watch.Elapsed.TotalSeconds;
        }
        catch (Exception e) when (e is IOException or InvalidDataException or InvalidOperationException or ArgumentException or UnauthorizedAccessException)
        {
            return Fail(e.Message);
        }

        using (nlp)
        {
            using var stanza = PythonStanza.Start(python, models, package, list, threads, out var pythonNote);
            var csRuns = new List<Dictionary<string, double>>();
            var pyRuns = new List<Dictionary<string, double>>();
            Document doc = null!;
            string? pyConllu = null;
            // A warm-up run each, then the timed runs, alternating: both sides see the same machine state.
            for (int run = 0; run <= runs; run++)
            {
                var input = run == 0 ? warmUp : text;
                Console.Error.Write(run == 0 ? "Warm-up run" : $"Run {run} of {runs}");
                stageLog.Stages.Clear();
                doc = nlp.Process(input);
                if (run > 0)
                    csRuns.Add(new(stageLog.Stages));
                Console.Error.Write($": StanzaSharp {stageLog.Stages.Values.Sum():F1} s");
                if (stanza != null)
                {
                    var reply = stanza.Send(new { op = "run", text = input });
                    var stages = reply.GetProperty("stages").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetDouble());
                    if (run > 0)
                        pyRuns.Add(stages);
                    pyConllu = reply.GetProperty("conllu").GetString();
                    Console.Error.Write($", Stanza {stages.Values.Sum():F1} s");
                }
                Console.Error.WriteLine();
            }

            // One short sentence per call: the text's sentences of 5-25 words, cycled; a few calls first to warm up.
            var sentences = doc.Sentences.Select(s => s.Text ?? "").Where(t => WordCount(t) is >= 5 and <= 25).ToList();
            if (sentences.Count == 0)
                sentences = [warmUp];
            var callTexts = Enumerable.Range(0, calls).Select(i => sentences[i % sentences.Count]).ToList();
            var warmUpCalls = callTexts.Take(5).ToList();
            Console.Error.WriteLine($"{calls} calls on one sentence each...");
            foreach (var t in warmUpCalls)
                nlp.Process(t);
            var csCalls = callTexts.Select(t =>
            {
                long start = Stopwatch.GetTimestamp();
                nlp.Process(t);
                return Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }).ToList();
            var cs = new Side(ManagedThreads.Count, load, MedianStages(csRuns), Median(csCalls), Percentile(csCalls, 0.9), PeakWorkingSetMB());

            Side? py = null;
            string? outputNote = null;
            if (stanza != null)
            {
                stanza.Send(new { op = "calls", texts = warmUpCalls });
                var reply = stanza.Send(new { op = "calls", texts = callTexts });
                var pyCalls = reply.GetProperty("ms").EnumerateArray().Select(e => e.GetDouble()).ToList();
                py = new Side(stanza.Threads, stanza.Load, MedianStages(pyRuns), Median(pyCalls), Percentile(pyCalls, 0.9), reply.GetProperty("peak_mb").GetDouble());
                var conllu = Conllu.Write(doc);
                if (text.Any(char.IsHighSurrogate))
                    conllu = ConlluDiff.ToCodePointOffsets(conllu, text);
                outputNote = ConlluDiff.Find(pyConllu!, conllu) is { } d
                    ? $"different: {d.DifferentSentences:N0} of {d.Sentences:N0} sentences differ (first at line {d.Line:N0} of the CoNLL-U; stanzasharp compare shows it)"
                    : "identical CoNLL-U on both sides";
                pythonNote = stanza.Versions;
            }

            var result = new Result(DescribeMachine(), package, list, doc.Sentences.Sum(s => s.Words.Count()), doc.Sentences.Count,
                runs, calls, cs, py, pythonNote, outputNote);
            output.Write(Format(result));
            if (json != null)
                File.WriteAllText(json, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
            return 0;
        }
    }

    /// <summary>The Markdown block the command prints.</summary>
    internal static string Format(Result r)
    {
        var m = r.Machine;
        var s = new StringBuilder();
        s.AppendLine("### StanzaSharp benchmark");
        s.AppendLine();
        s.AppendLine("| | |");
        s.AppendLine("|---|---|");
        var c = CultureInfo.InvariantCulture;
        s.AppendLine(c, $"| CPU | {m.Cpu}: {(m.Cores is { } cores ? $"{cores} cores, " : "")}{m.LogicalProcessors} logical processors |");
        s.AppendLine(c, $"| RAM | {m.RamGB:F1} GB |");
        s.AppendLine(c, $"| OS | {m.Os} |");
        s.AppendLine(c, $"| .NET | {m.DotNet}, SIMD {m.Simd} |");
        s.AppendLine(c, $"| StanzaSharp | {m.StanzaSharp}, managed backend |");
        s.AppendLine(c, $"| Python Stanza | {r.Python} |");
        s.AppendLine(c, $"| Run | {r.Package} package ({r.Processors}); {r.Words:N0} words in {r.Sentences:N0} sentences, median of " +
                        $"{r.Runs} timed {(r.Runs == 1 ? "run" : "runs")} after a warm-up{(r.Stanza != null ? ", the two sides alternating" : "")}; " +
                        $"{r.Calls} calls on one sentence each |");
        if (r.Output != null)
            s.AppendLine(c, $"| Output | {r.Output} |");
        s.AppendLine();

        var cs = r.StanzaSharp;
        var py = r.Stanza;
        s.AppendLine(py == null ? "| | StanzaSharp |" : "| | StanzaSharp | Python Stanza | StanzaSharp's advantage |");
        s.AppendLine(py == null ? "|---|---:|" : "|---|---:|---:|---:|");
        s.AppendLine(c, $"| threads | {cs.Threads} |{(py == null ? "" : $" {py.Threads} | |")}");
        void Row(string name, double a, double? b, string unit, string format, bool higherIsBetter = false)
        {
            string Value(double x) => x.ToString(format, CultureInfo.InvariantCulture) + unit;
            s.Append($"| {name} | {Value(a)} |");
            if (b is { } v)
            {
                double ratio = higherIsBetter ? a / v : v / a;
                s.Append($" {Value(v)} | {(double.IsFinite(ratio) && ratio > 0 ? ratio.ToString("F2", CultureInfo.InvariantCulture) + "x" : "-")} |");
            }
            s.AppendLine();
        }
        Row("load", cs.Load, py?.Load, " s", "F2");
        foreach (var stage in cs.Stages.Keys)
            Row(stage, cs.Stages[stage], py?.Stages.GetValueOrDefault(stage), " s", "F2");
        Row("**total**", cs.Total, py?.Total, " s", "F2");
        Row("words/s", r.Words / cs.Total, py == null ? null : r.Words / py.Total, "", "N0", higherIsBetter: true);
        Row("per call, median", cs.CallMedian, py?.CallMedian, " ms", "F1");
        Row("per call, p90", cs.CallP90, py?.CallP90, " ms", "F1");
        Row("peak memory", cs.PeakMB, py?.PeakMB, " MB", "N0");
        return s.ToString();
    }

    /// <summary>The benchmark's text: the golden texts, as samples/StanzaSharp.Benchmark builds it, repeated or cut after
    /// the paragraph that reaches <paramref name="words"/> (counted by whitespace, before tokenizing).</summary>
    internal static List<string> Paragraphs(int words)
    {
        var unit = new[] { "validation.txt", "corpus.txt", "tokenize_stress.txt" }
            .SelectMany(name =>
            {
                using var reader = new StreamReader(typeof(BenchmarkCommand).Assembly.GetManifestResourceStream("benchmark/" + name)!);
                return reader.ReadToEnd().Split("\n\n");
            })
            .Select(p => p.Trim('\n')).Where(p => p.Trim().Length > 0).ToList();
        var result = new List<string>();
        for (int count = 0, i = 0; count < words; i++)
        {
            result.Add(unit[i % unit.Count]);
            count += WordCount(result[^1]);
        }
        return result;
    }

    private static int WordCount(string s) => s.Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries).Length;

    private static Dictionary<string, double> MedianStages(List<Dictionary<string, double>> runs) =>
        runs[0].Keys.ToDictionary(stage => stage, stage => Median(runs.Select(r => r.GetValueOrDefault(stage)).ToList()));

    internal static double Median(List<double> xs)
    {
        var sorted = xs.Order().ToList();
        int n = sorted.Count;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
    }

    internal static double Percentile(List<double> xs, double p) => xs.Order().ElementAt((int)Math.Ceiling(p * xs.Count) - 1);

    private static int Fail(string message)
    {
        Console.Error.WriteLine(message);
        return 2;
    }

    // ---- The machine ----

    internal static Machine DescribeMachine()
    {
        var (cpu, cores) = OperatingSystem.IsWindows() ? (WindowsCpuName(), WindowsCores())
            : OperatingSystem.IsMacOS() ? (Command("sysctl", "-n machdep.cpu.brand_string")?.Trim(), int.TryParse(Command("sysctl", "-n hw.physicalcpu"), out var c) ? c : (int?)null)
            : Command("lscpu", "") is { } lscpu ? ParseLscpu(lscpu)
            : (ParseCpuInfo(File.Exists("/proc/cpuinfo") ? File.ReadAllText("/proc/cpuinfo") : ""), null);
        var version = typeof(Pipeline).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "?";
        return new Machine(string.IsNullOrWhiteSpace(cpu) ? RuntimeInformation.ProcessArchitecture + " CPU (model unknown)" : cpu, cores,
            Environment.ProcessorCount, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1073741824.0,
            $"{RuntimeInformation.OSDescription} ({RuntimeInformation.OSArchitecture})", RuntimeInformation.FrameworkDescription,
            Gemm.Detect().ToString(), version);
    }

    /// <summary>The model name(s) and physical cores from <c>lscpu</c> (run with LC_ALL=C).</summary>
    internal static (string? Cpu, int? Cores) ParseLscpu(string lscpu)
    {
        var fields = lscpu.Split('\n').Select(l => l.Split(':', 2)).Where(f => f.Length == 2)
            .Select(f => (Key: f[0].Trim(), Value: f[1].Trim())).ToList();
        string? First(string key) => fields.FirstOrDefault(f => f.Key == key).Value;
        // Hybrid CPUs list a model per core type.
        var names = fields.Where(f => f.Key == "Model name" && f.Value != "-").Select(f => f.Value).Distinct().ToList();
        // ponytail: the first block's cores × sockets; a hybrid CPU's other core types aren't added.
        int? cores = int.TryParse(First("Core(s) per socket") ?? First("Core(s) per cluster"), out var perSocket)
            ? perSocket * (int.TryParse(First("Socket(s)") ?? First("Cluster(s)"), out var sockets) ? sockets : 1) : null;
        return (names.Count > 0 ? string.Join(" + ", names) : null, cores);
    }

    /// <summary>The first "model name" of /proc/cpuinfo (x64; Arm has none).</summary>
    internal static string? ParseCpuInfo(string cpuinfo) =>
        cpuinfo.Split('\n').Select(l => l.Split(':', 2)).FirstOrDefault(f => f.Length == 2 && f[0].Trim() == "model name")?[1].Trim();

    private static string? Command(string file, string arguments)
    {
        try
        {
            var start = new ProcessStartInfo(file, arguments) { RedirectStandardOutput = true, RedirectStandardError = true };
            start.Environment["LC_ALL"] = "C";
            using var process = Process.Start(start)!;
            var text = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            return process.ExitCode == 0 ? text : null;
        }
        catch (Win32Exception)
        {
            return null;
        }
    }

    private static string? WindowsCpuName() =>
        OperatingSystem.IsWindows()
            ? (Microsoft.Win32.Registry.GetValue(@"HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString", null) as string)?.Trim()
            : null;

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformation(IntPtr buffer, ref uint length);

    /// <summary>Physical cores: the RelationProcessorCore records of GetLogicalProcessorInformation.</summary>
    private static int? WindowsCores()
    {
        // ponytail: the calling thread's processor group only (up to 64 logical processors); Ex for bigger machines.
        uint length = 0;
        GetLogicalProcessorInformation(IntPtr.Zero, ref length);
        var buffer = Marshal.AllocHGlobal((int)length);
        try
        {
            if (!GetLogicalProcessorInformation(buffer, ref length))
                return null;
            int size = IntPtr.Size == 8 ? 32 : 24; // SYSTEM_LOGICAL_PROCESSOR_INFORMATION; Relationship follows the mask
            return Enumerable.Range(0, (int)length / size).Count(i => Marshal.ReadInt32(buffer, i * size + IntPtr.Size) == 0);
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    [DllImport("libc", EntryPoint = "getrusage")]
    private static extern int GetRusage(int who, long[] usage);

    /// <summary>The process's peak working set; on macOS, where .NET doesn't report it, getrusage's ru_maxrss (bytes).</summary>
    private static double PeakWorkingSetMB()
    {
        if (!OperatingSystem.IsMacOS())
            return Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0;
        var usage = new long[18]; // struct rusage: two timevals (16 bytes each), then ru_maxrss
        return GetRusage(0, usage) == 0 ? usage[4] / 1048576.0 : 0;
    }

    /// <summary>Collects the pipeline's per-processor times ("{Processor} took {Milliseconds} ms", logged at Debug).</summary>
    private sealed class StageLog : ILogger
    {
        public readonly Dictionary<string, double> Stages = [];
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => logLevel == LogLevel.Debug;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            if (state is IReadOnlyList<KeyValuePair<string, object?>> values && values.Count == 3 && values[0].Key == "Processor")
                Stages[(string)values[0].Value!] = (double)values[1].Value! / 1000;
        }
    }

    /// <summary>benchmark.py, loaded once and answering one command per line; null from <see cref="Start"/> if unusable.</summary>
    private sealed class PythonStanza : IDisposable
    {
        private readonly Process _process;
        private readonly DirectoryInfo _dir;
        public double Load { get; private set; }
        public string Versions { get; private set; } = "";
        public int Threads { get; private set; }

        private PythonStanza(Process process, DirectoryInfo dir) => (_process, _dir) = (process, dir);

        public static PythonStanza? Start(string python, string models, string package, string list, int threads, out string note)
        {
            if (CompareCommand.StanzaModelsProblem(ref models, package, list) is { } problem)
            {
                note = "not run: " + problem;
                Console.Error.WriteLine($"Python Stanza: {note}");
                return null;
            }
            var dir = Directory.CreateTempSubdirectory("stanzasharp-benchmark-");
            CompareCommand.WriteResources(dir.FullName, "benchmark.py", "stanza_resources_en.json");
            var start = new ProcessStartInfo(python) { RedirectStandardInput = true, RedirectStandardOutput = true, StandardInputEncoding = new UTF8Encoding(false) };
            foreach (var arg in new[] { Path.Combine(dir.FullName, "benchmark.py"), models, package, list, threads.ToString(CultureInfo.InvariantCulture), ModelDownloader.StanzaVersion })
                start.ArgumentList.Add(arg);
            Console.Error.WriteLine($"Loading Stanza {ModelDownloader.StanzaVersion} with {python}...");
            PythonStanza? stanza = null;
            try
            {
                stanza = new PythonStanza(Process.Start(start)!, dir);
                var ready = stanza.Read();
                stanza.Load = ready.GetProperty("load").GetDouble();
                stanza.Threads = ready.GetProperty("threads").GetInt32();
                stanza.Versions = $"Stanza {ready.GetProperty("stanza").GetString()}, torch {ready.GetProperty("torch").GetString()}, " +
                                  $"Python {ready.GetProperty("python").GetString()}";
                var processors = string.Join(",", ready.GetProperty("processors").EnumerateArray().Select(e => e.GetString()));
                if (processors != list)
                    throw new InvalidDataException($"Stanza loaded {processors}, StanzaSharp {list}");
                note = stanza.Versions;
                return stanza;
            }
            catch (Exception e) when (e is Win32Exception or InvalidDataException)
            {
                stanza?.Dispose();
                if (stanza == null)
                    dir.Delete(recursive: true);
                note = $"not run: {(e is Win32Exception ? $"could not start {python}" : e.Message)}. " +
                       $"It needs Python with pip install stanza=={ModelDownloader.StanzaVersion} (--python PATH)";
                Console.Error.WriteLine($"Python Stanza: {note}");
                return null;
            }
        }

        public JsonElement Send(object command)
        {
            _process.StandardInput.WriteLine(JsonSerializer.Serialize(command));
            _process.StandardInput.Flush();
            return Read();
        }

        private JsonElement Read()
        {
            var line = _process.StandardOutput.ReadLine();
            if (line == null)
            {
                _process.WaitForExit();
                throw new InvalidDataException(_process.ExitCode == 3 ? "see the message above" : $"Python exited with code {_process.ExitCode}; its error is above");
            }
            return JsonDocument.Parse(line).RootElement;
        }

        public void Dispose()
        {
            try
            {
                _process.StandardInput.Close(); // benchmark.py ends when its input does
                if (!_process.WaitForExit(10_000))
                    _process.Kill();
            }
            catch (InvalidOperationException)
            {
            }
            _process.Dispose();
            _dir.Delete(recursive: true);
        }
    }
}
