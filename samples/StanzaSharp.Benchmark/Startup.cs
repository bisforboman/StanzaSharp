using System.Diagnostics;
using System.Globalization;
using System.Runtime;
using System.Security.Cryptography;
using System.Text;
using StanzaSharp;

/// <summary>
/// Startup cost in a fresh process (docs/performance.md, round 17): <see cref="Pipeline.Load"/>, then a few Process calls,
/// each with the methods the JIT compiled meanwhile (<see cref="JitInfo"/>). tools/startup.py runs it once per process
/// under each setting (ReadyToRun build, DOTNET_* knobs) and tabulates the medians.
/// <code>
/// StanzaSharp.Benchmark startup [--models DIR] [--text one|bench] [--calls N] [--threads N] [--hash]
/// </code>
/// <c>--text one</c>: one 13-word sentence per call; <c>bench</c>: the benchmark's 2,009 words (stanzasharp benchmark's
/// text). <c>--hash</c>: instead, the SHA-256 of the CoNLL-U of every tests/golden/*.txt, each processed alone.
/// Prints one line: <c>load=ms jit=methods/ms cpu=ms peak=MB calls=ms/methods/jitms/cpums,… hash=…</c> (cpu: the process's CPU time,
/// all threads, which other load on the machine disturbs less than the wall time).
/// </summary>
internal static class Startup
{
    public static int Run(string[] args, string repoRoot)
    {
        string modelDir = Path.Combine("models", "converted", "en"), text = "one";
        int calls = 5;
        int? threads = null;
        bool hash = false;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--models": modelDir = args[++i]; break;
                case "--text": text = args[++i]; break;
                case "--calls": calls = int.Parse(args[++i]); break;
                case "--threads": threads = int.Parse(args[++i]); break;
                case "--hash": hash = true; break;
                default: Console.Error.WriteLine($"Unknown option {args[i]}"); return 2;
            }
        }

        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture; // tools/startup.py parses the numbers
        string golden = Path.Combine(repoRoot, "tests", "golden");
        var (m0, j0) = Jit();
        var cpu0 = Process.GetCurrentProcess().TotalProcessorTime;
        long start = Stopwatch.GetTimestamp();
        using var nlp = Pipeline.Load(modelDir, new PipelineOptions { Threads = threads });
        double load = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        var (m1, j1) = Jit();
        double loadCpu = (Process.GetCurrentProcess().TotalProcessorTime - cpu0).TotalMilliseconds;
        double loadPeak = Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0; // MiB; 0 on macOS
        var line = new StringBuilder($"load={load:F0} jit={m1 - m0}/{j1 - j0:F0} cpu={loadCpu:F0} peak={loadPeak:F0}");

        if (hash)
        {
            var all = new StringBuilder();
            foreach (var file in Directory.GetFiles(golden, "*.txt").Order(StringComparer.Ordinal))
                all.Append(Conllu.Write(nlp.Process(Encoding.UTF8.GetString(File.ReadAllBytes(file)))));
            Console.WriteLine($"{line} hash={Sha(all.ToString())}");
            return 0;
        }

        string input = text == "one" ? "The quick brown fox jumped over the lazy dog near the river bank." : BenchText(golden);
        string? first = null;
        line.Append(" calls=");
        for (int i = 0; i < calls; i++)
        {
            var (ma, ja) = Jit();
            var cpu = Process.GetCurrentProcess().TotalProcessorTime;
            long s = Stopwatch.GetTimestamp();
            var doc = nlp.Process(input);
            double ms = Stopwatch.GetElapsedTime(s).TotalMilliseconds;
            var (mb, jb) = Jit();
            double cpuMs = (Process.GetCurrentProcess().TotalProcessorTime - cpu).TotalMilliseconds;
            line.Append(i == 0 ? "" : ",").Append($"{ms:F1}/{mb - ma}/{jb - ja:F0}/{cpuMs:F0}");
            first ??= Sha(Conllu.Write(doc));
        }

        Console.WriteLine($"{line} hash={first}");
        return 0;
    }

    private static (long Methods, double Ms) Jit() => (JitInfo.GetCompiledMethodCount(), JitInfo.GetCompilationTime().TotalMilliseconds);

    private static string Sha(string s) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(s)))[..16];

    // stanzasharp benchmark's text (BenchmarkCommand.Paragraphs): the golden texts cut after the paragraph that reaches 1,500 words.
    private static string BenchText(string golden)
    {
        var unit = new[] { "validation.txt", "corpus.txt", "tokenize_stress.txt" }
            .SelectMany(n => File.ReadAllText(Path.Combine(golden, n)).Split("\n\n"))
            .Select(p => p.Trim('\n'))
            .Where(p => p.Trim().Length > 0)
            .ToList();
        var paragraphs = new List<string>();
        for (int count = 0, i = 0; count < 1500; i++)
        {
            paragraphs.Add(unit[i % unit.Count]);
            count += paragraphs[^1].Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries).Length;
        }

        return string.Join("\n\n", paragraphs);
    }
}
