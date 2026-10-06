using System.Diagnostics;
using StanzaSharp;
using StanzaSharp.Constituency;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;
using TorchSharp;

// Times each pipeline stage on a large text built from the golden corpora. The Python counterpart is
// tools/benchmark.py; both build the same text and print the same report. See docs/performance.md.
const string Usage = """
    Usage: StanzaSharp.Benchmark [--models DIR] [--copies N] [--runs N] [--threads N] [--out FILE]

      --models DIR    converted models (default: models/converted/en)
      --copies N      copies of the golden texts in the input (default: 8, ~20k words)
      --runs N        timed runs after a warm-up run on one copy; medians are reported (default: 3)
      --threads N     torch intra-op threads (default: torch's default)
      --out FILE      write the last run's CoNLL-U here, to compare with Python's
    """;

string modelDir = Path.Combine("models", "converted", "en");
int copies = 8, runs = 3, threads = 0;
string? outFile = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models" when i + 1 < args.Length: modelDir = args[++i]; break;
        case "--copies" when i + 1 < args.Length: copies = int.Parse(args[++i]); break;
        case "--runs" when i + 1 < args.Length: runs = int.Parse(args[++i]); break;
        case "--threads" when i + 1 < args.Length: threads = int.Parse(args[++i]); break;
        case "--out" when i + 1 < args.Length: outFile = args[++i]; break;
        default:
            Console.Error.WriteLine(Usage);
            return 2;
    }
}
if (threads > 0)
    torch.set_num_threads(threads);

string text = BuildText(copies);
string Model(string relative) => Path.Combine(modelDir, relative);

var clock = Stopwatch.StartNew();
using var tokenizer = Tokenizer.Load(Model("tokenize/combined_nocharlm"));
using var mwt = MwtExpander.Load(Model("mwt/combined"));
using var pretrain = Pretrain.Load(Model("pretrain/conll17"));
using var charlmForward = CharLanguageModel.Load(Model("forward_charlm/1billion"));
using var charlmBackward = CharLanguageModel.Load(Model("backward_charlm/1billion"));
using var pos = PosTagger.Load(Model("pos/combined_charlm"), pretrain, charlmForward, charlmBackward);
using var parser = ConstituencyParser.Load(Model("constituency/ptb3-revised_charlm"), pretrain, charlmForward, charlmBackward);
double load = clock.Elapsed.TotalSeconds;

string[] stages = ["tokenize", "mwt", "pos", "constituency"];
var times = stages.ToDictionary(s => s, _ => new List<double>());
Document doc = null!;
for (int run = 0; run <= runs; run++)
{
    var input = run == 0 ? BuildText(1) : text; // run 0 warms up
    var timed = new Dictionary<string, double>();
    double Time(Action action)
    {
        clock.Restart();
        action();
        return clock.Elapsed.TotalSeconds;
    }
    timed["tokenize"] = Time(() => doc = tokenizer.Process(input));
    timed["mwt"] = Time(() => mwt.Process(doc));
    timed["pos"] = Time(() => pos.Process(doc));
    timed["constituency"] = Time(() => parser.Process(doc));
    if (run > 0)
        foreach (var s in stages)
            times[s].Add(timed[s]);
}
if (outFile != null)
    File.WriteAllText(outFile, Conllu.Write(doc));

int words = doc.Sentences.Sum(s => s.Words.Count());
Console.WriteLine($"C# StanzaSharp, torch threads {torch.get_num_threads()}, {copies} copies: " +
                  $"{text.Length} chars, {doc.Sentences.Count} sentences, {words} words, {runs} runs");
Console.WriteLine($"{"load",-14}{load,9:F2} s");
double total = 0;
foreach (var s in stages)
{
    double median = Median(times[s]);
    total += median;
    Console.WriteLine($"{s,-14}{median,9:F2} s {words / median,10:F0} words/s");
}
Console.WriteLine($"{"total",-14}{total,9:F2} s {words / total,10:F0} words/s");
Console.WriteLine($"{"peak memory",-14}{Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0,9:F0} MB (peak working set)");
return 0;

static string BuildText(int copies)
{
    // The same text as tools/benchmark.py: the golden texts, repeated, separated by blank lines.
    string golden = Path.Combine(FindRepoRoot(), "tests", "golden");
    string unit = string.Join("\n\n", new[] { "validation.txt", "corpus.txt", "tokenize_stress.txt" }.Select(f => File.ReadAllText(Path.Combine(golden, f))));
    return string.Join("\n\n", Enumerable.Repeat(unit, copies));
}

static string FindRepoRoot()
{
    foreach (var start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "StanzaSharp.slnx")))
                return dir.FullName;
    throw new DirectoryNotFoundException("Run from inside the repository: StanzaSharp.slnx not found");
}

static double Median(List<double> xs)
{
    var sorted = xs.Order().ToList();
    int n = sorted.Count;
    return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}
