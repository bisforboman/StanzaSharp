using System.Diagnostics;
using StanzaSharp;
using StanzaSharp.Constituency;
using StanzaSharp.Depparse;
using StanzaSharp.Lemma;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;
using TorchSharp;
using static TorchSharp.torch;

// Times each pipeline stage on a large text built from the golden corpora. The Python counterpart is
// tools/benchmark.py; both build the same text and print the same report. See docs/performance.md.
const string Usage = """
    Usage: StanzaSharp.Benchmark [--models DIR] [--copies N] [--runs N] [--threads N] [--out FILE]
                                 [--device cpu|cuda] [--no-tf32]

      --models DIR    converted models (default: models/converted/en)
      --copies N      copies of the golden texts in the input (default: 8, ~20k words)
      --runs N        timed runs after a warm-up run on one copy; medians are reported (default: 3)
      --threads N     torch intra-op threads (default: torch's default)
      --out FILE      write the last run's CoNLL-U here, to compare with Python's
      --device D      cpu (default) or cuda; cuda needs a build with STANZASHARP_CUDA=1 (docs/gpu.md)
      --no-tf32       cuda: turn off TF32 in cuDNN and cuBLAS (process-wide torch settings)
    """;

string modelDir = Path.Combine("models", "converted", "en");
int copies = 8, runs = 3, threads = 0;
string? outFile = null;
var device = torch.CPU;
bool noTf32 = false;
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
        default:
            Console.Error.WriteLine(Usage);
            return 2;
    }
}
if (threads > 0)
    torch.set_num_threads(threads);
if (noTf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = false;

string text = BuildText(copies);
string Model(string relative) => Path.Combine(modelDir, relative);

var clock = Stopwatch.StartNew();
using var tokenizer = Tokenizer.Load(Model("tokenize/combined_nocharlm"), device);
using var mwt = MwtExpander.Load(Model("mwt/combined"), device);
using var pretrain = Pretrain.Load(Model("pretrain/conll17"), device);
using var charlmForward = CharLanguageModel.Load(Model("forward_charlm/1billion"), device);
using var charlmBackward = CharLanguageModel.Load(Model("backward_charlm/1billion"), device);
using var pos = PosTagger.Load(Model("pos/combined_charlm"), pretrain, charlmForward, charlmBackward, device);
using var lemma = Lemmatizer.Load(Model("lemma/combined_nocharlm"), device);
using var depparse = DependencyParser.Load(Model("depparse/combined_charlm"), pretrain, charlmForward, charlmBackward, device);
using var parser = ConstituencyParser.Load(Model("constituency/ptb3-revised_charlm"), pretrain, charlmForward, charlmBackward, device);
if (device.type == DeviceType.CUDA)
    torch.cuda.synchronize();
double load = clock.Elapsed.TotalSeconds;

string[] stages = ["tokenize", "mwt", "pos", "lemma", "depparse", "constituency"];
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
    using var charlms = new CharlmCache();
    timed["tokenize"] = Time("tokenize", () => doc = tokenizer.Process(input));
    timed["mwt"] = Time("mwt", () => mwt.Process(doc));
    timed["pos"] = Time("pos", () => pos.Process(doc, charlms));
    timed["lemma"] = Time("lemma", () => lemma.Process(doc));
    timed["depparse"] = Time("depparse", () => depparse.Process(doc));
    timed["constituency"] = Time("constituency", () => parser.Process(doc, charlms));
    if (run > 0)
        foreach (var s in stages)
            times[s].Add(timed[s]);
}
if (outFile != null)
    File.WriteAllText(outFile, Conllu.Write(doc));

int words = doc.Sentences.Sum(s => s.Words.Count());
Console.WriteLine($"C# StanzaSharp on {device}, torch threads {torch.get_num_threads()}, {copies} copies: " +
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

static string FindRepoRoot()
{
    foreach (var start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "StanzaSharp.slnx")))
                return dir.FullName;
    throw new DirectoryNotFoundException("Run from inside the repository: StanzaSharp.slnx not found");
}

static double PeakMB() => Process.GetCurrentProcess().PeakWorkingSet64 / 1048576.0;

static double Median(List<double> xs)
{
    var sorted = xs.Order().ToList();
    int n = sorted.Count;
    return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}
