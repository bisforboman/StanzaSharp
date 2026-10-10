using System.Diagnostics;
using System.Text.Encodings.Web;
using System.Text.Json;
using StanzaSharp;
using StanzaSharp.Constituency;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;

/// <summary>
/// <c>constituency-divergence</c>: the C# side of tools/constituency_divergence.py (docs/backends.md, "constituency").
/// Runs tokenize,mwt,pos,constituency on each document of a JSON list of texts, then parses the document's words and XPOS
/// again with the step hook (charlms computed, as Stanza's parser does), and writes the same JSON as the Python side: each
/// sentence's words and XPOS, its tree, and per step [best legal transition, score, second legal transition, score].
/// </summary>
internal static class ConstituencyDivergence
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark constituency-divergence --docs DOCS.json --out OUT.json [--models DIR]
                                     [--backend torch|managed] [--double output,last-output,stacks,reduce] [--torch-charlm] [--threads N]

          --double  managed: those per-step layers' sums in double (ManagedConstituencyNet.Double, study only)
          --torch-charlm  managed: the parser reads TorchSharp charlm outputs instead of the managed charlms
        """;

    public static int Run(string[] args)
    {
        string modelDir = Path.Combine("models", "converted", "en");
        string? docsFile = null, outFile = null, variant = null;
        bool managed = true, torchCharlm = false;
        int threads = 0;
        var sums = ManagedConstituencyNet.DoubleSums.None;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--docs" when i + 1 < args.Length: docsFile = args[++i]; break;
                case "--out" when i + 1 < args.Length: outFile = args[++i]; break;
                case "--models" when i + 1 < args.Length: modelDir = args[++i]; break;
                case "--backend" when i + 1 < args.Length: managed = args[++i] == "managed"; break;
                case "--threads" when i + 1 < args.Length: threads = int.Parse(args[++i]); break;
                case "--torch-charlm": torchCharlm = true; break;
                case "--double" when i + 1 < args.Length:
                    variant = "double " + args[++i];
                    foreach (var part in args[i].Split(','))
                        sums |= Enum.Parse<ManagedConstituencyNet.DoubleSums>(part.Replace("-", ""), ignoreCase: true);
                    break;
                default:
                    Console.Error.WriteLine(Usage);
                    return 2;
            }
        }
        if (torchCharlm)
            variant = (variant == null ? "" : variant + ", ") + "torch charlm";
        if (docsFile == null || outFile == null || (sums != 0 || torchCharlm) && !managed)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        var texts = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(docsFile))!;
        ManagedConstituencyNet.Double = sums;
        using var nlp = Pipeline.Load(modelDir, new PipelineOptions
        {
            Processors = "tokenize,mwt,pos,constituency", Backend = managed ? PipelineBackend.Managed : CudaBackend.Cpu,
            Threads = threads > 0 ? threads : null,
        });
        var parser = nlp.Parser!;
        using var forwardLm = torchCharlm ? CharLanguageModel.Load(Path.Combine(modelDir, Pipeline.ForwardCharlmPath)) : null;
        using var backwardLm = torchCharlm ? CharLanguageModel.Load(Path.Combine(modelDir, Pipeline.BackwardCharlmPath)) : null;

        nlp.Process("Warm up the pipeline.");
        var docs = new List<object>(texts.Count);
        double seconds = 0, parseSeconds = 0;
        int cacheMismatches = 0;
        var clock = new Stopwatch();
        for (int n = 0; n < texts.Count; n++)
        {
            clock.Restart();
            var doc = nlp.Process(texts[n]);
            seconds += clock.Elapsed.TotalSeconds;
            var sentences = doc.Sentences.Where(s => s.Tokens.Count > 0).ToList();
            var tagged = sentences.Select(s => (IReadOnlyList<(string, string)>)s.Words.Select(w => (w.Text, w.Xpos!)).ToList()).ToList();
            var steps = sentences.Select(_ => new List<double[]>()).ToList();
            using var cache = torchCharlm ? TorchCharlms(forwardLm!, backwardLm!, sentences) : null;
            clock.Restart();
            var trees = parser.Parse(tagged, charlms: cache, cacheKeys: cache == null ? null : sentences,
                onStep: (i, row, legal) => steps[i].Add(Decision(row, legal)));
            parseSeconds += clock.Elapsed.TotalSeconds;
            var strings = trees.Select(t => t?.ToString()).ToList();
            // The pipeline's parser read the tagger's charlm cache; its trees should be the same.
            cacheMismatches += sentences.Where((s, i) => s.Constituency?.ToString() != strings[i]).Count();
            docs.Add(new
            {
                sents = tagged.Select(s => s.Select(x => new[] { x.Item1, x.Item2 })),
                trees = strings,
                steps,
            });
            if (n % 100 == 0)
                Console.Error.WriteLine($"{n}/{texts.Count} documents");
        }

        var meta = new
        {
            backend = managed ? "managed" : "torch", variant, threads = managed ? ManagedThreads.Count : torch.get_num_threads(),
            seconds, constituency_seconds = parseSeconds, cache_mismatches = cacheMismatches,
        };
        File.WriteAllText(outFile, JsonSerializer.Serialize(new { meta, docs }, new JsonSerializerOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        Console.WriteLine(JsonSerializer.Serialize(meta));
        return 0;
    }

    /// <summary>--torch-charlm: the sentences' charlm inputs from TorchSharp's charlms, as arrays the managed parser reads.</summary>
    private static CharlmCache TorchCharlms(CharLanguageModel forward, CharLanguageModel backward, List<Sentence> sentences)
    {
        var cache = new CharlmCache(int.MaxValue);
        using var scope = torch.NewDisposeScope();
        var words = sentences.Select(s => (IReadOnlyList<string>)s.Words.Select(w => w.Text).ToList()).ToList();
        var (f, b) = (forward.BuildCharRepresentation(words), backward.BuildCharRepresentation(words));
        for (int i = 0; i < sentences.Count; i++)
            cache.TryAdd(sentences[i], f[i].data<float>().ToArray(), b[i].data<float>().ToArray(), words[i].Count);
        return cache;
    }

    /// <summary>[best, score, second, score] over the legal transitions, best first (ties: the lower index, like the Python
    /// side's stable argsort); second -1 when only one is legal.</summary>
    private static double[] Decision(float[] row, bool[] legal)
    {
        int best = -1, second = -1;
        for (int j = 0; j < row.Length; j++)
        {
            if (!legal[j])
                continue;
            if (best < 0 || row[j] > row[best])
                (best, second) = (j, best);
            else if (second < 0 || row[j] > row[second])
                second = j;
        }
        return [best, best < 0 ? 0 : row[best], second, second < 0 ? 0 : row[second]];
    }
}
