using System.Diagnostics;
using System.Text.Encodings.Web;
using System.Text.Json;
using StanzaSharp;
using StanzaSharp.Lemma;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;

/// <summary>
/// <c>lemma-divergence</c>: the C# side of tools/lemma_divergence.py (docs/backends.md, "lemma"). Runs
/// tokenize,mwt,pos,lemma on each document of a JSON list of texts and writes the same JSON as the Python side: each
/// word's text, tags and lemma, and per seq2seq word its decoded string, edit logits and each decoder step's top 2.
/// </summary>
internal static class LemmaDivergence
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark lemma-divergence --docs DOCS.json --out OUT.json [--models DIR] [--package NAME]
                                     [--backend torch|managed] [--double-gates] [--no-dict] [--threads N]

          --double-gates  managed: the decoder's LSTMCell gate sums in double (ManagedLemmaNet.DoubleGates)
          --no-dict       every word through the seq2seq model (its lemma in the records; the document keeps the dictionary's)
        """;

    public static int Run(string[] args)
    {
        string modelDir = Path.Combine("models", "converted", "en"), package = Pipeline.DefaultPackage;
        string? docsFile = null, outFile = null;
        var backend = Backend.TorchSharp;
        bool doubleGates = false, noDict = false;
        int threads = 0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--docs" when i + 1 < args.Length: docsFile = args[++i]; break;
                case "--out" when i + 1 < args.Length: outFile = args[++i]; break;
                case "--models" when i + 1 < args.Length: modelDir = args[++i]; break;
                case "--package" when i + 1 < args.Length: package = args[++i]; break;
                case "--backend" when i + 1 < args.Length: backend = args[++i] == "managed" ? Backend.Managed : Backend.TorchSharp; break;
                case "--threads" when i + 1 < args.Length: threads = int.Parse(args[++i]); break;
                case "--double-gates": doubleGates = true; break;
                case "--no-dict": noDict = true; break;
                default:
                    Console.Error.WriteLine(Usage);
                    return 2;
            }
        }
        if (docsFile == null || outFile == null || doubleGates && backend != Backend.Managed)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        var texts = JsonSerializer.Deserialize<List<string>>(File.ReadAllText(docsFile))!;
        ManagedLemmaNet.DoubleGates = doubleGates;
        using var nlp = Pipeline.Load(modelDir, new PipelineOptions
        {
            Package = package, Processors = "tokenize,mwt,pos", Backend = backend == Backend.Managed ? PipelineBackend.Managed : PipelineBackend.TorchSharp, Threads = threads > 0 ? threads : null,
        });
        var lemmaModel = Pipeline.SelectModels(package, null, addRequired: false, "--package")["lemma"];
        using var lemmatizer = Lemmatizer.Load(Path.Combine(modelDir, "lemma", lemmaModel), null, backend);

        lemmatizer.Process(nlp.Process("Warm up the pipeline."));
        var docs = new List<object>(texts.Count);
        double seconds = 0, lemmaSeconds = 0;
        var clock = new Stopwatch();
        for (int n = 0; n < texts.Count; n++)
        {
            clock.Restart();
            var doc = nlp.Process(texts[n]);
            double before = clock.Elapsed.TotalSeconds;
            lemmatizer.Process(doc);
            double lemma = clock.Elapsed.TotalSeconds - before;
            var s2s = Diagnose(lemmatizer, doc, noDict, ref lemma); // --no-dict: the time of the seq2seq model on every word
            seconds += before + lemma;
            lemmaSeconds += lemma;
            docs.Add(new { sents = doc.Sentences.Select(s => s.Words.Select(w => new[] { w.Text, w.Upos, w.Xpos, w.Feats, w.Lemma })), s2s });
            if (n % 100 == 0)
                Console.Error.WriteLine($"{n}/{texts.Count} documents");
        }

        var meta = new
        {
            backend = backend == Backend.Managed ? "managed" : "torch", double_gates = doubleGates, package, no_dict = noDict,
            threads = backend == Backend.Managed ? ManagedThreads.Count : torch.get_num_threads(), seconds, lemma_seconds = lemmaSeconds,
        };
        File.WriteAllText(outFile, JsonSerializer.Serialize(new { meta, docs }, new JsonSerializerOptions { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
        Console.WriteLine(JsonSerializer.Serialize(meta));
        return 0;
    }

    /// <summary>
    /// Reruns the lemmatizer's seq2seq words (all words with <paramref name="noDict"/>) as Lemmatizer.Process does, the
    /// lemmas reset as they were before it, capturing each batch's steps and edit logits.
    /// </summary>
    private static List<object> Diagnose(Lemmatizer lemmatizer, Document doc, bool noDict, ref double noDictSeconds)
    {
        var words = doc.Sentences.SelectMany(s => s.Words).ToList();
        var saved = words.Select(w => w.Lemma).ToList();
        foreach (var w in words)
            w.Lemma = null;
        var misses = Enumerable.Range(0, words.Count).Where(i => noDict || lemmatizer.Lookup(words[i].Text, words[i].Upos) == null).ToList();
        var missWords = misses.Select(i => words[i]).ToList();
        if (noDict)
        {
            var watch = Stopwatch.StartNew();
            lemmatizer.Predict(missWords);
            noDictSeconds = watch.Elapsed.TotalSeconds;
        }
        var records = new List<(int I, int K, double[] El, double[][] Steps)>();
        var steps = new List<double[][]>(); // per step, per sorted row: top-1 id, log-prob, top-2 id, log-prob (null once done)
        int batchStart = 0;
        var (decoded, edits) = lemmatizer.Predict(missWords, onStep: (logProbs, columns, done) =>
        {
            var rows = new double[done.Length][];
            for (int r = 0; r < done.Length; r++)
            {
                if (done[r])
                    continue;
                var row = logProbs.AsSpan(r * columns, columns);
                int a = 0;
                for (int v = 1; v < columns; v++)
                    if (row[v] > row[a])
                        a = v;
                int b = a == 0 ? 1 : 0;
                for (int v = 0; v < columns; v++)
                    if (v != a && row[v] > row[b])
                        b = v;
                rows[r] = [a, row[a], b, row[b]];
            }
            steps.Add(rows);
        }, onEdit: editLogits =>
        {
            // The batch's rows in Lemmatizer.PredictBatch's order (data.sort_all: longest first, ties by later index first).
            int size = editLogits.Length / 3;
            var lengths = Enumerable.Range(0, size).Select(i => missWords[batchStart + i].Text.EnumerateRunes().Count()).ToArray();
            var order = Enumerable.Range(0, size).OrderByDescending(i => lengths[i]).ThenByDescending(i => i).ToArray();
            for (int r = 0; r < size; r++)
                records.Add((misses[batchStart + order[r]], batchStart + order[r],
                    editLogits.Skip(r * 3).Take(3).Select(x => (double)x).ToArray(),
                    steps.TakeWhile(s => s[r] != null).Select(s => s[r]).ToArray()));
            steps.Clear();
            batchStart += size;
        });

        for (int i = 0; i < words.Count; i++)
            words[i].Lemma = saved[i];
        // Add each word's decoded string, edit and lemma (Lemmatizer.Postprocess, then Process's "_" → null).
        return records.OrderBy(r => r.I).Select(r =>
        {
            var word = missWords[r.K].Text;
            var lemma = edits[r.K] switch { 1 => word, 2 => PyString.Lower(word), _ => decoded[r.K] };
            if (lemma.Length == 0 || lemma.Contains("<UNK>", StringComparison.Ordinal))
                lemma = word;
            return (object)new { i = r.I, lemma = lemma == "_" && word != "_" ? null : lemma, dec = decoded[r.K], edit = edits[r.K], el = r.El, steps = r.Steps };
        }).ToList();
    }
}
