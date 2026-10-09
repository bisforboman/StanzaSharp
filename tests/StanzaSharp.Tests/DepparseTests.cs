using System.Text.Json.Nodes;
using StanzaSharp.Depparse;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class DepparseTests(ITestOutputHelper output)
{
    private static readonly string Golden = Path.Combine(Repo.Golden, "depparse");

    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain;
        public readonly CharLanguageModel? Forward, Backward;
        public readonly DependencyParser Parser;

        public Models(bool managed = false)
        {
            Pretrain = Repo.LoadPretrain(managed);
            if (managed)
            {
                Parser = DependencyParser.LoadManaged(Repo.Model("depparse/combined_charlm"), Pretrain,
                    ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion")), ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion")));
                return;
            }
            Forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
            Backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
            Parser = DependencyParser.Load(Repo.Model("depparse/combined_charlm"), Pretrain, Forward, Backward);
        }

        public void Dispose()
        {
            Parser.Dispose();
            Forward?.Dispose();
            Backward?.Dispose();
            Pretrain.Dispose();
        }
    }

    /// <summary>
    /// One sentence's scores against the golden intermediates at 1e-4 (arc log-probs, -inf where Stanza has it, and label
    /// log-probs); returns the largest differences.
    /// </summary>
    internal static (float Arcs, float Labels) AssertScoresMatch(DepparseScores scores, float[] arcs, float[] labelLogProbs, string name)
    {
        Assert.Equal(arcs.Select(float.IsNegativeInfinity), scores.ArcLogProbs.Select(float.IsNegativeInfinity));
        float arcDiff = TokenizerTests.AssertClose(arcs.Select(Finite).ToArray(), scores.ArcLogProbs.Select(Finite).ToArray(), 1e-4f, $"{name}.unlabeled");
        return (arcDiff, TokenizerTests.AssertClose(labelLogProbs, LogSoftmax(scores.LabelScores!, scores.Relations), 1e-4f, $"{name}.deprel"));
    }

    /// <summary>log_softmax over each run of <paramref name="n"/> scores, in double.</summary>
    private static float[] LogSoftmax(float[] scores, int n)
    {
        var result = new float[scores.Length];
        for (int at = 0; at < scores.Length; at += n)
        {
            var row = scores.AsSpan(at, n);
            double max = double.NegativeInfinity, sum = 0;
            foreach (var x in row)
                max = Math.Max(max, x);
            foreach (var x in row)
                sum += Math.Exp(x - max);
            for (int i = 0; i < n; i++)
                result[at + i] = (float)(row[i] - max - Math.Log(sum));
        }
        return result;
    }

    /// <summary>tools/make_golden.py mst_scores.</summary>
    private static double[,] MstScores(int n, int seed)
    {
        var s = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                s[i, j] = ((i * 31 + j * 17 + seed * 7) * (i + 2 * j + seed + 1)) % 23 - 11;
        return s;
    }

    [Fact]
    [Trait("Backend", "Managed")]
    public void ChuLiuEdmonds_MatchesStanzaOnTiedScores()
    {
        var cases = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "mst.json")))!.AsArray();
        Assert.NotEmpty(cases);
        foreach (var c in cases)
        {
            int n = c!["n"]!.GetValue<int>(), seed = c["seed"]!.GetValue<int>();
            var expected = c["tree"]?.AsArray().Select(h => h!.GetValue<int>()).ToArray();
            if (expected == null)
                Assert.Throws<InvalidOperationException>(() => ChuLiuEdmonds.OneRoot(MstScores(n, seed)));
            else
                Assert.True(expected.SequenceEqual(ChuLiuEdmonds.OneRoot(MstScores(n, seed))), $"n={n} seed={seed}");
        }
    }

    [Fact]
    [Trait("Backend", "Managed")]
    public void ChuLiuEdmonds_BreaksACycle()
    {
        // Words 1 and 2 prefer each other; the cycle is broken by its cheapest entry from the root.
        double ninf = double.NegativeInfinity;
        var scores = new double[,] { { 0, ninf, ninf }, { -5, ninf, -1 }, { -3, -1, ninf } };
        Assert.Equal([0, 2, 0], ChuLiuEdmonds.OneRoot(scores));
        Assert.Single(ChuLiuEdmonds.Tarjan([0, 2, 1]));
        Assert.Empty(ChuLiuEdmonds.Tarjan([0, 0, 1]));
    }

    [Fact]
    [Trait("Backend", "Managed")]
    public void PairwiseSum_MatchesNumpyBlocking()
    {
        // np.sum adds in 8 interleaved accumulators from 8 elements on, so it differs from a plain loop.
        double[] a = [0.008791606182879854, -107178.74168774442, 9144672.031287812, -2.0063454615480424e-06,
            -0.12487488903344156, -3.138994719668478e-06, 54.10227877154389, 27.279133916445375, -98218.81249409777];
        double sequential = a.Aggregate(0.0, (x, y) => x + y);
        double pairwise = ChuLiuEdmonds.PairwiseSum(a, 0, a.Length);
        Assert.Equal(8939355.74243023, pairwise); // np.sum(a)
        Assert.NotEqual(sequential, pairwise);
    }

    [ModelFact]
    public void Batches_FollowStanzasSortingAndLimits()
    {
        using var models = new Models();
        // Lengths include ROOT. Over 150 goes alone; otherwise up to 5000 words per batch.
        int[] lengths = [3, 151, 5, 3, .. Enumerable.Repeat(130, 40)];
        var batches = models.Parser.Batches(lengths);
        Assert.Equal(3, batches.Count);
        Assert.Equal([1], batches[0]);
        // Sorted longest first with ties last-first, 38 x 130 fit; collate then flips the ties again.
        Assert.Equal(Enumerable.Range(6, 38), batches[1]);
        Assert.Equal([4, 5, 2, 0, 3], batches[2]);
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Scores_MatchGoldenIntermediates(bool managed)
    {
        using var models = new Models(managed);
        var golden = SafeTensorFile.Load(Path.Combine(Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Golden, "corpus.conllu")));
        Assert.Equal(3, index.Count);
        for (int i = 0; i < index.Count; i++)
        {
            var words = doc.Sentences[i].Words.ToList();
            Assert.Equal(index[i]!["words"]!.AsArray().Select(w => w!.GetValue<string>()), words.Select(w => w.Text));
            var (arcs, labels) = AssertScoresMatch(models.Parser.Scores([words], labelScores: true),
                golden.Read<float>($"s{i}.unlabeled"), golden.Read<float>($"s{i}.deprel"), $"s{i}");
            output.WriteLine($"s{i}: arc log-probs max |diff| {arcs:E2}, label log-probs {labels:E2}");
            var parsed = models.Parser.Parse([words])[0];
            Assert.Equal(index[i]!["heads"]!.AsArray().Select(h => h!.GetValue<int>()), parsed.Select(p => p.Head));
            Assert.Equal(index[i]!["deprels"]!.AsArray().Select(d => d!.GetValue<string>()), parsed.Select(p => p.Deprel));
        }
    }

    private static float Finite(float x) => float.IsNegativeInfinity(x) ? 0 : x;

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_ReproducesGoldenHeadsAndDeprels(bool managed)
    {
        using var models = new Models(managed);
        var files = Directory.GetFiles(Golden, "*.conllu").Order().ToList();
        Assert.Equal(13, files.Count); // corpus + 12 validation files
        var failures = new List<string>();
        foreach (var file in files)
        {
            // Stanza's own words, tags and lemmas; the parser must reproduce the rest of the file.
            var golden = File.ReadAllText(file);
            var doc = Conllu.Read(golden);
            foreach (var w in doc.Sentences.SelectMany(s => s.Words))
                (w.Head, w.Deprel) = (null, null);
            models.Parser.Process(doc);
            Compare(Path.GetFileName(file), golden, Conllu.Write(doc), failures);
        }
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pipeline_ReproducesGoldenFilesFromText(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,pos,lemma,depparse", Backend = Repo.PipelineBackendFor(managed) });
        var failures = new List<string>();
        var files = Directory.GetFiles(Golden, "*.conllu").Order().ToList();
        Assert.Equal(13, files.Count);
        foreach (var file in files)
        {
            var txt = Path.Combine(Repo.Golden, Path.GetFileNameWithoutExtension(file) + ".txt");
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(txt)));
            var golden = File.ReadAllText(file);
            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (Path.GetFileName(txt) == "validation_nonbmp.txt")
                (actual, golden) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(golden));
            Compare(Path.GetFileName(file), golden, actual, failures);
        }
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    /// <summary>
    /// A one-sentence document (a service's per-call case): the managed parser runs its backward charlm on from the
    /// tagger's final state (only ROOT left) instead of over the whole sentence. Every score must stay bit for bit.
    /// </summary>
    [ModelFact]
    [Trait("Backend", "Managed")]
    public void OneSentenceDocument_ContinuesTheTaggersBackwardCharlm_BitForBit()
    {
        using var models = new Models(managed: true);
        var (forward, backward) = (ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion")), ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion")));
        using var tagger = Pos.PosTagger.LoadManaged(Repo.Model("pos/combined_charlm"), models.Pretrain, forward, backward);
        var sentences = Directory.GetFiles(Golden, "*.conllu").Order().SelectMany(f => Conllu.Read(File.ReadAllText(f)).Sentences.Take(10)).ToList();
        int reused = 0;
        foreach (var sentence in sentences)
        {
            var doc = new Document();
            doc.Sentences.Add(sentence);
            using var cache = new CharlmCache();
            tagger.Process(doc, cache);
            if (!cache.TryGetBackwardState(sentence, out _))
                continue; // simplify_punct changed a word: the tagger keeps nothing
            reused++;
            var words = sentence.Words.ToList();
            var alone = models.Parser.Scores([words], labelScores: true);
            var continued = models.Parser.Scores([words], labelScores: true, charlms: cache, keys: [sentence]);
            Assert.Equal(alone.ArcLogProbs, continued.ArcLogProbs);
            Assert.Equal(alone.LabelScores, continued.LabelScores);
        }
        Assert.True(reused > sentences.Count * 3 / 4, $"only {reused} of {sentences.Count} sentences reused the tagger's state");
    }

    /// <summary>Records the first sentence where <paramref name="actual"/> differs from <paramref name="golden"/>.</summary>
    private static void Compare(string name, string golden, string actual, List<string> failures)
    {
        var expected = golden.Split("\n\n");
        var got = actual.Split("\n\n");
        int i = 0;
        while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
        if (i < expected.Length || i < got.Length)
            failures.Add($"{name}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");
    }
}
