using System.Text.Json.Nodes;
using StanzaSharp.Depparse;
using StanzaSharp.Nn;
using static TorchSharp.torch;

namespace StanzaSharp.Tests;

public class DepparseTests
{
    private static readonly string Golden = Path.Combine(Repo.Golden, "depparse");

    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        public readonly CharLanguageModel Forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        public readonly CharLanguageModel Backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        public readonly DependencyParser Parser;

        public Models() => Parser = DependencyParser.Load(Repo.Model("depparse/combined_charlm"), Pretrain, Forward, Backward);

        public void Dispose()
        {
            Parser.Dispose();
            Forward.Dispose();
            Backward.Dispose();
            Pretrain.Dispose();
        }
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

    [ModelFact]
    public void Scores_MatchGoldenIntermediates()
    {
        using var models = new Models();
        var golden = SafeTensorFile.Load(Path.Combine(Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Golden, "corpus.conllu")));
        Assert.Equal(3, index.Count);
        for (int i = 0; i < index.Count; i++)
        {
            var words = doc.Sentences[i].Words.ToList();
            Assert.Equal(index[i]!["words"]!.AsArray().Select(w => w!.GetValue<string>()), words.Select(w => w.Text));
            var (unlabeled, deprel) = models.Parser.Scores([words]);
            using (unlabeled)
            using (deprel)
            using (var labelLogProbs = nn.functional.log_softmax(deprel, 3))
            {
                var arcs = unlabeled.data<float>().ToArray();
                var expected = golden.Read<float>($"s{i}.unlabeled");
                Assert.Equal(expected.Select(float.IsNegativeInfinity), arcs.Select(float.IsNegativeInfinity));
                TokenizerTests.AssertClose(expected.Select(Finite).ToArray(), arcs.Select(Finite).ToArray(), 1e-4f, $"s{i}.unlabeled");
                TokenizerTests.AssertClose(golden.Read<float>($"s{i}.deprel"), labelLogProbs.data<float>().ToArray(), 1e-4f, $"s{i}.deprel");
            }
            var parsed = models.Parser.Parse([words])[0];
            Assert.Equal(index[i]!["heads"]!.AsArray().Select(h => h!.GetValue<int>()), parsed.Select(p => p.Head));
            Assert.Equal(index[i]!["deprels"]!.AsArray().Select(d => d!.GetValue<string>()), parsed.Select(p => p.Deprel));
        }
    }

    private static float Finite(float x) => float.IsNegativeInfinity(x) ? 0 : x;

    [ModelFact]
    public void Process_ReproducesGoldenHeadsAndDeprels()
    {
        using var models = new Models();
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

            var expected = golden.Split("\n\n");
            var got = Conllu.Write(doc).Split("\n\n");
            int i = 0;
            while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
            if (i < expected.Length || i < got.Length)
                failures.Add($"{Path.GetFileName(file)}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");
        }
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }
}
