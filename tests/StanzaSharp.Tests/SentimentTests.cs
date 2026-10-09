using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using StanzaSharp.Pos;
using StanzaSharp.Sentiment;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class SentimentTests(ITestOutputHelper output)
{
    private static readonly string Golden = Path.Combine(Repo.Golden, "sentiment");
    // On Arm64 (macOS, Apple Silicon) libtorch computes with Accelerate instead of the x64 kernels; the
    // classifier's large logits then drift up to ~1.1e-4 (labels still exact), so Arm64 gets 1e-3.
    private static readonly float Tolerance =
        System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture == System.Runtime.InteropServices.Architecture.Arm64 ? 1e-3f : 1e-4f;
    // Cached charlm outputs come from the tagger's batches and differ from fresh ones in the last bits.
    // The classifier amplifies that to about 1e-4 on logits of up to ~20 (the smallest top-2 margin is 1.6e-3).
    private const float CachedTolerance = 1e-3f;

    /// <summary>The classifier and the tagger (for the cache) on one backend, with that backend's charlms.</summary>
    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        private readonly CharLanguageModel? _forward, _backward;
        private readonly ManagedCharLanguageModel? _managedForward, _managedBackward;
        public readonly SentimentClassifier Classifier;

        public Models(bool managed = false)
        {
            if (managed)
            {
                _managedForward = ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
                _managedBackward = ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
                Classifier = SentimentClassifier.LoadManaged(Repo.Model("sentiment/sstplus_charlm"), Pretrain, _managedForward, _managedBackward);
            }
            else
            {
                _forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
                _backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
                Classifier = SentimentClassifier.Load(Repo.Model("sentiment/sstplus_charlm"), Pretrain, _forward, _backward);
            }
        }

        public PosTagger Tagger() => _managedForward != null
            ? PosTagger.LoadManaged(Repo.Model("pos/combined_charlm"), Pretrain, _managedForward, _managedBackward)
            : PosTagger.Load(Repo.Model("pos/combined_charlm"), Pretrain, _forward, _backward);

        public void Dispose()
        {
            Classifier.Dispose();
            _forward?.Dispose();
            _backward?.Dispose();
            Pretrain.Dispose();
        }
    }

    /// <summary>The texts make_golden.py's write_sentiment_golden reads, in its order: (name, text).</summary>
    private static List<(string Name, string Text)> Sources() =>
    [
        ("corpus", File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"))),
        .. Directory.GetFiles(Repo.Golden, "validation*.txt").Order(StringComparer.Ordinal)
            .Select(f => (Path.GetFileNameWithoutExtension(f), File.ReadAllText(f))),
        ("reviews", File.ReadAllText(Path.Combine(Golden, "reviews.txt"))),
    ];

    private static List<IReadOnlyList<string>> Tokens(Document doc) =>
        doc.Sentences.Select(s => (IReadOnlyList<string>)s.Tokens.Select(t => t.Text).ToList()).ToList();

    /// <summary>
    /// Checks labels exactly and each logit within <paramref name="tolerance"/> of Stanza's float32 value ("logits") or of
    /// the float64 one ("logits64", the same batches run in float64). The classifier is ill-conditioned on some sentences:
    /// on validation.txt sentence 41 Stanza's float32 logits are 1.49e-4 from the float64 ones, the managed backend's 1.2e-5.
    /// </summary>
    /// <returns>The largest difference from the float32 logits, and the largest accepted one (the nearer reference per logit).</returns>
    private static (float Float32, float Accepted) CompareToJson(string name, int[] labels, float[][] logits, List<string> failures, float? tolerance = null)
    {
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, name + ".json")))!["sentences"]!.AsArray();
        if (golden.Count != labels.Length)
        {
            failures.Add($"{name}: {labels.Length} sentences, expected {golden.Count}");
            return (float.NaN, float.NaN);
        }
        float worst32 = 0, worst = 0;
        for (int i = 0; i < labels.Length; i++)
        {
            var expected = golden[i]!["logits"]!.AsArray().Select(x => x!.GetValue<double>()).ToArray();
            var exact = golden[i]!["logits64"]!.AsArray().Select(x => x!.GetValue<double>()).ToArray();
            var diff32 = expected.Select((e, k) => (float)Math.Abs(e - logits[i][k])).ToArray();
            float diff = diff32.Select((d, k) => Math.Min(d, (float)Math.Abs(exact[k] - logits[i][k]))).Max();
            worst32 = Math.Max(worst32, diff32.Max());
            worst = Math.Max(worst, diff);
            if (golden[i]!["sentiment"]!.GetValue<int>() != labels[i] || diff > (tolerance ?? Tolerance))
                failures.Add($"{name}, sentence {i}: label {labels[i]} logits [{string.Join(", ", logits[i])}], expected {golden[i]}");
        }
        return (worst32, worst);
    }

    private static (float, float) Max((float A, float B) x, (float A, float B) y) => (Math.Max(x.A, y.A), Math.Max(x.B, y.B));

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pipeline_ReproducesGoldenFilesLabelsAndLogits(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,sentiment", Backend = Repo.PipelineBackendFor(managed) });
        using var models = new Models(managed);
        var failures = new List<string>();
        var sources = Sources();
        Assert.Equal(14, sources.Count);
        Assert.Equal(14, Directory.GetFiles(Golden, "*.conllu").Length);
        (float Float32, float Accepted) worst = (0, 0);
        foreach (var (name, text) in sources)
        {
            var doc = nlp.Process(text);
            var actual = Conllu.Write(doc);
            var golden = File.ReadAllText(Path.Combine(Golden, name + ".conllu"));
            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (name == "validation_nonbmp")
                (actual, golden) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(golden));
            var expected = golden.Split("\n\n");
            var got = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
            if (i < expected.Length || i < got.Length)
                failures.Add($"{name}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");

            var labels = models.Classifier.Classify(Tokens(doc), out var logits);
            Assert.Equal(doc.Sentences.Select(s => s.Sentiment!.Value), labels);
            worst = Max(worst, CompareToJson(name, labels, logits, failures));
        }
        output.WriteLine($"max |logit diff| vs float32 {worst.Float32}, accepted (nearer of float32 and float64) {worst.Accepted}");
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Classify_MatchesStanzaAcrossSeveralBatches(bool managed)
    {
        // Every text as one document: Stanza sorts its sentences by length and cuts them into 5000-token
        // batches, and each sentence's result depends on the padding of its batch.
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt" });
        using var models = new Models(managed);
        var doc = nlp.Process(string.Join("\n\n", Sources().Select(s => s.Text)));
        var tokens = Tokens(doc);
        var batches = SentimentClassifier.Batches(tokens.Select(t => t.Count).OrderDescending().ToArray());
        var json = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "all.json")))!;
        Assert.Equal(json["batches"]!.GetValue<int>(), batches.Count);
        Assert.True(batches.Count > 1);

        var labels = models.Classifier.Classify(tokens, out var logits);
        var failures = new List<string>();
        output.WriteLine($"{tokens.Count} sentences, {batches.Count} batches, max |logit diff| (vs float32, accepted) {CompareToJson("all", labels, logits, failures)}");
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void CharlmCache_FromTheTaggerGivesTheSameResults(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt" });
        using var models = new Models(managed);
        using var tagger = models.Tagger();
        var failures = new List<string>();
        int reused = 0, total = 0;
        (float Float32, float Accepted) worst = (0, 0);
        foreach (var (name, text) in Sources())
        {
            var doc = nlp.Process(text);
            using var cache = new CharlmCache();
            tagger.Process(doc, cache);
            var tokens = Tokens(doc);
            var plain = models.Classifier.Classify(tokens, out _);
            models.Classifier.Process(doc, cache);
            Assert.Equal(plain, doc.Sentences.Select(s => s.Sentiment!.Value));

            // The same with the logits, which must still match Stanza's.
            var keys = doc.Sentences.Select(SentimentClassifier.CacheKey).ToList();
            reused += keys.Count(k => k != null && cache.TryGetArrays(k, out _));
            var withCache = models.Classifier.Classify(tokens, out var cachedLogits, cache, keys);
            total += tokens.Count;
            Assert.Equal(plain, withCache);
            worst = Max(worst, CompareToJson(name, withCache, cachedLogits, failures, CachedTolerance));
        }
        output.WriteLine($"{reused}/{total} sentences reused the tagger's charlm outputs; max |logit diff| vs float32 {worst.Float32}, accepted {worst.Accepted}");
        Assert.True(reused > total / 2);
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pipeline_WithTaggerAndParser_WritesSentimentAfterConstituency(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,pos,constituency,sentiment", Backend = Repo.PipelineBackendFor(managed) });
        var conllu = Conllu.Write(nlp.Process("I love it. I don't like this movie at all."));
        // Python Stanza 1.15.0 with processors="tokenize,mwt,pos,constituency,sentiment".
        Assert.StartsWith("""
            # text = I love it.
            # sent_id = 0
            # constituency = (ROOT (S (NP (PRP I)) (VP (VBP love) (NP (PRP it))) (. .)))
            # sentiment = 2
            1	I	_	PRON	PRP	Case=Nom|Number=Sing|Person=1|PronType=Prs	0	_	_	start_char=0|end_char=1
            """.ReplaceLineEndings("\n"), conllu);
        Assert.Contains("# constituency = (ROOT (S (NP (PRP I)) (VP (VBP do) (RB n't) (VP (VB like) (NP (DT this) (NN movie)) (ADVP (IN at) (DT all)))) (. .)))\n# sentiment = 0\n", conllu);
    }

    [ModelFact]
    public void MapWord_TriesTheWordThenWithoutApostropheThenLowercased()
    {
        using var models = new Models();
        // CNNClassifier.map_word in Python on the conll17 pretrain. Without the apostrophe, the word is not
        // lowercased, so "DINOSAURS'" is unknown; "'" and "o'" are in the vocabulary as they are.
        (string Word, int Id)[] expected =
        [
            ("Paris", 1545), ("TERRIBLE", 4837), ("dinosaurs'", 14137), ("DINOSAURS'", 1), ("'", 61), ("o'", 2213),
            ("O'", 2213), ("zorbulatorqx", 1), ("İstanbul", 1), ("New York", 1), ("the", 6),
        ];
        Assert.Equal(expected, expected.Select(e => (e.Word, models.Classifier.MapWord(e.Word))));
    }

    [Fact]
    public void Batches_FollowSplitIntoBatches()
    {
        // Lengths sorted longest first; one over 5000 goes alone, the rest fill up to 5000 tokens.
        Assert.Equal([(0, 1), (1, 3), (3, 5)], SentimentClassifier.Batches([6000, 3000, 2000, 2500, 10]));
        Assert.Equal([(0, 2)], SentimentClassifier.Batches([4000, 1000]));
        Assert.Equal([(0, 1), (1, 2)], SentimentClassifier.Batches([4000, 1001]));
        Assert.Empty(SentimentClassifier.Batches([]));
    }

    [Fact]
    public void Conllu_ReadsAndWritesTheSentimentComment()
    {
        const string text = "# text = Fine.\n# sent_id = 0\n# sentiment = 1\n1\tFine\t_\t_\t_\t_\t0\t_\t_\tSpaceAfter=No\n2\t.\t_\t_\t_\t_\t1\t_\t_\tSpaceAfter=No\n";
        var doc = Conllu.Read(text);
        Assert.Equal(1, doc.Sentences[0].Sentiment);
        Assert.Equal(text, Conllu.Write(doc));
    }
}
