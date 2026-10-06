using System.Text.Json.Nodes;
using StanzaSharp.Nn;
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

    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        public readonly CharLanguageModel Forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        public readonly CharLanguageModel Backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        public readonly SentimentClassifier Classifier;

        public Models() => Classifier = SentimentClassifier.Load(Repo.Model("sentiment/sstplus_charlm"), Pretrain, Forward, Backward);

        public void Dispose()
        {
            Classifier.Dispose();
            Forward.Dispose();
            Backward.Dispose();
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

    /// <summary>Checks labels exactly and logits within <paramref name="tolerance"/>; returns the largest logit difference.</summary>
    private static float CompareToJson(string name, int[] labels, float[][] logits, List<string> failures, float tolerance = Tolerance)
    {
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, name + ".json")))!["sentences"]!.AsArray();
        if (golden.Count != labels.Length)
        {
            failures.Add($"{name}: {labels.Length} sentences, expected {golden.Count}");
            return float.NaN;
        }
        float worst = 0;
        for (int i = 0; i < labels.Length; i++)
        {
            var expected = golden[i]!["logits"]!.AsArray().Select(x => x!.GetValue<float>()).ToArray();
            float diff = expected.Zip(logits[i], (a, b) => Math.Abs(a - b)).Max();
            worst = Math.Max(worst, diff);
            if (golden[i]!["sentiment"]!.GetValue<int>() != labels[i] || diff > tolerance)
                failures.Add($"{name}, sentence {i}: label {labels[i]} logits [{string.Join(", ", logits[i])}], expected {golden[i]}");
        }
        return worst;
    }

    [ModelFact]
    public void Pipeline_ReproducesGoldenFilesLabelsAndLogits()
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,sentiment" });
        using var models = new Models();
        var failures = new List<string>();
        var sources = Sources();
        Assert.Equal(14, sources.Count);
        Assert.Equal(14, Directory.GetFiles(Golden, "*.conllu").Length);
        float worst = 0;
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
            worst = Math.Max(worst, CompareToJson(name, labels, logits, failures));
        }
        output.WriteLine($"max |logit diff| {worst}");
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelFact]
    public void Classify_MatchesStanzaAcrossSeveralBatches()
    {
        // Every text as one document: Stanza sorts its sentences by length and cuts them into 5000-token
        // batches, and each sentence's result depends on the padding of its batch.
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt" });
        using var models = new Models();
        var doc = nlp.Process(string.Join("\n\n", Sources().Select(s => s.Text)));
        var tokens = Tokens(doc);
        var batches = SentimentClassifier.Batches(tokens.Select(t => t.Count).OrderDescending().ToArray());
        var json = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "all.json")))!;
        Assert.Equal(json["batches"]!.GetValue<int>(), batches.Count);
        Assert.True(batches.Count > 1);

        var labels = models.Classifier.Classify(tokens, out var logits);
        var failures = new List<string>();
        output.WriteLine($"{tokens.Count} sentences, {batches.Count} batches, max |logit diff| {CompareToJson("all", labels, logits, failures)}");
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelFact]
    public void CharlmCache_FromTheTaggerGivesTheSameResults()
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt" });
        using var models = new Models();
        using var tagger = PosTagger.Load(Repo.Model("pos/combined_charlm"), models.Pretrain, models.Forward, models.Backward);
        var failures = new List<string>();
        int reused = 0, total = 0;
        float worst = 0;
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
            var withCache = models.Classifier.Classify(tokens, out var cachedLogits, i =>
            {
                var s = doc.Sentences[i];
                if (!s.Tokens.All(t => t.Words.Count == 1 && t.Words[0].Text == t.Text) || !cache.TryGet(s, out var r))
                    return null;
                reused++;
                return r;
            });
            total += tokens.Count;
            Assert.Equal(plain, withCache);
            worst = Math.Max(worst, CompareToJson(name, withCache, cachedLogits, failures, CachedTolerance));
        }
        output.WriteLine($"{reused}/{total} sentences reused the tagger's charlm outputs; max |logit diff| vs Stanza {worst}");
        Assert.True(reused > total / 2);
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelFact]
    public void Pipeline_WithTaggerAndParser_WritesSentimentAfterConstituency()
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,pos,constituency,sentiment" });
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
