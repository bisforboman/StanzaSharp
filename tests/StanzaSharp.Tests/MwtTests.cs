using System.Text.Json.Nodes;
using StanzaSharp.Mwt;
using StanzaSharp.Tokenize;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class MwtTests(ITestOutputHelper output)
{
    // managed: Backend.Managed (issue #29), else TorchSharp. Every golden test runs on both. English's model is the
    // character classifier, German's the seq2seq.
    private static MwtExpander Load(bool managed, string lang) => MwtExpander.Load(Repo.ModelOf(lang, "mwt"), backend: Repo.Backend(managed));

    private static JsonArray GoldenWords(string lang) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "mwt.json")))!.AsArray();

    private static List<string> Words(JsonNode? node) => node!.AsArray().Select(w => w!.GetValue<string>()).ToList();

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Expand_MatchesGoldenWithDictionary(bool managed, string lang)
    {
        using var mwt = Load(managed, lang);
        var golden = GoldenWords(lang);
        var words = golden.Select(g => g!["word"]!.GetValue<string>()).ToList();
        var expansions = mwt.Expand(words);
        for (int i = 0; i < words.Count; i++)
            Assert.Equal(Words(golden[i]!["pipeline"]), Split(expansions[i], words[i]));
    }

    /// <summary>The model without the dictionary: English's classifier, German's seq2seq (e.g. IM → "In deM").</summary>
    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void ModelAlone_MatchesGolden(bool managed, string lang)
    {
        using var mwt = Load(managed, lang);
        var golden = GoldenWords(lang);
        var words = golden.Select(g => g!["word"]!.GetValue<string>()).ToList();
        var predicted = mwt.Predict(words);
        for (int i = 0; i < words.Count; i++)
            Assert.Equal(Words(golden[i]!["model"]), Split(predicted[i], words[i]));
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void TokenizeAndMwt_MatchGoldenWords(bool managed, string lang)
    {
        using var tokenizer = Tokenizer.Load(Repo.ModelOf(lang, "tokenize"), backend: Repo.Backend(managed));
        using var mwt = Load(managed, lang);
        var golden = Repo.GoldenFor(lang);
        foreach (var (text, conllu) in new[] { ("corpus.txt", "pipeline.conllu"), ("validation_contractions.txt", "validation_contractions.conllu") })
        {
            var doc = tokenizer.Process(File.ReadAllText(Path.Combine(golden, text)));
            mwt.Process(doc);
            Assert.Equal(Describe(Conllu.Read(File.ReadAllText(Path.Combine(golden, conllu)))), Describe(doc));
            Assert.DoesNotContain(doc.Sentences.SelectMany(s => s.Tokens), t => t.IsMwtCandidate);
        }
    }

    /// <summary>
    /// German's seq2seq on both backends, on mwt.json's words and every token of the golden texts the tokenizer marks: the
    /// same decoder steps (every beam's row) and output. Reports the log-prob drift and the smallest top-2 margin of a row
    /// still decoding, like the lemmatizer's (ManagedBackendTests.LemmaNet_ManagedMatchesTorchSharp), and
    /// guards that the margin stays above 3 × the drift.
    /// </summary>
    [GermanFact]
    public void Seq2Seq_German_ManagedMatchesTorchSharp()
    {
        using var reference = Load(managed: false, "de");
        using var managed = Load(managed: true, "de");
        using var tokenizer = Tokenizer.LoadManaged(Repo.ModelOf("de", "tokenize"));
        var golden = Repo.GoldenFor("de");
        var calls = new List<List<string>> { GoldenWords("de").Select(g => g!["word"]!.GetValue<string>()).ToList() };
        foreach (var file in Directory.GetFiles(golden, "*.txt").Order())
            calls.Add(tokenizer.Process(File.ReadAllText(file)).Sentences.SelectMany(s => s.Tokens).Where(t => t.IsMwtCandidate).Select(t => t.Text).ToList());
        calls.RemoveAll(c => c.Count == 0);

        double diff = 0, margin = double.PositiveInfinity;
        int steps = 0;
        static double Margin(ReadOnlySpan<float> row)
        {
            float first = float.NegativeInfinity, second = float.NegativeInfinity;
            for (int v = 0; v < row.Length; v++)
            {
                if (v == 1)
                    continue; // never_decode_unk: <UNK> is never chosen
                if (row[v] > first)
                    (first, second) = (row[v], first);
                else if (row[v] > second)
                    second = row[v];
            }

            return first - second;
        }

        foreach (var tokens in calls)
        {
            var expectedSteps = new List<(float[] LogProbs, int Columns)>();
            var expected = reference.PredictSeq2Seq(tokens, (l, c, _) => expectedSteps.Add(((float[])l.Clone(), c)));
            int k = 0;
            var actual = managed.PredictSeq2Seq(tokens, (l, c, done) =>
            {
                var (e, columns) = expectedSteps[k++];
                Assert.Equal(columns, c);
                for (int r = 0; r < done.Length; r++)
                {
                    var er = e.AsSpan(r * c, c);
                    var ar = l.AsSpan(r * c, c);
                    for (int v = 0; v < c; v++)
                    {
                        if (v != 1) // never_decode_unk: <UNK> is -inf in both
                            diff = Math.Max(diff, Math.Abs(er[v] - ar[v]));
                    }

                    if (!done[r])
                        margin = Math.Min(margin, Math.Min(Margin(er), Margin(ar)));
                }
            });
            Assert.Equal(expectedSteps.Count, k);
            steps += k;
            Assert.Equal(expected, actual);
        }

        output.WriteLine($"{calls.Sum(c => c.Count)} tokens, {steps} decoder steps: log-probs max |diff| {diff:E2}, smallest top-2 margin {margin:E2}");
        Assert.True(margin >= 3 * diff, $"smallest top-2 margin {margin:E2} is under 3 × the backends' drift {diff:E2}");
    }

    /// <summary>How set_mwt_expansions turns an expansion into words.</summary>
    private static List<string> Split(string expansion, string token)
    {
        var words = expansion.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
        return words.Count <= 1 ? [token] : words;
    }

    private static List<string> Describe(Document doc) =>
        doc.Sentences.Select(s => string.Join(" ", s.Tokens.Select(t =>
            $"{t.Text}[{t.StartChar},{t.EndChar})=" + string.Join("+", t.Words.Select(w => $"{w.Id}:{w.Text}[{w.StartChar},{w.EndChar})"))))).ToList();
}
