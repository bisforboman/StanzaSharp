using System.Text.Json.Nodes;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

public class TokenizerTests
{
    private static Tokenizer Load() => Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"));

    [ModelFact]
    public void Logits_MatchGoldenIntermediates()
    {
        using var tokenizer = Load();
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!;

        foreach (var entry in index["sentences"]!.AsArray())
        {
            int i = entry!["sentence"]!.GetValue<int>();
            var expected = golden.Read<float>($"s{i}.tokenize.pred");
            var actual = tokenizer.ParagraphLogits(entry["text"]!.GetValue<string>(), out var shape);
            Assert.Equal(golden[$"s{i}.tokenize.pred"].Shape, shape);
            AssertClose(expected, actual, 1e-4f, $"sentence {i}");
        }
    }

    [ModelFact]
    public void Process_MatchesGoldenTokensAndSentences()
    {
        using var tokenizer = Load();
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));

        var doc = tokenizer.Process(text);

        Assert.Equal(Describe(golden), Describe(doc));
        foreach (var (g, a) in golden.Sentences.SelectMany(s => s.Tokens).Zip(doc.Sentences.SelectMany(s => s.Tokens)))
            if (g.IsMultiWord)
                Assert.True(a.IsMwtCandidate, $"'{a.Text}' at {a.StartChar} should be marked as a multi-word token");
    }

    [ModelFact]
    public void Process_MatchesGoldenOnLongParagraphsAndManyBatches()
    {
        using var tokenizer = Load();
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "tokenize_stress.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "tokenize_stress.conllu")));
        Assert.Equal(Describe(golden), Describe(tokenizer.Process(text)));
    }

    [ModelFact]
    public void Process_HandlesEmptyAndWhitespaceText()
    {
        using var tokenizer = Load();
        Assert.Empty(tokenizer.Process("").Sentences);
        Assert.Empty(tokenizer.Process(" \n\n \t").Sentences);
    }

    /// <summary>One line per sentence with its text, then each token as text[start,end) plus its trailing space.</summary>
    private static List<string> Describe(Document doc) =>
        doc.Sentences.Select(s => $"{s.SentId}: {s.Text} || " + string.Join(" ",
            s.Tokens.Select(t => $"{t.Text}[{t.StartChar},{t.EndChar}){Escape(t.SpaceAfter)}"))).ToList();

    private static string Escape(string space) => space == " " ? "" : "<" + space.Replace("\n", "\\n") + ">";

    internal static void AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worst = 0;
        for (int i = 1; i < expected.Length; i++)
            if (Math.Abs(expected[i] - actual[i]) > Math.Abs(expected[worst] - actual[worst]))
                worst = i;
        float diff = Math.Abs(expected[worst] - actual[worst]);
        Assert.True(diff <= tolerance, $"{what}: max |diff| {diff} at {worst} (expected {expected[worst]}, got {actual[worst]})");
    }
}
