using Xunit.Abstractions;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

public class TokenizerTests(ITestOutputHelper output)
{
    // managed: Backend.Managed (issue #29), else TorchSharp. Every golden test runs on both.
    private static Tokenizer Load(bool managed) =>
        Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"), backend: managed ? Backend.Managed : Backend.TorchSharp);

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Logits_MatchGoldenIntermediates(bool managed)
    {
        using var tokenizer = Load(managed);
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!;

        foreach (var entry in index["sentences"]!.AsArray())
        {
            int i = entry!["sentence"]!.GetValue<int>();
            var expected = golden.Read<float>($"s{i}.tokenize.pred");
            var actual = tokenizer.ParagraphLogits(entry["text"]!.GetValue<string>(), out var shape);
            Assert.Equal(golden[$"s{i}.tokenize.pred"].Shape, shape);
            output.WriteLine($"sentence {i}: max |diff| {AssertClose(expected, actual, 1e-4f, $"sentence {i}"):E2}");
        }
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_MatchesGoldenTokensAndSentences(bool managed)
    {
        using var tokenizer = Load(managed);
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));

        var doc = tokenizer.Process(text);

        Assert.Equal(Describe(golden), Describe(doc));
        foreach (var (g, a) in golden.Sentences.SelectMany(s => s.Tokens).Zip(doc.Sentences.SelectMany(s => s.Tokens)))
            if (g.IsMultiWord)
                Assert.True(a.IsMwtCandidate, $"'{a.Text}' at {a.StartChar} should be marked as a multi-word token");
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_MatchesGoldenOnLongParagraphsAndManyBatches(bool managed)
    {
        using var tokenizer = Load(managed);
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "tokenize_stress.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "tokenize_stress.conllu")));
        Assert.Equal(Describe(golden), Describe(tokenizer.Process(text)));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_ReplacesTokensLongerThanMaxSeqlen(bool managed)
    {
        using var tokenizer = Load(managed);
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "long_token.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "long_token.conllu")));
        Assert.Equal(Describe(golden), Describe(tokenizer.Process(text)));
    }

    /// <summary>All processors see <c>&lt;UNK&gt;</c> (tests/golden/long_token*.conllu), in bulk too; pretokenized tokens stay, as in Stanza.</summary>
    [ModelTheory]
    [InlineData("default", false)]
    [InlineData("default_fast", false)]
    [InlineData("default", true)]
    [InlineData("default_fast", true)]
    public void Pipeline_LongTokens_MatchGolden(string package, bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Package = package, Backend = Repo.Backend(managed) });
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "long_token.txt"));
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, package == "default" ? "long_token.conllu" : "long_token.fast.conllu"));
        Assert.Equal(golden, Conllu.Write(nlp.Process(text)));
        Assert.Equal(golden, Conllu.Write(nlp.Process([text])[0]));
        var url = new string('x', 300);
        Assert.Equal(url, nlp.Process([["see", url]]).Sentences[0].Tokens[1].Text);
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_HandlesEmptyAndWhitespaceText(bool managed)
    {
        using var tokenizer = Load(managed);
        Assert.Empty(tokenizer.Process("").Sentences);
        Assert.Empty(tokenizer.Process(" \n\n \t").Sentences);
    }

    /// <summary>One line per sentence with its text, then each token as text[start,end) plus its trailing space.</summary>
    private static List<string> Describe(Document doc) =>
        doc.Sentences.Select(s => $"{s.SentId}: {s.Text} || " + string.Join(" ",
            s.Tokens.Select(t => $"{t.Text}[{t.StartChar},{t.EndChar}){Escape(t.SpaceAfter)}"))).ToList();

    private static string Escape(string space) => space == " " ? "" : "<" + space.Replace("\n", "\\n") + ">";

    /// <returns>The largest difference.</returns>
    internal static float AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worst = 0;
        for (int i = 1; i < expected.Length; i++)
            if (Math.Abs(expected[i] - actual[i]) > Math.Abs(expected[worst] - actual[worst]))
                worst = i;
        float diff = Math.Abs(expected[worst] - actual[worst]);
        Assert.True(diff <= tolerance, $"{what}: max |diff| {diff} at {worst} (expected {expected[worst]}, got {actual[worst]})");
        return diff;
    }
}
