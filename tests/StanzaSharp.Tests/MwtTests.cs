using System.Text.Json.Nodes;
using StanzaSharp.Mwt;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

public class MwtTests
{
    // managed: Backend.Managed (issue #29), else TorchSharp. Every golden test runs on both.
    private static MwtExpander Load(bool managed) => MwtExpander.Load(Repo.Model("mwt/combined"), backend: Repo.Backend(managed));

    private static JsonArray GoldenWords() =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "mwt.json")))!.AsArray();

    private static List<string> Words(JsonNode? node) => node!.AsArray().Select(w => w!.GetValue<string>()).ToList();

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Expand_MatchesGoldenWithDictionary(bool managed)
    {
        using var mwt = Load(managed);
        var golden = GoldenWords();
        var words = golden.Select(g => g!["word"]!.GetValue<string>()).ToList();
        var expansions = mwt.Expand(words);
        for (int i = 0; i < words.Count; i++)
            Assert.Equal(Words(golden[i]!["pipeline"]), Split(expansions[i], words[i]));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ClassifierAlone_MatchesGolden(bool managed)
    {
        using var mwt = Load(managed);
        var golden = GoldenWords();
        var words = golden.Select(g => g!["word"]!.GetValue<string>()).ToList();
        var predicted = mwt.Predict(words);
        for (int i = 0; i < words.Count; i++)
            Assert.Equal(Words(golden[i]!["model"]), Split(predicted[i], words[i]));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void TokenizeAndMwt_MatchGoldenWords(bool managed)
    {
        using var tokenizer = Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"), backend: Repo.Backend(managed));
        using var mwt = Load(managed);
        var doc = tokenizer.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        mwt.Process(doc);
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));

        Assert.Equal(Describe(golden), Describe(doc));
        Assert.DoesNotContain(doc.Sentences.SelectMany(s => s.Tokens), t => t.IsMwtCandidate);
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
