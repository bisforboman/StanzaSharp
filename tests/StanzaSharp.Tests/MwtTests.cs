using System.Text.Json.Nodes;
using StanzaSharp.Mwt;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

public class MwtTests
{
    private static MwtExpander Load() => MwtExpander.Load(Repo.Model("mwt/combined"));

    private static JsonArray GoldenWords() =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "mwt.json")))!.AsArray();

    private static List<string> Words(JsonNode? node) => node!.AsArray().Select(w => w!.GetValue<string>()).ToList();

    [ModelFact]
    public void Expand_MatchesGoldenWithDictionary()
    {
        using var mwt = Load();
        var golden = GoldenWords();
        var words = golden.Select(g => g!["word"]!.GetValue<string>()).ToList();
        var expansions = mwt.Expand(words);
        for (int i = 0; i < words.Count; i++)
            Assert.Equal(Words(golden[i]!["pipeline"]), Split(expansions[i], words[i]));
    }

    [ModelFact]
    public void ClassifierAlone_MatchesGolden()
    {
        using var mwt = Load();
        var golden = GoldenWords();
        var words = golden.Select(g => g!["word"]!.GetValue<string>()).ToList();
        var predicted = mwt.Predict(words);
        for (int i = 0; i < words.Count; i++)
            Assert.Equal(Words(golden[i]!["model"]), Split(predicted[i], words[i]));
    }

    [ModelFact]
    public void TokenizeAndMwt_MatchGoldenWords()
    {
        using var tokenizer = Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"));
        using var mwt = Load();
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
