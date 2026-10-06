using System.Text.Json.Nodes;
using StanzaSharp.Lemma;

namespace StanzaSharp.Tests;

public class LemmaTests
{
    private static readonly string GoldenDir = Path.Combine(Repo.Golden, "lemma");

    private static (List<Word> Words, JsonArray Golden) GoldenWords()
    {
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(GoldenDir, "words.json")))!.AsArray();
        var words = golden.Select(g => new Word { Text = (string)g!["word"]!, Upos = (string)g["upos"]! }).ToList();
        return (words, golden);
    }

    [ModelFact]
    public void Lemmatize_MatchesGoldenWords()
    {
        using var lemma = Lemmatizer.Load(Repo.Model("lemma/combined_nocharlm"));
        var (words, golden) = GoldenWords();
        var doc = new Document();
        var sentence = new Sentence();
        foreach (var w in words)
            sentence.Tokens.Add(new Token { Text = w.Text, Words = { w } });
        doc.Sentences.Add(sentence);
        lemma.Process(doc);
        Assert.Equal(golden.Select(g => (string?)g!["lemma"]), words.Select(w => w.Lemma));
    }

    [ModelFact]
    public void Seq2SeqAlone_MatchesGoldenDecodingAndEdits()
    {
        using var lemma = Lemmatizer.Load(Repo.Model("lemma/combined_nocharlm"));
        var (words, golden) = GoldenWords();
        var (decoded, edits) = lemma.Predict(words);
        Assert.Equal(golden.Select(g => (string)g!["seq2seq"]!), decoded);
        Assert.Equal(golden.Select(g => (int)g!["edit"]!), edits);
        Assert.Equal(golden.Select(g => (string)g!["model"]!), lemma.Postprocess(words));
    }

    [ModelFact]
    public void Lookup_UsesPosEntryThenPosIndependentEntry()
    {
        using var lemma = Lemmatizer.Load(Repo.Model("lemma/combined_nocharlm"));
        Assert.Equal("see", lemma.Lookup("saw", "VERB"));
        Assert.Equal("saw", lemma.Lookup("saw", "NOUN"));
        Assert.Equal("see", lemma.Lookup("saw", null));
        Assert.Null(lemma.Lookup("flibbertigibbets", "NOUN"));
    }

    [ModelFact]
    public void Pipeline_ReproducesEveryGoldenFile()
    {
        using var nlp = Pipeline.Load(Repo.Models, "tokenize,mwt,pos,lemma");
        var failures = new List<string>();
        var names = Directory.GetFiles(GoldenDir, "*.conllu").Select(Path.GetFileNameWithoutExtension).Order().ToList();
        Assert.Equal(13, names.Count); // corpus + 12 validation files
        foreach (var name in names)
        {
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, name + ".txt"))));
            var golden = File.ReadAllText(Path.Combine(GoldenDir, name + ".conllu"));
            if (name == "validation_nonbmp") // offsets are UTF-16 in C#, code points in Stanza
                (actual, golden) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(golden));
            var expected = golden.Split("\n\n");
            var got = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
            if (i < expected.Length || i < got.Length)
                failures.Add($"{name}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");
        }
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }
}
