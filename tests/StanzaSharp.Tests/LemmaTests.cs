using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using StanzaSharp.Lemma;

namespace StanzaSharp.Tests;

public class LemmaTests
{
    private static string GoldenDir(string lang) => Path.Combine(Repo.GoldenFor(lang), "lemma");

    private static (List<Word> Words, JsonArray Golden) GoldenWords(string lang)
    {
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(GoldenDir(lang), "words.json")))!.AsArray();
        var words = golden.Select(g => new Word { Text = (string)g!["word"]!, Upos = (string)g["upos"]! }).ToList();
        return (words, golden);
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Lemmatize_MatchesGoldenWords(bool managed, string lang)
    {
        using var lemma = Lemmatizer.Load(Repo.ModelOf(lang, "lemma"), backend: Repo.Backend(managed));
        var (words, golden) = GoldenWords(lang);
        var doc = new Document();
        var sentence = new Sentence();
        foreach (var w in words)
            sentence.Tokens.Add(new Token { Text = w.Text, Words = { w } });
        doc.Sentences.Add(sentence);
        lemma.Process(doc);
        Assert.Equal(golden.Select(g => (string?)g!["lemma"]), words.Select(w => w.Lemma));
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Seq2SeqAlone_MatchesGoldenDecodingAndEdits(bool managed, string lang)
    {
        using var lemma = Lemmatizer.Load(Repo.ModelOf(lang, "lemma"), backend: Repo.Backend(managed));
        var (words, golden) = GoldenWords(lang);
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

    [ModelTheory]
    [InlineData("en")]
    [German]
    public void Dictionary_EqualsJsonSerializers(string lang)
    {
        // The dictionaries as the lemmatizer read them up to round 35: every entry must look up the same lemma.
        var root = Checkpoint.Load(Repo.ModelOf(lang, "lemma")).Root;
        var json = new MemoryStream();
        using (var gz = new GZipStream(new MemoryStream(Convert.FromBase64String(root["dicts"]!["$bytes"]!.GetValue<string>())), CompressionMode.Decompress))
            gz.CopyTo(json);
        var dict = new LemmaDictionary(json.GetBuffer(), (int)json.Length);
        var reference = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string?>>>(json.ToArray())!;
        Assert.True(reference.Values.Sum(d => d.Count) > 30_000);
        foreach (var (pos, words) in reference)
        {
            foreach (var (word, lemma) in words)
                Assert.Equal(lemma, dict.Lookup(pos, word));
        }
    }

    [Fact]
    public void Dictionary_ReadsJsonLikeJsonSerializer()
    {
        var text = " { \"*\" : {\"a\\u00e4\\\"\": \"x\\ny\", \"b\": null,\"c\":\"1\",\"c\":\"2\"},\r\n\t\"NOUN\": {}, \"VERB\": {\"d\": \"e\"}, \"VERB\": {\"f\": \"g\"} } ";
        var bytes = Encoding.UTF8.GetBytes(text);
        var dict = new LemmaDictionary(bytes, bytes.Length);
        Assert.Equal("x\ny", dict.Lookup("*", "aä\""));
        Assert.Null(dict.Lookup("*", "b"));
        Assert.Equal("2", dict.Lookup("*", "c"));
        Assert.Null(dict.Lookup("NOUN", "c"));
        Assert.Null(dict.Lookup("VERB", "d")); // a repeated POS replaces the earlier one, as in JsonSerializer
        Assert.Equal("g", dict.Lookup("VERB", "f"));
        Assert.Null(dict.Lookup("ADJ", "f"));
        Assert.Null(dict.Lookup("*", "\ud800"));
        var reference = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string?>>>(text)!;
        Assert.Equal(["*", "NOUN", "VERB"], reference.Keys);
        Assert.False(reference["VERB"].ContainsKey("d"));

        foreach (var bad in new[] { "", "{", "{\"*\": {\"a\": 1}}", "{\"*\": {\"a\": \"b\"}", "{\"*\": {\"a\": \"b\\", "{} x", "[]", "{\"*\": {\"a\": nul}}", "{\"*\": {\"a\\q\": \"b\"}}", "{\"*\": {\"\\ud800\": \"b\"}}" })
            Assert.Throws<InvalidDataException>(() => new LemmaDictionary(Encoding.UTF8.GetBytes(bad), Encoding.UTF8.GetByteCount(bad)));
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Pipeline_ReproducesEveryGoldenFile(bool managed, string lang)
    {
        using var nlp = Pipeline.Load(Repo.ModelsFor(lang), new PipelineOptions { Language = lang, Processors = "tokenize,mwt,pos,lemma", Backend = Repo.PipelineBackendFor(managed) });
        var failures = new List<string>();
        var names = Directory.GetFiles(GoldenDir(lang), "*.conllu").Select(Path.GetFileNameWithoutExtension).Order().ToList();
        Assert.Equal(13, names.Count); // corpus + 12 validation files
        foreach (var name in names)
        {
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), name + ".txt"))));
            var golden = File.ReadAllText(Path.Combine(GoldenDir(lang), name + ".conllu"));
            if (PipelineTests.HasNonBmp(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), name + ".txt")))) // offsets are UTF-16 in C#, code points in Stanza
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
