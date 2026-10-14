using System.Text.Json;
using System.Text.Json.Nodes;

namespace StanzaSharp.Tests;

/// <summary><see cref="Json"/>: Stanza's <c>str(doc)</c> (tests/golden/json and tests/golden/de/json).</summary>
public class JsonTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    private static string GoldenJson(string lang, string name) => File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "json", name + ".json"));

    // json/<name>.json: the package and processors of make_golden.py's JSON_OUTPUTS, on json/inputs.json's text.
    [ModelTheory]
    [Trait("Backend", "Managed")]
    [InlineData("default", null, "default", "en")]
    [InlineData("default_fast", null, "default_fast", "en")]
    [InlineData("default", "tokenize,mwt,pos", "pos", "en")]
    [German("default", null, "default")]
    [German("default_fast", null, "default_fast")]
    [German("default", "tokenize,mwt,pos", "pos")]
    public void Write_MatchesStanzasStr(string package, string? processors, string name, string lang)
    {
        var text = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "json", "inputs.json")))!["text"]!.GetValue<string>();
        var doc = pipelines.Get(package, processors: processors, lang: lang).Process(text);
        var golden = GoldenJson(lang, name);
        var actual = Json.Write(doc);
        Assert.True(golden == actual, $"--- expected\n{golden}\n--- actual\n{actual}");
    }

    [Theory]
    [InlineData("en", "default")]
    [InlineData("en", "default_fast")]
    [InlineData("en", "pos")]
    [InlineData("de", "default")]
    [InlineData("de", "default_fast")]
    [InlineData("de", "pos")]
    public void Read_RoundTripsStanzasJson(string lang, string name)
    {
        var golden = GoldenJson(lang, name);
        var doc = Json.Read(golden);
        Assert.Equal(golden, Json.Write(doc));
        Assert.Equal(golden, Json.Write(Json.Read(Json.Write(doc))));
        Assert.Equal(Enumerable.Range(0, doc.Sentences.Count).Select(i => i.ToString()), doc.Sentences.Select(s => s.SentId));
        Assert.All(doc.Sentences, s => Assert.Null(s.Text));
    }

    [Fact]
    public void Read_KeepsTokensWordsSpacingAndEntities()
    {
        var doc = Json.Read(GoldenJson("en", "default"));
        var first = doc.Sentences[0].Tokens[0];
        Assert.Equal(("\"", "", "  ", 2, 3), (first.Text, first.SpaceAfter, first.SpacesBefore, first.StartChar, first.EndChar));

        var dont = doc.Sentences[1].Tokens[1];
        Assert.Equal("don't", dont.Text);
        Assert.Equal(["do", "n't"], dont.Words.Select(w => w.Text));
        Assert.Equal("", dont.SpaceAfter);
        Assert.Equal((2, "do", "AUX", "VBP", 0, "root"), (dont.Words[0].Id, dont.Words[0].Lemma, dont.Words[0].Upos, dont.Words[0].Xpos, dont.Words[0].Head, dont.Words[0].Deprel));
        Assert.Contains(doc.Entities, e => e is { Text: "Barack Obama's", Type: "PERSON" });
        Assert.Equal("\r\n\r\n", doc.Sentences[2].Tokens[^1].SpaceAfter);
    }

    [Fact]
    public void Write_KeepsEveryTokenFieldOfConllu()
    {
        // Every field Conllu reads is in the JSON, and Conllu's document round-trips through it.
        var conllu = File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));
        var json = Json.Write(Conllu.Read(conllu));
        var doc = Json.Read(json);
        Assert.Equal(json, Json.Write(doc));
        foreach (var s in doc.Sentences)
            (s.Text, s.Constituency, s.Sentiment) = (null, null, null);
        var expected = Conllu.Read(conllu);
        foreach (var s in expected.Sentences)
            (s.Text, s.Constituency, s.Sentiment) = (null, null, null);
        Assert.Equal(Conllu.Write(expected), Conllu.Write(doc));
    }

    [Fact]
    public void Write_EscapesLikePython()
    {
        var doc = new Document();
        var sentence = new Sentence();
        var token = new Token { Text = "a\"\\\u0001\u001c\b\f\u007f é😀", SpaceAfter = "\t" };
        token.Words.Add(new Word { Id = 1, Text = token.Text });
        sentence.Tokens.Add(token);
        doc.Sentences.Add(sentence);
        Assert.Equal("""
            [
              [
                {
                  "id": 1,
                  "text": "a\"\\\u0001\u001c\b\f\u007f é😀",
                  "misc": "SpacesAfter=\\t"
                }
              ]
            ]
            """.Replace("\\u007f", "\u007f"), Json.Write(doc));
        Assert.Equal(token.Text, Json.Read(Json.Write(doc)).Sentences[0].Tokens[0].Text);
        Assert.Equal("[]", Json.Write(new Document()));
        Assert.Empty(Json.Read("[]").Sentences);
    }

    [Fact]
    public void Read_RejectsOtherJson()
    {
        Assert.ThrowsAny<JsonException>(() => Json.Read("{}"));
        Assert.ThrowsAny<JsonException>(() => Json.Read("[[{\"text\": \"a\"}]]"));
        Assert.ThrowsAny<JsonException>(() => Json.Read("[[{\"id\": 1}]]"));
        Assert.ThrowsAny<JsonException>(() => Json.Read("not json"));
        Assert.Throws<ArgumentNullException>(() => Json.Read(null!));
        Assert.Throws<ArgumentNullException>(() => Json.Write(null!));
    }
}
