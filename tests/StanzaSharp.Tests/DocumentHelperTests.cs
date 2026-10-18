using System.Text.Json.Nodes;

namespace StanzaSharp.Tests;

/// <summary>
/// The document helpers against Stanza's own (tests/golden/helpers, <c>make_golden.py --helpers-only</c>): num_tokens,
/// num_words, sentence.dependencies, dependencies_string(), tokens_string() and words_string() of documents read from
/// CoNLL-U, so no models are needed.
/// </summary>
public class DocumentHelperTests
{
    private static readonly JsonObject Golden =
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "helpers", "helpers.json")))!.AsObject();

    public static TheoryData<string> Files() => [.. Golden.Select(kv => kv.Key)];

    [Theory]
    [MemberData(nameof(Files))]
    public void Helpers_MatchStanzas(string file)
    {
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, file)));
        var golden = Golden[file]!;
        Assert.Equal(golden["num_tokens"]!.GetValue<int>(), doc.TokenCount);
        Assert.Equal(golden["num_words"]!.GetValue<int>(), doc.WordCount);
        var sentences = golden["sentences"]!.AsArray();
        Assert.Equal(sentences.Count, doc.Sentences.Count);
        for (int i = 0; i < sentences.Count; i++)
        {
            var sentence = doc.Sentences[i];
            Assert.Equal(sentences[i]!["dependencies"]!.GetValue<string>(), sentence.DependenciesString());
            Assert.Equal(sentences[i]!["tokens"]!.GetValue<string>(), sentence.TokensString());
            Assert.Equal(sentences[i]!["words"]!.GetValue<string>(), sentence.WordsString());
        }
    }

    [Fact]
    public void Dependencies_PointAtTheHeadWords_AndRootIsNull()
    {
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));
        var sentence = doc.Sentences[0]; // Barack Obama was born in Hawaii .
        var words = sentence.Words.ToList();
        var dependencies = sentence.Dependencies;
        Assert.Equal(words.Count, dependencies.Count);
        Assert.All(dependencies.Zip(words), x => Assert.Same(x.Second, x.First.Dependent));
        var root = Assert.Single(dependencies, d => d.Head == null);
        Assert.Equal(("born", "root"), (root.Dependent.Text, root.Relation));
        Assert.Same(root.Dependent, dependencies[0].Head); // Barack's head is born
    }

    [Fact]
    public void Dependencies_AreEmptyWithoutHeads_AndAHeadOutsideTheSentenceThrows()
    {
        var sentence = new Sentence();
        var token = new Token { Text = "Hi" };
        token.Words.Add(new Word { Id = 1, Text = "Hi" });
        sentence.Tokens.Add(token);
        Assert.Empty(sentence.Dependencies);
        Assert.Equal("", sentence.DependenciesString());

        (token.Words[0].Head, token.Words[0].Deprel) = (2, "dep");
        Assert.Throws<InvalidOperationException>(() => sentence.Dependencies);
    }

    [Theory]
    [InlineData("plain", "'plain'")]
    [InlineData("it's", "\"it's\"")]
    [InlineData("both '\"", "'both \\'\"'")]
    [InlineData("tab\there", "'tab\\there'")]
    [InlineData("new\nline", "'new\\nline'")]
    [InlineData("nul\0", "'nul\\x00'")]
    [InlineData("\u2028", "'\\u2028'")]
    [InlineData("\ue000", "'\\ue000'")]
    [InlineData("\U000E0001", "'\\U000e0001'")]
    public void Repr_IsPythons(string text, string expected) => Assert.Equal(expected, PyString.Repr(text));

    // Built in code: a lone surrogate in an InlineData argument doesn't survive xunit's serialization.
    [Fact]
    public void Repr_EscapesALoneSurrogate() => Assert.Equal("'a\\ud800'", PyString.Repr("a" + (char)0xD800));
}
