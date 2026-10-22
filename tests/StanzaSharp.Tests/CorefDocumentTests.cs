using System.Text.Json.Nodes;

namespace StanzaSharp.Tests;

/// <summary>
/// The coreference document model (docs/coref-plan.md, PR b): <see cref="Document.Coref"/>, <see cref="Word.CorefChains"/>
/// and <see cref="Sentence.EmptyWords"/> through <see cref="Conllu"/> and <see cref="Json"/>, against tests/golden/coref/
/// (Stanza's coref on the golden texts, and crafted.*: chains set by hand with empty words). No models needed.
/// </summary>
public class CorefDocumentTests
{
    // Each file by its path under tests/golden without the extension: coref/<name>, coref/zeros/<name> and de/coref/<name>.
    public static TheoryData<string> Names() =>
        [.. new[] { "coref", Path.Combine("coref", "zeros"), Path.Combine("de", "coref") }.SelectMany(folder => Directory.GetFiles(Path.Combine(Repo.Golden, folder), "*.conllu")
            .Select(f => Path.Combine(folder, Path.GetFileNameWithoutExtension(f)).Replace('\\', '/'))).Order(StringComparer.Ordinal)];

    private static string Golden(string name, string extension) => File.ReadAllText(Path.Combine(Repo.Golden, name + extension));

    [Theory]
    [MemberData(nameof(Names))]
    public void Conllu_RoundTripsStanzasOutput(string name)
    {
        var golden = Golden(name, ".conllu");
        var doc = Conllu.Read(golden);
        Assert.Equal(golden, Conllu.Write(doc));
    }

    // Conllu.Read rebuilds doc.coref from the coref_chains= marks alone: Stanza's indices, mentions in its order,
    // representative texts and indices.
    [Theory]
    [MemberData(nameof(Names))]
    public void Conllu_Read_RebuildsStanzasChains(string name)
    {
        var doc = Conllu.Read(Golden(name, ".conllu"));
        var expected = JsonNode.Parse(Golden(name, ".chains.json"))!.AsArray();
        Assert.Equal(expected.Count, doc.Coref.Count);
        for (int c = 0; c < expected.Count; c++)
        {
            var chain = doc.Coref[c];
            var golden = expected[c]!;
            Assert.Equal(golden["index"]!.GetValue<int>(), chain.Index);
            Assert.Equal(golden["representative_text"]!.GetValue<string>(), chain.RepresentativeText);
            Assert.Equal(golden["representative_index"]?.GetValue<int>(), chain.RepresentativeIndex);
            Assert.Equal(golden["mentions"]!.AsArray().Select(Mention), chain.Mentions);
        }

        if (expected.Count == 0)
            Assert.All(doc.Sentences.SelectMany(s => s.Words), w => Assert.Null(w.CorefChains));
        else
            Assert.All(doc.Sentences.SelectMany(s => s.Words.Concat(s.EmptyWords)), w => Assert.NotNull(w.CorefChains));
    }

    // [sentence, start, end], a zero mention's start and end being [word id, empty index].
    internal static CorefMention Mention(JsonNode? node)
    {
        var m = node!.AsArray();
        return m[1] is JsonArray zero
            ? new CorefMention(m[0]!.GetValue<int>(), zero[0]!.GetValue<int>(), zero[0]!.GetValue<int>(), zero[1]!.GetValue<int>())
            : new CorefMention(m[0]!.GetValue<int>(), m[1]!.GetValue<int>(), m[2]!.GetValue<int>());
    }

    // str(doc) of the same documents: every word's coref_chains (an empty list when in no mention), empty words in place.
    [Theory]
    [InlineData("coref/validation_dialogue")]
    [InlineData("coref/crafted")]
    public void Json_MatchesStanzasStr(string name)
    {
        var golden = Golden(name, ".str.json");

        // No parser ran, so the CoNLL-U's heads are the writer's dummies (id - 1), which str(doc) doesn't have.
        var doc = Conllu.Read(Golden(name, ".conllu"));
        foreach (var w in doc.Sentences.SelectMany(s => s.Words))
            w.Head = null;
        var actual = Json.Write(doc);
        Assert.True(golden == actual, $"--- expected\n{golden}\n--- actual\n{actual}");
        var read = Json.Read(golden);
        Assert.Equal(golden, Json.Write(read));
        Assert.Equal(doc.Coref, read.Coref, ChainComparer.Instance);
    }

    [Fact]
    public void Crafted_EmptyWordsAndAttachments()
    {
        var doc = Conllu.Read(Golden("coref/crafted", ".conllu"));
        Assert.Equal([(5, 1)], doc.Sentences[0].EmptyWords.Select(w => (w.Id, w.EmptyIndex!.Value)));
        Assert.Equal([(0, 1), (3, 2)], doc.Sentences[2].EmptyWords.Select(w => (w.Id, w.EmptyIndex!.Value)));
        Assert.All(doc.Sentences.SelectMany(s => s.EmptyWords), w => Assert.Equal(("_", true), (w.Text, w.IsEmpty)));

        // Empty words are not words: not counted, not in Words, so not in Dependencies.
        Assert.Equal(19, doc.WordCount);
        Assert.Equal(3, doc.Sentences[2].Words.Count());
        Assert.Equal("<Word id=(5, 1);text=_;lemma=_>", doc.Sentences[0].EmptyWords[0].ToString());

        // A zero mention's empty word is marked start and end, never representative; the chain of zeros has none.
        var zero = Assert.Single(doc.Sentences[0].EmptyWords[0].CorefChains!);
        Assert.Equal((0, true, true, false), (zero.Chain.Index, zero.IsStart, zero.IsEnd, zero.IsRepresentative));
        Assert.Equal(("_", (int?)null), (doc.Coref[2].RepresentativeText, doc.Coref[2].RepresentativeIndex));
        Assert.All(doc.Coref[2].Mentions, m => Assert.True(m.IsZero));

        // Every word of the representative mention is marked, as Stanza marks them.
        Assert.All(doc.Sentences[0].Words.Take(2), w => Assert.True(Assert.Single(w.CorefChains!).IsRepresentative));
    }

    [Fact]
    public void Coref_Setter_ThrowsForAMentionOutsideTheDocument()
    {
        var doc = Conllu.Read(Golden("coref/crafted", ".conllu"));
        var before = doc.Coref;
        Assert.Throws<ArgumentException>(() => doc.Coref = [new CorefChain(0, [new CorefMention(0, 7, 9)], "x", 0)]);
        Assert.Throws<ArgumentException>(() => doc.Coref = [new CorefChain(0, [new CorefMention(1, 2, 2, 1)], "_", null)]);
        Assert.Same(before, doc.Coref);
    }

    [Theory]
    [InlineData("1\tA\t_\t_\t_\t_\t0\t_\t_\tcoref_chains=end-id0\n")]
    [InlineData("1\tA\t_\t_\t_\t_\t0\t_\t_\tcoref_chains=start-id0\n")]
    [InlineData("1\tA\t_\t_\t_\t_\t0\t_\t_\tcoref_chains=unit-idx\n")]
    [InlineData("1\tA\t_\t_\t_\t_\t0\t_\t_\tcoref_chains=whole-id0\n")]
    public void Conllu_Read_ThrowsForMalformedCorefMarks(string conllu) =>
        Assert.Throws<FormatException>(() => Conllu.Read(conllu));

    private sealed class ChainComparer : IEqualityComparer<CorefChain>
    {
        public static readonly ChainComparer Instance = new();

        public bool Equals(CorefChain? x, CorefChain? y) =>
            x!.Index == y!.Index && x.RepresentativeText == y.RepresentativeText && x.RepresentativeIndex == y.RepresentativeIndex
            && x.Mentions.SequenceEqual(y.Mentions);

        public int GetHashCode(CorefChain obj) => obj.Index;
    }
}
