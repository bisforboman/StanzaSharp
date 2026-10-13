using System.Text.Json.Nodes;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class PosTests(ITestOutputHelper output, PosTests.Fixture fixture) : IClassFixture<PosTests.Fixture>
{
    // managed: Backend.Managed (issue #29), else TorchSharp. Every golden test runs on both.
    /// <summary>Each backend's and language's <see cref="Models"/>, loaded on first use and shared by the class's tests (read-only).</summary>
    public sealed class Fixture : IDisposable
    {
        private readonly ModelCache<Models> _models = new((managed, lang) => new Models(managed, lang));

        internal Models Get(bool managed = false, string lang = "en") => _models.Get(managed, lang);

        public void Dispose() => _models.Dispose();
    }

    internal sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain;
        public readonly CharLanguageModel? Forward, Backward;
        public readonly PosTagger Tagger;

        public Models(bool managed, string lang)
        {
            Pretrain = Repo.LoadPretrain(managed, lang);
            if (managed)
            {
                Tagger = PosTagger.LoadManaged(Repo.ModelOf(lang, "pos"), Pretrain,
                    ManagedCharLanguageModel.Load(Repo.Charlm(lang, forward: true)), ManagedCharLanguageModel.Load(Repo.Charlm(lang, forward: false)));
                return;
            }

            Forward = CharLanguageModel.Load(Repo.Charlm(lang, forward: true));
            Backward = CharLanguageModel.Load(Repo.Charlm(lang, forward: false));
            Tagger = PosTagger.Load(Repo.ModelOf(lang, "pos"), Pretrain, Forward, Backward);
        }

        public void Dispose()
        {
            Tagger.Dispose();
            Forward?.Dispose();
            Backward?.Dispose();
            Pretrain.Dispose();
        }
    }

    [Theory] // expected values from stanza.models.common.utils.simplify_punct
    [InlineData("??", "?")]
    [InlineData("?!?", "?")]
    [InlineData("!!!!!!", "!")]
    [InlineData("!?", "!")]
    [InlineData("？！", "?")]
    [InlineData("?", "?")]
    [InlineData("‼", "‼")]
    [InlineData("a??", "a??")]
    [Trait("Backend", "Managed")]
    public void SimplifyPunct_MatchesStanza(string word, string expected) =>
        Assert.Equal(expected, Punct.Simplify(word));

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void UposLogits_MatchGoldenIntermediates(bool managed, string lang)
    {
        var models = fixture.Get(managed, lang);
        var golden = SafeTensorFile.Load(Path.Combine(Repo.GoldenFor(lang), "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e => (IReadOnlyList<string>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList()).ToList();

        // One padded batch; the golden logits were computed one sentence at a time.
        var tags = models.Tagger.Predict(sentences, out var logits);
        for (int i = 0; i < sentences.Count; i++)
        {
            float diff = TokenizerTests.AssertClose(golden.Read<float>($"s{i}.pos.upos_logits"), logits[i], 1e-4f, $"s{i}");
            output.WriteLine($"s{i}: max |diff| {diff:E2}");
            Assert.Equal(index[i]!["xpos"]!.AsArray().Select(x => x!.GetValue<string>()), tags[i].Select(t => t.Xpos));
        }
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Pipeline_MatchesGoldenTags(bool managed, string lang)
    {
        var models = fixture.Get(managed, lang);
        using var tokenizer = Tokenizer.Load(Repo.ModelOf(lang, "tokenize"), backend: Repo.Backend(managed));
        using var mwt = MwtExpander.Load(Repo.ModelOf(lang, "mwt"), backend: Repo.Backend(managed));
        var doc = tokenizer.Process(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "corpus.txt")));
        mwt.Process(doc);
        models.Tagger.Process(doc);
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "pipeline.conllu")));

        // Through Conllu.Write, which sorts the feats like Stanza's writer (German's ExtPos, NumType, ... aren't in that order).
        Assert.Equal(Describe(golden), Describe(Conllu.Read(Conllu.Write(doc))));
    }

    private static string Describe(Document doc) =>
        string.Join("\n", doc.Sentences.SelectMany(s => s.Words.Select(w => $"{s.SentId}/{w.Id} {w.Text} {w.Upos} {w.Xpos} {w.Feats ?? "_"}")));
}
