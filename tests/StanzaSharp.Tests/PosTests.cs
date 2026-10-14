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

    /// <summary>
    /// Sentence <paramref name="i"/>'s UPOS logits: each within 1e-4 (<see cref="TokenizerTests.Allowed"/>: relative above
    /// |100|) of Stanza's float32 value ("upos_logits") or of the float64 one ("upos_logits64", the model run in float64), as
    /// for sentiment (owner's decision, 2026-10-13). German's logits reach |380|, and Stanza's own float32 ones are up to
    /// 1.01e-4 from float64.
    /// </summary>
    /// <returns>The largest difference from the float32 logits, and the largest accepted one (the nearer reference per logit).</returns>
    internal static (float Float32, float Accepted) AssertUposClose(SafeTensorFile golden, int i, float[] actual, string what)
    {
        var expected = golden.Read<float>($"s{i}.pos.upos_logits");
        var exact = golden.Read<double>($"s{i}.pos.upos_logits64");
        Assert.Equal(expected.Length, actual.Length);
        Assert.Equal(exact.Length, actual.Length);
        float worst32 = 0, worst = 0;
        int over = -1;
        for (int k = 0; k < actual.Length; k++)
        {
            float d32 = Math.Abs(expected[k] - actual[k]);
            double d64 = Math.Abs(exact[k] - actual[k]);
            worst32 = Math.Max(worst32, d32);
            worst = Math.Max(worst, Math.Min(d32, (float)d64));
            if (over < 0 && !(d32 <= TokenizerTests.Allowed(1e-4f, expected[k]) || d64 <= TokenizerTests.Allowed(1e-4f, exact[k])))
                over = k;
        }

        if (over >= 0)
            Assert.Fail($"{what}: logit {over} is {actual[over]}, expected {expected[over]} (float32) or {exact[over]} (float64), allowed {TokenizerTests.Allowed(1e-4f, exact[over]):E2}");
        return (worst32, worst);
    }

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
            var (diff, accepted) = AssertUposClose(golden, i, logits[i], $"s{i}");
            output.WriteLine($"s{i}: max |diff| vs float32 {diff:E2}, accepted (nearer of float32 and float64) {accepted:E2}");
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
