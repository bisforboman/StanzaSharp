using System.Text.Json.Nodes;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using Xunit.Abstractions;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

public class PosTests(ITestOutputHelper output)
{
    // managed: Backend.Managed (issue #29), else TorchSharp. Every golden test runs on both.
    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        public readonly CharLanguageModel Forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        public readonly CharLanguageModel Backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        public readonly PosTagger Tagger;

        public Models(bool managed = false) => Tagger = managed
            ? PosTagger.LoadManaged(Repo.Model("pos/combined_charlm"), Pretrain,
                ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion")), ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion")))
            : PosTagger.Load(Repo.Model("pos/combined_charlm"), Pretrain, Forward, Backward);

        public void Dispose()
        {
            Tagger.Dispose();
            Forward.Dispose();
            Backward.Dispose();
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
    public void SimplifyPunct_MatchesStanza(string word, string expected) =>
        Assert.Equal(expected, PosTagger.SimplifyPunct(word));

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void UposLogits_MatchGoldenIntermediates(bool managed)
    {
        using var models = new Models(managed);
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
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
    [InlineData(false)]
    [InlineData(true)]
    public void Pipeline_MatchesGoldenTags(bool managed)
    {
        using var models = new Models(managed);
        using var tokenizer = Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"), backend: Repo.Backend(managed));
        using var mwt = MwtExpander.Load(Repo.Model("mwt/combined"), backend: Repo.Backend(managed));
        var doc = tokenizer.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        mwt.Process(doc);
        models.Tagger.Process(doc);
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));

        Assert.Equal(Describe(golden), Describe(doc));
    }

    private static List<string> Describe(Document doc) =>
        doc.Sentences.SelectMany(s => s.Words.Select(w => $"{s.SentId}/{w.Id} {w.Text} {w.Upos} {w.Xpos} {w.Feats ?? "_"}")).ToList();
}
