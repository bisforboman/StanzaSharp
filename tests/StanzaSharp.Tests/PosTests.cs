using System.Text.Json.Nodes;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

public class PosTests
{
    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        public readonly CharLanguageModel Forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        public readonly CharLanguageModel Backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        public readonly PosTagger Tagger;

        public Models() => Tagger = PosTagger.Load(Repo.Model("pos/combined_charlm"), Pretrain, Forward, Backward);

        public void Dispose()
        {
            Tagger.Dispose();
            Forward.Dispose();
            Backward.Dispose();
            Pretrain.Dispose();
        }
    }

    [ModelFact]
    public void UposLogits_MatchGoldenIntermediates()
    {
        using var models = new Models();
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e => (IReadOnlyList<string>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList()).ToList();

        // One padded batch; the golden logits were computed one sentence at a time.
        var tags = models.Tagger.Predict(sentences, out var logits);
        for (int i = 0; i < sentences.Count; i++)
        {
            TokenizerTests.AssertClose(golden.Read<float>($"s{i}.pos.upos_logits"), logits[i], 1e-3f, $"s{i}");
            Assert.Equal(index[i]!["xpos"]!.AsArray().Select(x => x!.GetValue<string>()), tags[i].Select(t => t.Xpos));
        }
    }

    [ModelFact]
    public void Pipeline_MatchesGoldenTags()
    {
        using var models = new Models();
        using var tokenizer = Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"));
        using var mwt = MwtExpander.Load(Repo.Model("mwt/combined"));
        var doc = tokenizer.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        mwt.Process(doc);
        models.Tagger.Process(doc);
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));

        Assert.Equal(Describe(golden), Describe(doc));
    }

    private static List<string> Describe(Document doc) =>
        doc.Sentences.SelectMany(s => s.Words.Select(w => $"{s.SentId}/{w.Id} {w.Text} {w.Upos} {w.Xpos} {w.Feats ?? "_"}")).ToList();
}
