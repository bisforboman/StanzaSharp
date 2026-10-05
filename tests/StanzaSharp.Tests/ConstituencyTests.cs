using System.Text.Json.Nodes;
using StanzaSharp.Constituency;
using StanzaSharp.Mwt;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

public class ConstituencyTests
{
    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        public readonly CharLanguageModel Forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        public readonly CharLanguageModel Backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        public readonly ConstituencyParser Parser;

        public Models() => Parser = ConstituencyParser.Load(Repo.Model("constituency/ptb3-revised_charlm"), Pretrain, Forward, Backward);

        public void Dispose()
        {
            Parser.Dispose();
            Forward.Dispose();
            Backward.Dispose();
            Pretrain.Dispose();
        }
    }

    [ModelFact]
    public void TransitionScores_MatchGoldenIntermediates()
    {
        using var models = new Models();
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e =>
        {
            var words = e!["words"]!.AsArray().Select(w => w!.GetValue<string>());
            var tags = e["xpos"]!.AsArray().Select(t => t!.GetValue<string>());
            return (IReadOnlyList<(string, string)>)words.Zip(tags).ToList();
        }).ToList();

        // One batch; the golden scores were computed one sentence at a time.
        var scores = new List<List<float[]>>();
        var trees = models.Parser.Parse(sentences, scores);
        for (int i = 0; i < sentences.Count; i++)
        {
            Assert.Equal(index[i]!["tree"]!.GetValue<string>(), trees[i]?.ToString());
            Assert.Equal(golden[$"s{i}.constituency.scores"].Shape[0], scores[i].Count);
            TokenizerTests.AssertClose(golden.Read<float>($"s{i}.constituency.scores"), scores[i].SelectMany(r => r).ToArray(), 1e-3f, $"s{i}");
        }
    }

    [ModelFact]
    public void Pipeline_MatchesGoldenTrees()
    {
        using var models = new Models();
        using var tokenizer = Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"));
        using var mwt = MwtExpander.Load(Repo.Model("mwt/combined"));
        using var tagger = PosTagger.Load(Repo.Model("pos/combined_charlm"), models.Pretrain, models.Forward, models.Backward);

        var doc = tokenizer.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        mwt.Process(doc);
        tagger.Process(doc);
        models.Parser.Process(doc);
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));

        Assert.Equal(golden.Sentences.Select(s => s.Constituency?.ToString()), doc.Sentences.Select(s => s.Constituency?.ToString()));
    }
}
