using System.Text.Json.Nodes;
using StanzaSharp.Nn;

namespace StanzaSharp.Tests;

public class NnTests
{
    [ModelFact]
    public void CharLanguageModels_MatchGoldenRepresentations()
    {
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e => (IReadOnlyList<string>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList()).ToList();

        foreach (var direction in new[] { "forward", "backward" })
        {
            using var charlm = CharLanguageModel.Load(Repo.Model($"{direction}_charlm/1billion"));
            Assert.Equal(direction == "forward", charlm.IsForward);

            // All sentences in one padded batch: the golden tensors were computed one sentence at a time.
            var reps = charlm.BuildCharRepresentation(sentences);
            for (int i = 0; i < sentences.Count; i++)
            {
                using var rep = reps[i];
                Assert.Equal(golden[$"s{i}.charlm_{direction}"].Shape, rep.shape);
                TokenizerTests.AssertClose(golden.Read<float>($"s{i}.charlm_{direction}"), rep.data<float>().ToArray(), 1e-4f, $"{direction} s{i}");
            }
        }
    }

    [Fact]
    public void CharlmCache_KeepsAtMostMaxWords()
    {
        using var cache = new CharlmCache(maxWords: 5);
        Sentence a = new(), b = new(), c = new();
        cache.Add(a, TorchSharp.torch.zeros(3, 2), TorchSharp.torch.zeros(3, 2));
        cache.Add(b, TorchSharp.torch.zeros(3, 2), TorchSharp.torch.zeros(3, 2)); // 6 words: not kept
        cache.Add(c, TorchSharp.torch.zeros(2, 2), TorchSharp.torch.zeros(2, 2));
        Assert.True(cache.TryGet(a, out _));
        Assert.False(cache.TryGet(b, out _));
        Assert.True(cache.TryGet(c, out _));
        cache.Add(a, TorchSharp.torch.zeros(1, 2), TorchSharp.torch.zeros(1, 2)); // replacing frees a's words
        cache.Add(b, TorchSharp.torch.zeros(2, 2), TorchSharp.torch.zeros(2, 2));
        Assert.True(cache.TryGet(b, out _));
    }

    [ModelFact]
    public void Pretrain_LooksUpWordsAndVectors()
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        Assert.Equal(100, pretrain.Dim);
        Assert.Equal([250000L, 100L], pretrain.Embeddings.shape);

        // Values from stanza.models.common.pretrain.Pretrain on the original .pt file.
        Assert.Equal(6, pretrain.UnitToId("the"));
        Assert.Equal(5445, pretrain.UnitToId("stanford"));
        Assert.Equal(pretrain.UnkId, pretrain.UnitToId("The"));
        Assert.Equal(pretrain.UnkId, pretrain.UnitToId("qqqzzzxx"));
        Assert.Equal(1, pretrain.UnkId);

        using var row = pretrain.Embeddings[6];
        Assert.Equal(-0.1453009992837906f, row[0].item<float>());
        Assert.Equal(-0.11859499663114548f, row[99].item<float>());
    }
}
