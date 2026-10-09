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
    public void Pack_EqualsPackPaddedSequence()
    {
        using var scope = TorchSharp.torch.NewDisposeScope();
        long[] lengths = [3, 5, 1, 5, 3, 2]; // ties: the order must be pack_padded_sequence's
        var padded = TorchSharp.torch.randn(lengths.Length, 5, 4);
        var expected = TorchSharp.torch.nn.utils.rnn.pack_padded_sequence(padded, TorchSharp.torch.tensor(lengths), batch_first: true, enforce_sorted: false);
        var rows = Rnn.PackedOrder(lengths);
        var packed = Rnn.Pack(padded.reshape(-1, 4).index_select(0, TorchSharp.torch.tensor(rows)), lengths);
        Assert.Equal(expected.batch_sizes.ToArray<long>(), packed.batch_sizes.ToArray<long>());
        Assert.Equal(expected.data.data<float>().ToArray(), packed.data.data<float>().ToArray());
    }

    [Fact]
    public void CharlmCache_KeepsAtMostMaxWords()
    {
        using var cache = new CharlmCache(maxWords: 5);
        Sentence a = new(), b = new(), c = new();
        Assert.True(cache.TryAdd(a, TorchSharp.torch.zeros(3, 2), TorchSharp.torch.zeros(3, 2)));
        using (var rejected = TorchSharp.torch.zeros(3, 2))
        {
            Assert.False(cache.TryAdd(b, rejected, rejected)); // 6 words: not kept, and still the caller's
            Assert.Equal(0f, rejected.sum().item<float>()); // not disposed by the cache
        }
        Assert.True(cache.TryAdd(c, TorchSharp.torch.zeros(2, 2), TorchSharp.torch.zeros(2, 2)));
        Assert.True(cache.TryGet(a, out _));
        Assert.False(cache.TryGet(b, out _));
        Assert.True(cache.TryGet(c, out _));
        Assert.True(cache.TryAdd(a, TorchSharp.torch.zeros(1, 2), TorchSharp.torch.zeros(1, 2))); // replacing frees a's words
        Assert.True(cache.TryAdd(b, TorchSharp.torch.zeros(2, 2), TorchSharp.torch.zeros(2, 2)));
        Assert.True(cache.TryGet(b, out _));
    }

    [Fact]
    public void CharlmCache_ReadsEitherBackendsEntries()
    {
        using var cache = new CharlmCache(maxWords: 5);
        Sentence torchMade = new(), managedMade = new();
        float[] f = [1, 2, 3, 4, 5, 6], b = [6, 5, 4, 3, 2, 1];
        Assert.True(cache.TryAdd(torchMade, TorchSharp.torch.tensor(f, [3, 2]), TorchSharp.torch.tensor(b, [3, 2])));
        Assert.True(cache.TryGetArrays(torchMade, out var arrays)); // a copy for the managed backend
        Assert.Equal(f, arrays.Forward);
        Assert.Equal(b, arrays.Backward);
        Assert.False(cache.TryAdd(managedMade, f, b, 3)); // 6 words: managed entries count the same
        Assert.True(cache.TryAdd(managedMade, f[..4], b[..4], 2));
        Assert.True(cache.TryGetArrays(managedMade, out arrays));
        Assert.Same(arrays.Forward, cache.TryGetArrays(managedMade, out var again) ? again.Forward : null);
        Assert.True(cache.TryGet(managedMade, out var tensors)); // converted once, kept with the entry
        Assert.Equal([2L, 2L], tensors.Forward.shape);
        Assert.Equal(b[..4], tensors.Backward.data<float>().ToArray());
        Assert.True(cache.TryGet(managedMade, out var same));
        Assert.Same(tensors.Forward, same.Forward);
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

    /// <summary>The managed backend's pretrain: the same vocabulary and vectors as a plain array, no tensor.</summary>
    [ModelFact]
    public void Pretrain_LoadManaged_EqualsTheTensor()
    {
        using var tensor = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var managed = Pretrain.LoadManaged(Repo.Model("pretrain/conll17"));
        Assert.Equal((250000, 100), (managed.Count, managed.Dim));
        Assert.Equal(5445, managed.UnitToId("stanford"));
        Assert.True(tensor.CpuVectors().SequenceEqual(managed.CpuVectors()));
        Assert.Throws<InvalidOperationException>(() => managed.Embeddings);
    }
}
