using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Tests;

/// <summary>The managed-backend spike (issue #29, docs/managed-backend-spike.md) against TorchSharp and the golden data.</summary>
public class ManagedBackendTests
{
    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
    }

    private static float[] Unpad(float[] padded, int rows, int width, int paddedWidth) =>
        Enumerable.Range(0, rows).SelectMany(r => padded.AsSpan(r * paddedWidth, width).ToArray()).ToArray();

    [Theory]
    [InlineData(1, 7, 16, 1)]
    [InlineData(37, 300, 125, 4)] // ragged M, K over one K block, N not a whole panel
    [InlineData(200, 1024, 256, 8)]
    public void Gemm_MatchesLinear(int m, int k, int n, int threads)
    {
        var a = Random(m * k, 1);
        var w = Random(n * k, 2);
        var bias = Random(n, 3);
        var packed = new PackedMatrix(w, n, k);
        var paddedBias = new float[packed.PaddedN];
        bias.CopyTo(paddedBias, 0);
        int before = ManagedThreads.Count;
        ManagedThreads.Count = threads;
        try
        {
            var actual = Unpad(Gemm.Run(a, m, packed, paddedBias), m, n, packed.PaddedN);
            using var expected = nn.functional.linear(torch.tensor(a, [m, k]), torch.tensor(w, [n, k]), torch.tensor(bias));
            TokenizerTests.AssertClose(expected.data<float>().ToArray(), actual, 1e-4f, "gemm");
        }
        finally
        {
            ManagedThreads.Count = before;
        }
    }

    [Fact]
    public void PackedBiLstm_MatchesTorchLstm()
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        const int input = 10, hidden = 12;
        long[] lengths = [5, 3, 5, 1, 2, 4]; // unsorted, with ties: packing must be pack_padded_sequence's
        torch.manual_seed(7);
        var lstm = nn.LSTM(input, hidden, batchFirst: true, bidirectional: true);
        float[] P(string name) => lstm.get_parameter(name)!.data<float>().ToArray();
        var h0 = Random(2 * hidden, 4);
        var c0 = Random(2 * hidden, 5);

        var x = torch.randn(lengths.Length, lengths.Max(), input);
        var packed = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(lengths), batch_first: true, enforce_sorted: false);
        var state = (torch.tensor(h0, [2, 1, hidden]).expand(2, lengths.Length, hidden).contiguous(),
            torch.tensor(c0, [2, 1, hidden]).expand(2, lengths.Length, hidden).contiguous());
        var (expected, _, _) = lstm.call(packed, state);

        var (w, bias) = PackedLstm.PackInput(hidden, input, [P("weight_ih_l0"), P("weight_ih_l0_reverse")],
            [P("bias_ih_l0"), P("bias_ih_l0_reverse")], [P("bias_hh_l0"), P("bias_hh_l0_reverse")]);
        var managed = new PackedLstm(hidden, [P("weight_hh_l0"), P("weight_hh_l0_reverse")],
            [h0[..hidden], h0[hidden..]], [c0[..hidden], c0[hidden..]]);
        var data = packed.data.data<float>().ToArray();
        int rows = data.Length / input;
        var actual = managed.Recur(Gemm.Run(data, rows, w, bias), w.PaddedN, PackedLstm.BatchSizes(lengths));
        TokenizerTests.AssertClose(expected.data.data<float>().ToArray(), actual, 1e-5f, "bilstm");
    }

    [ModelFact]
    public void CharLanguageModels_MatchGoldenRepresentations()
    {
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e => (IReadOnlyList<string>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList()).ToList();
        foreach (var direction in new[] { "forward", "backward" })
        {
            var charlm = ManagedCharLanguageModel.Load(Repo.Model($"{direction}_charlm/1billion"));
            var reps = charlm.BuildCharRepresentation(sentences);
            for (int i = 0; i < sentences.Count; i++)
                TokenizerTests.AssertClose(golden.Read<float>($"s{i}.charlm_{direction}"), reps[i], 1e-4f, $"{direction} s{i}");
        }
    }

    [ModelFact]
    public void HighwayLstm_MatchesTorchSharp()
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var ckpt = Checkpoint.Load(Repo.Model("pos/combined_charlm"));
        var model = ckpt.Root["model"]!;
        using var reference = new HighwayLstm(ckpt, model, "taggerlstm", 2248, 200, 2);
        var managed = new ManagedHighwayLstm(ckpt, model, "taggerlstm", 2248, 200, 2);

        long[] lengths = [7, 30, 1, 12, 12];
        var input = Random((int)lengths.Sum() * 2248, 9);
        var expected = reference.Forward(Rnn.Pack(torch.tensor(input, [lengths.Sum(), 2248]), lengths)).data.data<float>().ToArray();
        TokenizerTests.AssertClose(expected, managed.Forward(input, lengths), 1e-4f, "highway");
    }
}
