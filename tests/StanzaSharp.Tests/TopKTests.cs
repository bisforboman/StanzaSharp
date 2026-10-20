using System.Text.Json.Nodes;
using StanzaSharp.Nn;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="TopK"/> against torch.topk(sorted=False) on Linux (tests/golden/coref/topk.json, make_golden.py --topk-only):
/// the same indices in the same order, ties and -inf rows included. No models needed.
/// </summary>
public class TopKTests
{
    public static TheoryData<string> Cases() =>
        [.. Golden().Select(c => c!["name"]!.GetValue<string>())];

    private static JsonArray Golden() => JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "coref", "topk.json")))!.AsArray();

    private static float Value(JsonNode? v) => v == null ? float.NegativeInfinity : v.GetValue<float>();

    [Theory]
    [MemberData(nameof(Cases))]
    public void Select_MatchesTorchsOrder(string name)
    {
        var c = Golden().First(c => c!["name"]!.GetValue<string>() == name)!;
        int k = c["k"]!.GetValue<int>();
        var rows = c["rows"]!.AsArray();
        for (int r = 0; r < rows.Count; r++)
        {
            var row = rows[r]!.AsArray().Select(Value).ToArray();
            var indices = new int[k];
            var values = new float[k];
            TopK.Select(row, k, indices, values);
            Assert.Equal(c["indices"]![r]!.AsArray().Select(i => i!.GetValue<int>()), indices);
            Assert.Equal(c["values"]![r]!.AsArray().Select(Value), values);
        }
    }
}
