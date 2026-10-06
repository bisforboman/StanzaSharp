using System.Text.Json.Nodes;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// stanza/models/common/biaffine.py <c>BiaffineScorer</c>: a bilinear layer over both inputs
/// with a constant 1 appended to each.
/// </summary>
internal sealed class Biaffine : IDisposable
{
    private readonly Bilinear _bilinear;

    /// <param name="prefix">State dict prefix of the scorer, e.g. <c>tag_clf.xpos.</c>.</param>
    public Biaffine(Checkpoint ckpt, JsonNode stateDict, string prefix, int input1, int input2, int output) =>
        _bilinear = nn.Bilinear(input1 + 1, input2 + 1, output).LoadFrom(ckpt, stateDict, prefix + "W_bilin.");

    public Tensor Forward(Tensor input1, Tensor input2)
    {
        using var scope = NewDisposeScope();
        return _bilinear.forward(AppendOne(input1), AppendOne(input2)).MoveToOuterDisposeScope();
    }

    private static Tensor AppendOne(Tensor x)
    {
        var shape = x.shape.ToArray();
        shape[^1] = 1;
        return cat([x, ones(shape, dtype: x.dtype, device: x.device)], -1);
    }

    public void Dispose() => _bilinear.Dispose();
}
