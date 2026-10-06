using System.Text.Json.Nodes;
using TorchSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>Copies weights from a converted checkpoint into TorchSharp modules.</summary>
public static class Weights
{
    /// <summary>Reads a float32 <c>$tensor</c> node as a TorchSharp tensor.</summary>
    public static Tensor ToTensor(this Checkpoint ckpt, JsonNode? node)
    {
        var t = torch.empty(ckpt.Shape(node), ScalarType.Float32);
        ckpt.ReadInto(node, t);
        return t;
    }

    /// <summary>Reads a float32 <c>$tensor</c> node straight into the CPU memory of <paramref name="target"/>, with no intermediate copy.</summary>
    private static void ReadInto(this Checkpoint ckpt, JsonNode? node, Tensor target)
    {
        var key = node?["$tensor"]?.GetValue<string>() ?? throw new ArgumentException($"Not a $tensor node: {node?.ToJsonString()}");
        if (ckpt.Tensors[key].Dtype != "F32")
            throw new InvalidOperationException($"Tensor '{key}' is {ckpt.Tensors[key].Dtype}, not F32");
        if (target.dtype != ScalarType.Float32 || target.device_type != DeviceType.CPU || !target.is_contiguous())
            throw new ArgumentException("Can only read into a contiguous float32 CPU tensor");
        ckpt.Tensors.ReadInto(key, target.bytes);
    }

    /// <summary>
    /// Fills every parameter of <paramref name="module"/> from <c>stateDict[prefix + name]</c> and
    /// switches it to eval mode. TorchSharp's built-in modules (Linear, Embedding, LSTM, ...) use the
    /// same parameter names as PyTorch, so a prefix like <c>"rnn."</c> is all the mapping needed.
    /// </summary>
    public static T LoadFrom<T>(this T module, Checkpoint ckpt, JsonNode stateDict, string prefix) where T : nn.Module
    {
        using var _ = torch.no_grad();
        foreach (var (name, param) in module.named_parameters())
        {
            var key = prefix + name;
            var node = stateDict[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'");
            var shape = ckpt.Shape(node);
            if (!param.shape.SequenceEqual(shape))
                throw new InvalidOperationException(
                    $"Weight '{key}' is [{string.Join(", ", shape)}], module expects [{string.Join(", ", param.shape)}]");
            ckpt.ReadInto(node, param);
        }
        module.eval();
        return module;
    }
}
