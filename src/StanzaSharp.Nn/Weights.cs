using System.Text.Json.Nodes;
using TorchSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>Copies weights from a converted checkpoint into TorchSharp modules.</summary>
public static class Weights
{
    /// <summary>Reads a float32 <c>$tensor</c> node as a TorchSharp tensor.</summary>
    public static Tensor ToTensor(this Checkpoint ckpt, JsonNode? node) =>
        torch.tensor(ckpt.Tensor<float>(node), ckpt.Shape(node));

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
            using var value = ckpt.ToTensor(node);
            if (!param.shape.SequenceEqual(value.shape))
                throw new InvalidOperationException(
                    $"Weight '{key}' is [{string.Join(", ", value.shape)}], module expects [{string.Join(", ", param.shape)}]");
            param.copy_(value);
        }
        module.eval();
        return module;
    }
}
