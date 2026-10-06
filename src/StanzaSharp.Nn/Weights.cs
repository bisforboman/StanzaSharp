using System.Text.Json.Nodes;
using TorchSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>Copies weights from a converted checkpoint into TorchSharp modules, on the device being loaded for.</summary>
public static class Weights
{
    // The device models are loaded onto, set around a load by On. Flow-local, so concurrent loads on
    // different threads don't see each other's device, and nothing process-global is touched
    // (unlike torch.set_default_device).
    private static readonly AsyncLocal<Device?> LoadDevice = new();

    /// <summary>
    /// The device of the load in progress (CPU outside one). Models capture it when constructed and
    /// create their input tensors there.
    /// </summary>
    public static Device Device => LoadDevice.Value ?? CPU;

    /// <summary>
    /// Runs <paramref name="load"/> with every weight it reads placed on <paramref name="device"/>.
    /// Null keeps the device of an enclosing load (CPU if none).
    /// </summary>
    public static T On<T>(Device? device, Func<T> load)
    {
        var outer = LoadDevice.Value;
        LoadDevice.Value = device ?? outer;
        try
        {
            return load();
        }
        finally
        {
            LoadDevice.Value = outer;
        }
    }

    /// <summary>Reads a float32 <c>$tensor</c> node as a TorchSharp tensor on <see cref="Device"/>.</summary>
    public static Tensor ToTensor(this Checkpoint ckpt, JsonNode? node)
    {
        var t = torch.empty(ckpt.Shape(node), ScalarType.Float32);
        ckpt.ReadInto(node, t);
        return Device.type == DeviceType.CPU ? t : t.to(Device, disposeAfter: true);
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
    /// Fills every parameter of <paramref name="module"/> from <c>stateDict[prefix + name]</c>, moves it to
    /// <see cref="Device"/> and switches it to eval mode. TorchSharp's built-in modules (Linear, Embedding,
    /// LSTM, ...) use the same parameter names as PyTorch, so a prefix like <c>"rnn."</c> is all the mapping needed.
    /// </summary>
    public static T LoadFrom<T>(this T module, Checkpoint ckpt, JsonNode stateDict, string prefix) where T : nn.Module
    {
        using (torch.no_grad())
            foreach (var (name, param) in module.named_parameters())
            {
                var key = prefix + name;
                var node = stateDict[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'");
                var shape = ckpt.Shape(node);
                if (!param.shape.SequenceEqual(shape))
                    throw new InvalidOperationException(
                        $"Weight '{key}' is [{string.Join(", ", shape)}], module expects [{string.Join(", ", param.shape)}]");
                ckpt.ReadInto(node, param); // parameters are on the CPU until module.to below
            }
        if (Device.type != DeviceType.CPU)
            module.to(Device);
        module.eval();
        return module;
    }

    /// <summary>A tensor's elements as an array, copied to the CPU first if needed.</summary>
    public static T[] ToArray<T>(this Tensor tensor) where T : unmanaged
    {
        if (tensor.device_type == DeviceType.CPU)
            return tensor.data<T>().ToArray();
        using var cpu = tensor.cpu();
        return cpu.data<T>().ToArray();
    }
}
