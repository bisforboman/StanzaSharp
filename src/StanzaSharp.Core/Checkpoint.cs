using System.Text.Json.Nodes;

namespace StanzaSharp;

/// <summary>
/// A checkpoint converted by tools/stanza_convert.py: <c>name.json</c> mirrors the original
/// structure, with each tensor replaced by <c>{"$tensor": key, "dtype", "shape"}</c> pointing into
/// <c>name.safetensors</c>. Non-JSON Python values stay tagged (<c>$tuple</c>, <c>$set</c>, <c>$dict</c>, ...).
/// An original PyTorch <c>name.pt</c> file loads into the same structure, without Python.
/// </summary>
public sealed class Checkpoint
{
    public JsonNode Root { get; }
    public SafeTensorFile Tensors { get; }

    private Checkpoint(JsonNode root, SafeTensorFile tensors)
    {
        Root = root;
        Tensors = tensors;
    }

    /// <summary>
    /// Loads <c>basePath.json</c> and <c>basePath.safetensors</c> if they exist, else the original
    /// <c>basePath.pt</c>. A path ending in <c>.pt</c> loads that file.
    /// </summary>
    public static Checkpoint Load(string basePath)
    {
        var pt = basePath.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) ? basePath : basePath + ".pt";
        if (pt == basePath || !File.Exists(basePath + ".json") && File.Exists(pt))
        {
            var (root, tensors) = TorchCheckpoint.Load(pt);
            return new Checkpoint(root, tensors);
        }
        var json = JsonNode.Parse(File.ReadAllText(basePath + ".json"))
            ?? throw new FormatException($"{basePath}.json is empty");
        return new Checkpoint(json, SafeTensorFile.Load(basePath + ".safetensors"));
    }

    /// <summary>Reads the tensor a <c>$tensor</c> node in the JSON points to.</summary>
    public T[] Tensor<T>(JsonNode? node) where T : unmanaged => Tensors.Read<T>(TensorKey(node));

    public long[] Shape(JsonNode? node) => Tensors[TensorKey(node)].Shape;

    /// <summary>Reads a Stanza vocab's <c>_unit2id</c> map, given the vocab's JSON node.</summary>
    public static Dictionary<string, int> UnitToId(JsonNode? vocab) =>
        (vocab?["_unit2id"] ?? throw new ArgumentException("Not a vocab node: no _unit2id")).AsObject()
            .ToDictionary(kv => kv.Key, kv => kv.Value!.GetValue<int>());

    private static string TensorKey(JsonNode? node) =>
        node?["$tensor"]?.GetValue<string>() ?? throw new ArgumentException($"Not a $tensor node: {node?.ToJsonString()}");
}
