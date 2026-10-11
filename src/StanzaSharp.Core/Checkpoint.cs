using System.Buffers;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace StanzaSharp;

/// <summary>
/// A checkpoint converted by tools/stanza_convert.py: <c>name.json</c> mirrors the original
/// structure, with each tensor replaced by <c>{"$tensor": key, "dtype", "shape"}</c> pointing into
/// <c>name.safetensors</c>. Non-JSON Python values stay tagged (<c>$tuple</c>, <c>$set</c>, <c>$dict</c>, ...).
/// An original PyTorch <c>name.pt</c> file loads into the same structure, without Python.
/// </summary>
internal sealed class Checkpoint
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
    /// <exception cref="InvalidDataException">
    /// The file is malformed, truncated or hostile (also thrown by later tensor reads and lookups). Loading takes time,
    /// memory and stack bounded by the file's size; nothing is instantiated by name.
    /// </exception>
    /// <exception cref="NotSupportedException">A well-formed big-endian .pt file.</exception>
    /// <exception cref="IOException">The operating system's errors (a missing or unreadable file) pass through.</exception>
    public static Checkpoint Load(string basePath)
    {
        var pt = basePath.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) ? basePath : basePath + ".pt";
        if (pt == basePath || !File.Exists(basePath + ".json") && File.Exists(pt))
        {
            var (root, tensors) = TorchCheckpoint.Load(pt);
            return new Checkpoint(root, tensors);
        }
        JsonNode json;
        using (var stream = File.OpenRead(basePath + ".json"))
        {
            try
            {
                json = JsonNode.Parse(stream) ?? throw new InvalidDataException($"{basePath}.json is null");
            }
            catch (JsonException e)
            {
                throw new InvalidDataException($"{basePath}.json: {e.Message}", e);
            }
        }
        return new Checkpoint(json, SafeTensorFile.Load(basePath + ".safetensors"));
    }

    /// <summary>Reads the tensor a <c>$tensor</c> node in the JSON points to.</summary>
    public T[] Tensor<T>(JsonNode? node) where T : unmanaged => Tensors.Read<T>(TensorKey(node));

    public long[] Shape(JsonNode? node) => Tensors[TensorKey(node)].Shape;

    /// <summary>Reads a Stanza vocab's <c>_unit2id</c> map, given the vocab's JSON node.</summary>
    /// <remarks>
    /// Reads the map as JSON text rather than enumerating the <see cref="JsonObject"/>, which would build
    /// a node per entry: the pretrain vocabulary has 250k of them.
    /// </remarks>
    public static Dictionary<string, int> UnitToId(JsonNode? vocab)
    {
        var map = (vocab as JsonObject)?["_unit2id"] ?? throw new InvalidDataException("Not a vocab node: no _unit2id");
        var buffer = new ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(buffer))
            map.WriteTo(writer);
        var reader = new Utf8JsonReader(buffer.WrittenSpan);
        if (!reader.Read() || reader.TokenType != JsonTokenType.StartObject)
            throw new InvalidDataException("_unit2id is not an object");
        var result = new Dictionary<string, int>();
        while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
        {
            var key = reader.GetString()!;
            reader.Read();
            result[key] = reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out int id) ? id
                : throw new InvalidDataException($"_unit2id['{key}'] is not an int");
        }
        return result;
    }

    private static string TensorKey(JsonNode? node) =>
        node is JsonObject o && o["$tensor"] is JsonValue key && key.TryGetValue(out string? s) ? s
            : throw new InvalidDataException($"Not a $tensor node: {node?.ToJsonString()}");
}
