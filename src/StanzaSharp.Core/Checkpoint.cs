using System.Buffers;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Unicode;

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
        if (pt == basePath || (!File.Exists(basePath + ".json") && File.Exists(pt)))
        {
            var (root, tensors) = TorchCheckpoint.Load(pt);
            return new Checkpoint(root, tensors);
        }

        JsonNode json;
        var bytes = File.ReadAllBytes(basePath + ".json");
        try
        {
            var text = bytes.AsSpan();
            if (text.StartsWith("﻿"u8))
                text = text[3..];
            ValidateStrings(text, basePath + ".json");
            json = JsonNode.Parse(text) ?? throw new InvalidDataException($"{basePath}.json is null");
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{basePath}.json: {e.Message}", e);
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
        if (map is JsonValue v && v.TryGetValue(out PyContainer? large))
        {
            if (large.Value is not PyDict dict)
                throw new InvalidDataException("_unit2id is not an object");
            var ids = new Dictionary<string, int>(dict.Count);
            foreach (var (key, id) in dict)
            {
                ids[(string)key] = id is long n && n is >= int.MinValue and <= int.MaxValue
                    ? (int)n
                    : throw new InvalidDataException($"_unit2id['{key}'] is not an int");
            }

            return ids;
        }

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
            result[key] = reader.TokenType == JsonTokenType.Number && reader.TryGetInt32(out int id)
                ? id
                : throw new InvalidDataException($"_unit2id['{key}'] is not an int");
        }

        return result;
    }

    /// <summary>
    /// Throws for invalid UTF-8 or an escaped lone surrogate. <see cref="JsonNode"/> decodes strings lazily, so either
    /// would otherwise throw <see cref="InvalidOperationException"/> later, wherever a model reads that name or value.
    /// </summary>
    private static void ValidateStrings(ReadOnlySpan<byte> json, string path)
    {
        if (!Utf8.IsValid(json))
            throw new InvalidDataException($"{path} is not valid UTF-8");
        var reader = new Utf8JsonReader(json);
        while (reader.Read())
        {
            if (reader.TokenType is JsonTokenType.PropertyName or JsonTokenType.String && reader.ValueIsEscaped)
            {
                try
                {
                    reader.GetString();
                }
                catch (InvalidOperationException e)
                {
                    throw new InvalidDataException($"{path}: {e.Message}", e);
                }
            }
        }
    }

    /// <summary>Reads a list of strings, such as a vocab's <c>_id2unit</c>.</summary>
    /// <remarks>
    /// Use this, not <see cref="JsonNode.AsArray"/>, for lists that can be long: a <c>.pt</c> file's lists of
    /// <see cref="PyContainer.MinCount"/> or more strings are a <see cref="JsonValue"/>, not a <see cref="JsonArray"/>.
    /// </remarks>
    public static string[] Strings(JsonNode? list) => list switch
    {
        JsonValue v when v.TryGetValue(out PyContainer? large) && large.Value is List<object?> items =>
            items.Select(x => (string)x!).ToArray(),
        JsonArray a => a.Select(x => x is JsonValue s && s.TryGetValue(out string? text)
            ? text
            : throw new InvalidDataException($"Not a string: {x?.GetPath()}")).ToArray(),
        _ => throw new InvalidDataException($"Not a list of strings: {list?.GetPath()}"),
    };

    private static string TensorKey(JsonNode? node) =>
        node is JsonObject o && o["$tensor"] is JsonValue key && key.TryGetValue(out string? s)
            ? s
            : throw new InvalidDataException($"Not a $tensor node: {node?.ToJsonString()}");
}
