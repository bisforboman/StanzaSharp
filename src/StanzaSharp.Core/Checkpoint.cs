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
    /// <c>basePath.pt</c>. A path ending in <c>.pt</c> loads that file, and so does one ending in <c>.bin</c> (a Hugging
    /// Face <c>pytorch_model.bin</c>: torch.save's zip format, like a newer <c>.pt</c>).
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// The file is malformed, truncated or hostile (also thrown by later tensor reads and lookups). Loading takes time,
    /// memory and stack bounded by the file's size; nothing is instantiated by name.
    /// </exception>
    /// <exception cref="NotSupportedException">A well-formed big-endian .pt file.</exception>
    /// <exception cref="IOException">The operating system's errors (a missing or unreadable file) pass through.</exception>
    public static Checkpoint Load(string basePath)
    {
        var pt = basePath.EndsWith(".pt", StringComparison.OrdinalIgnoreCase) || basePath.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)
            ? basePath
            : basePath + ".pt";
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
            json = Parse(text, basePath + ".json");
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
    /// Parses a converted checkpoint's JSON like a <c>.pt</c> file's (round 36): its large flat containers (a str → int
    /// object or a list of str, <see cref="PyContainer.IsLarge"/>) become <see cref="PyContainer"/> nodes, read in the
    /// same pass that validates the strings, and the rest is parsed into nodes with those left out. Throws for invalid
    /// UTF-8, an escaped lone surrogate or a property name repeated in one object: <see cref="JsonNode"/> decodes strings and
    /// builds an object's dictionary lazily, so these would otherwise throw <see cref="InvalidOperationException"/> or
    /// <see cref="ArgumentException"/> later, wherever a model reads that name or value.
    /// </summary>
    private static JsonNode Parse(ReadOnlySpan<byte> json, string path)
    {
        if (!Utf8.IsValid(json))
            throw new InvalidDataException($"{path} is not valid UTF-8");

        // Each open container's path step (an index, or the current property name), and the innermost one's items while
        // it may still be a large container (a container in it, or another kind of value, rules it out).
        var steps = new List<object>();
        var arrays = new List<bool>();
        var names = new List<HashSet<string>?>(); // each open object's property names (null for an array)
        var large = new List<(object[] Steps, int Start, int End, object Value)>();
        object? items = null;
        string? key = null;
        int start = 0;
        var reader = new Utf8JsonReader(json);
        try
        {
            while (reader.Read())
            {
                switch (reader.TokenType)
                {
                    case JsonTokenType.PropertyName:
                        key = reader.GetString()!;
                        if (!names[^1]!.Add(key))
                            throw new InvalidDataException($"{path}: the property name '{key}' is repeated in one object");
                        steps[^1] = key;
                        break;
                    case JsonTokenType.StartObject or JsonTokenType.StartArray:
                        bool tagged = arrays.Count > 0 && !arrays[^1] && key!.StartsWith('$'); // {"$tuple": [...]} and the like
                        Value();
                        bool array = reader.TokenType == JsonTokenType.StartArray;
                        items = arrays.Count == 0 || tagged ? null : array ? new List<object?>() : new PyDict();
                        start = (int)reader.TokenStartIndex;
                        steps.Add(-1);
                        arrays.Add(array);
                        names.Add(array ? null : new HashSet<string>(StringComparer.Ordinal));
                        break;
                    case JsonTokenType.EndObject or JsonTokenType.EndArray:
                        if (items is PyDict { Count: >= PyContainer.MinCount } or List<object?> { Count: >= PyContainer.MinCount })
                        {
                            large.Add((steps[..^1].ToArray(), start, (int)reader.BytesConsumed, items));
                        }

                        items = null;
                        steps.RemoveAt(steps.Count - 1);
                        arrays.RemoveAt(arrays.Count - 1);
                        names.RemoveAt(names.Count - 1);
                        break;
                    case JsonTokenType.String:
                        if (items is List<object?> strings)
                        {
                            strings.Add(reader.GetString());
                            Count();
                        }
                        else
                        {
                            if (reader.ValueIsEscaped)
                                reader.GetString();
                            Value();
                        }

                        break;
                    case JsonTokenType.Number:
                        if (items is PyDict dict && reader.TryGetInt64(out long id) && dict.TryAdd(key!, id))
                            Count();
                        else
                            Value();
                        break;
                    default:
                        Value();
                        break;
                }
            }
        }
        catch (InvalidOperationException e)
        {
            throw new InvalidDataException($"{path}: {e.Message}", e);
        }

        JsonNode root;
        if (large.Count == 0)
        {
            root = JsonNode.Parse(json) ?? throw new InvalidDataException($"{path} is null");
        }
        else
        {
            var rest = new ArrayBufferWriter<byte>(json.Length);
            int at = 0;
            foreach (var (_, from, end, _) in large)
            {
                rest.Write(json[at..from]);
                rest.Write("null"u8);
                at = end;
            }

            rest.Write(json[at..]);
            root = JsonNode.Parse(rest.WrittenSpan) ?? throw new InvalidDataException($"{path} is null");
            foreach (var (where, _, _, value) in large)
            {
                var parent = root;
                foreach (var step in where.AsSpan(0, where.Length - 1))
                    parent = (step is string name ? parent.AsObject()[name] : parent.AsArray()[(int)step])!;
                if (where[^1] is string last)
                    parent.AsObject()[last] = PyContainer.Node(value);
                else
                    parent.AsArray()[(int)where[^1]] = PyContainer.Node(value);
            }
        }

        return root;

        // A value (or a container's start) in the innermost container: its index there, and it rules that one out.
        void Value()
        {
            if (arrays.Count > 0 && arrays[^1])
                steps[^1] = (int)steps[^1] + 1;
            items = null;
        }

        void Count()
        {
            if (arrays[^1])
                steps[^1] = (int)steps[^1] + 1;
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
