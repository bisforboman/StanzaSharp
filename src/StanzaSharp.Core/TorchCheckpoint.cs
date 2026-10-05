using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace StanzaSharp;

/// <summary>
/// Reads a PyTorch <c>.pt</c> checkpoint directly and splits it exactly as tools/stanza_convert.py does:
/// the same JSON mirror (tensor keys, <c>$tuple</c>/<c>$dict</c>/... tagging, <c>optimizer</c> and
/// <c>scheduler</c> dropped) and the same contiguous little-endian tensor data.
/// </summary>
internal static class TorchCheckpoint
{
    private static readonly string[] Skip = ["optimizer", "scheduler"];

    // torch.serialization.MAGIC_NUMBER and PROTOCOL_VERSION of the legacy (non-zip) format.
    private static readonly BigInteger LegacyMagic = BigInteger.Parse("1950a86a20f9469cfc6c", NumberStyles.HexNumber);
    private const long LegacyProtocol = 1001;

    public static (JsonNode Root, SafeTensorFile Tensors) Load(string path)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("Tensor data is read as little-endian");
        var (obj, storages) = IsZip(path) ? ReadZip(path) : ReadLegacy(path);
        if (obj is PyDict top)
            foreach (var key in Skip)
                top.Remove(key);

        var buffer = new ArrayBufferWriter<byte>();
        var splitter = new Splitter(new Utf8JsonWriter(buffer));
        splitter.Walk(obj);
        splitter.Writer.Flush();
        var root = JsonNode.Parse(buffer.WrittenSpan) ?? throw new InvalidDataException($"{path}: checkpoint is None");
        return (root, Materialize(splitter.Tensors, storages));
    }

    private static bool IsZip(string path)
    {
        using var f = File.OpenRead(path);
        Span<byte> head = stackalloc byte[4];
        return f.ReadAtLeast(head, 4, throwOnEndOfStream: false) == 4 && head.SequenceEqual("PK\x03\x04"u8);
    }

    /// <summary>
    /// The legacy format (<c>_use_new_zipfile_serialization=False</c>), a sequence of pickles: magic number,
    /// protocol version, sys_info, the checkpoint itself, the list of storage keys; then for each of those
    /// keys an int64 element count followed by the storage's raw bytes.
    /// </summary>
    private static (object?, Dictionary<string, ArraySegment<byte>>) ReadLegacy(string path)
    {
        var bytes = File.ReadAllBytes(path);
        int pos = 0;
        if (Unpickler.Load(bytes, ref pos) is not BigInteger magic || magic != LegacyMagic)
            throw new InvalidDataException($"{path}: not a PyTorch checkpoint (bad magic number)");
        if (Unpickler.Load(bytes, ref pos) is not LegacyProtocol)
            throw new InvalidDataException($"{path}: unsupported legacy protocol version");
        if (Unpickler.Load(bytes, ref pos) is not PyDict sysInfo || !sysInfo.TryGetValue("little_endian", out var le) || le is not true)
            throw new NotSupportedException($"{path}: big-endian checkpoints are not supported");

        var roots = new Dictionary<string, TorchStorage>();
        // Persistent id: ("storage", storage_type, root_key, location, root_numel, view_metadata),
        // where view_metadata is None or (view_key, offset, numel) for a slice of the root storage.
        var obj = Unpickler.Load(bytes, ref pos, pid =>
        {
            if (pid is not PyTuple { Items: ["storage", TorchStorageType type, string key, string, long numel, var view] })
                throw new InvalidDataException($"{path}: unsupported persistent id {pid}");
            var root = Root(roots, new TorchStorage(type, key, numel, 0), path);
            return view switch
            {
                null => root,
                PyTuple { Items: [string, long offset, long size] } when offset >= 0 && size >= 0 && offset + size <= numel =>
                    root with { Numel = size, Offset = offset },
                _ => throw new InvalidDataException($"{path}: unsupported storage view {view}"),
            };
        });

        if (Unpickler.Load(bytes, ref pos) is not List<object?> keys)
            throw new InvalidDataException($"{path}: missing storage key list");
        var storages = new Dictionary<string, ArraySegment<byte>>();
        foreach (var k in keys)
        {
            if (k is not string key || !roots.TryGetValue(key, out var root))
                throw new InvalidDataException($"{path}: storage key {k} is not used by the checkpoint");
            if (pos > bytes.Length - 8 || BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(pos)) != root.Numel)
                throw new InvalidDataException($"{path}: storage {key} has the wrong size");
            pos += 8;
            long n = root.Numel * root.Type.ElementSize;
            if (n > bytes.Length - pos)
                throw new InvalidDataException($"{path}: storage {key} runs past the end of the file");
            storages[key] = new ArraySegment<byte>(bytes, pos, (int)n);
            pos += (int)n;
        }
        return (obj, storages);
    }

    /// <summary>
    /// The zip format (torch.save's default): <c>&lt;archive&gt;/data.pkl</c> holds the checkpoint, each
    /// storage is the entry <c>&lt;archive&gt;/data/&lt;key&gt;</c>, and <c>&lt;archive&gt;/byteorder</c> (if present) its byte order.
    /// </summary>
    private static (object?, Dictionary<string, ArraySegment<byte>>) ReadZip(string path)
    {
        using var zip = ZipFile.OpenRead(path);
        var pkl = zip.Entries.FirstOrDefault(e => e.FullName.EndsWith("/data.pkl"))
            ?? throw new InvalidDataException($"{path}: zip archive has no data.pkl");
        var prefix = pkl.FullName[..^"data.pkl".Length];
        if (zip.GetEntry(prefix + "byteorder") is { } order && Encoding.ASCII.GetString(ReadEntry(order)) != "little")
            throw new NotSupportedException($"{path}: big-endian checkpoints are not supported");

        // Persistent id: ("storage", storage_type, key, location, numel).
        var roots = new Dictionary<string, TorchStorage>();
        var obj = Unpickler.Load(ReadEntry(pkl), pid =>
            pid is PyTuple { Items: ["storage", TorchStorageType type, string key, string, long numel] }
                ? Root(roots, new TorchStorage(type, key, numel, 0), path)
                : throw new InvalidDataException($"{path}: unsupported persistent id {pid}"));

        var storages = new Dictionary<string, ArraySegment<byte>>();
        foreach (var (key, root) in roots)
        {
            var entry = zip.GetEntry(prefix + "data/" + key) ?? throw new InvalidDataException($"{path}: storage {key} is missing");
            var data = ReadEntry(entry);
            if (data.Length != root.Numel * root.Type.ElementSize)
                throw new InvalidDataException($"{path}: storage {key} has the wrong size");
            storages[key] = data;
        }
        return (obj, storages);
    }

    private static TorchStorage Root(Dictionary<string, TorchStorage> roots, TorchStorage storage, string path)
    {
        if (storage.Numel < 0)
            throw new InvalidDataException($"{path}: storage {storage.Key} has a negative size");
        if (!roots.TryAdd(storage.Key, storage) && roots[storage.Key] != storage)
            throw new InvalidDataException($"{path}: storage {storage.Key} is used with different types or sizes");
        return storage;
    }

    private static byte[] ReadEntry(ZipArchiveEntry entry)
    {
        using var s = entry.Open();
        var data = new byte[entry.Length];
        s.ReadExactly(data);
        return data;
    }

    /// <summary>Copies every tensor, in walk order, into one contiguous buffer (np.ascontiguousarray).</summary>
    private static SafeTensorFile Materialize(List<(string Key, TorchTensor Tensor)> tensors, Dictionary<string, ArraySegment<byte>> storages)
    {
        var infos = new Dictionary<string, TensorInfo>();
        long total = 0;
        foreach (var (key, t) in tensors)
        {
            long length = checked(Numel(t.Shape) * t.Storage.Type.ElementSize);
            infos[key] = new TensorInfo(t.Storage.Type.Dtype, t.Shape, total, length);
            total += length;
        }
        if (total > Array.MaxLength)
            throw new NotSupportedException($"Checkpoint tensors total {total} bytes, more than one array can hold");

        var data = new byte[total];
        foreach (var (key, t) in tensors)
            CopyContiguous(t, storages[t.Storage.Key], data.AsSpan((int)infos[key].Offset, (int)infos[key].Length), key);
        return new SafeTensorFile(data, infos);
    }

    private static void CopyContiguous(TorchTensor t, ReadOnlySpan<byte> root, Span<byte> dst, string key)
    {
        long numel = Numel(t.Shape);
        if (numel == 0)
            return;
        int size = t.Storage.Type.ElementSize;
        long first = t.Storage.Offset + t.Offset, last = first;
        bool contiguous = true;
        long expected = 1;
        for (int d = t.Shape.Length - 1; d >= 0; d--)
        {
            if (t.Shape[d] < 0 || t.Stride[d] < 0)
                throw new InvalidDataException($"Tensor {key} has a negative size or stride");
            last += (t.Shape[d] - 1) * t.Stride[d];
            contiguous &= t.Shape[d] == 1 || t.Stride[d] == expected;
            expected *= t.Shape[d];
        }
        if (t.Offset < 0 || last >= t.Storage.Offset + t.Storage.Numel)
            throw new InvalidDataException($"Tensor {key} reaches outside its storage");

        if (contiguous)
        {
            root.Slice((int)(first * size), dst.Length).CopyTo(dst);
            return;
        }
        // Strided view: walk the indices in row-major order like an odometer.
        var index = new long[t.Shape.Length];
        for (long n = 0; n < numel; n++)
        {
            long src = first;
            for (int d = 0; d < index.Length; d++)
                src += index[d] * t.Stride[d];
            root.Slice((int)(src * size), size).CopyTo(dst.Slice((int)(n * size), size));
            for (int d = index.Length - 1; d >= 0 && ++index[d] == t.Shape[d]; d--)
                index[d] = 0;
        }
    }

    private static long Numel(long[] shape) => shape.Aggregate(1L, (a, b) => checked(a * b));

    /// <summary>The converter's Splitter: writes the JSON mirror and names each tensor by its path.</summary>
    internal sealed class Splitter(Utf8JsonWriter writer)
    {
        public Utf8JsonWriter Writer { get; } = writer;
        public List<(string Key, TorchTensor Tensor)> Tensors { get; } = [];
        private readonly HashSet<string> _keys = [];
        private readonly List<string> _path = [];

        public void Walk(object? obj)
        {
            var w = Writer;
            switch (obj)
            {
                case TorchTensor t:
                    var key = NewKey();
                    if (t.Shape.Length == 0) // np.ascontiguousarray makes 0-d arrays 1-d
                        t = t with { Shape = [1], Stride = [1] };
                    Tensors.Add((key, t));
                    w.WriteStartObject();
                    w.WriteString("$tensor", key);
                    w.WriteString("dtype", t.Storage.Type.Dtype);
                    w.WriteStartArray("shape");
                    foreach (var d in t.Shape)
                        w.WriteNumberValue(d);
                    w.WriteEndArray();
                    w.WriteEndObject();
                    break;
                case null: w.WriteNullValue(); break;
                case bool b: w.WriteBooleanValue(b); break;
                case string s: w.WriteStringValue(s); break;
                case long l: w.WriteNumberValue(l); break;
                case BigInteger n: w.WriteRawValue(n.ToString(CultureInfo.InvariantCulture)); break;
                case double d when double.IsFinite(d): w.WriteRawValue(PythonRepr(d)); break;
                case double d: Tagged("$float", () => w.WriteStringValue(double.IsNaN(d) ? "nan" : d > 0 ? "inf" : "-inf")); break;
                case PyDict dict when dict.Keys.All(k => k is string):
                    w.WriteStartObject();
                    foreach (var (k, v) in dict)
                    {
                        w.WritePropertyName((string)k);
                        Child((string)k, v);
                    }
                    w.WriteEndObject();
                    break;
                case PyDict dict:
                    Tagged("$dict", () =>
                    {
                        w.WriteStartArray();
                        int i = 0;
                        foreach (var (k, v) in dict)
                        {
                            w.WriteStartArray();
                            Child($"k{i++}", k);
                            Child(PythonStr(k), v);
                            w.WriteEndArray();
                        }
                        w.WriteEndArray();
                    });
                    break;
                case List<object?> list: Items(list); break;
                case PyTuple tuple: Tagged("$tuple", () => Items(tuple.Items)); break;
                default:
                    throw new NotSupportedException($"Cannot convert {obj.GetType().Name} at '{string.Join(".", _path)}'");
            }
        }

        private void Items(IReadOnlyList<object?> items)
        {
            Writer.WriteStartArray();
            for (int i = 0; i < items.Count; i++)
                Child(i.ToString(CultureInfo.InvariantCulture), items[i]);
            Writer.WriteEndArray();
        }

        private void Tagged(string tag, Action writeValue)
        {
            Writer.WriteStartObject();
            Writer.WritePropertyName(tag);
            writeValue();
            Writer.WriteEndObject();
        }

        private void Child(string name, object? value)
        {
            _path.Add(name);
            Walk(value);
            _path.RemoveAt(_path.Count - 1);
        }

        private string NewKey()
        {
            var baseKey = _path.Count > 0 ? string.Join(".", _path) : "root";
            var key = baseKey;
            for (int n = 2; !_keys.Add(key); n++)
                key = $"{baseKey}#{n}";
            return key;
        }

        private static string PythonStr(object key) => key switch
        {
            string s => s,
            long or BigInteger => Convert.ToString(key, CultureInfo.InvariantCulture)!,
            bool b => b ? "True" : "False",
            double d => PythonRepr(d),
            _ => throw new NotSupportedException($"Cannot name a dict key of type {key.GetType().Name}"),
        };
    }

    /// <summary>
    /// Formats a finite double like Python's <c>repr</c> (which json.dump uses): shortest round-trip digits,
    /// scientific notation when the decimal exponent is below -4 or at least 16, else ".0" on whole numbers.
    /// </summary>
    internal static string PythonRepr(double value)
    {
        if (value == 0)
            return double.IsNegative(value) ? "-0.0" : "0.0";
        // .NET's "R" gives the same shortest digits, e.g. "1.5E-05" or "123.25"; take them apart.
        var r = Math.Abs(value).ToString("R", CultureInfo.InvariantCulture);
        int e = r.IndexOf('E');
        int exponent = e < 0 ? 0 : int.Parse(r[(e + 1)..], CultureInfo.InvariantCulture);
        var mantissa = e < 0 ? r : r[..e];
        int dot = mantissa.IndexOf('.');
        var digits = mantissa.Replace(".", "");
        int point = (dot < 0 ? mantissa.Length : dot) + exponent; // value = 0.<digits> * 10^point
        int lead = digits.Length - digits.TrimStart('0').Length;
        digits = digits.Trim('0');
        point -= lead;

        string text;
        if (point - 1 < -4 || point - 1 >= 16)
        {
            text = digits[..1] + (digits.Length > 1 ? "." + digits[1..] : "");
            int exp = point - 1;
            text += (exp < 0 ? "e-" : "e+") + Math.Abs(exp).ToString("00", CultureInfo.InvariantCulture);
        }
        else if (point <= 0)
            text = "0." + new string('0', -point) + digits;
        else if (point >= digits.Length)
            text = digits + new string('0', point - digits.Length) + ".0";
        else
            text = digits[..point] + "." + digits[point..];
        return value < 0 ? "-" + text : text;
    }
}
