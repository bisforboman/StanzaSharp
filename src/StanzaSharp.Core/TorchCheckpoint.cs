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

    /// <summary>Reads <c>destination.Length</c> bytes of the root storage <paramref name="key"/>, from <paramref name="offset"/> bytes into it.</summary>
    private delegate void StorageReader(string key, long offset, Span<byte> destination);

    public static (JsonNode Root, SafeTensorFile Tensors) Load(string path)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("Tensor data is read as little-endian");
        long fileLength = new FileInfo(path).Length;
        var (obj, stored, read) = IsZip(path) ? ReadZip(path, fileLength) : ReadLegacy(path);
        if (obj is PyDict top)
            foreach (var key in Skip)
                top.Remove(key);

        var buffer = new ArrayBufferWriter<byte>();
        // The memo lets a pickle reference one object many times, and the JSON copies it each time: cap the JSON
        // at 16 times the file (real checkpoints stay far below; tensors are mostly data).
        var splitter = new Splitter(new Utf8JsonWriter(buffer), maxBytes: 16 * fileLength + (1 << 20), path);
        splitter.Walk(obj);
        splitter.Writer.Flush();
        var root = JsonNode.Parse(buffer.WrittenSpan) ?? throw new InvalidDataException($"{path}: checkpoint is None");
        return (root, Index(splitter.Tensors, stored, read));
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
    /// keys an int64 element count followed by the storage's raw bytes. Only the storages' positions are
    /// noted here; their bytes are read from the file when a tensor is.
    /// </summary>
    private static (object?, ICollection<string>, StorageReader) ReadLegacy(string path)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16);
        if (Unpickler.Load(file) is not BigInteger magic || magic != LegacyMagic)
            throw new InvalidDataException($"{path}: not a PyTorch checkpoint (bad magic number)");
        if (Unpickler.Load(file) is not LegacyProtocol)
            throw new InvalidDataException($"{path}: unsupported legacy protocol version");
        if (Unpickler.Load(file) is not PyDict sysInfo || !sysInfo.TryGetValue("little_endian", out var le) || le is not true)
            throw new NotSupportedException($"{path}: big-endian checkpoints are not supported");

        var roots = new Dictionary<string, TorchStorage>();
        // Persistent id: ("storage", storage_type, root_key, location, root_numel, view_metadata),
        // where view_metadata is None or (view_key, offset, numel) for a slice of the root storage.
        var obj = Unpickler.Load(file, pid =>
        {
            if (pid is not PyTuple { Items: ["storage", TorchStorageType type, string key, string, long numel, var view] })
                throw new InvalidDataException($"{path}: unsupported persistent id {pid}");
            var root = Root(roots, new TorchStorage(type, key, numel, 0), path);
            return view switch
            {
                null => root,
                PyTuple { Items: [string, long offset, long size] } when offset >= 0 && size >= 0 && size <= numel - offset =>
                    root with { Numel = size, Offset = offset },
                _ => throw new InvalidDataException($"{path}: unsupported storage view {view}"),
            };
        });

        if (Unpickler.Load(file) is not List<object?> keys)
            throw new InvalidDataException($"{path}: missing storage key list");
        var starts = new Dictionary<string, long>();
        Span<byte> count = stackalloc byte[8];
        foreach (var k in keys)
        {
            if (k is not string key || !roots.TryGetValue(key, out var root))
                throw new InvalidDataException($"{path}: storage key {k} is not used by the checkpoint");
            if (file.ReadAtLeast(count, 8, throwOnEndOfStream: false) != 8 || BinaryPrimitives.ReadInt64LittleEndian(count) != root.Numel)
                throw new InvalidDataException($"{path}: storage {key} has the wrong size");
            if (root.Numel > (file.Length - file.Position) / root.Type.ElementSize) // not numel * size, which can overflow
                throw new InvalidDataException($"{path}: storage {key} runs past the end of the file");
            long n = root.Numel * root.Type.ElementSize;
            if (!starts.TryAdd(key, file.Position))
                throw new InvalidDataException($"{path}: storage {key} is listed twice");
            file.Seek(n, SeekOrigin.Current);
        }
        return (obj, starts.Keys, (key, offset, destination) =>
        {
            using var handle = File.OpenHandle(path);
            SafeTensorFile.ReadExactly(handle, destination, starts[key] + offset);
        });
    }

    /// <summary>
    /// The zip format (torch.save's default): <c>&lt;archive&gt;/data.pkl</c> holds the checkpoint, each
    /// storage is the entry <c>&lt;archive&gt;/data/&lt;key&gt;</c>, and <c>&lt;archive&gt;/byteorder</c> (if present) its byte order.
    /// The storages are read from their entries when a tensor is.
    /// </summary>
    private static (object?, ICollection<string>, StorageReader) ReadZip(string path, long fileLength)
    {
        string prefix;
        var roots = new Dictionary<string, TorchStorage>();
        object? obj;
        using (var zip = ZipFile.OpenRead(path))
        {
            var pkl = zip.Entries.FirstOrDefault(e => e.FullName.EndsWith("/data.pkl"))
                ?? throw new InvalidDataException($"{path}: zip archive has no data.pkl");
            prefix = pkl.FullName[..^"data.pkl".Length];
            if (zip.GetEntry(prefix + "byteorder") is { } order && Encoding.ASCII.GetString(ReadEntry(order, fileLength, path)) != "little")
                throw new NotSupportedException($"{path}: big-endian checkpoints are not supported");

            // Persistent id: ("storage", storage_type, key, location, numel).
            obj = Unpickler.Load(ReadEntry(pkl, fileLength, path), pid =>
                pid is PyTuple { Items: ["storage", TorchStorageType type, string key, string, long numel] }
                    ? Root(roots, new TorchStorage(type, key, numel, 0), path)
                    : throw new InvalidDataException($"{path}: unsupported persistent id {pid}"));

            foreach (var (key, root) in roots)
            {
                var entry = zip.GetEntry(prefix + "data/" + key) ?? throw new InvalidDataException($"{path}: storage {key} is missing");
                Stored(entry, fileLength, path);
                if (root.Numel > entry.Length / root.Type.ElementSize || entry.Length != root.Numel * root.Type.ElementSize)
                    throw new InvalidDataException($"{path}: storage {key} has the wrong size");
            }
        }
        return (obj, roots.Keys, (key, offset, destination) =>
        {
            using var zip = ZipFile.OpenRead(path);
            using var data = zip.GetEntry(prefix + "data/" + key)!.Open();
            if (data.CanSeek)
                data.Seek(offset, SeekOrigin.Begin);
            else
            {
                var skip = new byte[Math.Min(offset, 1 << 16)];
                for (int n; offset > 0; offset -= n)
                    if ((n = data.Read(skip, 0, (int)Math.Min(offset, skip.Length))) == 0)
                        throw new InvalidDataException($"{path}: storage {key} ends early");
            }
            if (data.ReadAtLeast(destination, destination.Length, throwOnEndOfStream: false) != destination.Length)
                throw new InvalidDataException($"{path}: storage {key} ends early");
        });
    }

    /// <summary>
    /// torch.save stores entries uncompressed. Requiring that bounds each entry by the file's size, so a header
    /// claiming gigabytes (or a zip bomb) is rejected before anything is allocated.
    /// </summary>
    private static void Stored(ZipArchiveEntry entry, long fileLength, string path)
    {
        if (entry.CompressedLength != entry.Length || entry.Length > fileLength)
            throw new InvalidDataException($"{path}: zip entry {entry.FullName} is compressed or larger than the file");
    }

    private static TorchStorage Root(Dictionary<string, TorchStorage> roots, TorchStorage storage, string path)
    {
        if (storage.Numel < 0)
            throw new InvalidDataException($"{path}: storage {storage.Key} has a negative size");
        if (!roots.TryAdd(storage.Key, storage) && roots[storage.Key] != storage)
            throw new InvalidDataException($"{path}: storage {storage.Key} is used with different types or sizes");
        return storage;
    }

    private static byte[] ReadEntry(ZipArchiveEntry entry, long fileLength, string path)
    {
        Stored(entry, fileLength, path);
        using var s = entry.Open();
        var data = new byte[entry.Length];
        if (s.ReadAtLeast(data, data.Length, throwOnEndOfStream: false) != data.Length)
            throw new InvalidDataException($"{path}: zip entry {entry.FullName} ends early");
        return data;
    }

    /// <summary>
    /// Lays the tensors out, in walk order, as the converter's contiguous data (np.ascontiguousarray). Each
    /// tensor's bytes are read from its storage when asked for, so no copy of the file is held in memory.
    /// </summary>
    private static SafeTensorFile Index(List<(string Key, TorchTensor Tensor)> tensors, ICollection<string> stored, StorageReader read)
    {
        var infos = new Dictionary<string, TensorInfo>();
        var views = new Dictionary<string, TorchTensor>();
        long total = 0;
        foreach (var (key, t) in tensors)
        {
            if (!stored.Contains(t.Storage.Key))
                throw new InvalidDataException($"Tensor {key}: storage {t.Storage.Key} is missing");
            if (t.Shape.Any(d => d < 0) || t.Stride.Any(s => s < 0))
                throw new InvalidDataException($"Tensor {key} has a negative size or stride");
            // No bigger than its storage: real views never are, and this bounds a stride-0 broadcast (or a size overflow).
            long numel = t.Shape.Contains(0) ? 0 : t.Shape.Aggregate(1L, (a, d) => a > t.Storage.Numel / d ? long.MaxValue : a * d);
            if (numel > t.Storage.Numel)
                throw new InvalidDataException($"Tensor {key} is larger than its storage");
            long length = numel * t.Storage.Type.ElementSize;
            if (length > 0)
                Extent(t, key); // checks the view before anything is read
            infos[key] = new TensorInfo(t.Storage.Type.Dtype, t.Shape, total, length);
            views[key] = t;
            total += length;
        }
        return new SafeTensorFile(infos, (key, destination) => CopyContiguous(views[key], read, destination, key));
    }

    /// <summary>A view's first and last element in its root storage, and whether it is contiguous.</summary>
    private static (long First, long Last, bool Contiguous) Extent(TorchTensor t, string key)
    {
        // Every term is non-negative (checked by Index) and the storage lies inside the file, so a sum that goes past
        // the storage's end is caught before it can overflow: each step adds at most (size - 1) * stride, compared first.
        long end = t.Storage.Offset + t.Storage.Numel; // the storage view's end in its root, at most the root's numel
        if (t.Offset < 0 || t.Offset >= t.Storage.Numel)
            throw new InvalidDataException($"Tensor {key} reaches outside its storage");
        long first = t.Storage.Offset + t.Offset, last = first;
        bool contiguous = true;
        long expected = 1;
        for (int d = t.Shape.Length - 1; d >= 0; d--)
        {
            long size = t.Shape[d] - 1, stride = t.Stride[d];
            if (size > 0 && stride > (end - 1 - last) / size)
                throw new InvalidDataException($"Tensor {key} reaches outside its storage");
            last += size * stride;
            contiguous &= t.Shape[d] == 1 || stride == expected;
            expected *= t.Shape[d]; // at most the tensor's numel, checked by Index
        }
        return (first, last, contiguous);
    }

    private static void CopyContiguous(TorchTensor t, StorageReader read, Span<byte> dst, string key)
    {
        if (dst.Length == 0)
            return;
        int size = t.Storage.Type.ElementSize;
        var (first, last, contiguous) = Extent(t, key);
        if (contiguous)
        {
            read(t.Storage.Key, first * size, dst);
            return;
        }
        // Strided view: read the elements it spans, then walk its indices in row-major order like an odometer.
        var root = new byte[(last - first + 1) * size];
        read(t.Storage.Key, first * size, root);
        var index = new long[t.Shape.Length];
        long numel = dst.Length / size;
        for (long n = 0; n < numel; n++)
        {
            long src = 0;
            for (int d = 0; d < index.Length; d++)
                src += index[d] * t.Stride[d];
            root.AsSpan((int)(src * size), size).CopyTo(dst.Slice((int)(n * size), size));
            for (int d = index.Length - 1; d >= 0 && ++index[d] == t.Shape[d]; d--)
                index[d] = 0;
        }
    }

    /// <summary>
    /// The converter's Splitter: writes the JSON mirror and names each tensor by its path. It refuses JSON deeper than
    /// <see cref="MaxDepth"/> (what <see cref="JsonNode.Parse(ReadOnlySpan{byte}, JsonNodeOptions?, JsonDocumentOptions)"/>
    /// reads; also a cyclic list) or longer than <paramref name="maxBytes"/>, so walking a pickle's object graph takes
    /// bounded stack, time and memory.
    /// </summary>
    internal sealed class Splitter(Utf8JsonWriter writer, long maxBytes, string path)
    {
        private const int MaxDepth = 60; // JsonNode.Parse's default limit is 64; a tensor adds 2 levels

        public Utf8JsonWriter Writer { get; } = writer;
        public List<(string Key, TorchTensor Tensor)> Tensors { get; } = [];
        private readonly HashSet<string> _keys = [];
        private readonly List<string> _path = [];

        public void Walk(object? obj)
        {
            var w = Writer;
            if (w.CurrentDepth > MaxDepth)
                throw new InvalidDataException($"{path}: checkpoint nested more than {MaxDepth} levels deep");
            if (w.BytesCommitted + w.BytesPending > maxBytes)
                throw new InvalidDataException($"{path}: checkpoint expands to more than {maxBytes:N0} bytes of JSON");
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
                case byte[] bytes: Tagged("$bytes", () => w.WriteStringValue(Convert.ToBase64String(bytes))); break;
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
                default: // a bare storage or global: not something torch.save writes
                    throw new InvalidDataException($"{path}: cannot convert {obj.GetType().Name} at '{string.Join(".", _path)}'");
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
            _ => throw new InvalidDataException($"Cannot name a dict key of type {key.GetType().Name}"), // Unpickler.Key allows no others
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
