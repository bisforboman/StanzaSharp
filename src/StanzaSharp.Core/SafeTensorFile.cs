using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace StanzaSharp;

internal sealed record TensorInfo(string Dtype, long[] Shape, long Offset, long Length);

/// <summary>
/// Reader for the safetensors format: u64 little-endian header length, a JSON header mapping
/// names to dtype/shape/data_offsets (relative to the data section), then raw little-endian data.
/// </summary>
/// <remarks>
/// Only the header is read up front; each tensor is read from the file when asked for, straight into
/// its destination, so loading never holds the whole file in memory.
/// </remarks>
internal sealed class SafeTensorFile
{
    private readonly Action<string, Span<byte>>? _read; // reads a tensor's bytes (.pt checkpoints), else they come from _path
    private readonly string? _path;
    private readonly long _dataStart;

    public IReadOnlyDictionary<string, TensorInfo> Tensors { get; }
    public IReadOnlyDictionary<string, string> Metadata { get; }

    private SafeTensorFile(string path, ReadOnlySpan<byte> header, long dataStart, long fileLength)
    {
        _path = path;
        _dataStart = dataStart;

        var tensors = new Dictionary<string, TensorInfo>();
        var metadata = new Dictionary<string, string>();
        try
        {
            var reader = new Utf8JsonReader(header);
            using var doc = JsonDocument.ParseValue(ref reader);
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Name == "__metadata__")
                {
                    foreach (var m in prop.Value.EnumerateObject())
                        metadata[m.Name] = m.Value.GetString() ?? "";
                    continue;
                }
                var v = prop.Value;
                var offsets = v.GetProperty("data_offsets");
                long begin = offsets[0].GetInt64(), end = offsets[1].GetInt64();
                var dtype = v.GetProperty("dtype").GetString()!;
                var shape = v.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray();
                if (!ElementSizes.TryGetValue(dtype, out int size))
                    throw Invalid($"Tensor '{prop.Name}' has an unknown dtype {dtype}");
                if (shape.Any(d => d < 0))
                    throw Invalid($"Tensor '{prop.Name}' has a negative size");
                if (begin < 0 || end < begin || end > fileLength - dataStart)
                    throw Invalid($"Tensor '{prop.Name}' has data_offsets outside the file");
                // numel * size == end - begin, without overflowing: the product is compared one factor at a time.
                long bytes = shape.Contains(0) ? 0 : shape.Aggregate((long)size, (a, d) => a > (end - begin) / d ? long.MaxValue : a * d);
                if (bytes != end - begin)
                    throw Invalid($"Tensor '{prop.Name}' has {end - begin} bytes, not what its dtype and shape need");
                if (!tensors.TryAdd(prop.Name, new TensorInfo(dtype, shape, begin, end - begin)))
                    throw Invalid($"Tensor '{prop.Name}' is listed twice");
            }
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or KeyNotFoundException or FormatException or IndexOutOfRangeException)
        {
            throw Invalid($"Malformed header: {e.Message}", e);
        }
        // Like the safetensors library: the tensors tile the data section, with no gaps, overlaps or trailing bytes.
        long next = 0;
        foreach (var (key, info) in tensors.OrderBy(t => t.Value.Offset).ThenBy(t => t.Value.Length))
        {
            if (info.Offset != next)
                throw Invalid($"Tensor '{key}' starts at {info.Offset}, not {next}: tensors overlap or leave a gap");
            next += info.Length;
        }
        if (next != fileLength - dataStart)
            throw Invalid($"The data section has {fileLength - dataStart} bytes, the tensors {next}");
        Tensors = tensors;
        Metadata = metadata;

        InvalidDataException Invalid(string message, Exception? inner = null) => new($"{path}: {message}", inner);
    }

    private static readonly Dictionary<string, int> ElementSizes = new()
    {
        ["F64"] = 8, ["F32"] = 4, ["F16"] = 2, ["BF16"] = 2, ["I64"] = 8, ["I32"] = 4, ["I16"] = 2, ["I8"] = 1,
        ["U64"] = 8, ["U32"] = 4, ["U16"] = 2, ["U8"] = 1, ["BOOL"] = 1,
    };

    /// <summary>Tensors whose bytes <paramref name="read"/> copies into a destination of their size, given their key.</summary>
    internal SafeTensorFile(Dictionary<string, TensorInfo> tensors, Action<string, Span<byte>> read)
    {
        _read = read;
        _dataStart = 0;
        Tensors = tensors;
        Metadata = new Dictionary<string, string>();
    }

    private const int MaxHeader = 100_000_000; // the safetensors library's limit

    public static SafeTensorFile Load(string path)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("safetensors data is little-endian");
        using var file = File.OpenHandle(path);
        long fileLength = RandomAccess.GetLength(file);
        if (fileLength < 8)
            throw new InvalidDataException($"{path}: file too short for a safetensors header");
        Span<byte> prefix = stackalloc byte[8];
        ReadExactly(file, prefix, 0);
        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (headerLength > (ulong)(fileLength - 8) || headerLength > MaxHeader)
            throw new InvalidDataException($"{path}: header length {headerLength} exceeds the file or {MaxHeader:N0} bytes");
        var header = new byte[headerLength];
        ReadExactly(file, header, 8);
        return new SafeTensorFile(path, header, 8 + (long)headerLength, fileLength);
    }

    public TensorInfo this[string key] =>
        Tensors.TryGetValue(key, out var info) ? info : throw new InvalidDataException($"No tensor '{key}' in the checkpoint");

    /// <summary>Copies a tensor's data out as a flat array. <typeparamref name="T"/> must match its dtype.</summary>
    public T[] Read<T>(string key) where T : unmanaged
    {
        var info = this[key];
        var expected = DtypeOf<T>();
        if (info.Dtype != expected)
            throw new InvalidOperationException($"Tensor '{key}' is {info.Dtype}, not {expected}");
        var result = new T[info.Length / System.Runtime.CompilerServices.Unsafe.SizeOf<T>()];
        ReadInto(key, MemoryMarshal.AsBytes(result.AsSpan()));
        return result;
    }

    /// <summary>Copies a tensor's raw bytes into <paramref name="destination"/>, which must be exactly its size.</summary>
    public void ReadInto(string key, Span<byte> destination)
    {
        var info = this[key];
        if (destination.Length != info.Length)
            throw new ArgumentException($"Tensor '{key}' has {info.Length} bytes, the destination {destination.Length}");
        if (_read != null)
        {
            _read(key, destination);
            return;
        }
        using var file = File.OpenHandle(_path!);
        ReadExactly(file, destination, _dataStart + info.Offset);
    }

    internal byte[] RawBytes(string key)
    {
        var bytes = new byte[this[key].Length];
        ReadInto(key, bytes);
        return bytes;
    }

    internal static void ReadExactly(Microsoft.Win32.SafeHandles.SafeFileHandle file, Span<byte> buffer, long offset)
    {
        while (buffer.Length > 0)
        {
            int n = RandomAccess.Read(file, buffer, offset);
            if (n == 0)
                throw new InvalidDataException("The file ends inside a tensor");
            buffer = buffer[n..];
            offset += n;
        }
    }

    private static string DtypeOf<T>() => typeof(T) switch
    {
        var t when t == typeof(float) => "F32",
        var t when t == typeof(double) => "F64",
        var t when t == typeof(Half) => "F16",
        var t when t == typeof(long) => "I64",
        var t when t == typeof(int) => "I32",
        var t when t == typeof(short) => "I16",
        var t when t == typeof(sbyte) => "I8",
        var t when t == typeof(byte) => "U8",
        var t when t == typeof(bool) => "BOOL",
        var t => throw new NotSupportedException($"No safetensors dtype for {t}"),
    };
}
