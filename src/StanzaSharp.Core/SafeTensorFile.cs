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
    private readonly byte[]? _bytes; // tensor data already in memory (.pt checkpoints), else read from _path
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
            if (begin < 0 || end < begin || dataStart + end > fileLength)
                throw new FormatException($"Tensor '{prop.Name}' has data_offsets outside the file");
            var shape = v.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray();
            tensors[prop.Name] = new TensorInfo(v.GetProperty("dtype").GetString()!, shape, begin, end - begin);
        }
        Tensors = tensors;
        Metadata = metadata;
    }

    /// <summary>Wraps tensor data already in memory; <paramref name="tensors"/> offsets index into <paramref name="data"/>.</summary>
    internal SafeTensorFile(byte[] data, Dictionary<string, TensorInfo> tensors)
    {
        _bytes = data;
        _dataStart = 0;
        Tensors = tensors;
        Metadata = new Dictionary<string, string>();
    }

    public static SafeTensorFile Load(string path)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("safetensors data is little-endian");
        using var file = File.OpenHandle(path);
        long fileLength = RandomAccess.GetLength(file);
        if (fileLength < 8)
            throw new FormatException("File too short for a safetensors header");
        Span<byte> prefix = stackalloc byte[8];
        ReadExactly(file, prefix, 0);
        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (headerLength > (ulong)(fileLength - 8) || headerLength > int.MaxValue)
            throw new FormatException($"Header length {headerLength} exceeds file size");
        var header = new byte[headerLength];
        ReadExactly(file, header, 8);
        return new SafeTensorFile(path, header, 8 + (long)headerLength, fileLength);
    }

    public TensorInfo this[string key] =>
        Tensors.TryGetValue(key, out var info) ? info : throw new KeyNotFoundException($"No tensor '{key}'");

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
        if (_bytes != null)
        {
            _bytes.AsSpan((int)(_dataStart + info.Offset), (int)info.Length).CopyTo(destination);
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

    private static void ReadExactly(Microsoft.Win32.SafeHandles.SafeFileHandle file, Span<byte> buffer, long offset)
    {
        while (buffer.Length > 0)
        {
            int n = RandomAccess.Read(file, buffer, offset);
            if (n == 0)
                throw new EndOfStreamException("safetensors file ends inside a tensor");
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
