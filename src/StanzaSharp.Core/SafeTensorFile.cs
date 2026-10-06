using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace StanzaSharp;

public sealed record TensorInfo(string Dtype, long[] Shape, long Offset, long Length);

/// <summary>
/// Reader for the safetensors format: u64 little-endian header length, a JSON header mapping
/// names to dtype/shape/data_offsets (relative to the data section), then raw little-endian data.
/// </summary>
public sealed class SafeTensorFile
{
    private readonly byte[] _bytes;
    private readonly int _dataStart;

    public IReadOnlyDictionary<string, TensorInfo> Tensors { get; }
    public IReadOnlyDictionary<string, string> Metadata { get; }

    private SafeTensorFile(byte[] bytes)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("safetensors data is little-endian");
        if (bytes.Length < 8)
            throw new FormatException("File too short for a safetensors header");

        ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(bytes);
        if (headerLength > (ulong)(bytes.Length - 8))
            throw new FormatException($"Header length {headerLength} exceeds file size");
        _bytes = bytes;
        _dataStart = 8 + (int)headerLength;

        var tensors = new Dictionary<string, TensorInfo>();
        var metadata = new Dictionary<string, string>();
        using var header = JsonDocument.Parse(bytes.AsMemory(8, (int)headerLength));
        foreach (var prop in header.RootElement.EnumerateObject())
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
            if (begin < 0 || end < begin || _dataStart + end > bytes.Length)
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

    public static SafeTensorFile Load(string path) => new(File.ReadAllBytes(path));

    public TensorInfo this[string key] =>
        Tensors.TryGetValue(key, out var info) ? info : throw new KeyNotFoundException($"No tensor '{key}'");

    /// <summary>Copies a tensor's data out as a flat array. <typeparamref name="T"/> must match its dtype.</summary>
    public T[] Read<T>(string key) where T : unmanaged
    {
        var info = this[key];
        var expected = DtypeOf<T>();
        if (info.Dtype != expected)
            throw new InvalidOperationException($"Tensor '{key}' is {info.Dtype}, not {expected}");
        return MemoryMarshal.Cast<byte, T>(RawBytes(key)).ToArray();
    }

    internal ReadOnlySpan<byte> RawBytes(string key)
    {
        var info = this[key];
        return _bytes.AsSpan(_dataStart + (int)info.Offset, (int)info.Length);
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
