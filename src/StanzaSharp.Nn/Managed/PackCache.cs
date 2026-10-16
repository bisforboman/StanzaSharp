using System.IO.MemoryMappedFiles;
using Microsoft.Win32.SafeHandles;
using System.Runtime.InteropServices;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// Round 34 prototype (not for merge): a model's packed weights cached on disk. A model's load runs on one thread and
/// packs its matrices in a fixed order, so the cache is the list of arrays it produced, in that order. On a hit each
/// <see cref="PackedMatrix"/> (and each <see cref="Unit"/>) takes the next array instead of computing it, and the
/// checkpoint tensors that only fed them (dead after the load, at least 256 KB) are not read: their arrays are left
/// uninitialized, which the bit checks would catch if anything read them.
/// </summary>
/// <remarks>
/// <c>STANZASHARP_PACK_CACHE=DIR</c> turns it on; <c>STANZASHARP_PACK_CACHE_MODE=map</c> maps the file and points the
/// packed matrices into it instead of reading them into arrays.
/// </remarks>
internal sealed unsafe class PackCache
{
    private const int Version = 1, Align = 64;

    [ThreadStatic]
    private static PackCache? _current;

    private static readonly List<object> Mapped = []; // the views stay alive for the process (prototype)

    private readonly string _path;
    private readonly long _sourceSize, _sourceTime;
    private readonly bool _hit;
    private int _depth, _next;

    // Miss: what the load produced.
    private readonly List<(float[] Data, bool Packed)> _entries = [];
    private readonly List<(string Key, long Bytes, WeakReference Array)> _reads = [];

    // Hit: the file's entries.
    private readonly List<(long Offset, int Length, bool Dummy)> _index = [];
    private readonly HashSet<string> _skip = [];
    private SafeFileHandle? _file;
    private byte* _base;

    private PackCache(string path, long size, long time)
    {
        (_path, _sourceSize, _sourceTime) = (path, size, time);
        _hit = File.Exists(path) && Open();
    }

    /// <summary>The cache of the model this thread is loading, outside any <see cref="Unit"/>.</summary>
    public static PackCache? Current => _current is { _depth: 0 } c ? c : null;

    public static string? Directory { get; } = Environment.GetEnvironmentVariable("STANZASHARP_PACK_CACHE");

    private static readonly bool Map = Environment.GetEnvironmentVariable("STANZASHARP_PACK_CACHE_MODE") == "map";

    public bool IsHit => _hit;

    /// <summary>Starts this thread's model load, given the model's file (.pt or .safetensors).</summary>
    public static PackCache? Begin(string name, string source)
    {
        if (Directory == null || !File.Exists(source))
            return null;
        using var _ = LoadProfile.Measure(LoadProfile.Stage.CacheRead);
        var info = new FileInfo(source);
        var path = Path.Combine(Directory, name.Replace('/', '_') + $".{Gemm.Path}.pack");
        var cache = new PackCache(path, info.Length, info.LastWriteTimeUtc.Ticks);
        _current = cache;
        if (cache._hit)
            TensorHooks.Skip = cache._skip.Contains;
        else
            TensorHooks.Read = (key, array) => cache._reads.Add((key, (long)Buffer.ByteLength(array), new WeakReference(array)));
        return cache;
    }

    /// <summary>Ends the load (<paramref name="model"/> alive): on a miss, writes the file.</summary>
    public void End(object model)
    {
        _current = null;
        TensorHooks.Skip = null;
        TensorHooks.Read = null;
        if (_hit)
        {
            if (_next != _index.Count)
                throw new InvalidDataException($"{_path}: {_index.Count} entries, {_next} used");
            if (!Map)
                _file?.Dispose();
            return;
        }

        using var _ = LoadProfile.Measure(LoadProfile.Stage.CacheWrite);
        GC.Collect();
        GC.KeepAlive(model);
        var skip = _reads.Where(r => r.Bytes >= 256 * 1024 && !r.Array.IsAlive).Select(r => r.Key).Distinct().ToList();
        var dummy = _entries.Select(e => !e.Packed && !_kept.Contains(e.Data)).ToList();
        System.IO.Directory.CreateDirectory(Directory!);
        var tmp = _path + ".tmp";
        using (var stream = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
        using (var w = new BinaryWriter(stream))
        {
            w.Write("SSPC"u8);
            w.Write(Version);
            w.Write(_sourceSize);
            w.Write(_sourceTime);
            w.Write(PackedMatrix.NR);
            w.Write(_entries.Count);
            for (int i = 0; i < _entries.Count; i++)
            {
                w.Write(_entries[i].Data.Length);
                w.Write(dummy[i]);
            }

            w.Write(skip.Count);
            foreach (var key in skip)
                w.Write(key);
            for (int i = 0; i < _entries.Count; i++)
            {
                if (dummy[i])
                    continue;
                w.Flush();
                stream.Position = (stream.Position + Align - 1) / Align * Align;
                stream.Write(MemoryMarshal.AsBytes(_entries[i].Data.AsSpan()));
            }
        }

        File.Move(tmp, _path, overwrite: true);
    }

    // Unit outputs the caller keeps; the others only feed a PackedMatrix, so they are not stored.
    private readonly HashSet<float[]> _kept = new(ReferenceEqualityComparer.Instance);

    /// <summary>The next packed array: from the file on a hit.</summary>
    public bool TryHit(int length, out float[]? data, out float* mapped)
    {
        data = null;
        mapped = null;
        if (!_hit)
            return false;
        using var _ = LoadProfile.Measure(LoadProfile.Stage.CacheRead);
        var (offset, len, dummy) = _index[_next++];
        if (len != length || dummy)
            throw new InvalidDataException($"{_path}: entry {_next - 1} has {len} floats, expected {length}");
        if (Map)
            mapped = (float*)(_base + offset);
        else
            data = ReadArray(offset, len);
        return true;
    }

    public void Record(float[] data)
    {
        if (!_hit)
            _entries.Add((data, true));
    }

    /// <summary>
    /// A load-time computation over checkpoint tensors whose outputs don't depend on the kernel path (or that the file
    /// name keys by <see cref="Gemm.Path"/>): computed and stored on a miss, read on a hit.
    /// </summary>
    public static float[][] Unit(Func<float[][]> compute, params bool[] keep)
    {
        var cache = Current;
        if (cache == null)
            return compute();
        if (cache._hit)
        {
            using var _ = LoadProfile.Measure(LoadProfile.Stage.CacheRead);
            var (_, count, _) = cache._index[cache._next++];
            var result = new float[count][];
            for (int i = 0; i < count; i++)
            {
                var (offset, len, dummy) = cache._index[cache._next++];
                result[i] = dummy ? GC.AllocateUninitializedArray<float>(len, pinned: true) : cache.ReadArray(offset, len);
            }

            return result;
        }

        cache._depth++;
        float[][] outputs;
        try
        {
            outputs = compute();
        }
        finally
        {
            cache._depth--;
        }

        cache._entries.Add((new float[outputs.Length], true)); // the count, as a packed (never dummy) entry
        for (int i = 0; i < outputs.Length; i++)
        {
            cache._entries.Add((outputs[i], false));
            if (keep[i])
                cache._kept.Add(outputs[i]);
        }

        return outputs;
    }

    private float[] ReadArray(long offset, int length)
    {
        var data = GC.AllocateUninitializedArray<float>(length, pinned: true);
        if (Map)
            new ReadOnlySpan<float>(_base + offset, length).CopyTo(data);
        else
            SafeTensorFile.ReadExactly(_file!, MemoryMarshal.AsBytes(data.AsSpan()), offset);
        return data;
    }

    private bool Open()
    {
        using var stream = new FileStream(_path, FileMode.Open, FileAccess.Read, FileShare.Read, 4096);
        using var r = new BinaryReader(stream);
        if (!r.ReadBytes(4).AsSpan().SequenceEqual("SSPC"u8) || r.ReadInt32() != Version || r.ReadInt64() != _sourceSize
            || r.ReadInt64() != _sourceTime || r.ReadInt32() != PackedMatrix.NR)
            return false;
        int count = r.ReadInt32();
        var entries = new List<(int Length, bool Dummy)>();
        for (int i = 0; i < count; i++)
            entries.Add((r.ReadInt32(), r.ReadBoolean()));
        int skip = r.ReadInt32();
        for (int i = 0; i < skip; i++)
            _skip.Add(r.ReadString());
        long position = stream.Position;
        foreach (var (length, dummy) in entries)
        {
            if (!dummy)
                position = (position + Align - 1) / Align * Align;
            _index.Add((position, length, dummy));
            if (!dummy)
                position += 4L * length;
        }

        r.Close();
        if (Map)
        {
            var mmf = MemoryMappedFile.CreateFromFile(_path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* p = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            _base = p;
            lock (Mapped)
                Mapped.AddRange([mmf, view]);
        }
        else
        {
            _file = File.OpenHandle(_path);
        }

        return true;
    }
}
