using System.Buffers.Binary;
using System.Diagnostics;
using System.IO.Compression;
using System.Text;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// Malformed, truncated and hostile model files must fail cleanly: <see cref="Checkpoint.Load"/> and every tensor read
/// either succeed or throw <see cref="InvalidDataException"/> (<see cref="NotSupportedException"/> for a well-formed but
/// big-endian file), within bounded time, memory and stack.
/// </summary>
/// <remarks>
/// The short fuzz runs in CI with a fixed seed. Longer runs: <c>STANZASHARP_FUZZ_ITERATIONS=N</c> (per file, default 300),
/// <c>STANZASHARP_FUZZ_SEED</c>, and <c>STANZASHARP_FUZZ_MODELS=all</c> to mutate every Stanza checkpoint rather than
/// the three small ones. Failing inputs are saved under <c>%TEMP%/stanzasharp-fuzz</c>, and with
/// <c>STANZASHARP_FUZZ_REPORT=FILE</c> every failure is appended to FILE.
/// </remarks>
[Trait("Backend", "Managed")]
public class CheckpointFuzzTests(ITestOutputHelper output)
{
    private int _runs;

    private static readonly int Iterations =
        int.TryParse(Environment.GetEnvironmentVariable("STANZASHARP_FUZZ_ITERATIONS"), out var n) ? n : 300;
    private static readonly int Seed =
        int.TryParse(Environment.GetEnvironmentVariable("STANZASHARP_FUZZ_SEED"), out var s) ? s : 20261010;

    // ---------------------------------------------------------------- harness

    /// <summary>
    /// Loads a checkpoint and reads every tensor on a thread with a small stack. Returns null when it loaded or failed
    /// cleanly, else what went wrong.
    /// </summary>
    private static string? Check(string loadPath, long fileBytes)
    {
        string? problem = null;
        var thread = new Thread(() =>
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            try
            {
                var ckpt = Checkpoint.Load(loadPath);
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                if (allocated > 32 * fileBytes + (64 << 20))
                    problem = $"Load allocated {allocated:N0} bytes for a {fileBytes:N0}-byte file";
                foreach (var (key, info) in ckpt.Tensors.Tensors)
                {
                    if (info.Length > fileBytes)
                    {
                        problem = $"tensor {key} claims {info.Length:N0} bytes, more than the file";
                        break;
                    }
                    // Callers allocate by shape and read Length bytes into it.
                    long size = info.Dtype switch { "F64" or "I64" or "U64" => 8, "F32" or "I32" or "U32" => 4, "F16" or "BF16" or "I16" or "U16" => 2, _ => 1 };
                    if (info.Shape.Any(d => d < 0) || info.Shape.Aggregate(size, (a, d) => a * d) != info.Length)
                    {
                        problem = $"tensor {key}: shape [{string.Join(", ", info.Shape)}] {info.Dtype} disagrees with its {info.Length} bytes";
                        break;
                    }
                    ckpt.Tensors.RawBytes(key);
                }
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException)
            {
                long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
                if (allocated > 32 * fileBytes + (64 << 20))
                    problem = $"allocated {allocated:N0} bytes before failing on a {fileBytes:N0}-byte file";
            }
            catch (Exception e)
            {
                problem = $"{e.GetType().Name}: {e.Message}\n{e.StackTrace}";
            }
        }, maxStackSize: 256 << 10);
        thread.Start();
        if (!thread.Join(TimeSpan.FromSeconds(30)))
            return "no result after 30 s (hang)";
        return problem;
    }

    private static readonly long[] Extremes =
        [0, 1, -1, 0x7f, 0xff, int.MaxValue, int.MinValue, uint.MaxValue, 1L << 31, 1L << 40, long.MaxValue, long.MinValue];

    /// <summary>Applies one random mutation, near the start, the end or anywhere; returns a description.</summary>
    private static string Mutate(Random rng, ref byte[] data)
    {
        int len = data.Length;
        int Pos() => len == 0 ? 0 : rng.Next(3) switch
        {
            0 => rng.Next(Math.Min(len, 4096)),
            1 => Math.Max(0, len - 1 - rng.Next(Math.Min(len, 1024))),
            _ => rng.Next(len),
        };
        int p = Pos();
        switch (len == 0 ? 2 : rng.Next(7))
        {
            case 0:
                int bits = 1 + rng.Next(8);
                for (int i = 0; i < bits; i++)
                    data[Pos()] ^= (byte)(1 << rng.Next(8));
                return $"flip {bits} bits";
            case 1:
                byte b = (byte)new[] { 0, 1, 0x7f, 0x80, 0xff }[rng.Next(5)];
                data[p] = b;
                return $"byte {p} = {b}";
            case 2:
                int keep = len == 0 ? 0 : rng.Next(len);
                data = data[..keep];
                return $"truncate to {keep}";
            case 3:
                long v = Extremes[rng.Next(Extremes.Length)];
                int width = rng.Next(2) == 0 ? 4 : 8;
                p = Math.Min(p, Math.Max(0, len - width));
                if (len < width)
                    return "too short";
                if (width == 4)
                    BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(p), (int)v);
                else
                    BinaryPrimitives.WriteInt64LittleEndian(data.AsSpan(p), v);
                return $"int{width * 8} at {p} = {v}";
            case 4:
                int cut = Math.Min(1 + rng.Next(16), len - p);
                data = [.. data.AsSpan(0, p), .. data.AsSpan(p + cut)];
                return $"delete {cut} at {p}";
            case 5:
                int from = rng.Next(len), count = Math.Min(1 + rng.Next(64), len - from);
                data = [.. data.AsSpan(0, p), .. data.AsSpan(from, count), .. data.AsSpan(p)];
                return $"insert {count} from {from} at {p}";
            default:
                // A copy of an ASCII digit run with one digit changed: JSON offsets, shapes and lengths.
                int digit = Array.FindIndex(data, p, x => x is >= (byte)'0' and <= (byte)'9');
                if (digit < 0)
                    return "no digit";
                data[digit] = (byte)('0' + rng.Next(10));
                return $"digit at {digit}";
        }
    }

    private sealed record Target(string Name, string Source, string[] Companions);

    /// <summary>
    /// Fuzzes <paramref name="source"/> (a .pt, .json or .safetensors file): writes each mutant next to its unmutated
    /// <paramref name="companions"/> in a temp directory and loads it there.
    /// </summary>
    private List<string> Fuzz(string source, int iterations, int seed, bool everyTruncation = false, params string[] companions)
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-fuzz-");
        try
        {
            var original = File.ReadAllBytes(source);
            var name = Path.GetFileName(source);
            var target = Path.Combine(dir.FullName, name);
            foreach (var c in companions)
                File.Copy(c, Path.Combine(dir.FullName, Path.GetFileName(c)));
            string loadPath = name.EndsWith(".pt") ? target : Path.Combine(dir.FullName, Path.GetFileNameWithoutExtension(name));
            long companionBytes = companions.Sum(c => new FileInfo(c).Length);

            var failures = new List<string>();
            void Run(byte[] data, string what)
            {
                File.WriteAllBytes(target, data);
                _runs++;
                if (Check(loadPath, data.Length + companionBytes) is { } problem)
                {
                    var saved = Path.Combine(Path.GetTempPath(), "stanzasharp-fuzz", $"{seed}-{failures.Count}-{name}");
                    Directory.CreateDirectory(Path.GetDirectoryName(saved)!);
                    File.WriteAllBytes(saved, data);
                    failures.Add($"{name} {what}: {problem} (saved as {saved})");
                }
            }

            Run(original, "unmutated");
            if (everyTruncation)
                // Every length within 512 bytes of either end (headers, central directory), every 8th in between.
                for (int keep = 0; keep < original.Length; keep += keep < 512 || keep >= original.Length - 512 ? 1 : 8)
                    Run(original[..keep], $"truncated to {keep}");
            var rng = new Random(seed);
            for (int i = 0; i < iterations; i++)
            {
                var data = (byte[])original.Clone();
                var what = string.Join(", ", Enumerable.Range(0, 1 + rng.Next(3)).Select(_ => Mutate(rng, ref data)));
                Run(data, $"#{i} ({what})");
            }
            return failures;
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    private void AssertNoFailures(List<string> failures, Stopwatch timer)
    {
        output.WriteLine($"{_runs} runs in {timer.Elapsed.TotalSeconds:F1} s, seed {Seed}");
        if (Environment.GetEnvironmentVariable("STANZASHARP_FUZZ_REPORT") is { Length: > 0 } report)
            File.AppendAllLines(report, failures);
        Assert.True(failures.Count == 0, $"{failures.Count} unclean failures:\n" + string.Join("\n", failures.Take(20)));
    }

    [Fact]
    public void Fuzz_PtFixtures()
    {
        var timer = Stopwatch.StartNew();
        var pt = Path.Combine(Repo.Golden, "pt");
        var failures = new List<string>();
        foreach (var name in new[] { "tiny_legacy", "tiny_zip" })
        {
            var basePath = Path.Combine(pt, name);
            failures.AddRange(Fuzz(basePath + ".pt", Iterations, Seed, everyTruncation: true));
            failures.AddRange(Fuzz(basePath + ".safetensors", Iterations, Seed, everyTruncation: true, basePath + ".json"));
            failures.AddRange(Fuzz(basePath + ".json", Iterations, Seed, everyTruncation: false, basePath + ".safetensors"));
        }
        AssertNoFailures(failures, timer);
    }

    [PtModelFact]
    public void Fuzz_StanzaCheckpoints()
    {
        var timer = Stopwatch.StartNew();
        var names = Environment.GetEnvironmentVariable("STANZASHARP_FUZZ_MODELS") == "all"
            ? ((IEnumerable<object[]>)TorchCheckpointTests.StanzaCheckpoints).Select(row => (string)row[0]).ToArray()
            : new[] { "mwt/combined", "tokenize/combined_nocharlm", "lemma/combined_nocharlm" };
        int iterations = Math.Max(1, Iterations / 10);
        var failures = new List<string>();
        foreach (var name in names)
            failures.AddRange(Fuzz(Path.Combine(Repo.StanzaModels, name + ".pt"), iterations, Seed));
        AssertNoFailures(failures, timer);
    }

    // ---------------------------------------------------------------- crafted inputs

    /// <summary>Pickle bytes: ints are single bytes, strings ASCII, byte arrays as they are.</summary>
    private static byte[] P(params object[] parts) =>
        parts.SelectMany(p => p switch
        {
            string s => Encoding.ASCII.GetBytes(s),
            int b => [(byte)b],
            byte[] bytes => bytes,
            _ => throw new ArgumentException(p.ToString()),
        }).ToArray();

    private static byte[] Str(string s) => P("X", Le32(Encoding.UTF8.GetByteCount(s)), Encoding.UTF8.GetBytes(s));
    private static byte[] Le32(int v) { var b = new byte[4]; BinaryPrimitives.WriteInt32LittleEndian(b, v); return b; }
    private static byte[] Le64(long v) { var b = new byte[8]; BinaryPrimitives.WriteInt64LittleEndian(b, v); return b; }
    private static byte[] Int(long v) => P(0x8a, 8, Le64(v)); // LONG1
    private static byte[] Tuple(params long[] items) => P("(", items.Select(Int).SelectMany(b => b).ToArray(), "t");

    /// <summary><c>_rebuild_tensor_v2</c> over a FloatStorage <paramref name="key"/> of <paramref name="numel"/> elements.</summary>
    private static byte[] Tensor(string key, long numel, long offset, long[] shape, long[] stride, byte[]? view = null) =>
        P("ctorch._utils\n_rebuild_tensor_v2\n(",
            "(", Str("storage"), "ctorch\nFloatStorage\n", Str(key), Str("cpu"), Int(numel), view ?? P("N"), "tQ",
            Int(offset), Tuple(shape), Tuple(stride), 0x89, "ccollections\nOrderedDict\n)R", "tR");

    /// <summary>A legacy-format .pt: header pickles, <paramref name="body"/> (a pickled object, without PROTO/STOP), the key list, the storages.</summary>
    private static byte[] Legacy(byte[] body, params (string Key, long Count, int Bytes)[] storages) =>
        P(0x80, 2, 0x8a, 10, new byte[] { 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 }, ".",
            0x80, 2, "M", 0xe9, 0x03, ".",
            0x80, 2, "}", Str("little_endian"), 0x88, "s.",
            0x80, 2, body, ".",
            0x80, 2, "](", storages.SelectMany(s => Str(s.Key)).ToArray(), "e.",
            storages.SelectMany(s => P(Le64(s.Count), new byte[s.Bytes])).ToArray());

    /// <summary>A legacy .pt holding <c>{"w": tensor}</c> over storage "0" of 4 floats.</summary>
    private static byte[] OneTensor(long numel, long offset, long[] shape, long[] stride, byte[]? view = null, long storedCount = 4) =>
        Legacy(P("}", Str("w"), Tensor("0", numel, offset, shape, stride, view), "s"), ("0", storedCount, 16));

    private static InvalidDataException LoadFails(byte[] file, string name = "crafted.pt")
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-crafted-");
        try
        {
            var path = Path.Combine(dir.FullName, name);
            File.WriteAllBytes(path, file);
            Assert.Null(Check(path, file.Length)); // clean, bounded, no hang
            return Assert.Throws<InvalidDataException>(() =>
            {
                var ckpt = Checkpoint.Load(path);
                foreach (var key in ckpt.Tensors.Tensors.Keys)
                    ckpt.Tensors.RawBytes(key);
            });
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void Crafted_ValidTensorLoads()
    {
        // The builders themselves are right: the same file with sane numbers loads.
        var dir = Directory.CreateTempSubdirectory("stanzasharp-crafted-");
        try
        {
            var path = Path.Combine(dir.FullName, "ok.pt");
            File.WriteAllBytes(path, OneTensor(4, 1, [3], [1]));
            var ckpt = Checkpoint.Load(path);
            Assert.Equal([3L], ckpt.Tensors["w"].Shape);
            Assert.Equal(12, ckpt.Tensors.RawBytes("w").Length);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void Crafted_StorageSizeOverflow()
    {
        // 2^62 floats are 2^64 bytes, which wrapped to 0 and passed the "fits in the file" check.
        long huge = 1L << 62;
        LoadFails(Legacy(P("}", Str("w"), Tensor("0", huge, 0, [1], [1]), "s"), ("0", huge, 0)));
    }

    [Fact]
    public void Crafted_StorageViewOverflow()
    {
        // A storage view at offset long.MaxValue: offset + size wrapped negative and passed "offset + size <= numel".
        LoadFails(OneTensor(4, 0, [1], [1], view: P("(", Str("v"), Int(long.MaxValue), Int(2), "t")));
    }

    [Theory]
    [InlineData(new long[] { -1, 0 }, new long[] { 1, 1 })]        // negative size hidden by a zero: was accepted
    [InlineData(new long[] { 2, 2 }, new long[] { long.MaxValue, 1 })] // the extent wrapped negative
    [InlineData(new long[] { 1L << 40 }, new long[] { 0 })]         // a 4 TB broadcast of one float
    [InlineData(new long[] { 1L << 32, 1L << 32 }, new long[] { 0, 0 })] // the byte length overflows
    [InlineData(new long[] { 2 }, new long[] { -1 })]
    public void Crafted_BadShapesAndStrides(long[] shape, long[] stride) =>
        LoadFails(OneTensor(4, 0, shape, stride));

    [Theory]
    [InlineData(long.MaxValue)]
    [InlineData(-1L)]
    [InlineData(4L)]
    public void Crafted_BadTensorOffset(long offset) => LoadFails(OneTensor(4, offset, [1], [1]));

    [Fact]
    public void Crafted_DuplicateStorageKey() =>
        LoadFails(Legacy(P("}", Str("w"), Tensor("0", 4, 0, [4], [1]), "s"), ("0", 4, 16), ("0", 4, 16)));

    [Fact]
    public void Crafted_DeepNesting()
    {
        // 100,000 nested lists: the unpickler is iterative, the JSON writer recursive.
        int depth = 100_000;
        LoadFails(Legacy(P(Enumerable.Repeat((byte)']', depth).ToArray(), Enumerable.Repeat((byte)'a', depth - 1).ToArray())));
        // Nested tuples (TUPLE1), which also nest the JSON twice as deep ($tuple objects).
        LoadFails(Legacy(P("N", Enumerable.Repeat((byte)0x85, depth).ToArray())));
    }

    [Fact]
    public void Crafted_CyclicList() =>
        // l = []; l.append(l)
        LoadFails(Legacy(P("]q", 0, "h", 0, "a")));

    [Fact]
    public void Crafted_MemoAmplification()
    {
        // A 64 KB string referenced 100,000 times through the memo: 6.5 GB of JSON from a 264 KB file.
        LoadFails(Legacy(P("]q", 0, Str(new string('a', 65536)), "q", 1, "a",
            Enumerable.Range(0, 100_000).SelectMany(_ => P("h", 0, "h", 1, "a")).ToArray())));
    }

    [Fact]
    public void Crafted_NestedTupleDictKey()
    {
        // A dict key nested 100,000 tuples deep, hashed recursively, used to overflow the stack.
        LoadFails(Legacy(P("}", "N", Enumerable.Repeat((byte)0x85, 100_000).ToArray(), "Ns")));
    }

    [Fact]
    public void Unpickler_MarksAreConstantTime()
    {
        // 200,000 items, then 200,000 x MARK + TUPLE: each MARK copied the whole stack (4e10 copies).
        var bytes = P(0x80, 2, "(", Enumerable.Repeat((byte)'N', 200_000).ToArray(),
            Enumerable.Range(0, 200_000).SelectMany(_ => P("(t")).ToArray(), "t.");
        var timer = Stopwatch.StartNew();
        var tuple = Assert.IsType<PyTuple>(Unpickler.Load(bytes));
        Assert.Equal(400_000, tuple.Items.Length);
        Assert.True(timer.Elapsed < TimeSpan.FromSeconds(5), $"took {timer.Elapsed}");
        // Items below a MARK cannot be popped before the MARK is closed.
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(P(0x80, 2, "N(", 0x85, "t.")));
    }

    [Theory]
    [InlineData("__builtin__\neval\n")]
    [InlineData("builtins\ngetattr\n")]
    [InlineData("torch\nload\n")]
    [InlineData("torch._utils\n_rebuild_parameter\n")]
    [InlineData("numpy.core.multiarray\n_reconstruct\n")]
    public void Unpickler_RejectsOtherGlobals(string global) =>
        Assert.Contains("is not allowed",
            Assert.Throws<InvalidDataException>(() => Unpickler.Load(P(0x80, 2, "c", global, ")R."))).Message);

    [Theory]
    [InlineData(0x8b)] // LONG4
    [InlineData(0x93)] // STACK_GLOBAL (protocol 4)
    [InlineData((int)'i')] // INST
    [InlineData((int)'o')] // OBJ
    [InlineData(0x81)] // NEWOBJ
    [InlineData((int)'P')] // PERSID (text)
    [InlineData(0x82)] // EXT1
    public void Unpickler_RejectsUnsupportedOpcodes(int op) =>
        Assert.Contains("unsupported opcode", Assert.Throws<InvalidDataException>(() => Unpickler.Load(P(0x80, 2, op, 0, 0, 0, 0, "."))).Message);

    [Fact]
    public void Unpickler_Long1InRangeIsALong()
    {
        // LONG1 values that fit were boxed as BigInteger (the ternary's type), so a storage size pickled that way
        // (any numel of 2^31 or more) did not match the persistent id pattern.
        Assert.Equal(5L, Unpickler.Load(P(0x80, 2, Int(5), ".")));
        Assert.Equal(1L << 40, Unpickler.Load(P(0x80, 2, Int(1L << 40), ".")));
    }

    [Fact]
    public void Unpickler_HugeStringLength() =>
        // BINUNICODE of 4 GB - 1 in a 10-byte pickle: rejected before any allocation.
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(P(0x80, 2, "X", 0xff, 0xff, 0xff, 0xff, "ab.")));

    [Fact]
    public void Crafted_MalformedPersistentIds()
    {
        // Too short, not a tuple, a negative size.
        LoadFails(Legacy(P("}", Str("w"), "(", Str("storage"), "ctorch\nFloatStorage\n", Str("0"), "tQs")));
        LoadFails(Legacy(P("}", Str("w"), Str("storage"), "Qs")));
        LoadFails(OneTensor(-4, 0, [1], [1]));
    }

    // ---------------------------------------------------------------- zip

    private static byte[] Zip(params (string Name, byte[] Data)[] entries)
    {
        using var ms = new MemoryStream();
        using (var zip = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
            foreach (var (name, data) in entries)
            {
                using var s = zip.CreateEntry(name, CompressionLevel.NoCompression).Open();
                s.Write(data);
            }
        return ms.ToArray();
    }

    /// <summary>Sets every size field of the zip's headers that equals <paramref name="from"/> to <paramref name="to"/>.</summary>
    private static byte[] PatchSizes(byte[] zip, uint from, uint to)
    {
        var z = (byte[])zip.Clone();
        for (int i = 0; i + 4 <= z.Length; i++)
        {
            uint sig = BinaryPrimitives.ReadUInt32LittleEndian(z.AsSpan(i));
            int[] fields = sig == 0x04034b50 ? [18, 22] : sig == 0x02014b50 ? [20, 24] : [];
            foreach (var f in fields)
                if (BinaryPrimitives.ReadUInt32LittleEndian(z.AsSpan(i + f)) == from)
                    BinaryPrimitives.WriteUInt32LittleEndian(z.AsSpan(i + f), to);
        }
        return z;
    }

    private static readonly byte[] ZipPickle = P(0x80, 2, "}", Str("w"),
        "ctorch._utils\n_rebuild_tensor_v2\n(", "(", Str("storage"), "ctorch\nFloatStorage\n", Str("0"), Str("cpu"), Int(4), "tQ",
        Int(0), Tuple(4), Tuple(1), 0x89, "ccollections\nOrderedDict\n)R", "tRs.");

    [Fact]
    public void Crafted_ZipValidLoads()
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-crafted-");
        try
        {
            var path = Path.Combine(dir.FullName, "ok.pt");
            File.WriteAllBytes(path, Zip(("archive/data.pkl", ZipPickle), ("archive/data/0", new byte[16])));
            Assert.Equal(16, Checkpoint.Load(path).Tensors.RawBytes("w").Length);
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void Crafted_ZipEntrySizes()
    {
        var zip = Zip(("archive/data.pkl", ZipPickle), ("archive/data/0", new byte[16]));
        // data.pkl claiming 2 GB: it was allocated before anything was read.
        LoadFails(PatchSizes(zip, (uint)ZipPickle.Length, int.MaxValue));
        // A storage entry claiming more or less than its numel.
        LoadFails(PatchSizes(zip, 16, 1u << 31));
        LoadFails(PatchSizes(zip, 16, 8));
        // A missing storage, and a compressed one (torch.save stores them uncompressed).
        LoadFails(Zip(("archive/data.pkl", ZipPickle)));
        using var ms = new MemoryStream();
        using (var z = new ZipArchive(ms, ZipArchiveMode.Create, leaveOpen: true))
        {
            using (var s = z.CreateEntry("archive/data.pkl", CompressionLevel.NoCompression).Open())
                s.Write(ZipPickle);
            using (var s = z.CreateEntry("archive/data/0", CompressionLevel.Optimal).Open())
                s.Write(new byte[16]);
        }
        LoadFails(ms.ToArray());
    }

    // ---------------------------------------------------------------- safetensors

    private static byte[] SafeTensors(string header, int dataBytes)
    {
        var h = Encoding.UTF8.GetBytes(header);
        return P(Le64(h.Length), h, new byte[dataBytes]);
    }

    private static InvalidDataException SafeTensorsFails(string header, int dataBytes, byte[]? file = null)
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-crafted-");
        try
        {
            var basePath = Path.Combine(dir.FullName, "m");
            File.WriteAllText(basePath + ".json", "{}");
            var bytes = file ?? SafeTensors(header, dataBytes);
            File.WriteAllBytes(basePath + ".safetensors", bytes);
            Assert.Null(Check(basePath, bytes.Length + 2));
            return Assert.Throws<InvalidDataException>(() =>
            {
                var ckpt = Checkpoint.Load(basePath);
                foreach (var key in ckpt.Tensors.Tensors.Keys)
                    ckpt.Tensors.RawBytes(key);
            });
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Theory]
    [InlineData("""{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}""", 4)] // past the end
    [InlineData("""{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}""", 12)] // trailing bytes
    [InlineData("""{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}""", 8)] // shape and size disagree
    [InlineData("""{"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}""", 8)]
    [InlineData("""{"a":{"dtype":"F32","shape":[4611686018427387904, 4],"data_offsets":[0,8]}}""", 8)] // overflow
    [InlineData("""{"a":{"dtype":"Q7","shape":[2],"data_offsets":[0,8]}}""", 8)] // unknown dtype
    [InlineData("""{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}""", 8)]
    [InlineData("""{"a":{"dtype":"F32","shape":[2],"data_offsets":[-8,0]}}""", 8)]
    [InlineData("""{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}""", 12)] // overlap
    [InlineData("""{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}""", 12)] // hole
    [InlineData("""{"a":{"dtype":"F32","shape":[2]}}""", 8)] // no offsets
    [InlineData("""{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8.5]}}""", 8)]
    [InlineData("""{"a":{"dtype":7,"shape":[2],"data_offsets":[0,8]}}""", 8)]
    [InlineData("""{"a":[1,2]}""", 0)]
    [InlineData("""[1,2]""", 0)]
    [InlineData("""{"__metadata__":{"format":7}}""", 0)]
    [InlineData("""{"a":""", 0)] // truncated JSON
    public void Crafted_SafeTensorsHeaders(string header, int dataBytes) => SafeTensorsFails(header, dataBytes);

    [Fact]
    public void Crafted_SafeTensorsHugeHeaderLength()
    {
        SafeTensorsFails("", 0, P(Le64(long.MaxValue), "{}"));
        SafeTensorsFails("", 0, P(Le64(1L << 31), "{}"));
        SafeTensorsFails("", 0, P(1, 2, 3));
    }

    [Fact]
    public void Crafted_ConvertedJson()
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-crafted-");
        try
        {
            var basePath = Path.Combine(dir.FullName, "m");
            File.WriteAllBytes(basePath + ".safetensors", SafeTensors("{}", 0));
            foreach (var json in new[] { "", "{", "null", new string('[', 10_000) + new string(']', 10_000) })
            {
                File.WriteAllText(basePath + ".json", json);
                Assert.Throws<InvalidDataException>(() => Checkpoint.Load(basePath));
            }
            // A $tensor node naming a tensor the safetensors file lacks.
            File.WriteAllText(basePath + ".json", """{"w":{"$tensor":"w","dtype":"F32","shape":[1]}}""");
            var ckpt = Checkpoint.Load(basePath);
            Assert.Throws<InvalidDataException>(() => ckpt.Tensor<float>(ckpt.Root["w"]));
            Assert.Throws<InvalidDataException>(() => ckpt.Shape(ckpt.Root["w"]));
            // A vocabulary whose ids are not integers.
            Assert.Throws<InvalidDataException>(() => Checkpoint.UnitToId(System.Text.Json.Nodes.JsonNode.Parse("""{"_unit2id":{"a":"x"}}""")));
            Assert.Throws<InvalidDataException>(() => Checkpoint.UnitToId(System.Text.Json.Nodes.JsonNode.Parse("""{"_unit2id":[1]}""")));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }
}
