using System.Diagnostics;
using System.Text;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class TorchCheckpointTests(ITestOutputHelper output)
{
    public static readonly TheoryData<string> StanzaCheckpoints =
    [
        "tokenize/combined_nocharlm", "mwt/combined", "pos/combined_charlm", "lemma/combined_nocharlm", "constituency/ptb3-revised_charlm",
        "pretrain/conll17", "forward_charlm/1billion", "backward_charlm/1billion",
    ];

    [Theory]
    [InlineData("tiny_legacy")]
    [InlineData("tiny_zip")]
    public void LoadsPtFixtureLikeTheConverter(string name)
    {
        var basePath = Path.Combine(Repo.Golden, "pt", name);
        AssertSameAsConverted(Checkpoint.Load(basePath + ".pt"), Checkpoint.Load(basePath));
    }

    [PtModelTheory]
    [MemberData(nameof(StanzaCheckpoints))]
    public void LoadsStanzaCheckpointLikeTheConverter(string name)
    {
        var timer = Stopwatch.StartNew();
        var pt = Checkpoint.Load(Path.Combine(Repo.StanzaModels, name)); // no .json there, so this reads name.pt
        var ptTime = timer.Elapsed;
        timer.Restart();
        var converted = Checkpoint.Load(Repo.Model(name));
        output.WriteLine($"{name}: .pt {ptTime.TotalMilliseconds:F0} ms, converted {timer.Elapsed.TotalMilliseconds:F0} ms");
        AssertSameAsConverted(pt, converted);
    }

    [PtModelFact]
    public void Pipeline_RunsFromStanzaModelDirectory()
    {
        using var nlp = Pipeline.Load(Repo.StanzaModels);
        var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        Assert.Equal(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")), Conllu.Write(doc));
    }

    private static void AssertSameAsConverted(Checkpoint pt, Checkpoint converted)
    {
        Assert.Equal(converted.Root.ToJsonString(), pt.Root.ToJsonString());
        Assert.Equal(converted.Tensors.Tensors.Keys.Order(), pt.Tensors.Tensors.Keys.Order());
        foreach (var (key, expected) in converted.Tensors.Tensors)
        {
            Assert.Equal(expected.Dtype, pt.Tensors[key].Dtype);
            Assert.Equal(expected.Shape, pt.Tensors[key].Shape);
            Assert.True(converted.Tensors.RawBytes(key).SequenceEqual(pt.Tensors.RawBytes(key)), $"tensor {key} differs");
        }
    }

    /// <summary>Concatenates pickle bytes: ints are single bytes, strings ASCII.</summary>
    private static byte[] Pickle(params object[] parts) =>
        parts.SelectMany(p => p switch
        {
            string s => Encoding.ASCII.GetBytes(s),
            int b => [(byte)b],
            _ => throw new ArgumentException(p.ToString()),
        }).ToArray();

    [Fact]
    public void Unpickler_ReadsPlainData()
    {
        // {'a': 7, 'b': [1.5, (True, None), -2], 'c': 'b'}, the last 'b' fetched from the memo.
        var obj = Unpickler.Load(Pickle(0x80, 2, "}q", 0, "(",
            "X", 1, 0, 0, 0, "a", "K", 7,
            "X", 1, 0, 0, 0, "bq", 2, "](G", 0x3f, 0xf8, 0, 0, 0, 0, 0, 0, 0x88, "N", 0x86, "J", 0xfe, 0xff, 0xff, 0xff, "e",
            "X", 1, 0, 0, 0, "ch", 2, "u."));

        var dict = Assert.IsType<PyDict>(obj);
        Assert.Equal(["a", "b", "c"], dict.Keys);
        Assert.Equal(7L, dict["a"]);
        Assert.Equal([1.5, new PyTuple([true, null]), -2L], Assert.IsType<List<object?>>(dict["b"]));
        Assert.Equal("b", dict["c"]);
    }

    [Fact]
    public void Unpickler_RebuildsTensorsFromPersistentStorages()
    {
        // _rebuild_tensor_v2(<storage>, 1, (2, 2), (1, 2), False, OrderedDict())
        var bytes = Pickle(0x80, 2, "ctorch._utils\n_rebuild_tensor_v2\n(",
            "(X", 7, 0, 0, 0, "storagectorch\nFloatStorage\nX", 1, 0, 0, 0, "0X", 3, 0, 0, 0, "cpuK", 6, "NtQ",
            "K", 1, "K", 2, "K", 2, 0x86, "K", 1, "K", 2, 0x86, 0x89, "ccollections\nOrderedDict\n)Rt", "R.");
        var tensor = Assert.IsType<TorchTensor>(Unpickler.Load(bytes, pid =>
            pid is PyTuple { Items: ["storage", TorchStorageType type, string key, "cpu", long numel, null] }
                ? new TorchStorage(type, key, numel, 0) : throw new InvalidDataException()));

        Assert.Equal(("F32", 4, "0", 6L), (tensor.Storage.Type.Dtype, tensor.Storage.Type.ElementSize, tensor.Storage.Key, tensor.Storage.Numel));
        Assert.Equal(1L, tensor.Offset);
        Assert.Equal([2L, 2], tensor.Shape);
        Assert.Equal([1L, 2], tensor.Stride);

        // Without a persistent_load, storages are rejected.
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(bytes));
    }

    [Fact]
    public void Unpickler_RejectsGlobalsOutsideTheAllowlist()
    {
        // os.system('ls'): the classic pickle exploit.
        var e = Assert.Throws<InvalidDataException>(() =>
            Unpickler.Load(Pickle(0x80, 2, "cos\nsystem\nX", 2, 0, 0, 0, "ls", 0x85, "R.")));
        Assert.Contains("'os.system' is not allowed", e.Message);
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(Pickle(0x80, 2, "cbuiltins\nset\n)R.")));
        // An allowed global can still only be called the way torch uses it.
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(Pickle(0x80, 2, "ctorch\nFloatStorage\n)R.")));
    }

    [Fact]
    public void Unpickler_RejectsMalformedData()
    {
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(Pickle(0x80, 2, "X", 5, 0, 0, 0, "ab"))); // truncated
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(Pickle(0x80, 2, "N", 0xff))); // unknown opcode
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(Pickle(0x80, 2, "."))); // empty stack
        Assert.Throws<InvalidDataException>(() => Unpickler.Load(Pickle(0x80, 2, "h", 3, "."))); // missing memo
    }

    [Theory]
    [InlineData(1e-05, "1e-05")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(1e16, "1e+16")]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(2.0, "2.0")]
    [InlineData(100.0, "100.0")]
    [InlineData(-0.0, "-0.0")]
    [InlineData(0.1, "0.1")]
    [InlineData(1.0 / 3, "0.3333333333333333")]
    [InlineData(123.456, "123.456")]
    [InlineData(1.4e-08, "1.4e-08")]
    [InlineData(-2.5e-07, "-2.5e-07")]
    [InlineData(1.7976931348623157e308, "1.7976931348623157e+308")]
    [InlineData(5e-324, "5e-324")]
    [InlineData(12345678901234567890.0, "1.2345678901234567e+19")]
    public void PythonRepr_MatchesPython(double value, string expected) =>
        Assert.Equal(expected, TorchCheckpoint.PythonRepr(value));
}
