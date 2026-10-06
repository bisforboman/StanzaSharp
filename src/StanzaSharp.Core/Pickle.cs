using System.Buffers.Binary;
using System.Collections;
using System.Numerics;
using System.Text;

namespace StanzaSharp;

/// <summary>A Python tuple.</summary>
internal sealed record PyTuple(object?[] Items)
{
    public bool Equals(PyTuple? other) =>
        other != null && StructuralComparisons.StructuralEqualityComparer.Equals(Items, other.Items);

    public override int GetHashCode() => StructuralComparisons.StructuralEqualityComparer.GetHashCode(Items);
}

/// <summary>A Python dict (insertion-ordered, like Python's), also used for <c>collections.OrderedDict</c>.</summary>
internal sealed class PyDict : OrderedDictionary<object, object?>;

/// <summary>A legacy typed storage class such as <c>torch.FloatStorage</c>, mapped to its safetensors dtype.</summary>
internal sealed record TorchStorageType(string Name, string Dtype, int ElementSize);

/// <summary>
/// A storage reference from a checkpoint's persistent ids: <paramref name="Key"/> names the root storage
/// holding the data, and the referenced elements start <paramref name="Offset"/> elements into it.
/// </summary>
internal sealed record TorchStorage(TorchStorageType Type, string Key, long Numel, long Offset);

/// <summary>A tensor from <c>torch._utils._rebuild_tensor_v2</c>: a strided view into a storage.</summary>
internal sealed record TorchTensor(TorchStorage Storage, long Offset, long[] Shape, long[] Stride);

/// <summary>
/// A restricted unpickler for PyTorch checkpoints. It reads the opcodes <c>torch.load(weights_only=True)</c>
/// reads (pickle protocol 2, which torch.save uses) and builds only plain data (None, bool, int, float,
/// str, bytes, list, tuple, dict) plus the few globals in <see cref="Allowed"/>, a subset of torch's own
/// allowlist; any other global or opcode throws. Nothing is ever executed or instantiated by name.
/// </summary>
internal sealed class Unpickler
{
    private static readonly Dictionary<string, object> Allowed = new[]
    {
        new TorchStorageType("FloatStorage", "F32", 4), new("DoubleStorage", "F64", 8), new("HalfStorage", "F16", 2),
        new("LongStorage", "I64", 8), new("IntStorage", "I32", 4), new("ShortStorage", "I16", 2),
        new("CharStorage", "I8", 1), new("ByteStorage", "U8", 1), new("BoolStorage", "BOOL", 1),
    }.ToDictionary(t => "torch." + t.Name, t => (object)t);

    static Unpickler()
    {
        Allowed["collections.OrderedDict"] = "collections.OrderedDict";
        Allowed["torch._utils._rebuild_tensor_v2"] = "torch._utils._rebuild_tensor_v2";
        Allowed["_codecs.encode"] = "_codecs.encode"; // protocol 2 pickles bytes as encode(latin-1 str, 'latin1')
    }

    private readonly byte[] _data;
    private readonly Func<object?, object?>? _persistentLoad;
    private readonly List<object?> _stack = [];
    private readonly Stack<List<object?>> _marks = new();
    private readonly Dictionary<long, object?> _memo = [];
    private int _pos;

    private Unpickler(byte[] data, int pos, Func<object?, object?>? persistentLoad)
    {
        _data = data;
        _pos = pos;
        _persistentLoad = persistentLoad;
    }

    /// <summary>Unpickles one object from <paramref name="data"/> starting at <paramref name="pos"/>, advancing it past STOP.</summary>
    /// <param name="persistentLoad">Resolves persistent ids (BINPERSID); without it they are rejected.</param>
    public static object? Load(byte[] data, ref int pos, Func<object?, object?>? persistentLoad = null)
    {
        var u = new Unpickler(data, pos, persistentLoad);
        var result = u.Run();
        pos = u._pos;
        return result;
    }

    public static object? Load(byte[] data, Func<object?, object?>? persistentLoad = null)
    {
        int pos = 0;
        return Load(data, ref pos, persistentLoad);
    }

    private object? Run()
    {
        while (true)
        {
            byte op = Bytes(1)[0];
            switch (op)
            {
                case 0x80: Bytes(1); break; // PROTO
                case (byte)'.': // STOP
                    return Pop();
                case (byte)'(': _marks.Push(new List<object?>(_stack)); _stack.Clear(); break; // MARK
                case (byte)'N': Push(null); break;
                case 0x88: Push(true); break; // NEWTRUE
                case 0x89: Push(false); break; // NEWFALSE
                case (byte)'J': Push((long)BinaryPrimitives.ReadInt32LittleEndian(Bytes(4))); break; // BININT
                case (byte)'K': Push((long)Bytes(1)[0]); break; // BININT1
                case (byte)'M': Push((long)BinaryPrimitives.ReadUInt16LittleEndian(Bytes(2))); break; // BININT2
                case 0x8a: // LONG1: little-endian two's complement
                {
                    var n = new BigInteger(Bytes(Bytes(1)[0]), isUnsigned: false, isBigEndian: false);
                    Push(n >= long.MinValue && n <= long.MaxValue ? (long)n : n);
                    break;
                }
                case (byte)'G': Push(BinaryPrimitives.ReadDoubleBigEndian(Bytes(8))); break; // BINFLOAT
                case (byte)'X': Push(Utf8(BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4)))); break; // BINUNICODE
                case (byte)'U': Push(Utf8(Bytes(1)[0])); break; // SHORT_BINSTRING (Python 2 str; torch.load decodes it as UTF-8)
                case (byte)')': Push(new PyTuple([])); break; // EMPTY_TUPLE
                case (byte)'t': Push(new PyTuple(PopMark().ToArray())); break; // TUPLE
                case 0x85: Push(new PyTuple([Pop()])); break; // TUPLE1
                case 0x86: { var b = Pop(); var a = Pop(); Push(new PyTuple([a, b])); break; } // TUPLE2
                case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); Push(new PyTuple([a, b, c])); break; } // TUPLE3
                case (byte)']': Push(new List<object?>()); break; // EMPTY_LIST
                case (byte)'a': { var v = Pop(); Peek<List<object?>>().Add(v); break; } // APPEND
                case (byte)'e': { var items = PopMark(); Peek<List<object?>>().AddRange(items); break; } // APPENDS
                case (byte)'}': Push(new PyDict()); break; // EMPTY_DICT
                case (byte)'s': { var v = Pop(); var k = Pop(); Peek<PyDict>()[Key(k)] = v; break; } // SETITEM
                case (byte)'u': // SETITEMS
                {
                    var items = PopMark();
                    if (items.Count % 2 != 0)
                        throw Error("SETITEMS with an odd number of items");
                    var dict = Peek<PyDict>();
                    for (int i = 0; i < items.Count; i += 2)
                        dict[Key(items[i])] = items[i + 1];
                    break;
                }
                case (byte)'h': Push(MemoGet(Bytes(1)[0])); break; // BINGET
                case (byte)'j': Push(MemoGet(BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4)))); break; // LONG_BINGET
                case (byte)'q': _memo[Bytes(1)[0]] = Top(); break; // BINPUT
                case (byte)'r': _memo[BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4))] = Top(); break; // LONG_BINPUT
                case (byte)'c': Push(Global(Line(), Line())); break; // GLOBAL
                case (byte)'R': { var args = Pop() as PyTuple ?? throw Error("REDUCE arguments are not a tuple"); Push(Reduce(Pop(), args.Items)); break; }
                case (byte)'b': // BUILD: torch sets attributes on OrderedDicts (e.g. a state_dict's _metadata); dict items are unaffected
                    Pop();
                    Peek<PyDict>();
                    break;
                case 0x51: // BINPERSID
                    Push((_persistentLoad ?? throw Error("persistent id without a persistent_load"))(Pop()));
                    break;
                default:
                    throw Error($"unsupported opcode 0x{op:x2}");
            }
        }
    }

    private static object Global(string? module, string? name) =>
        Allowed.TryGetValue($"{module}.{name}", out var g) ? g
            : throw new InvalidDataException(
                $"Pickle global '{module}.{name}' is not allowed: only plain data, OrderedDict and tensors are loaded");

    private object Reduce(object? callable, object?[] args)
    {
        switch (callable)
        {
            case "collections.OrderedDict" when args.Length == 0:
                return new PyDict();
            case "torch._utils._rebuild_tensor_v2" when args.Length is 6 or 7:
                // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
                return args[0] is TorchStorage storage && args[1] is long offset
                    && args[2] is PyTuple size && args[3] is PyTuple stride && size.Items.Length == stride.Items.Length
                    ? new TorchTensor(storage, offset, Longs(size), Longs(stride))
                    : throw Error("malformed _rebuild_tensor_v2 arguments");
            case "_codecs.encode" when args is [string text, "latin1"]:
                return text.All(c => c < 256) ? Encoding.Latin1.GetBytes(text) : throw Error("bytes text is not latin-1");
            default:
                throw Error($"cannot call {callable ?? "None"} with {args.Length} arguments");
        }
    }

    private long[] Longs(PyTuple t) =>
        t.Items.Select(x => x as long? ?? throw Error("tensor size/stride is not an int")).ToArray();

    private object Key(object? key) => key ?? throw Error("dict key None is not supported");

    private void Push(object? v) => _stack.Add(v);

    private object? Pop()
    {
        if (_stack.Count == 0)
            throw Error("stack underflow");
        var v = _stack[^1];
        _stack.RemoveAt(_stack.Count - 1);
        return v;
    }

    private object? Top() => _stack.Count > 0 ? _stack[^1] : throw Error("stack underflow");

    private T Peek<T>() where T : class =>
        _stack.Count > 0 && _stack[^1] is T t ? t : throw Error($"expected a {typeof(T).Name} on the stack");

    private List<object?> PopMark()
    {
        if (!_marks.TryPop(out var outer))
            throw Error("no MARK on the stack");
        var items = new List<object?>(_stack);
        _stack.Clear();
        _stack.AddRange(outer);
        return items;
    }

    private object? MemoGet(long i) => _memo.TryGetValue(i, out var v) ? v : throw Error($"memo key {i} not found");

    private ReadOnlySpan<byte> Bytes(int n)
    {
        if (n < 0 || _pos > _data.Length - n)
            throw Error("unexpected end of data");
        _pos += n;
        return _data.AsSpan(_pos - n, n);
    }

    private string Utf8(uint n) => n <= int.MaxValue ? Encoding.UTF8.GetString(Bytes((int)n)) : throw Error($"string length {n} too large");

    private string Line()
    {
        int end = Array.IndexOf(_data, (byte)'\n', _pos);
        if (end < 0)
            throw Error("unterminated GLOBAL");
        var s = Encoding.UTF8.GetString(_data, _pos, end - _pos);
        _pos = end + 1;
        return s;
    }

    private InvalidDataException Error(string message) => new($"Invalid pickle at byte {_pos}: {message}");
}
