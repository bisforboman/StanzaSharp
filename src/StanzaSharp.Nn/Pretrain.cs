using StanzaSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// Pretrained word vectors (stanza/models/common/pretrain.py). Callers choose the lookup policy:
/// the tagger lowercases every word, the parser tries the word as written and then lowercased.
/// </summary>
/// <remarks>
/// Two forms: a TorchSharp tensor (<see cref="Load"/>) or, for the managed backend, a plain float array
/// (<see cref="LoadManaged"/>), which touches no TorchSharp type and so loads no native libtorch.
/// </remarks>
internal sealed class Pretrain : IDisposable
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Tensor? _embeddings;
    private readonly float[]? _vectors;
    private float[]? _cpuCopy;

    public int UnkId { get; }

    /// <summary>The number of vectors (rows).</summary>
    public int Count { get; }

    public int Dim { get; }

    /// <summary>[vocab, dim] embedding matrix (TorchSharp form only).</summary>
    public Tensor Embeddings => _embeddings ?? throw new InvalidOperationException("This pretrain was loaded for the managed backend (LoadManaged) and has no tensor");

    /// <summary>
    /// The embedding matrix's floats, row-major, for the managed backend: the array itself, or the CPU tensor's own
    /// memory (not a copy; valid while this object is). On another device, a copy made on first use.
    /// </summary>
    public ReadOnlySpan<float> CpuVectors() => _vectors
        ?? (_embeddings!.device_type == DeviceType.CPU
            ? System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(_embeddings.bytes)
            : _cpuCopy ??= _embeddings.ToArray<float>()); // a race only makes the same copy twice

    private Pretrain(Checkpoint ckpt, bool managed)
    {
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        if (ckpt.Root["vocab"]!["lower"]?.GetValue<bool>() == true)
            throw new NotSupportedException("Lowercasing pretrain vocabularies are not ported");
        UnkId = _vocab["<UNK>"];
        var emb = ckpt.Root["emb"];
        var shape = ckpt.Shape(emb);
        (Count, Dim) = ((int)shape[0], (int)shape[1]);
        if (managed)
            _vectors = ckpt.Tensor<float>(emb);
        else
            _embeddings = ckpt.ToTensor(emb);
    }

    /// <summary>Loads e.g. <c>models/converted/en/pretrain/conll17</c> as a tensor.</summary>
    /// <param name="device">Where the embedding matrix lives; CPU by default.</param>
    public static Pretrain Load(string basePath, Device? device = null) => Weights.On(device, () => new Pretrain(Checkpoint.Load(basePath), managed: false));

    /// <summary>Loads the vectors as a float array, for the managed backend: no tensor, no TorchSharp call.</summary>
    public static Pretrain LoadManaged(string basePath) => new(Checkpoint.Load(basePath), managed: true);

    /// <summary>PretrainedWordVocab.unit2id: spaces inside a word are stored as U+00A0.</summary>
    public int UnitToId(string word) =>
        _vocab.TryGetValue(word.Replace(' ', ' '), out var id) ? id : UnkId;

    public void Dispose() => _embeddings?.Dispose();
}
