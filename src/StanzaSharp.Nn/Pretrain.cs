using System.Text.Json.Nodes;

namespace StanzaSharp.Nn;

/// <summary>
/// Pretrained word vectors (stanza/models/common/pretrain.py). Callers choose the lookup policy:
/// the tagger lowercases every word, the parser tries the word as written and then lowercased.
/// </summary>
/// <remarks>
/// Two forms: for the managed backend a plain float array (<see cref="LoadManaged"/>), which touches no TorchSharp type
/// and so loads no native libtorch; or a TorchSharp tensor (<c>Pretrain.Load</c> and <c>Embeddings</c>, extensions in
/// StanzaSharp.Cuda), held in <see cref="Native"/>.
/// </remarks>
internal sealed class Pretrain : IDisposable
{
    private readonly Dictionary<string, int> _vocab;
    private readonly float[]? _vectors;

    public int UnkId { get; }

    /// <summary>The number of vectors (rows).</summary>
    public int Count { get; }

    public int Dim { get; }

    /// <summary>The TorchSharp form's vectors, or null for the managed form.</summary>
    internal IPretrainVectors? Native { get; }

    /// <summary>
    /// The embedding matrix's floats, row-major, for the managed backend: the array itself, or the TorchSharp form's
    /// (<see cref="IPretrainVectors.CpuVectors"/>).
    /// </summary>
    public ReadOnlySpan<float> CpuVectors() => _vectors ?? Native!.CpuVectors();

    /// <param name="native">Makes the TorchSharp form from the checkpoint and its <c>emb</c> node; null for the managed form.</param>
    internal Pretrain(Checkpoint ckpt, Func<Checkpoint, JsonNode?, IPretrainVectors>? native)
    {
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        if (ckpt.Root["vocab"]!["lower"]?.GetValue<bool>() == true)
            throw new NotSupportedException("Lowercasing pretrain vocabularies are not ported");
        UnkId = _vocab["<UNK>"];
        var emb = ckpt.Root["emb"];
        var shape = ckpt.Shape(emb);
        (Count, Dim) = ((int)shape[0], (int)shape[1]);
        if (native == null)
            _vectors = ckpt.Tensor<float>(emb);
        else
            Native = native(ckpt, emb);
    }

    /// <summary>Loads the vectors as a float array, for the managed backend: no tensor, no TorchSharp call.</summary>
    public static Pretrain LoadManaged(string basePath) => new(Checkpoint.Load(basePath), native: null);

    /// <summary>PretrainedWordVocab.unit2id: spaces inside a word are stored as U+00A0.</summary>
    public int UnitToId(string word) =>
        _vocab.TryGetValue(word.Replace(' ', ' '), out var id) ? id : UnkId;

    public void Dispose() => Native?.Dispose();
}

/// <summary>A backend's own form of the pretrain vectors (StanzaSharp.Cuda: a tensor).</summary>
internal interface IPretrainVectors : IDisposable
{
    /// <summary>The [count, dim] floats, row-major, on the CPU (valid while this object is).</summary>
    ReadOnlySpan<float> CpuVectors();
}
