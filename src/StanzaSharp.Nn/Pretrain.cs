using StanzaSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// Pretrained word vectors (stanza/models/common/pretrain.py). Callers choose the lookup policy:
/// the tagger lowercases every word, the parser tries the word as written and then lowercased.
/// </summary>
public sealed class Pretrain : IDisposable
{
    private readonly Dictionary<string, int> _vocab;

    public int UnkId { get; }

    /// <summary>[vocab, dim] embedding matrix.</summary>
    public Tensor Embeddings { get; }

    public int Dim => (int)Embeddings.shape[1];

    private Pretrain(Checkpoint ckpt)
    {
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        if (ckpt.Root["vocab"]!["lower"]?.GetValue<bool>() == true)
            throw new NotSupportedException("Lowercasing pretrain vocabularies are not ported");
        UnkId = _vocab["<UNK>"];
        Embeddings = ckpt.ToTensor(ckpt.Root["emb"]);
    }

    /// <summary>Loads e.g. <c>models/converted/en/pretrain/conll17</c>.</summary>
    public static Pretrain Load(string basePath) => new(Checkpoint.Load(basePath));

    /// <summary>PretrainedWordVocab.unit2id: spaces inside a word are stored as U+00A0.</summary>
    public int UnitToId(string word) =>
        _vocab.TryGetValue(word.Replace(' ', ' '), out var id) ? id : UnkId;

    public void Dispose() => Embeddings.Dispose();
}
