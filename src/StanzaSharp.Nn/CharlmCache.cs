using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// Per-sentence charlm representations, kept by the tagger for the parser. Both run the same words
/// through the same forward and backward charlms, which is most of their work, so the pipeline keeps
/// one of these per document and each sentence goes through the charlms once.
/// </summary>
/// <remarks>
/// ponytail: holds 2 x 1024 floats (8 KB) per word until the document is parsed; for very large
/// documents, process them in parts or free entries as the parser consumes them.
/// </remarks>
public sealed class CharlmCache : IDisposable
{
    private readonly Dictionary<Sentence, (Tensor Forward, Tensor Backward)> _reps = [];

    /// <summary>Stores a sentence's [words, dim] representations; the cache takes ownership.</summary>
    public void Add(Sentence sentence, Tensor forward, Tensor backward)
    {
        if (_reps.Remove(sentence, out var old))
        {
            old.Forward.Dispose();
            old.Backward.Dispose();
        }
        _reps[sentence] = (forward, backward);
    }

    public bool TryGet(Sentence sentence, out (Tensor Forward, Tensor Backward) reps) => _reps.TryGetValue(sentence, out reps);

    public void Dispose()
    {
        foreach (var (f, b) in _reps.Values)
        {
            f.Dispose();
            b.Dispose();
        }
        _reps.Clear();
    }
}
