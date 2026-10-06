using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// Per-sentence charlm representations, kept by the tagger for the parser. Both run the same words
/// through the same forward and backward charlms, which is most of their work, so the pipeline keeps
/// one of these per document and each sentence goes through the charlms once.
/// </summary>
/// <remarks>
/// Each word costs 2 x 1024 floats (8 KB) with the English charlms, held until the cache is disposed.
/// To bound that on large documents the cache keeps at most <see cref="MaxWords"/> words; sentences
/// past that are not kept, and consumers compute them again as they do for any missing sentence.
/// </remarks>
internal sealed class CharlmCache(int maxWords = CharlmCache.DefaultMaxWords) : IDisposable
{
    /// <summary>About 256 MB with the English charlms.</summary>
    public const int DefaultMaxWords = 32_768;

    private readonly Dictionary<Sentence, (Tensor Forward, Tensor Backward)> _reps = [];
    private long _words;

    public int MaxWords { get; } = maxWords;

    /// <summary>
    /// Keeps a sentence's [words, dim] representations if they fit under <see cref="MaxWords"/>.
    /// When kept, the cache owns them: it detaches them from their dispose scope and disposes them
    /// itself. When not kept, nothing happens and they stay with the caller (and its dispose scope).
    /// </summary>
    /// <returns>Whether the cache kept them.</returns>
    public bool TryAdd(Sentence sentence, Tensor forward, Tensor backward)
    {
        if (_reps.Remove(sentence, out var old))
        {
            _words -= old.Forward.shape[0];
            old.Forward.Dispose();
            old.Backward.Dispose();
        }
        if (_words + forward.shape[0] > MaxWords)
            return false;
        _words += forward.shape[0];
        _reps[sentence] = (forward.DetachFromDisposeScope(), backward.DetachFromDisposeScope());
        return true;
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
        _words = 0;
    }
}
