using TorchSharp;
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
/// <para>
/// Backend-neutral (issue #29): an entry holds what its producer made, TorchSharp tensors or float arrays, and each
/// consumer reads the form its backend uses. Tensors read as arrays are copied per read (the caller owns the copy);
/// arrays read as tensors are converted once, on <paramref name="device"/>, and kept with the entry. A pipeline whose
/// producer and consumers share a backend never converts.
/// </para>
/// </remarks>
/// <param name="device">Where tensors made from float entries go: the TorchSharp processors' device (CPU by default).</param>
internal sealed class CharlmCache(int maxWords = CharlmCache.DefaultMaxWords, Device? device = null) : IDisposable
{
    /// <summary>About 256 MB with the English charlms.</summary>
    public const int DefaultMaxWords = 32_768;

    private sealed class Entry
    {
        public int Words;
        public Tensor? Forward, Backward;         // TorchSharp producer, or converted for a TorchSharp consumer
        public float[]? ForwardData, BackwardData; // managed producer: [words, dim] row-major
        public float[]? BackwardFinalC;            // managed producer that ran the sentence alone: see TryGetBackwardState
    }

    private readonly Dictionary<Sentence, Entry> _reps = [];
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
        if (!Reserve(sentence, (int)forward.shape[0]))
            return false;
        _reps[sentence] = new Entry { Words = (int)forward.shape[0], Forward = forward.DetachFromDisposeScope(), Backward = backward.DetachFromDisposeScope() };
        return true;
    }

    /// <summary>
    /// <see cref="TryAdd(Sentence, Tensor, Tensor)"/> for a managed producer: [<paramref name="words"/>, dim] arrays,
    /// which the cache keeps (the caller must not change them afterwards).
    /// </summary>
    /// <param name="backwardFinalC">Only when the backward charlm ran this sentence alone (a batch of one): its cell
    /// state after the last character (see <see cref="TryGetBackwardState"/>).</param>
    public bool TryAdd(Sentence sentence, float[] forward, float[] backward, int words, float[]? backwardFinalC = null)
    {
        if (!Reserve(sentence, words))
            return false;
        _reps[sentence] = new Entry { Words = words, ForwardData = forward, BackwardData = backward, BackwardFinalC = backwardFinalC };
        return true;
    }

    /// <summary>
    /// The backward charlm's state at the end of a sentence that a managed producer ran alone: the representations
    /// (row 0, the first word's, is the final h) and the final cell state. The dependency parser continues from it over its
    /// ROOT word, which comes last backward, instead of running the whole sentence again; run alone too, it gets the
    /// same bits (<see cref="Managed.ManagedCharLanguageModel.Continue"/>).
    /// </summary>
    public bool TryGetBackwardState(Sentence sentence, out (float[] Backward, float[] FinalC) state)
    {
        state = _reps.TryGetValue(sentence, out var e) && e.BackwardFinalC != null ? (e.BackwardData!, e.BackwardFinalC) : default;
        return state.Backward != null;
    }

    /// <summary>Whether a new sentence of <paramref name="words"/> words would fit, so a producer can skip making copies that won't be kept.</summary>
    public bool HasRoom(int words) => _words + words <= MaxWords;

    /// <summary>Drops any old entry for <paramref name="sentence"/>, then whether <paramref name="words"/> more fit.</summary>
    private bool Reserve(Sentence sentence, int words)
    {
        if (_reps.Remove(sentence, out var old))
        {
            _words -= old.Words;
            Dispose(old);
        }
        if (_words + words > MaxWords)
            return false;
        _words += words;
        return true;
    }

    /// <summary>A sentence's [words, dim] tensors, owned by the cache (don't dispose them).</summary>
    public bool TryGet(Sentence sentence, out (Tensor Forward, Tensor Backward) reps)
    {
        if (!_reps.TryGetValue(sentence, out var e))
        {
            reps = default;
            return false;
        }
        if (e.Forward is null)
        {
            Tensor Convert(float[] data)
            {
                using var cpu = torch.tensor(data, [e.Words, data.Length / e.Words]);
                return (device == null || device.type == DeviceType.CPU ? cpu.clone() : cpu.to(device)).DetachFromDisposeScope();
            }
            e.Forward = Convert(e.ForwardData!);
            e.Backward = Convert(e.BackwardData!);
        }
        reps = (e.Forward!, e.Backward!);
        return true;
    }

    /// <summary>A sentence's representations as [words, dim] row-major arrays, for the managed backend (don't change them).</summary>
    public bool TryGetArrays(Sentence sentence, out (float[] Forward, float[] Backward) reps)
    {
        if (!_reps.TryGetValue(sentence, out var e))
        {
            reps = default;
            return false;
        }
        reps = e.ForwardData != null ? (e.ForwardData, e.BackwardData!) : (e.Forward!.ToArray<float>(), e.Backward!.ToArray<float>());
        return true;
    }

    private static void Dispose(Entry e)
    {
        e.Forward?.Dispose();
        e.Backward?.Dispose();
    }

    public void Dispose()
    {
        foreach (var e in _reps.Values)
            Dispose(e);
        _reps.Clear();
        _words = 0;
    }
}
