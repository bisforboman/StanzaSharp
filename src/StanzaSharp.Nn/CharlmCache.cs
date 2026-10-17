namespace StanzaSharp.Nn;

/// <summary>
/// Per-sentence charlm representations, kept by the tagger for the processors after it (constituency, sentiment, NER,
/// and depparse on the managed backend). They run the same words through the same forward and backward charlms, which
/// is most of their work, so the pipeline keeps one of these per document and each sentence goes through the charlms once.
/// </summary>
/// <remarks>
/// Each word costs 2 x 1024 floats (8 KB) with the English charlms, held until the cache is disposed, plus 4 KB per
/// sentence from a managed tagger (the backward charlm's final cell state, for depparse), not counted in
/// <see cref="MaxWords"/>. To bound that on large documents the cache keeps at most <see cref="MaxWords"/> words; sentences
/// past that are not kept, and consumers compute them again as they do for any missing sentence.
/// <para>
/// A managed producer's entries are copied into 4 MB chunks rented from <c>ArrayPool&lt;float&gt;.Shared</c>, which
/// <see cref="Dispose()"/> gives back, so the next call reuses them (round 41). The pool then keeps them after the call
/// until it trims them.
/// </para>
/// <para>
/// Backend-neutral (issue #29): an entry holds what its producer made, TorchSharp tensors or float arrays, and each
/// consumer reads the form its backend uses. Tensors read as arrays are copied per read (the caller owns the copy);
/// arrays read as tensors are converted once, on the reader's device, and kept with the entry. A pipeline whose
/// producer and consumers share a backend never converts. The tensor half (<c>TryAdd</c> and <c>TryGet</c> with
/// tensors) is in StanzaSharp.Cuda (Nn/TensorForms.cs); it stores an <see cref="ICharlmReps"/>.
/// </para>
/// </remarks>
internal sealed class CharlmCache(int maxWords = CharlmCache.DefaultMaxWords) : IDisposable
{
    /// <summary>About 256 MB with the English charlms.</summary>
    public const int DefaultMaxWords = 32_768;

    internal sealed class Entry
    {
        public int Words;
        public ICharlmReps? Native;                // TorchSharp producer, or converted for a TorchSharp consumer
        public ArraySegment<float> ForwardData, BackwardData; // managed producer: [words, dim] row-major
        public ArraySegment<float> BackwardFinalC;            // managed producer: the backward pass's final cell state, [dim]
    }

    private readonly Dictionary<Sentence, Entry> _reps = [];

    // A power of two, so the pool's arrays are exactly this size (no rounding up). Sentences are copied in end to end.
    private const int ChunkFloats = 1 << 20;
    private readonly List<float[]> _chunks = [];
    private int _used; // floats used in the last chunk
    private long _words;

    public int MaxWords { get; } = maxWords;

    /// <summary>
    /// Keeps a sentence's [<paramref name="words"/>, dim] representations if they fit under <see cref="MaxWords"/>:
    /// only then is <paramref name="make"/> called, and the cache owns (and disposes) what it returns.
    /// </summary>
    /// <returns>Whether the cache kept them.</returns>
    internal bool TryAdd(Sentence sentence, int words, Func<ICharlmReps> make)
    {
        if (!Reserve(sentence, words))
            return false;
        _reps[sentence] = new Entry { Words = words, Native = make() };
        return true;
    }

    /// <summary>
    /// <see cref="TryAdd(Sentence, int, Func{ICharlmReps})"/> for a managed producer: [<paramref name="words"/>, dim] rows,
    /// which the cache copies.
    /// </summary>
    /// <param name="backwardFinalC">Optional: the backward pass's final cell state (see <see cref="TryGetBackward"/>).</param>
    public bool TryAdd(Sentence sentence, ReadOnlySpan<float> forward, ReadOnlySpan<float> backward, int words, ReadOnlySpan<float> backwardFinalC = default)
    {
        if (!Reserve(sentence, words))
            return false;
        _reps[sentence] = new Entry
        {
            Words = words,
            ForwardData = Copy(forward),
            BackwardData = Copy(backward),
            BackwardFinalC = backwardFinalC.IsEmpty ? default : Copy(backwardFinalC),
        };
        return true;
    }

    /// <summary>A copy of <paramref name="data"/> in the cache's rented chunks.</summary>
    private ArraySegment<float> Copy(ReadOnlySpan<float> data)
    {
        if (_chunks.Count == 0 || _used + data.Length > _chunks[^1].Length)
        {
            _chunks.Add(System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(ChunkFloats, data.Length)));
            _used = 0;
        }

        var copy = new ArraySegment<float>(_chunks[^1], _used, data.Length);
        data.CopyTo(copy);
        _used += data.Length;
        return copy;
    }

    /// <summary>
    /// What a managed charlm call on one sentence alone computed: [words, dim] arrays and the backward pass's final cell
    /// state (see <see cref="Managed.ManagedCharLanguageModel.Continue"/>).
    /// </summary>
    internal sealed record Alone(float[] Forward, float[] Backward, float[] BackwardFinalC);

    // By the sentence's texts. A managed charlm step gives a row the same arithmetic in any batch (Gemm.KernelBlocked), so
    // the outputs depend on the texts only: any other call on the same texts computes the same bits. (A TorchSharp
    // producer's entries above are not exact: there a row's kernel depends on its batch.)
    private readonly Dictionary<string, Alone> _alone = [];

    /// <summary>Keeps a single-sentence call's outputs (<see cref="TryGetAlone"/>) if they fit under <see cref="MaxWords"/>.</summary>
    public void AddAlone(IReadOnlyList<string> texts, Alone outputs)
    {
        if (!HasRoom(texts.Count) || !_alone.TryAdd(AloneKey(texts), outputs))
            return;
        _words += texts.Count;
    }

    /// <summary>
    /// The outputs of an earlier single-sentence charlm call on exactly these texts: exactly what a single-sentence call
    /// would compute now.
    /// </summary>
    public bool TryGetAlone(IReadOnlyList<string> texts, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out Alone outputs) =>
        _alone.TryGetValue(AloneKey(texts), out outputs);

    /// <summary>
    /// The backward charlm's [words, dim] outputs over <paramref name="texts"/> and its final cell state, from a managed
    /// producer: <paramref name="sentence"/>'s entry (kept with words as written, which are then its texts) or a
    /// single-sentence call on the texts. Exactly what a managed charlm call on the texts computes, in any batch.
    /// </summary>
    public bool TryGetBackward(Sentence? sentence, IReadOnlyList<string> texts, out ArraySegment<float> backward, out ArraySegment<float> finalC)
    {
        if (sentence != null && _reps.TryGetValue(sentence, out var e) && e.BackwardFinalC.Array != null && e.Words == texts.Count)
        {
            (backward, finalC) = (e.BackwardData, e.BackwardFinalC);
            return true;
        }

        if (TryGetAlone(texts, out var alone))
        {
            (backward, finalC) = (alone.Backward, alone.BackwardFinalC);
            return true;
        }

        (backward, finalC) = (default, default);
        return false;
    }

    private static string AloneKey(IReadOnlyList<string> texts) => string.Concat(texts.Select(t => $"{t.Length}:{t}"));

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

    /// <summary>A sentence's entry, for the TorchSharp half.</summary>
    internal bool TryGetEntry(Sentence sentence, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out Entry? entry) => _reps.TryGetValue(sentence, out entry);

    /// <summary>A sentence's representations as [words, dim] row-major arrays, for the managed backend (don't change them).</summary>
    public bool TryGetArrays(Sentence sentence, out (ArraySegment<float> Forward, ArraySegment<float> Backward) reps)
    {
        if (!_reps.TryGetValue(sentence, out var e))
        {
            reps = default;
            return false;
        }

        reps = e.ForwardData.Array != null ? (e.ForwardData, e.BackwardData) : e.Native!.ToArrays();
        return true;
    }

    /// <summary>
    /// Drops <paramref name="sentence"/>'s entry once its last reader is done with it (NER, the pipeline's last
    /// processor): a TorchSharp entry's tensors are freed at once, a managed entry's chunk space only at
    /// <see cref="Dispose()"/>. It doesn't make room for new ones.
    /// </summary>
    public void Release(Sentence sentence)
    {
        if (_reps.Remove(sentence, out var e))
            Dispose(e);
    }

    private static void Dispose(Entry e) => e.Native?.Dispose();

    public void Dispose()
    {
        foreach (var e in _reps.Values)
            Dispose(e);
        _reps.Clear();
        _alone.Clear();
        foreach (var chunk in _chunks)
            System.Buffers.ArrayPool<float>.Shared.Return(chunk);
        _chunks.Clear();
        _used = 0;
        _words = 0;
    }
}

/// <summary>A backend's own form of a sentence's charlm representations (StanzaSharp.Cuda: two tensors).</summary>
internal interface ICharlmReps : IDisposable
{
    /// <summary>Copies of the [words, dim] representations, row-major.</summary>
    (float[] Forward, float[] Backward) ToArrays();
}
