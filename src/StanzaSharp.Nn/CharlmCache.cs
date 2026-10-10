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
/// arrays read as tensors are converted once, on the reader's device, and kept with the entry. A pipeline whose
/// producer and consumers share a backend never converts. The tensor half (<c>TryAdd</c> and <c>TryGet</c> with
/// tensors) is in StanzaSharp.Cuda; it stores an <see cref="ICharlmReps"/>.
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
        public float[]? ForwardData, BackwardData; // managed producer: [words, dim] row-major
    }

    private readonly Dictionary<Sentence, Entry> _reps = [];
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
    /// <see cref="TryAdd(Sentence, int, Func{ICharlmReps})"/> for a managed producer: [<paramref name="words"/>, dim] arrays,
    /// which the cache keeps (the caller must not change them afterwards).
    /// </summary>
    public bool TryAdd(Sentence sentence, float[] forward, float[] backward, int words)
    {
        if (!Reserve(sentence, words))
            return false;
        _reps[sentence] = new Entry { Words = words, ForwardData = forward, BackwardData = backward };
        return true;
    }

    /// <summary>
    /// What a managed charlm call on one sentence alone computed: [words, dim] arrays and the backward pass's final cell
    /// state (see <see cref="Managed.ManagedCharLanguageModel.Continue"/>).
    /// </summary>
    internal sealed record Alone(float[] Forward, float[] Backward, float[] BackwardFinalC);

    // By the sentence's texts. Run alone, each step of the charlms runs one row through the same kernel, so the outputs
    // depend on the texts only: any other single-sentence call on the same texts would compute the same bits. (In a batch
    // a row's kernel depends on the other rows, which is why the entries above are not exact.)
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
    public bool TryGetArrays(Sentence sentence, out (float[] Forward, float[] Backward) reps)
    {
        if (!_reps.TryGetValue(sentence, out var e))
        {
            reps = default;
            return false;
        }
        reps = e.ForwardData != null ? (e.ForwardData, e.BackwardData!) : e.Native!.ToArrays();
        return true;
    }

    /// <summary>
    /// Drops <paramref name="sentence"/>'s entry once its last reader is done with it (NER, the pipeline's last
    /// processor), so a large document doesn't hold all of it to the end of the call. It doesn't make room for new ones.
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
        _words = 0;
    }
}

/// <summary>A backend's own form of a sentence's charlm representations (StanzaSharp.Cuda: two tensors).</summary>
internal interface ICharlmReps : IDisposable
{
    /// <summary>Copies of the [words, dim] representations, row-major.</summary>
    (float[] Forward, float[] Backward) ToArrays();
}
