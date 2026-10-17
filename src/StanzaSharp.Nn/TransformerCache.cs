namespace StanzaSharp.Nn;

/// <summary>
/// One call's transformer states per id sequence, so the processors that read the same frozen transformer (pos, depparse
/// and ner of <c>default_accurate</c>, docs/transformer-plan.md) run it once per sentence: pos and depparse send the same
/// simplify_punct'ed words (so the same ids), ner the token texts, equal whenever a sentence has no multi-word token and no
/// punctuation run. Stanza runs the encoder once per processor; sharing is exact because a sentence's states don't depend
/// on its batch (bitwise on the managed backend; TorchSharp pads, which moves only the last bits).
/// </summary>
/// <param name="net">The encoder whose states it holds: <see cref="TransformerEmbedder.Embed"/> ignores the cache for any other
/// (sentiment's LoRA encoder gives other states for the same ids).</param>
/// <remarks>
/// An entry is the last <see cref="Layers"/> layers of every piece, [Layers, pieces, hidden] (16 KB a piece with ELECTRA-large),
/// so each reader mixes them its own way. At most <see cref="MaxPieces"/> pieces are kept (sentences past that are not, and
/// their readers run the encoder again), so a long document can't hold gigabytes between the processors.
/// </remarks>
internal sealed class TransformerCache(ITransformerNet net, int maxPieces = TransformerCache.DefaultMaxPieces)
{
    public ITransformerNet Net { get; } = net;

    /// <summary>Every reader's mix reads at most the last 4 layers (the legacy mix three of them).</summary>
    public const int Layers = 4;

    /// <summary>About 256 MB with ELECTRA-large, like <see cref="CharlmCache.DefaultMaxWords"/>.</summary>
    public const int DefaultMaxPieces = 16_384;

    private readonly Dictionary<int[], float[]> _states = new(IdsComparer.Instance);
    private long _pieces;

    public int MaxPieces { get; } = maxPieces;

    public bool TryGet(int[] ids, out float[] states) => _states.TryGetValue(ids, out states!);

    /// <summary>Keeps <paramref name="states"/> ([<see cref="Layers"/>, ids, hidden]) if they fit; the caller must not change them afterwards.</summary>
    public bool TryAdd(int[] ids, float[] states)
    {
        if (_states.ContainsKey(ids) || _pieces + ids.Length > MaxPieces)
            return false;
        _states[ids] = states;
        _pieces += ids.Length;
        return true;
    }

    private sealed class IdsComparer : IEqualityComparer<int[]>
    {
        public static readonly IdsComparer Instance = new();

        public bool Equals(int[]? x, int[]? y) => x.AsSpan().SequenceEqual(y);

        public int GetHashCode(int[] ids)
        {
            var hash = default(HashCode);
            hash.AddBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(ids.AsSpan()));
            return hash.ToHashCode();
        }
    }
}
