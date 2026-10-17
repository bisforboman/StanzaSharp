using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>A batch's transformer word vectors (TransformerEmbedder.Embed, one [words, hidden] array per sentence) as one tensor.</summary>
internal static class TransformerRows
{
    /// <summary>[Σ words, <paramref name="hidden"/>], sentences in order.</summary>
    public static Tensor Tensor(IReadOnlyList<float[]> sentences, int hidden, Device device)
    {
        long total = 0;
        foreach (var s in sentences)
            total += s.Length;
        var flat = new float[total];
        long at = 0;
        foreach (var s in sentences)
        {
            s.CopyTo(flat, at);
            at += s.Length;
        }

        return tensor(flat, [total / hidden, hidden], device: device);
    }
}
