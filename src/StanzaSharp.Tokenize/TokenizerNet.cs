using System.Text.Json.Nodes;
using StanzaSharp.Nn;

namespace StanzaSharp.Tokenize;

/// <summary>The tokenizer's network, per <see cref="Backend"/>: <c>TokenizerNet</c> or <see cref="ManagedTokenizerNet"/>.</summary>
internal interface ITokenizerNet : IDisposable
{
    /// <param name="ids">[rows, width] character ids.</param>
    /// <param name="feats">[rows, width, feat_dim] features.</param>
    /// <param name="lengths">Per-row length both LSTMs are packed at, as model.forward does.</param>
    /// <returns>[rows, width, 5] log-probabilities, row-major.</returns>
    float[] Forward(long[] ids, float[] feats, int rows, int width, long[] lengths, CancellationToken ct);
}

