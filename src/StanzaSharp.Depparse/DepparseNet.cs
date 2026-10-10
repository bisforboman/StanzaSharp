using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Depparse;

/// <summary>
/// One parser batch: row b of each [size, width] id array is ROOT (id 3) then sentence b's words, 0 past its length.
/// <see cref="Texts"/> are the words after simplify_punct, without ROOT. <see cref="Charlms"/> (optional): the tagger's charlm
/// outputs (see <see cref="DependencyParser.Process"/>).
/// </summary>
internal sealed record DepparseBatch(IReadOnlyList<IReadOnlyList<string>> Texts, int Width, long[] Lengths,
    long[] Word, long[] Lemma, long[] Upos, long[] Xpos, long[] Pretrained, CharlmCache? Charlms = null);

/// <summary>
/// A batch's scores, [size, width, width] with [b, i, j] = dependent i, head j (0 is ROOT):
/// <see cref="ArcLogProbs"/> is the log-softmax over heads, padding columns included as in Stanza (defined for rows
/// i &lt; length); <see cref="Labels"/> the argmax relation (without the vocab prefix) and, if asked for,
/// <see cref="LabelScores"/> [size, width, width, relations] (both defined for i, j &lt; length).
/// </summary>
internal sealed record DepparseScores(int Width, int Relations, float[] ArcLogProbs, int[] Labels, float[]? LabelScores);

/// <summary>The parser's network, per <see cref="Backend"/>: <c>DepparseNet</c> or <see cref="ManagedDepparseNet"/>.</summary>
internal interface IDepparseNet : IDisposable
{
    /// <summary>GraphParser.forward_scores, then predict's log-softmax over heads and label argmax.</summary>
    /// <param name="labelScores">Also return every pair's label scores (tests).</param>
    /// <param name="ct">Checked after each charlm pass (or the character model), inside the LSTM and between the scorers' chunks.</param>
    DepparseScores Forward(DepparseBatch batch, bool labelScores, CancellationToken ct);
}

