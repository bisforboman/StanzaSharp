using static TorchSharp.torch;

namespace StanzaSharp.Constituency;

internal enum TransitionKind { Shift, Close, Open }

/// <param name="Index">Position in the model's transition list (the output layer's columns).</param>
/// <param name="OpenIndex">For Open: index into constituent_opens (the open/dummy embedding row).</param>
internal sealed record Transition(int Index, TransitionKind Kind, string? Label = null, int OpenIndex = -1);

/// <summary>
/// A persistent stack whose nodes also carry an LSTM state, like Stanza's TreeStack of
/// lstm_tree_stack.Node: pushing runs the LSTM one step from the parent's state.
/// </summary>
internal sealed class StackNode<T>(T value, StackNode<T>? parent, Tensor hx, Tensor cx, Tensor output)
{
    public T Value { get; } = value;
    public StackNode<T>? Parent { get; } = parent;
    public int Length { get; } = (parent?.Length ?? 0) + 1;

    /// <summary>[layers, hidden]</summary>
    public Tensor Hx { get; } = hx;
    public Tensor Cx { get; } = cx;

    /// <summary>[hidden]: the last layer's output.</summary>
    public Tensor Output { get; } = output;
}

/// <summary>
/// An item on the constituent stack: a finished subtree, or the marker left by an Open
/// transition (Stanza's Dummy). <see cref="Hx"/> is its [hidden] vector.
/// </summary>
internal sealed class Constituent(Tree? tree, string? openLabel, Tensor? hx)
{
    public Tree? Tree { get; } = tree;
    public string? OpenLabel { get; } = openLabel;
    public bool IsOpenMarker => OpenLabel != null;
    public Tensor? Hx { get; } = hx;
}

/// <summary>Parsing state for one sentence (stanza/models/constituency/state.py), mutated in place.</summary>
internal sealed class ParserState
{
    public required int SentenceLength;
    public required Tree[] Preterminals;
    /// <summary>[SentenceLength + 2, hidden]: start sentinel, one row per word, end sentinel.</summary>
    public required Tensor WordHx;
    public required StackNode<Transition?> Transitions;
    public required StackNode<Constituent> Constituents;
    public int WordPosition;
    public int NumOpens;
    public bool Broken;

    public bool EmptyWordQueue => WordPosition == SentenceLength;
    public bool EmptyConstituents => Constituents.Parent == null;
    public bool HasOneConstituent => Constituents.Length == 2;
    public int NumTransitions => Transitions.Length - 1;
    public Transition? TopTransition => Transitions.Value;

    public bool Finished(IReadOnlySet<string> rootLabels) =>
        EmptyWordQueue && HasOneConstituent && Constituents.Value.Tree is { } tree && rootLabels.Contains(tree.Label);

    /// <summary>The IN_ORDER rules of Shift/OpenConstituent/CloseConstituent.is_legal in parse_transitions.py.</summary>
    public bool IsLegal(Transition t, IReadOnlySet<string> rootLabels, int unaryLimit)
    {
        switch (t.Kind)
        {
            case TransitionKind.Shift:
                if (EmptyWordQueue)
                    return false;
                return NumOpens != 0 || EmptyConstituents;

            case TransitionKind.Open:
                if (NumOpens > SentenceLength + 10 || EmptyConstituents || TopTransition?.Kind == TransitionKind.Open)
                    return false;
                if (rootLabels.Contains(t.Label!))
                    return NumOpens == 0 && EmptyWordQueue;
                return !((NumOpens > 0 || EmptyWordQueue) && TooManyUnaryNodes(Constituents.Value.Tree, unaryLimit));

            case TransitionKind.Close:
                if (NumOpens <= 0)
                    return false;
                if (TopTransition?.Kind != TransitionKind.Open || NumOpens > 1 || EmptyWordQueue)
                    return true;
                return !TooManyUnaryNodes(Constituents.Parent!.Value.Tree, unaryLimit);

            default:
                throw new ArgumentOutOfRangeException(nameof(t));
        }
    }

    /// <summary>True if the tree starts with more than <paramref name="limit"/> unary nodes in a row.</summary>
    private static bool TooManyUnaryNodes(Tree? tree, int limit)
    {
        if (tree == null)
            return false;
        for (int i = 0; i <= limit; i++)
        {
            if (tree.Children.Count != 1)
                return false;
            tree = tree.Children[0];
        }
        return true;
    }
}
