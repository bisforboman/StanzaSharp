using System.Text;

namespace StanzaSharp;

/// <summary>
/// A constituency tree in PTB bracket form, e.g. <c>(ROOT (S (NP (PRP He)) (VP (VBD left))))</c>.
/// Labels hold the raw text: a "(" word is "(" in the tree and prints as "-LRB-".
/// </summary>
public sealed class Tree(string label, IReadOnlyList<Tree>? children = null)
{
    public string Label { get; } = label;
    public IReadOnlyList<Tree> Children { get; } = children ?? [];

    public bool IsLeaf => Children.Count == 0;
    public bool IsPreterminal => Children.Count == 1 && Children[0].IsLeaf;

    public IEnumerable<Tree> Leaves() =>
        IsLeaf ? [this] : Children.SelectMany(c => c.Leaves());

    public override string ToString()
    {
        var sb = new StringBuilder();
        Write(sb);
        return sb.ToString();
    }

    // Like Stanza's Tree.__format__, brackets inside labels and words print as -LRB-/-RRB-.
    private static string Escape(string label) => label.Replace("(", "-LRB-").Replace(")", "-RRB-");

    private void Write(StringBuilder sb)
    {
        if (IsLeaf)
        {
            sb.Append(Escape(Label));
            return;
        }
        sb.Append('(').Append(Escape(Label));
        foreach (var child in Children)
            child.Write(sb.Append(' '));
        sb.Append(')');
    }

    /// <summary>Parses one bracketed tree. Labels and leaves are whitespace/bracket-delimited atoms.</summary>
    public static Tree Parse(string text)
    {
        int pos = 0;
        var tree = ParseNode(text, ref pos);
        SkipSpace(text, ref pos);
        if (pos != text.Length)
            throw new FormatException($"Trailing text after tree at {pos}: {text}");
        return tree;
    }

    private static Tree ParseNode(string text, ref int pos)
    {
        SkipSpace(text, ref pos);
        if (pos >= text.Length)
            throw new FormatException($"Unexpected end of tree: {text}");
        if (text[pos] != '(')
            return new Tree(ReadAtom(text, ref pos));

        pos++;
        SkipSpace(text, ref pos);
        var label = ReadAtom(text, ref pos);
        var children = new List<Tree>();
        while (true)
        {
            SkipSpace(text, ref pos);
            if (pos >= text.Length)
                throw new FormatException($"Unbalanced brackets in tree: {text}");
            if (text[pos] == ')')
            {
                pos++;
                return new Tree(label, children);
            }
            children.Add(ParseNode(text, ref pos));
        }
    }

    private static string ReadAtom(string text, ref int pos)
    {
        int start = pos;
        while (pos < text.Length && text[pos] is not ('(' or ')') && !char.IsWhiteSpace(text[pos]))
            pos++;
        if (pos == start)
            throw new FormatException($"Expected a label at {start}: {text}");
        return text[start..pos];
    }

    private static void SkipSpace(string text, ref int pos)
    {
        while (pos < text.Length && char.IsWhiteSpace(text[pos]))
            pos++;
    }
}
