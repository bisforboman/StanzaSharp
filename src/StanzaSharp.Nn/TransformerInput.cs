namespace StanzaSharp.Nn;

/// <summary>
/// One sentence's transformer input as Stanza builds it (stanza/models/common/bert_embedding.py,
/// extract_base_embeddings): the words through <see cref="WordPiece"/>, a word that yields no piece replaced by "-"
/// (fix_blank_tokens), and convert_to_position_list's positions. Like Stanza's tokenizer call it doesn't truncate:
/// <see cref="Ids"/> can be longer than the model's 512 positions, and the encoder then runs overlapping windows
/// (build_cloned_features).
/// </summary>
internal sealed record TransformerInput(string[] Words, int[] Ids, int[] Positions)
{
    /// <param name="words">The words a processor sends (pos and depparse: simplify_punct'ed; ner and sentiment: token texts).</param>
    public static TransformerInput Build(WordPiece tokenizer, IReadOnlyList<string> words)
    {
        var (ids, owners) = tokenizer.Encode(words);
        string[] text = [.. words];
        var counts = new int[words.Count];
        foreach (int owner in owners)
        {
            if (owner >= 0)
                counts[owner]++;
        }

        if (counts.Contains(0))
        {
            // fix_blank_tokens: every word whose own tokenization is [CLS] [SEP] becomes "-", and the sentence is
            // tokenized again (Stanza does this for the whole batch, but a word's pieces don't depend on its neighbours).
            for (int w = 0; w < text.Length; w++)
            {
                if (counts[w] == 0)
                    text[w] = "-";
            }

            (ids, owners) = tokenizer.Encode(text);
        }

        return new TransformerInput(text, ids, PositionList(owners, words.Count));
    }

    // convert_to_position_list: [0] = [CLS], [w + 1] = word w's last piece, [n + 1] = one past the last word's last piece.
    private static int[] PositionList(int[] owners, int words)
    {
        var positions = new int[words + 2];
        Array.Fill(positions, -1);
        for (int p = 0; p < owners.Length; p++)
        {
            if (owners[p] >= 0)
                positions[owners[p] + 1] = p;
        }

        positions[0] = 0;
        for (int i = words; i >= 0; i--)
        {
            if (positions[i] >= 0)
            {
                positions[^1] = positions[i] + 1;
                break;
            }
        }

        if (positions.Contains(-1))
            throw new InvalidOperationException("A word has no transformer piece after fix_blank_tokens");
        return positions;
    }
}
