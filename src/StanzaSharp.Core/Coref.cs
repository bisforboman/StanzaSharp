namespace StanzaSharp;

/// <summary>
/// A coreference chain: the mentions of one entity across the document (Stanza's <c>CorefChain</c>, an item of
/// <see cref="Document.Coref"/>).
/// </summary>
/// <param name="Index">The chain's 0-based position in <see cref="Document.Coref"/>; CoNLL-U's <c>id</c> in <c>coref_chains=</c>.</param>
/// <param name="Mentions">The chain's mentions, in document order.</param>
/// <param name="RepresentativeText">The text of the representative mention, the longest one (ties broken by the most proper
/// nouns, then the earliest); <c>_</c> when every mention is a zero mention.</param>
/// <param name="RepresentativeIndex">The representative mention's index in <paramref name="Mentions"/>; null when every
/// mention is a zero mention.</param>
public sealed record CorefChain(int Index, IReadOnlyList<CorefMention> Mentions, string RepresentativeText, int? RepresentativeIndex);

/// <summary>
/// One mention of a <see cref="CorefChain"/>: a span of words within one sentence, or a zero mention (a dropped pronoun) on
/// an empty word (Stanza's <c>CorefMention</c>). Indices are 0-based, as in Stanza, unlike <see cref="Word.Id"/>.
/// </summary>
/// <param name="Sentence">The 0-based index of the mention's sentence in <see cref="Document.Sentences"/>.</param>
/// <param name="StartWord">The 0-based index of the mention's first word in <see cref="StanzaSharp.Sentence.Words"/>; for a
/// zero mention, its empty word's <see cref="Word.Id"/> (the word it follows, 0 before the first word).</param>
/// <param name="EndWord">The 0-based index just past the mention's last word; for a zero mention, <paramref name="StartWord"/>.</param>
/// <param name="EmptyIndex">For a zero mention, its empty word's <see cref="Word.EmptyIndex"/>; null for a span of words.</param>
public sealed record CorefMention(int Sentence, int StartWord, int EndWord, int? EmptyIndex = null)
{
    /// <summary>Whether the mention is a zero mention, on an empty word of <see cref="StanzaSharp.Sentence.EmptyWords"/>.</summary>
    public bool IsZero => EmptyIndex != null;
}

/// <summary>
/// A word's place in a <see cref="CorefChain"/> (Stanza's <c>CorefAttachment</c>, an item of <see cref="Word.CorefChains"/>):
/// one per mention of the chain that covers the word.
/// </summary>
/// <param name="Chain">The chain.</param>
/// <param name="IsStart">Whether the word is the mention's first word.</param>
/// <param name="IsEnd">Whether the word is the mention's last word.</param>
/// <param name="IsRepresentative">Whether the mention is the chain's representative mention (set on each of its words, as
/// Stanza does).</param>
public sealed record CorefAttachment(CorefChain Chain, bool IsStart, bool IsEnd, bool IsRepresentative)
{
    // dict_to_conll_text: "unit-", "start-", "end-" or "middle-", then "repr-" if representative, then "id" + the chain's index.
    internal string ToConllu() =>
        (IsStart && IsEnd ? "unit-" : IsStart ? "start-" : IsEnd ? "end-" : "middle-") + (IsRepresentative ? "repr-" : "") + "id" + Chain.Index;
}

/// <summary>The coreference parts of reading and writing a document: attachments, and chains rebuilt from them.</summary>
internal static class CorefLinks
{
    /// <summary>
    /// Document._attach_coref_mentions: every word (empty words too) gets an empty list, then each mention's words an
    /// attachment, a zero mention's empty word one marked start and end but not representative.
    /// </summary>
    internal static void Attach(Document doc, IReadOnlyList<CorefChain> chains)
    {
        var lists = new Dictionary<Word, List<CorefAttachment>>(ReferenceEqualityComparer.Instance);
        foreach (var sentence in doc.Sentences)
        {
            foreach (var word in sentence.Words.Concat(sentence.EmptyWords))
                lists[word] = [];
        }

        var words = new Dictionary<int, List<Word>>();
        foreach (var chain in chains)
        {
            for (int m = 0; m < chain.Mentions.Count; m++)
            {
                var mention = chain.Mentions[m];
                if (mention.Sentence < 0 || mention.Sentence >= doc.Sentences.Count)
                    throw new ArgumentException($"Chain {chain.Index}: the document has no sentence {mention.Sentence}");
                var sentence = doc.Sentences[mention.Sentence];
                if (mention.EmptyIndex is int empty)
                {
                    var word = sentence.EmptyWords.FirstOrDefault(w => w.Id == mention.StartWord && w.EmptyIndex == empty)
                        ?? throw new ArgumentException($"Chain {chain.Index}: sentence {mention.Sentence} has no empty word {mention.StartWord}.{empty}");
                    lists[word].Add(new CorefAttachment(chain, true, true, false));
                    continue;
                }

                if (!words.TryGetValue(mention.Sentence, out var list))
                    words[mention.Sentence] = list = [.. sentence.Words];
                if (mention.StartWord < 0 || mention.EndWord > list.Count || mention.StartWord >= mention.EndWord)
                    throw new ArgumentException($"Chain {chain.Index}: words {mention.StartWord}..{mention.EndWord} are not in sentence {mention.Sentence}");
                for (int w = mention.StartWord; w < mention.EndWord; w++)
                    lists[list[w]].Add(new CorefAttachment(chain, w == mention.StartWord, w == mention.EndWord - 1, m == chain.RepresentativeIndex));
            }
        }

        foreach (var (word, list) in lists)
            word.CorefChains = list;
    }

    /// <summary>One attachment as CoNLL-U writes it, e.g. <c>start-repr-id3</c>: (position, representative, chain index).</summary>
    internal static (bool Start, bool End, bool Representative, int Index) ParseConllu(string piece)
    {
        var rest = piece.AsSpan();
        (bool start, bool end) = rest.StartsWith("unit-") ? (true, true)
            : rest.StartsWith("start-") ? (true, false)
            : rest.StartsWith("end-") ? (false, true)
            : rest.StartsWith("middle-") ? (false, false)
            : throw new FormatException($"Not a coref attachment: {piece}");
        rest = rest[(rest.IndexOf('-') + 1)..];
        bool representative = rest.StartsWith("repr-");
        if (representative)
            rest = rest[5..];
        if (!rest.StartsWith("id") || !int.TryParse(rest[2..], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int index))
            throw new FormatException($"Not a coref attachment: {piece}");
        return (start, end, representative, index);
    }

    /// <summary>
    /// The chains a document's attachments describe (CoNLL-U's <c>coref_chains=</c> or JSON's <c>coref_chains</c>, read into
    /// <paramref name="read"/> per word): each chain's mentions from its start and end marks, a word marked start and end
    /// being a mention by itself and an empty word a zero mention; mentions in document order; the representative mention
    /// the one whose words are marked representative, its text rebuilt as Stanza's extract_text builds it
    /// (<see cref="MentionText"/>). Null when no word has an attachment list.
    /// </summary>
    internal static List<CorefChain>? Rebuild(Document doc, IReadOnlyDictionary<Word, List<(bool Start, bool End, bool Representative, int Index)>> read)
    {
        if (read.Count == 0)
            return null;

        var mentions = new SortedDictionary<int, List<(CorefMention Mention, bool Representative)>>();
        for (int s = 0; s < doc.Sentences.Count; s++)
        {
            var sentence = doc.Sentences[s];
            var words = sentence.Words.ToList();
            var open = new Dictionary<int, Stack<int>>();
            foreach (var (word, position) in AllWords(sentence, words))
            {
                if (!read.TryGetValue(word, out var pieces))
                    continue;
                foreach (var (start, end, representative, index) in pieces)
                {
                    if (!mentions.TryGetValue(index, out var list))
                        mentions[index] = list = [];
                    if (word.EmptyIndex != null)
                    {
                        list.Add((new CorefMention(s, word.Id, word.Id, word.EmptyIndex), false));
                        continue;
                    }

                    // Each mention's last word says whether it is the representative one (every word of it is marked).
                    if (start)
                    {
                        if (!open.TryGetValue(index, out var stack))
                            open[index] = stack = new();
                        stack.Push(position);
                    }

                    if (end)
                    {
                        if (!open.TryGetValue(index, out var stack) || stack.Count == 0)
                            throw new FormatException($"Sentence {s}: chain {index} ends a mention at word {word.Id} that no word started");
                        list.Add((new CorefMention(s, stack.Pop(), position + 1), representative));
                    }
                }
            }

            if (open.FirstOrDefault(o => o.Value.Count > 0) is { Value: not null } unclosed)
                throw new FormatException($"Sentence {s}: chain {unclosed.Key} starts a mention that no word ends");
        }

        var chains = new List<CorefChain>(mentions.Count);
        foreach (var (index, list) in mentions)
        {
            var ordered = list.OrderBy(m => m.Mention.Sentence).ThenBy(m => m.Mention.StartWord).ThenBy(m => m.Mention.EndWord).ThenBy(m => m.Mention.EmptyIndex ?? 0).ToList();
            int r = ordered.FindIndex(m => m.Representative);
            var text = r < 0 ? "_" : MentionText(doc.Sentences[ordered[r].Mention.Sentence], ordered[r].Mention);
            chains.Add(new CorefChain(index, [.. ordered.Select(m => m.Mention)], text, r < 0 ? null : r));
        }

        return chains;
    }

    // Sentence.all_words: the words and the empty words in id order, an empty word i.m after word i; each word's 0-based
    // index (an empty word's is unused).
    private static IEnumerable<(Word Word, int Position)> AllWords(Sentence sentence, List<Word> words)
    {
        var empties = sentence.EmptyWords.OrderBy(w => w.Id).ThenBy(w => w.EmptyIndex).ToList();
        int e = 0;
        while (e < empties.Count && empties[e].Id == 0)
            yield return (empties[e++], -1);
        for (int i = 0; i < words.Count; i++)
        {
            yield return (words[i], i);
            while (e < empties.Count && empties[e].Id == words[i].Id)
                yield return (empties[e++], -1);
        }

        while (e < empties.Count)
            yield return (empties[e++], -1);
    }

    /// <summary>
    /// coref_processor.extract_text: the mention's words, a token's text in place of its words when the mention covers the
    /// whole token, joined by single spaces. Stanza means to join them by each piece's spacing, but reads it from the
    /// token's <c>misc</c>, which holds none in its documents (the spacing lives in <c>spaces_after</c>), so
    /// <c>misc_to_space_after</c> always gives one space: <c>$ 4.50</c>, <c>HARBORVIEW , Ore.</c>.
    /// </summary>
    internal static string MentionText(Sentence sentence, CorefMention mention)
    {
        var pieces = new List<string>();
        int w = 0;
        foreach (var token in sentence.Tokens)
        {
            int first = w, last = w + token.Words.Count - 1;
            w += token.Words.Count;
            if (last < mention.StartWord || first >= mention.EndWord)
                continue;
            if (!token.IsMultiWord || (first >= mention.StartWord && last < mention.EndWord))
            {
                pieces.Add(token.Text);
                continue;
            }

            for (int i = Math.Max(first, mention.StartWord); i <= Math.Min(last, mention.EndWord - 1); i++)
                pieces.Add(token.Words[i - first].Text);
        }

        return string.Join(' ', pieces);
    }
}
