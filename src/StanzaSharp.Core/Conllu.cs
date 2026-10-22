using System.Text;
using System.Text.RegularExpressions;

namespace StanzaSharp;

/// <summary>
/// CoNLL-U reading and writing in the dialect Stanza produces: token offsets as
/// <c>start_char</c>/<c>end_char</c> in MISC, whitespace as <c>SpaceAfter=No</c> / <c>SpacesAfter=</c>
/// on the token line, NER tags as <c>ner=</c> in MISC, coreference as <c>coref_chains=</c> in MISC, empty nodes
/// (<c>i.m</c>) as <see cref="Sentence.EmptyWords"/>, and the parse as a <c># constituency =</c> comment.
/// </summary>
public static class Conllu
{
    /// <summary>
    /// Parses CoNLL-U text into a document: tokens, multi-word tokens, words with their columns, offsets and
    /// spacing from MISC, empty nodes (<c>i.m</c>), and the <c>text</c>, <c>sent_id</c>, <c>constituency</c> and
    /// <c>sentiment</c> comments. Entities are rebuilt from <c>ner=</c> tags, and <see cref="Document.Coref"/> from the
    /// <c>coref_chains=</c> marks (which Stanza writes but does not read): each chain's mentions from their first and last
    /// words' marks, in document order, the representative one from its <c>repr</c> marks, its text from its words as Stanza
    /// builds it (tokens and words joined by single spaces). A zero mention is placed by its empty word; the coref processor
    /// places it by the span the model predicted, which the marks don't record, so the two orders differ only when a chain has
    /// a mention starting inside that span. <see cref="Document.Text"/> stays null.
    /// </summary>
    /// <exception cref="FormatException">A line is not CoNLL-U, or a <c>coref_chains=</c> mark is malformed or ends a mention
    /// no mark started.</exception>
    public static Document Read(string text)
    {
        var doc = new Document();
        Sentence? sent = null;
        Token? mwt = null;
        int mwtEnd = 0;
        var coref = new Dictionary<Word, List<(bool, bool, bool, int)>>(ReferenceEqualityComparer.Instance);

        foreach (var raw in text.Split('\n'))
        {
            var line = raw.TrimEnd('\r');
            if (line.Length == 0)
            {
                sent = null;
                continue;
            }

            if (sent == null)
            {
                sent = new Sentence();
                doc.Sentences.Add(sent);
            }

            if (line.StartsWith('#'))
            {
                ReadComment(sent, line);
                continue;
            }

            var cols = line.Split('\t');
            if (cols.Length != 10)
                throw new FormatException($"Expected 10 columns: {line}");
            var misc = ParseMisc(cols[9]);

            int dash = cols[0].IndexOf('-');
            if (dash > 0)
            {
                mwt = new Token { Text = cols[1] };
                ApplyTokenMisc(mwt, misc);
                mwtEnd = int.Parse(cols[0][(dash + 1)..]);
                sent.Tokens.Add(mwt);
                continue;
            }

            int dot = cols[0].IndexOf('.');
            var word = new Word
            {
                Id = int.Parse(dot > 0 ? cols[0][..dot] : cols[0]),
                EmptyIndex = dot > 0 ? int.Parse(cols[0][(dot + 1)..]) : null,
                Text = cols[1],
                Lemma = cols[1] == "_" ? cols[2] : Field(cols[2]), // Word.lemma keeps "_" for a word whose text is "_" (an empty word's)
                Upos = Field(cols[3]),
                Xpos = Field(cols[4]),
                Feats = Field(cols[5]),
                Head = cols[6] == "_" ? null : int.Parse(cols[6]),
                Deprel = Field(cols[7]),
                StartChar = misc.Start,
                EndChar = misc.End,
            };
            if (misc.Coref != null)
                coref[word] = [.. misc.Coref.Split(',').Select(CorefLinks.ParseConllu)];
            if (word.EmptyIndex != null)
            {
                sent.EmptyWords.Add(word);
                continue;
            }

            if (mwt != null && word.Id <= mwtEnd)
            {
                mwt.Words.Add(word);
                if (word.Id == mwtEnd)
                    mwt = null;
            }
            else
            {
                var token = new Token { Text = word.Text };
                ApplyTokenMisc(token, misc);
                token.Words.Add(word);
                sent.Tokens.Add(token);
            }
        }

        foreach (var s in doc.Sentences.Where(s => s.Tokens.Any(t => t.Ner != null)))
            Entity.Build(s, doc.Text);
        if (CorefLinks.Rebuild(doc, coref) is { } chains)
            doc.Coref = chains;
        return doc;
    }

    /// <summary>Formats a document as CoNLL-U, byte for byte as Stanza's <c>"{:C}".format(doc)</c> writes it (plus a final newline).</summary>
    public static string Write(Document doc)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < doc.Sentences.Count; i++)
        {
            if (i > 0)
                sb.Append('\n');
            WriteSentence(sb, doc.Sentences[i]);
        }

        return sb.ToString();
    }

    private static void WriteSentence(StringBuilder sb, Sentence sent)
    {
        // Stanza writes ' '.join(text.split()): every run of Python whitespace (which, unlike .NET's,
        // includes U+001C..U+001F) in the comment becomes one space.
        if (sent.Text != null) sb.Append("# text = ").AppendJoin(' ', PyWhitespace.Split(sent.Text).Where(w => w.Length > 0)).Append('\n');
        if (sent.SentId != null) sb.Append("# sent_id = ").Append(sent.SentId).Append('\n');
        if (sent.Constituency != null) sb.Append("# constituency = ").Append(sent.Constituency).Append('\n');

        // Stanza runs sentiment after constituency, and its comments keep the order they were set in.
        if (sent.Sentiment != null) sb.Append("# sentiment = ").Append(sent.Sentiment.Value).Append('\n');

        // Sentence._ordered_dicts: an empty word i.m right after word i (inside a multi-word token too), i = 0 first.
        var empties = sent.EmptyWords.OrderBy(w => w.Id).ThenBy(w => w.EmptyIndex).ToList();
        int e = 0;
        while (e < empties.Count && empties[e].Id == 0)
            WriteEmpty(sb, empties[e++]);

        foreach (var token in sent.Tokens)
        {
            if (token.IsMultiWord)
            {
                sb.Append(token.Words[0].Id).Append('-').Append(token.Words[^1].Id).Append('\t')
                  .Append(token.Text)
                  .Append("\t_\t_\t_\t_\t_\t_\t_\t")
                  .Append(Misc(token.SpaceAfter, token.SpacesBefore, token.StartChar, token.EndChar, token.Ner, null))
                  .Append('\n');
            }

            foreach (var w in token.Words)
            {
                var (after, before) = token.IsMultiWord ? (" ", "") : (token.SpaceAfter, token.SpacesBefore);
                sb.Append(w.Id).Append('\t').Append(w.Text).Append('\t')
                  .Append(w.Lemma ?? "_")
                  .Append('\t')
                  .Append(w.Upos ?? "_")
                  .Append('\t')
                  .Append(w.Xpos ?? "_")
                  .Append('\t')
                  .Append(SortFeats(w.Feats) ?? "_")
                  .Append('\t')

                  // Like Stanza, a word without a head gets the dummy head id - 1 (the UD eval script needs an int).
                  .Append(w.Head ?? w.Id - 1)
                  .Append('\t')
                  .Append(w.Deprel ?? "_")
                  .Append("\t_\t")

                  // The NER tag belongs to the token: on the range line of an MWT, on the word otherwise.
                  .Append(Misc(after, before, w.StartChar, w.EndChar, token.IsMultiWord ? null : token.Ner, w.CorefChains))
                  .Append('\n');
                while (e < empties.Count && empties[e].Id == w.Id)
                    WriteEmpty(sb, empties[e++]);
            }
        }

        while (e < empties.Count)
            WriteEmpty(sb, empties[e++]);
    }

    // dict_to_conll_text for an empty word: "i.m", its fields, no dummy head, no spacing (it has no token).
    private static void WriteEmpty(StringBuilder sb, Word w) =>
        sb.Append(w.Id).Append('.').Append(w.EmptyIndex).Append('\t').Append(w.Text).Append('\t')
          .Append(w.Lemma ?? "_")
          .Append('\t')
          .Append(w.Upos ?? "_")
          .Append('\t')
          .Append(w.Xpos ?? "_")
          .Append('\t')
          .Append(SortFeats(w.Feats) ?? "_")
          .Append('\t')
          .Append(w.Head?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "_")
          .Append('\t')
          .Append(w.Deprel ?? "_")
          .Append("\t_\t")
          .Append(Misc(" ", "", w.StartChar, w.EndChar, null, w.CorefChains))
          .Append('\n');

    private static readonly Regex PyWhitespace = new(@"[\s\x1c-\x1f]+");

    private static void ReadComment(Sentence sent, string line)
    {
        int eq = line.IndexOf(" = ", StringComparison.Ordinal);
        if (eq < 0)
            return;
        var key = line[1..eq].Trim();
        var value = line[(eq + 3)..];
        switch (key)
        {
            case "text": sent.Text = value; break;
            case "sent_id": sent.SentId = value; break;
            case "constituency": sent.Constituency = Tree.Parse(value); break;
            case "sentiment": sent.Sentiment = int.Parse(value); break;
        }
    }

    private static string? Field(string value) => value == "_" ? null : value;

    /// <summary>Stanza's writer sorts UFeats case-insensitively (key=str.casefold).</summary>
    private static string? SortFeats(string? feats) =>
        feats == null ? null : string.Join('|', feats.Split('|').OrderBy(p => p.ToLowerInvariant(), StringComparer.Ordinal));

    internal readonly record struct MiscFields(string Space, string SpacesBefore, int? Start, int? End, string? Ner, string? Coref = null);

    internal static MiscFields ParseMisc(string misc)
    {
        string space = " ", before = "";
        int? start = null, end = null;
        string? ner = null, coref = null;
        if (misc == "_")
            return new(space, before, start, end, ner);
        foreach (var piece in misc.Split('|'))
        {
            int eq = piece.IndexOf('=');
            if (eq < 0)
                continue;
            var value = piece[(eq + 1)..];
            switch (piece[..eq])
            {
                case "SpaceAfter" when value == "No": space = ""; break;
                case "SpacesAfter": space = UnescapeSpace(value); break;
                case "SpacesBefore": before = UnescapeSpace(value); break;
                case "start_char": start = int.Parse(value); break;
                case "end_char": end = int.Parse(value); break;
                case "ner": ner = value; break;
                case "coref_chains": coref = value; break;
            }
        }

        return new(space, before, start, end, ner, coref);
    }

    private static void ApplyTokenMisc(Token token, MiscFields misc)
    {
        token.SpaceAfter = misc.Space;
        token.SpacesBefore = misc.SpacesBefore;
        token.StartChar = misc.Start;
        token.EndChar = misc.End;
        token.Ner = misc.Ner;
    }

    // Stanza sorts the space pieces (SpaceAfter/SpacesAfter sort before SpacesBefore), then adds the offsets, the NER tag and
    // the coref marks (none written for a word in no mention).
    internal static string Misc(string spaceAfter, string spacesBefore, int? start, int? end, string? ner, IReadOnlyList<CorefAttachment>? coref = null)
    {
        var pieces = new List<string>(4);
        if (spaceAfter.Length == 0)
            pieces.Add("SpaceAfter=No");
        else if (spaceAfter != " ")
            pieces.Add("SpacesAfter=" + EscapeSpace(spaceAfter));
        if (spacesBefore.Length > 0)
            pieces.Add("SpacesBefore=" + EscapeSpace(spacesBefore));
        if (start != null) pieces.Add($"start_char={start}");
        if (end != null) pieces.Add($"end_char={end}");
        if (ner != null) pieces.Add("ner=" + ner);
        if (coref is { Count: > 0 }) pieces.Add("coref_chains=" + string.Join(',', coref.Select(a => a.ToConllu())));
        return pieces.Count == 0 ? "_" : string.Join('|', pieces);
    }

    // Same escapes as stanza.models.common.utils.escape_misc_space / unescape_misc_space.
    private static string EscapeSpace(string space)
    {
        var sb = new StringBuilder();
        foreach (var c in space)
        {
            sb.Append(c switch
            {
                ' ' => "\\s",
                '\t' => "\\t",
                '\r' => "\\r",
                '\n' => "\\n",
                '|' => "\\p",
                '\\' => "\\\\",
                ' ' => "\\u00A0",
                _ => c.ToString(),
            });
        }

        return sb.ToString();
    }

    private static string UnescapeSpace(string escaped)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < escaped.Length; i++)
        {
            if (escaped[i] == '\\' && i + 1 < escaped.Length)
            {
                if (escaped.AsSpan(i).StartsWith("\\u00A0"))
                {
                    sb.Append(' ');
                    i += 5;
                    continue;
                }

                char? c = escaped[i + 1] switch
                {
                    's' => ' ',
                    't' => '\t',
                    'r' => '\r',
                    'n' => '\n',
                    'p' => '|',
                    '\\' => '\\',
                    _ => null,
                };
                if (c != null)
                {
                    sb.Append(c.Value);
                    i++;
                    continue;
                }
            }

            sb.Append(escaped[i]);
        }

        return sb.ToString();
    }
}
