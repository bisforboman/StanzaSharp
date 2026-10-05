using System.Text;

namespace StanzaSharp;

/// <summary>
/// CoNLL-U reading and writing in the dialect Stanza produces: token offsets as
/// <c>start_char</c>/<c>end_char</c> in MISC, whitespace as <c>SpaceAfter=No</c> / <c>SpacesAfter=</c>
/// on the token line, and the parse as a <c># constituency =</c> comment.
/// </summary>
public static class Conllu
{
    public static Document Read(string text)
    {
        var doc = new Document();
        Sentence? sent = null;
        Token? mwt = null;
        int mwtEnd = 0;

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
            if (cols[0].Contains('.'))
                continue; // empty nodes (enhanced UD) are not produced by our pipeline

            var word = new Word
            {
                Id = int.Parse(cols[0]),
                Text = cols[1],
                Lemma = Field(cols[2]),
                Upos = Field(cols[3]),
                Xpos = Field(cols[4]),
                Feats = Field(cols[5]),
                Head = cols[6] == "_" ? null : int.Parse(cols[6]),
                Deprel = Field(cols[7]),
                StartChar = misc.Start,
                EndChar = misc.End,
            };
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
        return doc;
    }

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
        // Stanza writes ' '.join(text.split()): every whitespace run in the comment becomes one space.
        if (sent.Text != null) sb.Append("# text = ").AppendJoin(' ', sent.Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).Append('\n');
        if (sent.SentId != null) sb.Append("# sent_id = ").Append(sent.SentId).Append('\n');
        if (sent.Constituency != null) sb.Append("# constituency = ").Append(sent.Constituency).Append('\n');

        foreach (var token in sent.Tokens)
        {
            if (token.IsMultiWord)
            {
                sb.Append(token.Words[0].Id).Append('-').Append(token.Words[^1].Id).Append('\t')
                  .Append(token.Text).Append("\t_\t_\t_\t_\t_\t_\t_\t")
                  .Append(Misc(token.SpaceAfter, token.SpacesBefore, token.StartChar, token.EndChar)).Append('\n');
            }
            foreach (var w in token.Words)
            {
                var (after, before) = token.IsMultiWord ? (" ", "") : (token.SpaceAfter, token.SpacesBefore);
                sb.Append(w.Id).Append('\t').Append(w.Text).Append('\t')
                  .Append(w.Lemma ?? "_").Append('\t').Append(w.Upos ?? "_").Append('\t')
                  .Append(w.Xpos ?? "_").Append('\t').Append(SortFeats(w.Feats) ?? "_").Append('\t')
                  // Like Stanza, a word without a head gets the dummy head id - 1 (the UD eval script needs an int).
                  .Append(w.Head ?? w.Id - 1).Append('\t').Append(w.Deprel ?? "_").Append("\t_\t")
                  .Append(Misc(after, before, w.StartChar, w.EndChar)).Append('\n');
            }
        }
    }

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
        }
    }

    private static string? Field(string value) => value == "_" ? null : value;

    /// <summary>Stanza's writer sorts UFeats case-insensitively (key=str.casefold).</summary>
    private static string? SortFeats(string? feats) =>
        feats == null ? null : string.Join('|', feats.Split('|').OrderBy(p => p.ToLowerInvariant(), StringComparer.Ordinal));

    private readonly record struct MiscFields(string Space, string SpacesBefore, int? Start, int? End);

    private static MiscFields ParseMisc(string misc)
    {
        string space = " ", before = "";
        int? start = null, end = null;
        if (misc == "_")
            return new(space, before, start, end);
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
            }
        }
        return new(space, before, start, end);
    }

    private static void ApplyTokenMisc(Token token, MiscFields misc)
    {
        token.SpaceAfter = misc.Space;
        token.SpacesBefore = misc.SpacesBefore;
        token.StartChar = misc.Start;
        token.EndChar = misc.End;
    }

    // Stanza sorts the space pieces (SpaceAfter/SpacesAfter sort before SpacesBefore), then adds the offsets.
    private static string Misc(string spaceAfter, string spacesBefore, int? start, int? end)
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
                if (escaped.AsSpan(i).StartsWith("\\u00A0")) { sb.Append(' '); i += 5; continue; }
                char? c = escaped[i + 1] switch
                {
                    's' => ' ', 't' => '\t', 'r' => '\r', 'n' => '\n', 'p' => '|', '\\' => '\\', _ => null,
                };
                if (c != null) { sb.Append(c.Value); i++; continue; }
            }
            sb.Append(escaped[i]);
        }
        return sb.ToString();
    }
}
