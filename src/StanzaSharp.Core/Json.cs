using System.Text;
using System.Text.Json;

namespace StanzaSharp;

/// <summary>
/// JSON in Stanza's format: <c>str(doc)</c>, a list of sentences, each a list of token and word objects with
/// <c>id</c>, <c>text</c>, <c>lemma</c>, <c>upos</c>, <c>xpos</c>, <c>feats</c>, <c>head</c>, <c>deprel</c>,
/// <c>start_char</c>, <c>end_char</c>, <c>ner</c>, <c>multi_ner</c> and <c>misc</c> (the spacing). Like Stanza's,
/// it holds no sentence-level results: no text, sentence id, constituency tree or sentiment (use <see cref="Conllu"/>).
/// </summary>
public static class Json
{
    /// <summary>
    /// Formats a document byte for byte as Python Stanza's <c>str(doc)</c> (<c>json.dumps(doc.to_dict(), indent=2,
    /// ensure_ascii=False)</c>): a field is left out when it has no value, a multi-word token comes before its words
    /// with its range as <c>"id": [first, last]</c>, and a token's NER tag goes on its line (a single word's for a
    /// one-word token). No final newline.
    /// </summary>
    /// <example>
    /// <code>
    /// File.WriteAllText("doc.json", Json.Write(nlp.Process("Barack Obama was born in Hawaii.")));
    /// </code>
    /// </example>
    public static string Write(Document doc)
    {
        ArgumentNullException.ThrowIfNull(doc);
        var sb = new StringBuilder();
        List(sb, 0, doc.Sentences, (sent, depth) => List(sb, depth, Entries(sent), (entry, d) => Entry(sb, d, entry)));
        return sb.ToString();
    }

    /// <summary>
    /// Parses Stanza's JSON (<see cref="Write"/>'s output, or Python's <c>str(doc)</c>) into a document, as Stanza's
    /// <c>Document(json.loads(text))</c> does: tokens, multi-word tokens, words with their fields, and the spacing from
    /// <c>misc</c>. Sentence ids are the sentences' indices, as Stanza numbers them; entities are rebuilt from the
    /// <c>ner</c> tags, as <see cref="Conllu.Read"/> does. <see cref="Document.Text"/> and <see cref="Sentence.Text"/>
    /// stay null. <c>Json.Write(Json.Read(json))</c> equals <c>json</c> for any <see cref="Write"/> output.
    /// </summary>
    /// <exception cref="JsonException">Not valid JSON, or not a list of lists of objects with <c>id</c> and
    /// <c>text</c>.</exception>
    public static Document Read(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var parsed = JsonDocument.Parse(json);
        var doc = new Document();
        try
        {
            foreach (var entries in parsed.RootElement.EnumerateArray())
                doc.Sentences.Add(ReadSentence(entries, doc.Sentences.Count));
        }
        catch (Exception e) when (e is InvalidOperationException or KeyNotFoundException or FormatException)
        {
            throw new JsonException("Not a Stanza document: " + e.Message, e);
        }

        foreach (var s in doc.Sentences.Where(s => s.Tokens.Any(t => t.Ner != null)))
            Entity.Build(s, doc.Text);
        return doc;
    }

    private static Sentence ReadSentence(JsonElement entries, int index)
    {
        var sent = new Sentence { SentId = index.ToString(System.Globalization.CultureInfo.InvariantCulture) };
        Token? mwt = null;
        int mwtEnd = 0, lastWord = 0;
        foreach (var entry in entries.EnumerateArray())
        {
            var id = entry.GetProperty("id");
            var misc = Conllu.ParseMisc(String(entry, "misc") ?? "_");
            string? ner = String(entry, "ner");
            if (id.ValueKind == JsonValueKind.Array)
            {
                int first = id[0].GetInt32();
                if (first <= lastWord)
                    continue; // an empty node (enhanced UD), which our pipeline never makes
                mwt = NewToken(entry, misc, ner);
                mwtEnd = id[id.GetArrayLength() - 1].GetInt32();
                sent.Tokens.Add(mwt);
                continue;
            }

            var word = new Word
            {
                Id = lastWord = id.GetInt32(),
                Text = entry.GetProperty("text").GetString()!,
                Lemma = String(entry, "lemma"),
                Upos = String(entry, "upos"),
                Xpos = String(entry, "xpos"),
                Feats = String(entry, "feats"),
                Head = entry.TryGetProperty("head", out var head) ? head.GetInt32() : null,
                Deprel = String(entry, "deprel"),
                StartChar = Int(entry, "start_char"),
                EndChar = Int(entry, "end_char"),
            };
            if (mwt != null && word.Id <= mwtEnd)
            {
                mwt.Words.Add(word);
                continue;
            }

            mwt = null;
            var token = NewToken(entry, misc, ner);
            token.Words.Add(word);
            sent.Tokens.Add(token);
        }

        return sent;
    }

    private static Token NewToken(JsonElement entry, Conllu.MiscFields misc, string? ner) => new()
    {
        Text = entry.GetProperty("text").GetString()!,
        StartChar = Int(entry, "start_char"),
        EndChar = Int(entry, "end_char"),
        SpaceAfter = misc.Space,
        SpacesBefore = misc.SpacesBefore,
        Ner = ner,
    };

    private static string? String(JsonElement entry, string name) =>
        entry.TryGetProperty(name, out var value) && value.ValueKind != JsonValueKind.Null ? value.GetString() : null;

    private static int? Int(JsonElement entry, string name) =>
        entry.TryGetProperty(name, out var value) && value.ValueKind != JsonValueKind.Null ? value.GetInt32() : null;

    // Sentence.to_dict: per token, a multi-word token's own line first, then its words. Each entry is its fields in
    // DEFAULT_OUTPUT_FIELDS order, with a non-null value.
    private static IEnumerable<List<(string Key, object Value)>> Entries(Sentence sent)
    {
        foreach (var token in sent.Tokens)
        {
            var misc = Conllu.Misc(token.SpaceAfter, token.SpacesBefore, null, null, null);
            if (token.IsMultiWord)
            {
                var entry = new List<(string, object)> { ("id", new[] { token.Words[0].Id, token.Words[^1].Id }), ("text", token.Text) };
                AddTokenFields(entry, token, misc);
                yield return entry;
            }

            foreach (var w in token.Words)
            {
                var entry = new List<(string, object)> { ("id", w.Id), ("text", w.Text) };
                Add(entry, "lemma", w.Lemma);
                Add(entry, "upos", w.Upos);
                Add(entry, "xpos", w.Xpos);
                Add(entry, "feats", w.Feats);
                Add(entry, "head", w.Head);
                Add(entry, "deprel", w.Deprel);
                Add(entry, "start_char", w.StartChar);
                Add(entry, "end_char", w.EndChar);
                if (!token.IsMultiWord)
                {
                    Add(entry, "ner", token.Ner);
                    Add(entry, "multi_ner", token.Ner == null ? null : new[] { token.Ner });
                    Add(entry, "misc", misc == "_" ? null : misc);
                }

                yield return entry;
            }
        }
    }

    private static void AddTokenFields(List<(string, object)> entry, Token token, string misc)
    {
        Add(entry, "start_char", token.StartChar);
        Add(entry, "end_char", token.EndChar);
        Add(entry, "ner", token.Ner);

        // One NER model: multi_ner is (ner,), as Stanza sets it.
        Add(entry, "multi_ner", token.Ner == null ? null : new[] { token.Ner });
        Add(entry, "misc", misc == "_" ? null : misc);
    }

    private static void Add(List<(string, object)> entry, string key, object? value)
    {
        if (value != null)
            entry.Add((key, value));
    }

    // json.dumps with indent=2: "[]" when empty, else one item per line, indented two spaces per level.
    private static void List<T>(StringBuilder sb, int depth, IEnumerable<T> items, Action<T, int> item)
    {
        bool any = false;
        foreach (var x in items)
        {
            sb.Append(any ? ",\n" : "[\n").Append(' ', 2 * (depth + 1));
            item(x, depth + 1);
            any = true;
        }

        if (!any)
        {
            sb.Append("[]");
            return;
        }

        sb.Append('\n').Append(' ', 2 * depth).Append(']');
    }

    private static void Entry(StringBuilder sb, int depth, List<(string Key, object Value)> fields)
    {
        sb.Append("{\n");
        for (int i = 0; i < fields.Count; i++)
        {
            sb.Append(' ', 2 * (depth + 1));
            String(sb, fields[i].Key);
            sb.Append(": ");
            Value(sb, depth + 1, fields[i].Value);
            sb.Append(i + 1 < fields.Count ? ",\n" : "\n");
        }

        sb.Append(' ', 2 * depth).Append('}');
    }

    private static void Value(StringBuilder sb, int depth, object value)
    {
        switch (value)
        {
            case string s: String(sb, s); break;
            case int n: sb.Append(n); break;
            case int[] ids: List(sb, depth, ids, (n, _) => sb.Append(n)); break;
            case string[] tags: List(sb, depth, tags, (s, _) => String(sb, s)); break;
            default: throw new InvalidOperationException(value.GetType().Name);
        }
    }

    // Python's json with ensure_ascii=False: only '"', '\' and the C0 controls are escaped.
    private static void String(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (var c in s)
        {
            _ = c switch
            {
                '"' => sb.Append("\\\""),
                '\\' => sb.Append("\\\\"),
                '\n' => sb.Append("\\n"),
                '\r' => sb.Append("\\r"),
                '\t' => sb.Append("\\t"),
                '\b' => sb.Append("\\b"),
                '\f' => sb.Append("\\f"),
                < ' ' => sb.Append("\\u").Append(((int)c).ToString("x4", System.Globalization.CultureInfo.InvariantCulture)),
                _ => sb.Append(c),
            };
        }

        sb.Append('"');
    }
}
