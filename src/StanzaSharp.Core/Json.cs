using System.Text;
using System.Text.Json;

namespace StanzaSharp;

/// <summary>
/// JSON in Stanza's format: <c>str(doc)</c>, a list of sentences, each a list of token and word objects with
/// <c>id</c>, <c>text</c>, <c>lemma</c>, <c>upos</c>, <c>xpos</c>, <c>feats</c>, <c>head</c>, <c>deprel</c>,
/// <c>start_char</c>, <c>end_char</c>, <c>coref_chains</c>, <c>ner</c>, <c>multi_ner</c> and <c>misc</c> (the spacing), and
/// the empty words (<c>"id": [i, m]</c>) after the word they follow. Like Stanza's,
/// it holds no sentence-level results: no text, sentence id, constituency tree or sentiment (use <see cref="Conllu"/>).
/// </summary>
public static class Json
{
    /// <summary>
    /// Formats a document byte for byte as Python Stanza's <c>str(doc)</c> (<c>json.dumps(doc.to_dict(), indent=2,
    /// ensure_ascii=False)</c>): a field is left out when it has no value, a multi-word token comes before its words
    /// with its range as <c>"id": [first, last]</c>, and a token's NER tag goes on its line (a single word's for a
    /// one-word token). Once <see cref="Document.Coref"/> is set, every word has its <c>coref_chains</c> (an empty list
    /// for a word in no mention), each attachment as Stanza's <c>to_json()</c>. No final newline.
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
    /// <c>Document(json.loads(text))</c> does: tokens, multi-word tokens, words with their fields, empty words, and the
    /// spacing from <c>misc</c>. Sentence ids are the sentences' indices, as Stanza numbers them; entities are rebuilt from
    /// the <c>ner</c> tags and <see cref="Document.Coref"/> from the <c>coref_chains</c> attachments, as
    /// <see cref="Conllu.Read"/> does. <see cref="Document.Text"/> and <see cref="Sentence.Text"/> stay null. <c>Json.Write(Json.Read(json))</c> equals <c>json</c> for any <see cref="Write"/> output.
    /// </summary>
    /// <exception cref="JsonException">Not valid JSON, or not a list of lists of objects with <c>id</c> and
    /// <c>text</c>.</exception>
    public static Document Read(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var parsed = JsonDocument.Parse(json);
        var doc = new Document();
        var coref = new Dictionary<Word, List<(bool, bool, bool, int)>>(ReferenceEqualityComparer.Instance);
        try
        {
            foreach (var entries in parsed.RootElement.EnumerateArray())
                doc.Sentences.Add(ReadSentence(entries, doc.Sentences.Count, coref));
        }
        catch (Exception e) when (e is InvalidOperationException or KeyNotFoundException or FormatException)
        {
            throw new JsonException("Not a Stanza document: " + e.Message, e);
        }

        foreach (var s in doc.Sentences.Where(s => s.Tokens.Any(t => t.Ner != null)))
            Entity.Build(s, doc.Text);
        try
        {
            if (CorefLinks.Rebuild(doc, coref) is { } chains)
                doc.Coref = chains;
        }
        catch (Exception e) when (e is FormatException or ArgumentException)
        {
            throw new JsonException("Not a Stanza document: " + e.Message, e);
        }

        return doc;
    }

    private static Sentence ReadSentence(JsonElement entries, int index, Dictionary<Word, List<(bool, bool, bool, int)>> coref)
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
                {
                    // An empty word: a range starts past the words seen so far, and never at 0 (Sentence._process_tokens).
                    var empty = NewWord(entry, first, coref);
                    empty.EmptyIndex = id[1].GetInt32();
                    sent.EmptyWords.Add(empty);
                    continue;
                }

                mwt = NewToken(entry, misc, ner);
                mwtEnd = id[id.GetArrayLength() - 1].GetInt32();
                sent.Tokens.Add(mwt);
                continue;
            }

            var word = NewWord(entry, lastWord = id.GetInt32(), coref);
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

    private static Word NewWord(JsonElement entry, int id, Dictionary<Word, List<(bool, bool, bool, int)>> coref)
    {
        var word = new Word
        {
            Id = id,
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

        // CorefAttachment.to_json: the chain's index and representative text, and each flag only when true.
        if (entry.TryGetProperty("coref_chains", out var chains) && chains.ValueKind == JsonValueKind.Array)
        {
            coref[word] = [.. chains.EnumerateArray().Select(a => (Flag(a, "is_start"), Flag(a, "is_end"), Flag(a, "is_representative"),
                a.GetProperty("index").GetInt32()))];
        }

        return word;
    }

    private static bool Flag(JsonElement attachment, string name) => attachment.TryGetProperty(name, out var value) && value.GetBoolean();

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
        // Sentence._ordered_dicts: an empty word i.m right after word i (inside a multi-word token too), i = 0 first.
        var empties = sent.EmptyWords.OrderBy(w => w.Id).ThenBy(w => w.EmptyIndex).ToList();
        int e = 0;
        while (e < empties.Count && empties[e].Id == 0)
            yield return WordFields(empties[e], new[] { empties[e].Id, empties[e++].EmptyIndex!.Value });

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
                var entry = WordFields(w, w.Id);
                if (!token.IsMultiWord)
                {
                    Add(entry, "ner", token.Ner);
                    Add(entry, "multi_ner", token.Ner == null ? null : new[] { token.Ner });
                    Add(entry, "misc", misc == "_" ? null : misc);
                }

                yield return entry;
                while (e < empties.Count && empties[e].Id == w.Id)
                    yield return WordFields(empties[e], new[] { empties[e].Id, empties[e++].EmptyIndex!.Value });
            }
        }

        while (e < empties.Count)
            yield return WordFields(empties[e], new[] { empties[e].Id, empties[e++].EmptyIndex!.Value });
    }

    // Word.to_dict: its own fields, coref_chains last (the token's ner, multi_ner and misc follow, Token.to_dict).
    private static List<(string Key, object Value)> WordFields(Word w, object id)
    {
        var entry = new List<(string, object)> { ("id", id), ("text", w.Text) };
        Add(entry, "lemma", w.Lemma);
        Add(entry, "upos", w.Upos);
        Add(entry, "xpos", w.Xpos);
        Add(entry, "feats", w.Feats);
        Add(entry, "head", w.Head);
        Add(entry, "deprel", w.Deprel);
        Add(entry, "start_char", w.StartChar);
        Add(entry, "end_char", w.EndChar);
        Add(entry, "coref_chains", w.CorefChains);
        return entry;
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

    // CorefAttachment.to_json: index and representative_text, then each flag only when true.
    private static List<(string Key, object Value)> Attachment(CorefAttachment a)
    {
        var fields = new List<(string, object)> { ("index", a.Chain.Index), ("representative_text", a.Chain.RepresentativeText) };
        if (a.IsStart) fields.Add(("is_start", true));
        if (a.IsEnd) fields.Add(("is_end", true));
        if (a.IsRepresentative) fields.Add(("is_representative", true));
        return fields;
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
            case bool b: sb.Append(b ? "true" : "false"); break;
            case IReadOnlyList<CorefAttachment> attachments: List(sb, depth, attachments, (a, d) => Entry(sb, d, Attachment(a))); break;
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
