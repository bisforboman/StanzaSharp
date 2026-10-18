using System.Globalization;
using System.Text;
using System.Text.Json;

namespace StanzaSharp.Nn;

/// <summary>
/// The BERT/ELECTRA tokenizer of Hugging Face's <c>tokenizer.json</c> (BertNormalizer, BertPreTokenizer, WordPiece), or
/// the one transformers builds from a <c>vocab.txt</c> and <c>tokenizer_config.json</c> (german-nlp-group's ELECTRA), for
/// words already split, like transformers' <c>is_split_into_words=True</c>: every word is normalized and
/// pre-tokenized alone, so its pieces never join a neighbour's. The normalizer's options are the file's (lowercase,
/// accents stripped or not, Chinese characters split; docs/transformer-plan.md, docs/german-accurate-plan.md), with the character classes of
/// the tokenizers library (Rust): control = Cc, Cf, Co, Cs but tab, newline and carriage return (removed), whitespace =
/// Unicode White_Space (a space), punctuation = ASCII punctuation or any P* category (a piece of its own). The added
/// (special) tokens' texts are matched first in the raw word, leftmost and longest, like the library's added
/// vocabulary: a word "[CLS]" is id 101, not "[", "cl", "##s", "]".
/// </summary>
internal sealed class WordPiece
{
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _lookup;
    private readonly int _maxChars;
    private readonly (string Content, int Id)[] _added; // longest first
    private readonly bool _lowercase, _stripAccents, _chinese;

    public int Cls { get; }
    public int Sep { get; }
    public int Unk { get; }
    public int Pad { get; }

    private WordPiece(Dictionary<string, int> vocab, int maxChars, (string Content, int Id)[] added, bool lowercase, bool stripAccents, bool chinese)
    {
        (_lowercase, _stripAccents, _chinese) = (lowercase, stripAccents, chinese);
        _added = [.. added.OrderByDescending(a => a.Content.Length)];
        _lookup = vocab.GetAlternateLookup<ReadOnlySpan<char>>();
        _maxChars = maxChars;
        Cls = vocab["[CLS]"];
        Sep = vocab["[SEP]"];
        Unk = vocab["[UNK]"];
        Pad = vocab["[PAD]"];
    }

    public static WordPiece Load(string tokenizerJson)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        var model = doc.RootElement.GetProperty("model");
        if (!model.TryGetProperty("continuing_subword_prefix", out var prefix) || prefix.GetString() != "##")
            throw new InvalidDataException($"{tokenizerJson}: not a WordPiece model with ## continuations");
        var vocab = new Dictionary<string, int>();
        foreach (var entry in model.GetProperty("vocab").EnumerateObject())
            vocab[entry.Name] = entry.Value.GetInt32();
        var added = new List<(string, int)>();
        foreach (var token in doc.RootElement.GetProperty("added_tokens").EnumerateArray())
        {
            if (token.GetProperty("normalized").GetBoolean() || token.GetProperty("lstrip").GetBoolean() || token.GetProperty("rstrip").GetBoolean()
                || token.GetProperty("single_word").GetBoolean())
            {
                throw new InvalidDataException($"{tokenizerJson}: added token {token.GetProperty("content")} uses options StanzaSharp doesn't support");
            }

            added.Add((token.GetProperty("content").GetString()!, token.GetProperty("id").GetInt32()));
        }

        var normalizer = doc.RootElement.GetProperty("normalizer");
        if (normalizer.GetProperty("type").GetString() != "BertNormalizer" || !normalizer.GetProperty("clean_text").GetBoolean()
            || doc.RootElement.GetProperty("pre_tokenizer").GetProperty("type").GetString() != "BertPreTokenizer")
        {
            throw new InvalidDataException($"{tokenizerJson}: a normalizer or pre-tokenizer StanzaSharp doesn't support (BertNormalizer with clean_text, BertPreTokenizer)");
        }

        bool lowercase = normalizer.GetProperty("lowercase").GetBoolean();
        var strip = normalizer.GetProperty("strip_accents");
        return new WordPiece(vocab, model.GetProperty("max_input_chars_per_word").GetInt32(), [.. added], lowercase,
            strip.ValueKind == JsonValueKind.Null ? lowercase : strip.GetBoolean(), normalizer.GetProperty("handle_chinese_chars").GetBoolean());
    }

    /// <summary>
    /// The tokenizer of a transformer directory: its <c>tokenizer.json</c>, else what transformers' BertTokenizer converter
    /// builds from <c>vocab.txt</c> (line i is id i, a repeated line keeping the last) and <c>tokenizer_config.json</c>
    /// (<c>do_lower_case</c> and <c>tokenize_chinese_chars</c> true unless set, <c>strip_accents</c> following the
    /// lowercasing unless set), with the special tokens [PAD], [UNK], [CLS], [SEP] and [MASK] as added tokens and at most
    /// 100 characters per word.
    /// </summary>
    public static WordPiece LoadDirectory(string directory)
    {
        var json = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(json))
            return Load(json);
        var vocabTxt = Path.Combine(directory, "vocab.txt");
        var vocab = new Dictionary<string, int>();
        int id = 0;
        foreach (var line in File.ReadLines(vocabTxt))
            vocab[line] = id++;
        var configPath = Path.Combine(directory, "tokenizer_config.json");
        using var config = JsonDocument.Parse(File.ReadAllBytes(configPath));
        bool? Flag(string key) => config.RootElement.TryGetProperty(key, out var v) && v.ValueKind != JsonValueKind.Null ? v.GetBoolean() : null;
        bool lowercase = Flag("do_lower_case") ?? true;
        string[] specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
        var missing = specials.FirstOrDefault(t => !vocab.ContainsKey(t));
        if (missing != null)
            throw new InvalidDataException($"{vocabTxt}: no {missing}");
        return new WordPiece(vocab, 100, [.. specials.Select(t => (t, vocab[t]))], lowercase, Flag("strip_accents") ?? lowercase,
            Flag("tokenize_chinese_chars") ?? true);
    }

    /// <summary>
    /// [CLS] + every word's pieces + [SEP], and for each id the index of its word (-1 for [CLS] and [SEP]). A word
    /// that normalizes to nothing gets no piece (Stanza's fix_blank_tokens then replaces it with "-").
    /// </summary>
    public (int[] Ids, int[] Words) Encode(IReadOnlyList<string> words)
    {
        var ids = new List<int> { Cls };
        var owners = new List<int> { -1 };
        var normalized = new StringBuilder();
        for (int w = 0; w < words.Count; w++)
        {
            var word = words[w];
            int start = 0;
            while (start <= word.Length)
            {
                var (at, added) = FindAdded(word, start);
                int end = at < 0 ? word.Length : at;
                if (end > start)
                {
                    Normalize(word[start..end], normalized);
                    foreach (var piece in PreTokenize(normalized.ToString()))
                    {
                        int before = ids.Count;
                        AddPieces(piece, ids);
                        for (int i = before; i < ids.Count; i++)
                            owners.Add(w);
                    }
                }

                if (at < 0)
                    break;
                ids.Add(added.Id);
                owners.Add(w);
                start = at + added.Content.Length;
            }
        }

        ids.Add(Sep);
        owners.Add(-1);
        return (ids.ToArray(), owners.ToArray());
    }

    // The leftmost added token at or after start (the longest one there), or -1.
    private (int At, (string Content, int Id) Token) FindAdded(string word, int start)
    {
        int best = -1;
        (string, int) token = default;
        foreach (var added in _added)
        {
            int at = word.IndexOf(added.Content, start, StringComparison.Ordinal);
            if (at >= 0 && (best < 0 || at < best))
                (best, token) = (at, added);
        }

        return (best, token);
    }

    // BertNormalizer: clean_text, handle_chinese_chars, strip_accents, lowercase (Rust's char::to_lowercase, one character at
    // a time: like .NET's simple mapping but for U+0130 İ, whose lowercase is i + U+0307; NFD has split it before when
    // accents are stripped).
    private void Normalize(string text, StringBuilder output)
    {
        output.Clear();
        var cleaned = new StringBuilder(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            int c = rune.Value;
            if (c == 0 || c == 0xFFFD || IsControl(rune))
                continue;
            if (IsWhitespace(rune))
                cleaned.Append(' ');
            else if (_chinese && IsChinese(c))
                cleaned.Append(' ').Append(rune.ToString()).Append(' ');
            else
                cleaned.Append(rune.ToString());
        }

        var text2 = _stripAccents ? cleaned.ToString().Normalize(NormalizationForm.FormD) : cleaned.ToString();
        foreach (var rune in text2.EnumerateRunes())
        {
            if (_stripAccents && Rune.GetUnicodeCategory(rune) == UnicodeCategory.NonSpacingMark)
                continue;
            if (!_lowercase)
                output.Append(rune.ToString());
            else if (rune.Value == 0x130)
                output.Append("i\u0307");
            else
                output.Append(Rune.ToLowerInvariant(rune).ToString());
        }
    }

    // BertPreTokenizer: split on whitespace; every punctuation character is a piece of its own.
    private static IEnumerable<string> PreTokenize(string text)
    {
        var current = new StringBuilder();
        foreach (var rune in text.EnumerateRunes())
        {
            if (IsWhitespace(rune) || IsPunctuation(rune))
            {
                if (current.Length > 0)
                    yield return current.ToString();
                current.Clear();
                if (IsPunctuation(rune))
                    yield return rune.ToString();
            }
            else
            {
                current.Append(rune.ToString());
            }
        }

        if (current.Length > 0)
            yield return current.ToString();
    }

    // Greedy longest match, "##" for continuations; any unmatched rest makes the whole word [UNK].
    private void AddPieces(string word, List<int> ids)
    {
        var runes = word.EnumerateRunes().Select(r => r.ToString()).ToArray();
        if (runes.Length > _maxChars)
        {
            ids.Add(Unk);
            return;
        }

        int before = ids.Count, start = 0;
        var candidate = new StringBuilder();
        while (start < runes.Length)
        {
            int end = runes.Length, found = -1;
            while (start < end)
            {
                candidate.Clear();
                if (start > 0)
                    candidate.Append("##");
                for (int i = start; i < end; i++)
                    candidate.Append(runes[i]);
                if (_lookup.TryGetValue(candidate.ToString().AsSpan(), out found))
                    break;
                found = -1;
                end--;
            }

            if (found < 0)
            {
                ids.RemoveRange(before, ids.Count - before);
                ids.Add(Unk);
                return;
            }

            ids.Add(found);
            start = end;
        }
    }

    private static bool IsWhitespace(Rune r) => Rune.IsWhiteSpace(r);

    // tokenizers' is_control: Cc, Cf, Co, Cs except '\t', '\n', '\r'. Its is_other names Cn too, but unassigned code
    // points come out as [UNK] there (U+0378 in the golden cases), so they are kept.
    private static bool IsControl(Rune r) =>
        r.Value is not ('\t' or '\n' or '\r')
        && Rune.GetUnicodeCategory(r) is UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.PrivateUse or UnicodeCategory.Surrogate;

    private static bool IsPunctuation(Rune r) =>
        r.Value is (>= 33 and <= 47) or (>= 58 and <= 64) or (>= 91 and <= 96) or (>= 123 and <= 126)
        || Rune.GetUnicodeCategory(r) is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation
            or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;

    private static bool IsChinese(int c) =>
        c is (>= 0x4E00 and <= 0x9FFF) or (>= 0x3400 and <= 0x4DBF) or (>= 0x20000 and <= 0x2A6DF) or (>= 0x2A700 and <= 0x2B73F)
            or (>= 0x2B740 and <= 0x2B81F) or (>= 0x2B820 and <= 0x2CEAF) or (>= 0xF900 and <= 0xFAFF) or (>= 0x2F800 and <= 0x2FA1F);
}
