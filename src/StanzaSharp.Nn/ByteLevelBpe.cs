using System.Globalization;
using System.Text;
using System.Text.Json;

namespace StanzaSharp.Nn;

/// <summary>
/// A transformer's tokenizer for words already split (transformers' <c>is_split_into_words=True</c>): the start token, every
/// word's pieces, the end token, and for each id the index of its word (-1 for the start and end tokens). A word that
/// yields no piece is left out (Stanza's fix_blank_tokens then replaces it with "-").
/// </summary>
internal interface ITransformerTokenizer
{
    (int[] Ids, int[] Words) Encode(IReadOnlyList<string> words);
}

/// <summary>
/// RoBERTa's (GPT-2's) byte-level BPE tokenizer from Hugging Face's <c>tokenizer.json</c>, as Stanza's load_tokenizer sets it
/// up: <c>add_prefix_space=True</c> (stanza/models/common/bert_embedding.py) whatever the file says. Per word: the added
/// (special) tokens' texts first, leftmost and longest (<c>lstrip</c> takes the whitespace before one, <c>rstrip</c> after);
/// every other piece of the word gets a space in front unless it starts with one (the tokenizers library's ByteLevel
/// pre-tokenizer checks U+0020 only), is split by GPT-2's pattern
/// (<c>'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+</c>, scanned here by code point rather than
/// run as a .NET regex, which would see a non-BMP letter as two surrogates), turned into the byte-to-unicode alphabet
/// and merged by rank (lowest first, leftmost on a tie, like the library's BPE without dropout).
/// </summary>
internal sealed class ByteLevelBpe : ITransformerTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<(string, string), int> _ranks;
    private readonly (string Content, int Id, bool Lstrip, bool Rstrip)[] _added; // longest first
    private readonly int _start, _end;

    private static readonly string[] ByteChars = BytesToUnicode();

    private ByteLevelBpe(Dictionary<string, int> vocab, Dictionary<(string, string), int> ranks, (string, int, bool, bool)[] added, int start, int end)
    {
        (_vocab, _ranks, _start, _end) = (vocab, ranks, start, end);
        _added = [.. added.OrderByDescending(a => a.Item1.Length)];
    }

    /// <summary>
    /// Whether <paramref name="tokenizerJson"/> holds a BPE model (else a WordPiece one): its <c>type</c>, or, in files that
    /// leave it out (roberta-base's), its <c>merges</c>, as the tokenizers library infers it.
    /// </summary>
    public static bool IsBpe(string tokenizerJson)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        return IsBpe(doc.RootElement.GetProperty("model"));
    }

    private static bool IsBpe(JsonElement model) => model.TryGetProperty("type", out var type) ? type.GetString() == "BPE" : model.TryGetProperty("merges", out _);

    public static ByteLevelBpe Load(string tokenizerJson)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        string? Text(JsonElement e, string key) => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        bool Flag(JsonElement e, string key) => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.True;
        if (!IsBpe(model) || (model.TryGetProperty("dropout", out var dropout) && dropout.ValueKind != JsonValueKind.Null)
            || Flag(model, "byte_fallback") || Flag(model, "ignore_merges") || !string.IsNullOrEmpty(Text(model, "continuing_subword_prefix"))
            || !string.IsNullOrEmpty(Text(model, "end_of_word_suffix")))
        {
            throw new InvalidDataException($"{tokenizerJson}: a BPE model with options StanzaSharp doesn't support (dropout, byte fallback, prefixes, ignore_merges)");
        }

        var pre = root.GetProperty("pre_tokenizer");
        var post = root.GetProperty("post_processor");
        if ((root.TryGetProperty("normalizer", out var normalizer) && normalizer.ValueKind != JsonValueKind.Null) || Text(pre, "type") != "ByteLevel"
            || (pre.TryGetProperty("use_regex", out var regex) && regex.ValueKind == JsonValueKind.False) || Text(post, "type") != "RobertaProcessing")
        {
            throw new InvalidDataException($"{tokenizerJson}: a normalizer, pre-tokenizer or post-processor StanzaSharp doesn't support (none, ByteLevel, RobertaProcessing)");
        }

        var vocab = new Dictionary<string, int>();
        foreach (var entry in model.GetProperty("vocab").EnumerateObject())
            vocab[entry.Name] = entry.Value.GetInt32();
        var ranks = new Dictionary<(string, string), int>();
        int rank = 0;
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
        {
            // "a b" (older files) or ["a", "b"]
            var (a, b) = merge.ValueKind == JsonValueKind.String
                ? merge.GetString()!.Split(' ') is [var x, var y] ? (x, y) : throw new InvalidDataException($"{tokenizerJson}: merge {merge} is not two symbols")
                : (merge[0].GetString()!, merge[1].GetString()!);
            ranks.TryAdd((a, b), rank++);
        }

        var added = new List<(string, int, bool, bool)>();
        foreach (var token in root.GetProperty("added_tokens").EnumerateArray())
        {
            if (Flag(token, "single_word"))
                throw new InvalidDataException($"{tokenizerJson}: added token {token.GetProperty("content")} is single_word, which StanzaSharp doesn't support");
            added.Add((token.GetProperty("content").GetString()!, token.GetProperty("id").GetInt32(), Flag(token, "lstrip"), Flag(token, "rstrip")));
        }

        int Special(string key) => post.GetProperty(key)[1].GetInt32();
        return new ByteLevelBpe(vocab, ranks, [.. added], Special("cls"), Special("sep"));
    }

    public (int[] Ids, int[] Words) Encode(IReadOnlyList<string> words)
    {
        var ids = new List<int> { _start };
        var owners = new List<int> { -1 };
        for (int w = 0; w < words.Count; w++)
        {
            int before = ids.Count;
            Word(words[w], ids);
            for (int i = before; i < ids.Count; i++)
                owners.Add(w);
        }

        ids.Add(_end);
        owners.Add(-1);
        return ([.. ids], [.. owners]);
    }

    // One word: its added tokens, and the text between them through the pre-tokenizer and BPE.
    private void Word(string word, List<int> ids)
    {
        int start = 0;
        while (start < word.Length)
        {
            var (at, length, token) = FindAdded(word, start);
            if (token == null)
            {
                Text(word[start..], ids);
                return;
            }

            int textEnd = at, next = at + length;
            if (token.Value.Lstrip)
            {
                while (textEnd > start && char.IsWhiteSpace(word[textEnd - 1]))
                    textEnd--;
            }

            if (token.Value.Rstrip)
            {
                while (next < word.Length && char.IsWhiteSpace(word[next]))
                    next++;
            }

            Text(word[start..textEnd], ids);
            ids.Add(token.Value.Id);
            start = next;
        }
    }

    // The leftmost added token at or after start (the longest there), or none.
    private (int At, int Length, (string Content, int Id, bool Lstrip, bool Rstrip)? Token) FindAdded(string word, int start)
    {
        for (int i = start; i < word.Length; i++)
        {
            foreach (var token in _added)
            {
                if (i + token.Content.Length <= word.Length && string.CompareOrdinal(word, i, token.Content, 0, token.Content.Length) == 0)
                    return (i, token.Content.Length, token);
            }
        }

        return (word.Length, 0, null);
    }

    private void Text(string text, List<int> ids)
    {
        if (text.Length == 0)
            return;
        if (text[0] != ' ')
            text = " " + text; // add_prefix_space
        var runes = text.EnumerateRunes().ToArray();
        Span<byte> utf8 = stackalloc byte[4];
        int i = 0;
        while (i < runes.Length)
        {
            int end = NextPiece(runes, i);
            var piece = new StringBuilder();
            for (int r = i; r < end; r++)
            {
                int n = runes[r].EncodeToUtf8(utf8);
                for (int b = 0; b < n; b++)
                    piece.Append(ByteChars[utf8[b]]);
            }

            Bpe(piece.ToString(), ids);
            i = end;
        }
    }

    private enum Kind
    {
        Space,
        Letter,
        Number,
        Other,
    }

    private static Kind Of(Rune r)
    {
        if (Rune.IsWhiteSpace(r))
            return Kind.Space;
        return Rune.GetUnicodeCategory(r) switch
        {
            UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter or UnicodeCategory.ModifierLetter
                or UnicodeCategory.OtherLetter => Kind.Letter,
            UnicodeCategory.DecimalDigitNumber or UnicodeCategory.LetterNumber or UnicodeCategory.OtherNumber => Kind.Number,
            _ => Kind.Other,
        };
    }

    // The end of GPT-2's pattern's match at i (the alternatives in order, leftmost first, each greedy).
    private static int NextPiece(Rune[] runes, int i)
    {
        int n = runes.Length;

        // 's|'t|'re|'ve|'m|'ll|'d (lowercase only)
        if (runes[i].Value == '\'' && i + 1 < n)
        {
            int c = runes[i + 1].Value, d = i + 2 < n ? runes[i + 2].Value : -1;
            if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') || (c == 'l' && d == 'l'))
                return i + 3;
            if (c is 's' or 't' or 'm' or 'd')
                return i + 2;
        }

        //  ?\p{L}+ |  ?\p{N}+ |  ?[^\s\p{L}\p{N}]+
        int first = runes[i].Value == ' ' && i + 1 < n && Of(runes[i + 1]) != Kind.Space ? i + 1 : i;
        var kind = Of(runes[first]);
        if (kind != Kind.Space)
        {
            int j = first + 1;
            while (j < n && Of(runes[j]) == kind)
                j++;
            return j;
        }

        // \s+(?!\S): the whitespace run, less its last character when something follows it; else \s+ (that one character)
        int end = i + 1;
        while (end < n && Of(runes[end]) == Kind.Space)
            end++;
        return end == n || end - i == 1 ? end : end - 1;
    }

    private void Bpe(string piece, List<int> ids)
    {
        var symbols = new List<string>();
        foreach (var r in piece.EnumerateRunes())
            symbols.Add(r.ToString());
        while (symbols.Count > 1)
        {
            int best = -1, bestRank = int.MaxValue;
            for (int k = 0; k + 1 < symbols.Count; k++)
            {
                if (_ranks.TryGetValue((symbols[k], symbols[k + 1]), out int rank) && rank < bestRank)
                    (best, bestRank) = (k, rank);
            }

            if (best < 0)
                break;
            symbols[best] += symbols[best + 1];
            symbols.RemoveAt(best + 1);
        }

        foreach (var symbol in symbols)
            ids.Add(_vocab.TryGetValue(symbol, out int id) ? id : throw new InvalidDataException($"BPE symbol {symbol} is not in the vocabulary"));
    }

    // GPT-2's bytes_to_unicode: printable Latin-1 bytes stand for themselves, the rest map to U+0100 on.
    private static string[] BytesToUnicode()
    {
        var map = new string[256];
        int extra = 0;
        for (int b = 0; b < 256; b++)
        {
            bool printable = b is (>= '!' and <= '~') or (>= 0xA1 and <= 0xAC) or (>= 0xAE and <= 0xFF);
            map[b] = ((char)(printable ? b : 256 + extra++)).ToString();
        }

        return map;
    }
}
