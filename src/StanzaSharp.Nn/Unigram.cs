using System.Globalization;
using System.Text;
using System.Text.Json;

namespace StanzaSharp.Nn;

/// <summary>
/// XLM-RoBERTa's SentencePiece Unigram tokenizer from Hugging Face's <c>tokenizer.json</c>, as the tokenizers library runs
/// it (tokenizers 0.23.3, spm_precompiled 0.1.4). Per word: the added (special) tokens' texts first, leftmost and longest
/// (<c>lstrip</c>/<c>rstrip</c> as in <see cref="ByteLevelBpe"/>); every other piece of the word through
/// <list type="number">
/// <item>the <c>Precompiled</c> normalizer: sentencepiece's double-array trie over the UTF-8 bytes (the base64
/// <c>precompiled_charsmap</c>), applied per extended grapheme cluster; a cluster shorter than 6 bytes is replaced whole by
/// its <em>shortest</em> trie match if it has one (spm_precompiled's <c>transform</c> takes the first match, whatever its
/// length), else each of its characters by its own match, or kept;</item>
/// <item><c>WhitespaceSplit</c> (Unicode White_Space), then <c>Metaspace</c>: spaces become <c>▁</c>, a <c>▁</c> is
/// prepended unless the piece starts with one, and the piece is split before every <c>▁</c>;</item>
/// <item>the Unigram model: the best path by summed piece scores (a character no piece covers is <c>&lt;unk&gt;</c> at the
/// lowest score − 10; a later start replaces an earlier one only with a strictly higher score), adjacent unknowns fused
/// into one <c>&lt;unk&gt;</c>.</item>
/// </list>
/// </summary>
internal sealed class Unigram : ITransformerTokenizer
{
    private const double UnknownPenalty = 10;
    private const char Meta = '▁';

    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _lookup;
    private readonly double[] _scores;
    private readonly int _maxPieceChars, _unk, _start, _end;
    private readonly double _unkScore;
    private readonly (string Content, int Id, bool Lstrip, bool Rstrip)[] _added; // longest first
    private readonly Precompiled? _normalizer;

    private Unigram(Dictionary<string, int> ids, double[] scores, int unk, int start, int end, (string, int, bool, bool)[] added, Precompiled? normalizer)
    {
        (_scores, _unk, _start, _end, _normalizer) = (scores, unk, start, end, normalizer);
        _lookup = ids.GetAlternateLookup<ReadOnlySpan<char>>();
        _maxPieceChars = ids.Keys.Max(k => k.Length);
        _unkScore = scores.Min() - UnknownPenalty;
        _added = [.. added.OrderByDescending(a => a.Item1.Length)];
    }

    /// <summary>
    /// Whether <paramref name="tokenizerJson"/> holds a Unigram model: its <c>type</c>, or, in files that leave it out
    /// (xlm-roberta-large's), a <c>vocab</c> of [piece, score] pairs, as the tokenizers library infers it.
    /// </summary>
    public static bool IsUnigram(string tokenizerJson)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        return IsUnigram(doc.RootElement.GetProperty("model"));
    }

    private static bool IsUnigram(JsonElement model) => model.TryGetProperty("type", out var type)
        ? type.GetString() == "Unigram"
        : model.TryGetProperty("vocab", out var vocab) && vocab.ValueKind == JsonValueKind.Array;

    public static Unigram Load(string tokenizerJson)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(tokenizerJson));
        var root = doc.RootElement;
        string? Text(JsonElement e, string key) => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        bool Flag(JsonElement e, string key) => e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.True;

        var model = root.GetProperty("model");
        if (!IsUnigram(model) || Flag(model, "byte_fallback") || !model.TryGetProperty("unk_id", out var unkId) || unkId.ValueKind != JsonValueKind.Number)
            throw new InvalidDataException($"{tokenizerJson}: a Unigram model with options StanzaSharp doesn't support (byte fallback, no unk_id)");

        Precompiled? normalizer = null;
        if (root.TryGetProperty("normalizer", out var norm) && norm.ValueKind != JsonValueKind.Null)
        {
            normalizer = Text(norm, "type") == "Precompiled"
                ? Precompiled.Load(Convert.FromBase64String(Text(norm, "precompiled_charsmap") ?? ""), tokenizerJson)
                : throw new InvalidDataException($"{tokenizerJson}: normalizer {Text(norm, "type")}; StanzaSharp's Unigram supports Precompiled only");
        }

        // WhitespaceSplit + Metaspace(▁, prepend always, split), as xlm-roberta-large's file has them
        var pre = root.GetProperty("pre_tokenizer");
        var steps = Text(pre, "type") == "Sequence" ? [.. pre.GetProperty("pretokenizers").EnumerateArray()] : new[] { pre };
        var meta = steps.Length == 2 && Text(steps[0], "type") == "WhitespaceSplit" && Text(steps[1], "type") == "Metaspace" ? steps[1] : default;
        if (meta.ValueKind != JsonValueKind.Object || Text(meta, "replacement") != Meta.ToString()
            || (Text(meta, "prepend_scheme") ?? "always") != "always" || (meta.TryGetProperty("add_prefix_space", out var aps) && aps.ValueKind == JsonValueKind.False)
            || (meta.TryGetProperty("split", out var split) && split.ValueKind == JsonValueKind.False))
        {
            throw new InvalidDataException($"{tokenizerJson}: a pre-tokenizer StanzaSharp doesn't support (WhitespaceSplit, then Metaspace ▁ prepended always and split)");
        }

        var vocab = model.GetProperty("vocab");
        var ids = new Dictionary<string, int>(vocab.GetArrayLength(), StringComparer.Ordinal);
        var scores = new double[vocab.GetArrayLength()];
        int id = 0;
        foreach (var entry in vocab.EnumerateArray())
        {
            ids.TryAdd(entry[0].GetString()!, id); // the first of a repeated piece, like the library's map (last insert wins there; none repeat here)
            scores[id++] = entry[1].GetDouble();
        }

        var added = new List<(string, int, bool, bool)>();
        foreach (var token in root.GetProperty("added_tokens").EnumerateArray())
        {
            if (Flag(token, "single_word") || Flag(token, "normalized"))
                throw new InvalidDataException($"{tokenizerJson}: added token {token.GetProperty("content")} is single_word or normalized, which StanzaSharp doesn't support");
            added.Add((token.GetProperty("content").GetString()!, token.GetProperty("id").GetInt32(), Flag(token, "lstrip"), Flag(token, "rstrip")));
        }

        // TemplateProcessing: <s> A </s>
        var post = root.GetProperty("post_processor");
        var single = Text(post, "type") == "TemplateProcessing" ? [.. post.GetProperty("single").EnumerateArray()] : Array.Empty<JsonElement>();
        string? Special(JsonElement e) => e.TryGetProperty("SpecialToken", out var s) ? Text(s, "id") : null;
        if (single.Length != 3 || Special(single[0]) is not { } first || Special(single[2]) is not { } last || !single[1].TryGetProperty("Sequence", out _))
            throw new InvalidDataException($"{tokenizerJson}: a post-processor StanzaSharp doesn't support (TemplateProcessing: special, sequence, special)");
        int SpecialId(string name) => post.GetProperty("special_tokens").GetProperty(name).GetProperty("ids")[0].GetInt32();
        return new Unigram(ids, scores, unkId.GetInt32(), SpecialId(first), SpecialId(last), [.. added], normalizer);
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

    // One word: its added tokens, and the text between them through the normalizer, the pre-tokenizer and the model.
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
        var normalized = _normalizer?.Normalize(text) ?? text;

        // WhitespaceSplit, then Metaspace on each split: ▁ in front unless it starts with one, split before every ▁
        int i = 0;
        while (i < normalized.Length)
        {
            while (i < normalized.Length && IsWhiteSpace(normalized, i))
                i += char.IsSurrogatePair(normalized, i) ? 2 : 1;
            int end = i;
            while (end < normalized.Length && !IsWhiteSpace(normalized, end))
                end += char.IsSurrogatePair(normalized, end) ? 2 : 1;
            if (end > i)
            {
                var split = normalized[i] == Meta ? normalized[i..end] : Meta + normalized[i..end];
                int pieceStart = 0;
                for (int k = 1; k <= split.Length; k++)
                {
                    if (k == split.Length || split[k] == Meta)
                    {
                        Model(split.AsSpan(pieceStart, k - pieceStart), ids);
                        pieceStart = k;
                    }
                }
            }

            i = end;
        }
    }

    // Rust's char::is_whitespace (the White_Space property), which Rune.IsWhiteSpace matches.
    private static bool IsWhiteSpace(string s, int i) => Rune.IsWhiteSpace(Rune.GetRuneAt(s, i));

    // encode_optimized: the best path over the piece's characters (pieces end on whole characters).
    private void Model(ReadOnlySpan<char> text, List<int> ids)
    {
        int n = text.Length;
        var best = new double[n + 1];
        var from = new int[n + 1];
        var node = new int[n + 1];
        Array.Fill(from, -1);
        for (int start = 0; start < n; start += char.IsSurrogatePair(text[start], start + 1 < n ? text[start + 1] : '\0') ? 2 : 1)
        {
            int unit = start + 1 < n && char.IsSurrogatePair(text[start], text[start + 1]) ? 2 : 1;
            bool single = false;
            for (int length = 1; length <= Math.Min(_maxPieceChars, n - start); length++)
            {
                // pieces end on whole characters: skip a length that splits a surrogate pair
                if (char.IsHighSurrogate(text[start + length - 1]) && start + length < n && char.IsLowSurrogate(text[start + length]))
                    continue;
                if (!_lookup.TryGetValue(text.Slice(start, length), out int id))
                    continue;
                Relax(start + length, start, id, _scores[id]);
                if (length == unit)
                    single = true;
            }

            if (!single)
                Relax(start + unit, start, _unk, _unkScore);
        }

        // Backtrack, fusing adjacent unknowns into one <unk>.
        var path = new List<int>();
        bool lastUnk = false;
        for (int end = n; end > 0; end = from[end])
        {
            bool unk = node[end] == _unk;
            if (!(unk && lastUnk))
                path.Add(node[end]);
            lastUnk = unk;
        }

        for (int k = path.Count - 1; k >= 0; k--)
            ids.Add(path[k]);

        void Relax(int target, int start, int id, double score)
        {
            double candidate = best[start] + score;
            if (from[target] < 0 || candidate > best[target])
                (best[target], from[target], node[target]) = (candidate, start, id);
        }
    }

    /// <summary>sentencepiece's precompiled normalization charsmap: a double-array trie over UTF-8 bytes into a pool of NUL-terminated replacements.</summary>
    private sealed class Precompiled
    {
        private readonly uint[] _array;
        private readonly byte[] _pool;

        private Precompiled(uint[] array, byte[] pool) => (_array, _pool) = (array, pool);

        public static Precompiled Load(byte[] charsmap, string path)
        {
            if (charsmap.Length < 4)
                throw new InvalidDataException($"{path}: precompiled_charsmap is shorter than its header");
            uint trieBytes = BitConverter.ToUInt32(charsmap, 0); // little-endian, as sentencepiece writes it
            if (!BitConverter.IsLittleEndian || trieBytes % 4 != 0 || trieBytes > charsmap.Length - 4 || trieBytes == 0)
                throw new InvalidDataException($"{path}: precompiled_charsmap's trie size {trieBytes} doesn't fit the map");
            var array = new uint[trieBytes / 4];
            Buffer.BlockCopy(charsmap, 4, array, 0, (int)trieBytes);
            return new Precompiled(array, charsmap[(4 + (int)trieBytes)..]);
        }

        public string Normalize(string text)
        {
            var result = new StringBuilder(text.Length);
            Span<byte> utf8 = stackalloc byte[16];
            var graphemes = StringInfo.GetTextElementEnumerator(text);
            while (graphemes.MoveNext())
            {
                var grapheme = (string)graphemes.Current;
                int bytes = Encoding.UTF8.GetByteCount(grapheme);
                if (bytes < 6)
                {
                    Encoding.UTF8.GetBytes(grapheme, utf8);
                    if (Transform(utf8[..bytes], result))
                        continue;
                }

                foreach (var rune in grapheme.EnumerateRunes())
                {
                    int n = rune.EncodeToUtf8(utf8);
                    if (!Transform(utf8[..n], result))
                        result.Append(rune.ToString());
                }
            }

            return result.ToString();
        }

        // spm_precompiled's transform: the first (shortest) prefix match's replacement, or none.
        private bool Transform(ReadOnlySpan<byte> key, StringBuilder result)
        {
            int value = FirstPrefixMatch(key);
            if (value < 0)
                return false;
            if (value >= _pool.Length)
                throw new InvalidDataException($"precompiled_charsmap: replacement offset {value} past the pool ({_pool.Length} bytes)");
            int end = Array.IndexOf(_pool, (byte)0, value);
            result.Append(Encoding.UTF8.GetString(_pool, value, (end < 0 ? _pool.Length : end) - value));
            return true;
        }

        // Darts' common_prefix_search, stopped at the first result.
        private int FirstPrefixMatch(ReadOnlySpan<byte> key)
        {
            long pos = 0;
            uint unit = At(pos);
            pos ^= Offset(unit);
            foreach (byte c in key)
            {
                if (c == 0)
                    break;
                pos ^= c;
                unit = At(pos);
                if (Label(unit) != c)
                    return -1;
                pos ^= Offset(unit);
                if (((unit >> 8) & 1) == 1)
                    return (int)(At(pos) & ((1U << 31) - 1));
            }

            return -1;
        }

        private uint At(long pos) => pos >= 0 && pos < _array.Length ? _array[pos] : throw new InvalidDataException($"precompiled_charsmap: trie index {pos} out of range");

        private static uint Label(uint unit) => unit & ((1U << 31) | 0xFF);

        private static long Offset(uint unit) => (long)(unit >> 10) << (int)((unit & (1U << 9)) >> 6);
    }
}
