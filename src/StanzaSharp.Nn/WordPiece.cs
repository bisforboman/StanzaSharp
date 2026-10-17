using System.Globalization;
using System.Text;
using System.Text.Json;

namespace StanzaSharp.Nn;

/// <summary>
/// The BERT/ELECTRA tokenizer of Hugging Face's <c>tokenizer.json</c> (BertNormalizer, BertPreTokenizer, WordPiece)
/// for words already split, like transformers' <c>is_split_into_words=True</c>: every word is normalized and
/// pre-tokenized alone, so its pieces never join a neighbour's. Spike code (docs/transformer-plan.md): only the
/// options google/electra-large-discriminator uses (lowercase, accents stripped, Chinese characters split).
/// </summary>
internal sealed class WordPiece
{
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _lookup;
    private readonly int _maxChars;

    public int Cls { get; }
    public int Sep { get; }
    public int Unk { get; }
    public int Pad { get; }

    private WordPiece(Dictionary<string, int> vocab, int maxChars)
    {
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
        return new WordPiece(vocab, model.GetProperty("max_input_chars_per_word").GetInt32());
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
            Normalize(words[w], normalized);
            foreach (var piece in PreTokenize(normalized.ToString()))
            {
                int before = ids.Count;
                AddPieces(piece, ids);
                for (int i = before; i < ids.Count; i++)
                    owners.Add(w);
            }
        }

        ids.Add(Sep);
        owners.Add(-1);
        return (ids.ToArray(), owners.ToArray());
    }

    // BertNormalizer: clean_text, handle_chinese_chars, strip_accents (follows lowercase), lowercase.
    private static void Normalize(string text, StringBuilder output)
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
            else if (IsChinese(c))
                cleaned.Append(' ').Append(rune.ToString()).Append(' ');
            else
                cleaned.Append(rune.ToString());
        }

        foreach (var rune in cleaned.ToString().Normalize(NormalizationForm.FormD).EnumerateRunes())
        {
            if (Rune.GetUnicodeCategory(rune) == UnicodeCategory.NonSpacingMark)
                continue;
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

    private static bool IsWhitespace(Rune r) => r.Value is ' ' or '\t' or '\n' or '\r' || Rune.GetUnicodeCategory(r) == UnicodeCategory.SpaceSeparator;

    private static bool IsControl(Rune r) =>
        r.Value is not ('\t' or '\n' or '\r')
        && Rune.GetUnicodeCategory(r) is UnicodeCategory.Control or UnicodeCategory.Format;

    private static bool IsPunctuation(Rune r) =>
        r.Value is (>= 33 and <= 47) or (>= 58 and <= 64) or (>= 91 and <= 96) or (>= 123 and <= 126)
        || Rune.GetUnicodeCategory(r) is UnicodeCategory.ConnectorPunctuation or UnicodeCategory.DashPunctuation
            or UnicodeCategory.OpenPunctuation or UnicodeCategory.ClosePunctuation or UnicodeCategory.InitialQuotePunctuation
            or UnicodeCategory.FinalQuotePunctuation or UnicodeCategory.OtherPunctuation;

    private static bool IsChinese(int c) =>
        c is (>= 0x4E00 and <= 0x9FFF) or (>= 0x3400 and <= 0x4DBF) or (>= 0x20000 and <= 0x2A6DF) or (>= 0x2A700 and <= 0x2B73F)
            or (>= 0x2B740 and <= 0x2B81F) or (>= 0x2B820 and <= 0x2CEAF) or (>= 0xF900 and <= 0xFAFF) or (>= 0x2F800 and <= 0x2FA1F);
}
