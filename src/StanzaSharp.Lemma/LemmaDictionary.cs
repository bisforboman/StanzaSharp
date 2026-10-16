using System.Buffers;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.Unicode;

namespace StanzaSharp.Lemma;

/// <summary>
/// The lemmatizer's dictionaries, <c>{pos: {word: lemma or null}}</c> JSON, kept as the JSON's UTF-8 bytes with a hash
/// table of (POS, word) → lemma offsets, so no string is made at load (docs/performance/round-35-german-lemma-dictionary.md).
/// A repeated key keeps the last value, as JsonSerializer's dictionaries did.
/// </summary>
internal sealed class LemmaDictionary
{
    private readonly byte[] _text;          // the JSON, then the unescaped forms of escaped strings
    private readonly string[] _pos;
    private readonly int[] _entryPos, _wordStart, _wordLength, _lemmaStart, _lemmaLength; // lemma start -1: null
    private readonly int[] _hashes, _buckets; // _buckets: entry + 1, 0 empty; open addressing, a power of two

    public LemmaDictionary(byte[] json, int length)
    {
        var extra = new MemoryStream();
        var pos = new List<string>();
        var entries = new List<(int Pos, int WordStart, int WordLength, int LemmaStart, int LemmaLength)>();

        // json.dumps output: {"pos": {"word": "lemma" or null, ...}, ...}, whitespace allowed between tokens.
        int at = 0;
        Expect(Next() == '{');
        if (Peek() != '}')
        {
            do
            {
                var (posStart, posLength) = String();
                pos.Add(Encoding.UTF8.GetString(Bytes(posStart, posLength))); // a repeated POS replaces the earlier one (Lookup takes the last)
                Expect(Next() == ':' && Next() == '{');
                if (Peek() != '}')
                {
                    do
                    {
                        var (wordStart, wordLength) = String();
                        Expect(Next() == ':');
                        var lemma = (Start: -1, Length: 0);
                        if (Peek() == 'n')
                        {
                            Expect(json.AsSpan(at, length - at).StartsWith("null"u8));
                            at += 4;
                        }
                        else
                        {
                            lemma = String();
                        }

                        entries.Add((pos.Count - 1, wordStart, wordLength, lemma.Start, lemma.Length));
                    }
                    while (Next() == ',');
                    at--;
                }

                Expect(Next() == '}');
            }
            while (Next() == ',');
            at--;
        }

        Expect(Next() == '}' && Peek() == -1);

        _text = json;
        if (extra.Length > 0)
        {
            _text = new byte[length + extra.Length];
            json.AsSpan(0, length).CopyTo(_text);
            extra.GetBuffer().AsSpan(0, (int)extra.Length).CopyTo(_text.AsSpan(length));
        }

        _pos = [.. pos];
        int n = entries.Count;
        _entryPos = new int[n];
        _wordStart = new int[n];
        _wordLength = new int[n];
        _lemmaStart = new int[n];
        _lemmaLength = new int[n];
        _hashes = new int[n];
        _buckets = new int[Math.Max(16, (int)BitOperations.RoundUpToPowerOf2((uint)n * 2))];
        int mask = _buckets.Length - 1;
        for (int i = 0; i < n; i++)
        {
            var e = entries[i];
            var word = _text.AsSpan(e.WordStart, e.WordLength);
            int hash = Hash(e.Pos, word);
            for (int b = hash & mask; ; b = (b + 1) & mask)
            {
                int j = _buckets[b] - 1;
                if (j < 0 || (_hashes[j] == hash && _entryPos[j] == e.Pos && word.SequenceEqual(_text.AsSpan(_wordStart[j], _wordLength[j]))))
                {
                    _buckets[b] = i + 1; // a repeated key: the last value wins
                    break;
                }
            }

            (_entryPos[i], _wordStart[i], _wordLength[i], _lemmaStart[i], _lemmaLength[i], _hashes[i]) = (e.Pos, e.WordStart, e.WordLength, e.LemmaStart, e.LemmaLength, hash);
        }

        int Peek()
        {
            while (at < length && json[at] is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t')
                at++;
            return at < length ? json[at] : -1;
        }

        int Next()
        {
            int c = Peek();
            at++;
            return c;
        }

        ReadOnlySpan<byte> Bytes(int start, int count) => start < length ? json.AsSpan(start, count) : extra.GetBuffer().AsSpan(start - length, count);

        // A string's UTF-8 in _text: its bytes in the JSON, or, when it has escapes, its unescaped form appended after it.
        (int Start, int Length) String()
        {
            Expect(Next() == '"');
            int start = at;
            bool escaped = false;
            while (true)
            {
                int i = json.AsSpan(at, length - at).IndexOfAny((byte)'"', (byte)'\\');
                Expect(i >= 0);
                at += i + 1;
                if (json[at - 1] == '"')
                    break;
                escaped = true;
                Expect(at < length);
                at++; // the escaped character (\uXXXX's digits need no skipping)
            }

            if (!escaped)
                return (start, at - 1 - start);
            var reader = new Utf8JsonReader(json.AsSpan(start - 1, at - start + 1));
            reader.Read();
            var bytes = Encoding.UTF8.GetBytes(reader.GetString()!);
            int offset = length + (int)extra.Length;
            extra.Write(bytes);
            return (offset, bytes.Length);
        }

        static void Expect(bool ok)
        {
            if (!ok)
                throw new InvalidDataException("The lemma dictionaries are not {pos: {word: lemma}}");
        }
    }

    /// <summary>The lemma of (<paramref name="pos"/>, <paramref name="word"/>), or null if it is missing or null.</summary>
    public string? Lookup(string pos, string word)
    {
        int p = Array.LastIndexOf(_pos, pos);
        if (p < 0)
            return null;
        byte[]? rented = null;
        Span<byte> utf8 = word.Length <= 128 ? stackalloc byte[word.Length * 3] : (rented = ArrayPool<byte>.Shared.Rent(word.Length * 3));
        try
        {
            // A word with a lone surrogate equals no key.
            if (Utf8.FromUtf16(word, utf8, out _, out int written, replaceInvalidSequences: false) != OperationStatus.Done)
                return null;
            var key = utf8[..written];
            int hash = Hash(p, key), mask = _buckets.Length - 1;
            for (int b = hash & mask; ; b = (b + 1) & mask)
            {
                int j = _buckets[b] - 1;
                if (j < 0)
                    return null;
                if (_hashes[j] == hash && _entryPos[j] == p && key.SequenceEqual(_text.AsSpan(_wordStart[j], _wordLength[j])))
                    return _lemmaStart[j] < 0 ? null : Encoding.UTF8.GetString(_text, _lemmaStart[j], _lemmaLength[j]);
            }
        }
        finally
        {
            if (rented != null)
                ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static int Hash(int pos, ReadOnlySpan<byte> word)
    {
        var hash = default(HashCode);
        hash.Add(pos);
        hash.AddBytes(word);
        return hash.ToHashCode() & int.MaxValue;
    }
}
