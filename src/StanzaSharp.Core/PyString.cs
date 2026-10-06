using System.Globalization;
using System.Buffers;
using System.Text;

namespace StanzaSharp;

/// <summary>Python <c>str</c> semantics where .NET's differ. Stanza's behavior depends on these.</summary>
internal static class PyString
{
    /// <summary>
    /// <c>str.lower()</c>: per code point, with Python's full mapping of U+0130 (İ → i̇) and the
    /// Final_Sigma rule (Σ → ς at the end of a word), which <c>ToLowerInvariant</c> lacks.
    /// </summary>
    public static string Lower(string s)
    {
        var sb = new StringBuilder(s.Length);
        int i = 0;
        foreach (var rune in s.EnumerateRunes())
        {
            if (rune.Value == 0x130)
                sb.Append("i̇");
            else if (rune.Value == 0x3A3)
                sb.Append(IsFinalSigma(s, i) ? 'ς' : 'σ');
            else
                sb.Append(Rune.ToLowerInvariant(rune).ToString());
            i += rune.Utf16SequenceLength;
        }
        return sb.ToString();
    }

    /// <summary><c>str.isupper()</c>: at least one cased code point, and no lowercase or titlecase ones.</summary>
    public static bool IsUpper(string s) => HasOnlyCase(s, upper: true);

    /// <summary><c>str.islower()</c>.</summary>
    public static bool IsLower(string s) => HasOnlyCase(s, upper: false);

    /// <summary><c>str.isspace()</c> for one char: .NET's whitespace plus the separators U+001C..U+001F.</summary>
    public static bool IsSpace(char c) => char.IsWhiteSpace(c) || c is >= '\x1c' and <= '\x1f';

    private static bool HasOnlyCase(string s, bool upper)
    {
        bool any = false;
        foreach (var rune in s.EnumerateRunes())
        {
            if (IsUpperRune(rune) == upper && (upper || IsLowerRune(rune)))
                any = true;
            else if (IsCased(rune))
                return false;
        }
        return any;
    }

    // Python's Uppercase/Lowercase properties: categories Lu/Ll plus Other_Uppercase/Other_Lowercase,
    // which .NET does not expose (ranges from Python 3.10's Unicode 13 database). E.g. Ⅰ, Ⓐ; ª, ʰ, ⅰ.
    private static bool IsUpperRune(Rune r) =>
        Rune.GetUnicodeCategory(r) == UnicodeCategory.UppercaseLetter
        || r.Value is >= 0x2160 and <= 0x216F or >= 0x24B6 and <= 0x24CF or >= 0x1F130 and <= 0x1F149
            or >= 0x1F150 and <= 0x1F169 or >= 0x1F170 and <= 0x1F189;

    private static bool IsLowerRune(Rune r) =>
        Rune.GetUnicodeCategory(r) == UnicodeCategory.LowercaseLetter
        || r.Value is 0xAA or 0xBA or >= 0x2B0 and <= 0x2B8 or 0x2C0 or 0x2C1 or >= 0x2E0 and <= 0x2E4
            or 0x345 or 0x37A or >= 0x1D2C and <= 0x1D6A or 0x1D78 or >= 0x1D9B and <= 0x1DBF or 0x2071
            or 0x207F or >= 0x2090 and <= 0x209C or >= 0x2170 and <= 0x217F or >= 0x24D0 and <= 0x24E9
            or 0x2C7C or 0x2C7D or 0xA69C or 0xA69D or 0xA770 or 0xA7F8 or 0xA7F9 or >= 0xAB5C and <= 0xAB5F;

    /// <summary>Python's <c>Py_UNICODE_ISUPPER</c> for one code point (the tokenizer's "capitalized" feature).</summary>
    public static bool IsUpper(Rune r) => IsUpperRune(r);

    private static bool IsCased(Rune r) =>
        IsUpperRune(r) || IsLowerRune(r) || Rune.GetUnicodeCategory(r) == UnicodeCategory.TitlecaseLetter;

    // CPython handle_capital_sigma: preceded by a cased letter and not followed by one, skipping
    // case-ignorable code points in both directions.
    private static bool IsFinalSigma(string s, int at)
    {
        int j = at;
        Rune r = default;
        bool found = false;
        while (j > 0 && Rune.DecodeLastFromUtf16(s.AsSpan(0, j), out r, out int n) == OperationStatus.Done)
        {
            j -= n;
            if (!IsCaseIgnorable(r)) { found = true; break; }
        }
        if (!found || !IsCased(r))
            return false;
        for (j = at + 1; j < s.Length;)
        {
            Rune.DecodeFromUtf16(s.AsSpan(j), out r, out int n);
            j += n;
            if (!IsCaseIgnorable(r))
                return !IsCased(r);
        }
        return true;
    }

    // Unicode Case_Ignorable: Mn, Me, Cf, Lm, Sk, plus Word_Break MidLetter/MidNumLet/Single_Quote.
    private static bool IsCaseIgnorable(Rune r) =>
        Rune.GetUnicodeCategory(r) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark
            or UnicodeCategory.Format or UnicodeCategory.ModifierLetter or UnicodeCategory.ModifierSymbol
        || r.Value is '\'' or '.' or ':' or '·' or '‘' or '’' or '․' or '‧' or 0x387 or 0x55F or 0x5F4
            or 0xFE13 or 0xFE52 or 0xFE55 or 0xFF07 or 0xFF0E or 0xFF1A;
}
