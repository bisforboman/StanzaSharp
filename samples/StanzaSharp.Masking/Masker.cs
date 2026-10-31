using System.Text;
using System.Text.RegularExpressions;

namespace StanzaSharp.Masking;

/// <summary>A span of the text suggested for masking.</summary>
/// <param name="Start">The UTF-16 offset of the span's first character.</param>
/// <param name="Length">The span's length in UTF-16 characters.</param>
/// <param name="Type">An NER type (PER, LOC, ORG, TME) or a pattern's type (PNR, ORGNR, PHONE, EMAIL, ADDRESS).</param>
/// <param name="Source">"ner", or the name of the pattern that found it.</param>
/// <param name="Text">The masked text.</param>
public sealed record MaskSpan(int Start, int Length, string Type, string Source, string Text)
{
    /// <summary>The offset just past the span.</summary>
    public int End => Start + Length;
}

/// <summary>
/// Suggests spans to mask in Swedish text: StanzaSharp's named entities plus patterns for what NER doesn't tag
/// (personnummer, organisationsnummer, phone numbers, e-mail, street and postal addresses). The result is a
/// suggestion for a person to review, not a guarantee: indirect identifiers ("hans syster, rektor på X-skolan")
/// are out of reach of both.
/// </summary>
public static partial class Masker
{
    /// <summary>The NER types masked by default. TME (dates and times) is opt-in, MSR (amounts) never masked.</summary>
    public static readonly IReadOnlySet<string> DefaultNerTypes = new HashSet<string> { "PER", "LOC", "ORG" };

    /// <summary>
    /// Finds the spans of <paramref name="text"/> to mask: the pattern matches plus the document's entities of
    /// <paramref name="nerTypes"/>, overlapping spans merged.
    /// </summary>
    public static IReadOnlyList<MaskSpan> Find(string text, Document? document, IReadOnlySet<string>? nerTypes = null)
    {
        nerTypes ??= DefaultNerTypes;
        var spans = new List<MaskSpan>(FindPatterns(text));
        if (document != null)
        {
            foreach (var entity in document.Entities)
            {
                if (!nerTypes.Contains(entity.Type) || entity.StartChar is not int start || entity.EndChar is not int end)
                    continue;
                spans.Add(new MaskSpan(start, end - start, entity.Type, "ner", text[start..end]));
            }
        }

        return Merge(text, spans);
    }

    /// <summary>The pattern matches alone (no model needed).</summary>
    public static IEnumerable<MaskSpan> FindPatterns(string text)
    {
        foreach (Match m in Email().Matches(text))
            yield return Span(m, "EMAIL", "email");
        foreach (Match m in IdNumber().Matches(text))
        {
            var digits = m.Groups["century"].Value + m.Groups["date"].Value + m.Groups["serial"].Value;
            var ten = digits[^10..];
            if (!Luhn(ten))
                continue;

            // The third digit of an organisationsnummer is 2 or more; a personnummer's month never is.
            if (ten[2] >= '2' && m.Groups["century"].Value is "" or "16")
                yield return Span(m, "ORGNR", "organisationsnummer");
            else if (IsDate(ten))
                yield return Span(m, "PNR", "personnummer");
        }

        foreach (Match m in Phone().Matches(text))
            yield return Span(m, "PHONE", "phone");
        foreach (Match m in Street().Matches(text))
            yield return Span(m, "ADDRESS", "street");
        foreach (Match m in PostalCode().Matches(text))
            yield return Span(m, "ADDRESS", "postal-code");
    }

    /// <summary>
    /// The text with each span replaced by a placeholder, the same one wherever the same text of the same
    /// type recurs (<c>[PER-1]</c> for every "Karin Lindqvist"), so a reviewer can still follow who is who.
    /// </summary>
    public static string Apply(string text, IReadOnlyList<MaskSpan> spans)
    {
        var numbers = new Dictionary<(string Type, string Text), int>();
        var counts = new Dictionary<string, int>();
        var result = new StringBuilder(text.Length);
        var at = 0;
        foreach (var span in spans)
        {
            result.Append(text, at, span.Start - at);
            if (!numbers.TryGetValue((span.Type, span.Text), out var n))
            {
                n = counts[span.Type] = counts.GetValueOrDefault(span.Type) + 1;
                numbers[(span.Type, span.Text)] = n;
            }

            result.Append('[').Append(span.Type).Append('-').Append(n).Append(']');
            at = span.End;
        }

        return result.Append(text, at, text.Length - at).ToString();
    }

    /// <summary>The Luhn check of Swedish personnummer and organisationsnummer, over their last ten digits.</summary>
    public static bool Luhn(string digits)
    {
        var sum = 0;
        for (var i = 0; i < digits.Length; i++)
        {
            var d = (digits[i] - '0') * (i % 2 == 0 ? 2 : 1);
            sum += d / 10 + d % 10;
        }

        return sum % 10 == 0;
    }

    // Sorted by start; overlapping spans become one, typed by the pattern if one is in it (a pattern is more
    // specific than NER: "Storgatan 14 B" is an ADDRESS, not the LOC "Storgatan"), an id number before any other
    // pattern (a personnummer born in the 2000s, 050101-1233, also looks like a phone number), else the longest span.
    private static List<MaskSpan> Merge(string text, List<MaskSpan> spans)
    {
        var merged = new List<MaskSpan>();
        foreach (var span in spans.OrderBy(s => s.Start).ThenByDescending(s => s.Length))
        {
            if (merged.Count > 0 && span.Start < merged[^1].End)
            {
                var last = merged[^1];
                var end = Math.Max(last.End, span.End);
                var winner = Rank(span) > Rank(last) ? span : last;
                merged[^1] = new MaskSpan(last.Start, end - last.Start, winner.Type, winner.Source, text[last.Start..end]);
            }
            else
            {
                merged.Add(span);
            }
        }

        return merged;

        static int Rank(MaskSpan s) => s.Length + s.Type switch
        {
            "PNR" or "ORGNR" => 2_000_000,
            _ when s.Source != "ner" => 1_000_000,
            _ => 0,
        };
    }

    private static MaskSpan Span(Match m, string type, string source) => new(m.Index, m.Length, type, source, m.Value);

    // A personnummer's date: month 01-12, day 01-31, or 61-91 for a samordningsnummer.
    private static bool IsDate(string ten)
    {
        var month = int.Parse(ten.AsSpan(2, 2));
        var day = int.Parse(ten.AsSpan(4, 2));
        if (day > 60)
            day -= 60;
        return month is >= 1 and <= 12 && day is >= 1 and <= 31;
    }

    // YYYYMMDD-NNNN, YYMMDD-NNNN, YYMMDD+NNNN (100 or older), with or without the separator; an organisationsnummer
    // is NNNNNN-NNNN (optionally with the prefix 16).
    [GeneratedRegex(@"(?<![\d-])(?<century>(?:19|20|16)?)(?<date>\d{6})[-+]?(?<serial>\d{4})(?![\d-])")]
    private static partial Regex IdNumber();

    // 07x-xxx xx xx, 08-xxx xx xx, 018-12 34 56, +46 70 123 45 67: a leading 0 or +46, then groups of digits.
    [GeneratedRegex(@"(?<![\w+-])(?:\+46[ -]?|0)\d{1,3}(?:[ -]?\d{2,4}){2,4}(?![\w-])")]
    private static partial Regex Phone();

    [GeneratedRegex(@"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")]
    private static partial Regex Email();

    // "Storgatan 14 B", "Västra Ringvägen 7": a word ending in a street suffix, optionally after a capitalized word,
    // then a number and an optional entrance letter.
    [GeneratedRegex(@"\b(?:\p{Lu}\p{Ll}+ )?\p{Lu}\p{Ll}*(?:gatan|vägen|gränd|torget|backen|stigen|allén|leden|platsen|esplanaden) \d+(?: ?[A-Z]\b)?")]
    private static partial Regex Street();

    // "753 20 Uppsala": a postal code and the town after it.
    [GeneratedRegex(@"(?<![\d-])\d{3} \d{2} \p{Lu}\p{Ll}+(?: \p{Lu}\p{Ll}+)?")]
    private static partial Regex PostalCode();
}
