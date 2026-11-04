using System.Globalization;
using System.Text.RegularExpressions;
using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes dates of birth in Swedish text: a date right after <c>född</c>, <c>föddes</c>, <c>f.</c>,
/// <c>födelsedatum</c> or <c>födelsedag</c>, written <c>1985-03-12</c>, <c>19850312</c>, <c>12 mars 1985</c>,
/// <c>12:e mars 1985</c>, <c>12/3 1985</c>, <c>12.3.1985</c> or <c>12/3-85</c>, optionally after <c>den</c>.
/// </summary>
/// <remarks>
/// <para>The word before the date is part of the pattern (it is not in the span: "född &lt;DATE_OF_BIRTH&gt;"), so a
/// match that is a real calendar date, not in the future, counts with full confidence. Any other date is left to
/// <see cref="StanzaNerRecognizer"/>, which tags dates as <c>DATE_TIME</c>; a separate type lets dates of birth be masked
/// while other dates are kept. A <c>den</c> before the date is taken into the span, so it also covers the date NER tags
/// with it.</para>
/// <para>A year alone (<c>född 1985</c>) and a date without the word before it are not covered.</para>
/// </remarks>
public sealed partial class SeDateOfBirthRecognizer : PatternRecognizer
{
    private const string Months =
        "januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december"
        + "|jan|feb|mar|apr|jun|jul|aug|sept|sep|okt|nov|dec";

    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern(
            "Date of birth (after född, födelsedatum, ...)",
            @"(?<=(?<![\p{L}\d])(?:född|föddes|f\.|födelsedatum|födelsedag)(?: den)?:? )(?:den )?"
            + @"(?:(?:19|20)\d{2}-?[01]\d-?[0-3]\d"
            + $@"|[0-3]?\d(?::[ae])? (?:{Months})\.? (?:19|20)\d{{2}}"
            + @"|[0-3]?\d[./][01]?\d(?:[./ -](?:19|20)\d{2}|[./-]\d{2}))"
            + @"(?![\p{L}\d-]|[./]\d)",
            0.6),
    ];

    private static readonly string[] MonthNames =
        ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug", "sep", "okt", "nov", "dec"];

    /// <summary>The entity type it detects, <c>DATE_OF_BIRTH</c>.</summary>
    public const string EntityType = "DATE_OF_BIRTH";

    /// <summary>Initializes a new instance of the <see cref="SeDateOfBirthRecognizer"/> class.</summary>
    public SeDateOfBirthRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, supportedLanguage: supportedLanguage, countryCode: "se")
    {
    }

    /// <summary>
    /// Promotes a real calendar date that is not in the future (a two-digit year counts as either century) and rejects
    /// any other.
    /// </summary>
    public override bool? ValidateResult(string patternText)
    {
        var m = DateParts().Match(patternText);
        if (!m.Success)
        {
            return false;
        }

        var day = int.Parse(m.Groups["d"].Value, CultureInfo.InvariantCulture);
        var month = m.Groups["mon"].Success
            ? Array.IndexOf(MonthNames, m.Groups["mon"].Value[..3].ToLowerInvariant()) + 1
            : int.Parse(m.Groups["m"].Value, CultureInfo.InvariantCulture);
        var year = int.Parse(m.Groups["y"].Value, CultureInfo.InvariantCulture);
        if (year < 100)
        {
            // either century may be meant: 2000 is a leap year and 1900 isn't, so a 29 February counts
            year += 2000;
        }

        if (month is < 1 or > 12 || day < 1 || day > DateTime.DaysInMonth(year, month))
        {
            return false;
        }

        return m.Groups["y"].Value.Length == 2 || new DateOnly(year, month, day) <= DateOnly.FromDateTime(DateTime.Today);
    }

    // The parts of a matched date: y-m-d, d mon y, or d/m y.
    [GeneratedRegex(
        @"^(?:den )?(?:(?<y>\d{4})-?(?<m>\d{2})-?(?<d>\d{2})"
        + @"|(?<d>\d{1,2})(?::[ae])? (?<mon>\p{L}+)\.? (?<y>\d{4})"
        + @"|(?<d>\d{1,2})[./](?<m>\d{1,2})[./ -](?<y>\d{2}|\d{4}))$",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex DateParts();
}
