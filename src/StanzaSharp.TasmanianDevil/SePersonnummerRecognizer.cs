using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish personal identity numbers (personnummer) and coordination numbers (samordningsnummer): a birth date
/// <c>YYMMDD</c> or <c>YYYYMMDD</c>, a four-digit serial whose last digit is a Luhn check digit, written with
/// <c>-</c> (or <c>+</c> from the year someone turns 100) or without a separator.
/// </summary>
/// <remarks>
/// <para>Validation drops a match whose check digit or date fails. The rules come from the public specifications:
/// folkbokföringslagen (1991:481) 18 § (birth date <c>YYMMDD</c>, a three-digit birth number, a check digit, the
/// <c>-</c> that becomes <c>+</c> the year someone turns 100, and, when a date's numbers run out, a nearby day of the
/// same month, so the date is always a real one), lagen (2022:1697) om samordningsnummer 8 § (the same, with 60 added to
/// the day), and Skatteverket's brochure SKV 704 (the check digit: modulus 10 with weights 2, 1, 2, ... over the nine
/// digits before it). The date must exist: 29 February only in a leap year (a ten-digit number: if either century it
/// could have makes it one).</para>
/// <para>Written with its separator, a valid number is distinctive enough to promote to full confidence. A bare run of ten
/// or twelve digits is not: the Luhn check passes for one in ten numbers and phone numbers such as <c>0701234567</c>
/// look like dates, so a passing bare number abstains and keeps its low base score, and a context word
/// (<c>personnummer</c>, <c>pnr</c>, <c>född</c>, ...) is needed to clear the threshold, as for <c>NL_BSN</c>.</para>
/// <para>Organisation numbers have a third digit of 2 or more (no month can), so they never validate here; see
/// <see cref="SeOrganisationsnummerRecognizer"/>.</para>
/// <para>Register it next to <see cref="StanzaNerRecognizer"/> (for <c>sv</c> by default: pass the analysis
/// language).</para>
/// </remarks>
public sealed class SePersonnummerRecognizer : PatternRecognizer
{
    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern("Personnummer (with separator)", @"(?<![\d+-])(?:(?:19|20)\d{6}|\d{6})[-+]\d{4}(?![\d-])", 0.5),
        new Pattern("Personnummer (12 digits, context required)", @"(?<![\d+-])(?:19|20)\d{10}(?![\d-])", 0.1),
        new Pattern("Personnummer (10 digits, context required)", @"(?<![\d+-])\d{10}(?![\d-])", 0.05),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "personnummer", "personnr", "pnr", "samordningsnummer", "samordningsnr", "född", "födelsenummer",
        "personal identity number",
    ];

    /// <summary>The entity type it detects, <c>SE_PERSONNUMMER</c>.</summary>
    public const string EntityType = "SE_PERSONNUMMER";

    /// <summary>Initializes a new instance of the <see cref="SePersonnummerRecognizer"/> class.</summary>
    public SePersonnummerRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage, countryCode: "se")
    {
    }

    /// <summary>
    /// Rejects a value whose date or Luhn check digit fails; promotes a valid number written with its separator, and
    /// abstains (returns <see langword="null"/>) for a valid bare number, which still needs context.
    /// </summary>
    public override bool? ValidateResult(string patternText)
    {
        var digits = SwedishIdNumber.LastTenDigits(patternText, out var hasSeparator);
        if (digits == null || !IsDate(patternText, digits) || !SwedishIdNumber.Luhn(digits))
        {
            return false;
        }

        return hasSeparator ? true : null;
    }

    // YYMMDD (YYYYMMDD in the twelve-digit form): a real date, the day + 60 for a samordningsnummer
    private static bool IsDate(string patternText, string digits)
    {
        var month = ((digits[2] - '0') * 10) + (digits[3] - '0');
        var day = ((digits[4] - '0') * 10) + (digits[5] - '0');
        if (day > 60)
        {
            day -= 60;
        }

        if (month is < 1 or > 12 || day < 1)
        {
            return false;
        }

        var yy = ((digits[0] - '0') * 10) + (digits[1] - '0');
        var leap = SwedishIdNumber.Century(patternText) is { } century ? DateTime.IsLeapYear((century * 100) + yy) : yy % 4 == 0;
        var days = month == 2 ? (leap ? 29 : 28) : DateTime.DaysInMonth(2001, month);
        return day <= days;
    }
}
