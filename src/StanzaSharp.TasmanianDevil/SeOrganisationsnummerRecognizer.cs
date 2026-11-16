using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish organisation numbers (organisationsnummer, also the VAT and tax number of a company): ten digits
/// <c>NNNNNN-NNNN</c>, optionally prefixed <c>16</c>, whose third digit is 2 or more and whose last is a Luhn check digit.
/// </summary>
/// <remarks>
/// <para>Validation drops a match whose third digit or check digit fails. The rules come from the public specifications:
/// lagen (1974:174) om identitetsbeteckning för juridiska personer m.fl. 4 § (ten digits, the last a check digit),
/// Skatteverket's brochure SKV 709 (the third digit is at least 2, so that no organisation number looks like a
/// personnummer) and its field list for the Pisa XML files (the twelve-digit form adds two fictitious century digits,
/// <c>16</c>). Neither names the check digit's method; it is checked with the personnummer's modulus 10 (SKV 704), which
/// the organisation numbers Skatteverket publishes for its test environments all pass.</para>
/// <para>Written with its hyphen, a valid number is promoted to full confidence; a bare run of ten digits abstains and keeps
/// its low base score, so a context word (<c>organisationsnummer</c>, <c>orgnr</c>, ...) is needed.</para>
/// <para>Register it next to <see cref="StanzaNerRecognizer"/> (for <c>sv</c> by default: pass the analysis
/// language).</para>
/// </remarks>
public sealed class SeOrganisationsnummerRecognizer : PatternRecognizer
{
    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern("Organisationsnummer (with hyphen)", @"(?<![\d+-])(?:16)?\d{6}-\d{4}(?![\d-])", 0.5),
        new Pattern("Organisationsnummer (10 digits, context required)", @"(?<![\d+-])(?:16)?\d{10}(?![\d-])", 0.05),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "organisationsnummer", "organisationsnr", "orgnr", "org.nr", "org nr", "momsregistreringsnummer", "bolagsverket",
        "organisation number",
    ];

    /// <summary>The entity type it detects, <c>SE_ORGANISATIONSNUMMER</c>.</summary>
    public const string EntityType = "SE_ORGANISATIONSNUMMER";

    /// <summary>Initializes a new instance of the <see cref="SeOrganisationsnummerRecognizer"/> class.</summary>
    public SeOrganisationsnummerRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage, countryCode: "se")
    {
    }

    /// <summary>
    /// Rejects a value whose third digit is below 2 or whose Luhn check digit fails; promotes a valid hyphenated number, and
    /// abstains (returns <see langword="null"/>) for a valid bare number, which still needs context.
    /// </summary>
    public override bool? ValidateResult(string patternText)
    {
        var digits = SwedishIdNumber.LastTenDigits(patternText, out var hasSeparator);
        if (digits == null || digits[2] < '2' || !SwedishIdNumber.Luhn(digits) || patternText.Contains('+'))
        {
            return false;
        }

        // a 12-digit match must carry the 16 prefix, not a century
        if (patternText.Replace("-", "", StringComparison.Ordinal).Length == 12 && !patternText.StartsWith("16", StringComparison.Ordinal))
        {
            return false;
        }

        return hasSeparator ? true : null;
    }
}
