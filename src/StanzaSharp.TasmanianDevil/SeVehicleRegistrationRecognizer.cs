using System.Text.RegularExpressions;
using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish vehicle registration numbers (registreringsnummer): three capital letters and three digits, or
/// since 2019 two digits and a letter, with or without a space, e.g. <c>ABC 123</c> or <c>ABC 12D</c>.
/// </summary>
/// <remarks>
/// <para>The letters leave out those Transportstyrelsen doesn't use (I, Q, V, Å, Ä, Ö; also O in the last place), and the
/// pattern is case-sensitive. The shape is still shared with currency amounts and codes (<c>EUR 100</c>, <c>ABC 123</c>),
/// so a match keeps a base score below TasmanianDevil's default threshold and needs a context word
/// (<c>registreringsnummer</c>, <c>regnr</c>, <c>bilen</c>, ...); currency codes are never taken. Personalized plates
/// (<c>MAMMA1</c>) are not covered.</para>
/// <para>Register it next to <see cref="StanzaNerRecognizer"/> (for <c>sv</c> by default: pass the analysis
/// language).</para>
/// </remarks>
public sealed class SeVehicleRegistrationRecognizer : PatternRecognizer
{
    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern(
            "Registreringsnummer (context required)",
            @"(?<![\p{L}\d-])[A-HJ-PR-UW-Z]{3} ?\d{2}[\dA-HJ-NPR-UW-Z](?![\p{L}\d-])",
            0.3),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "registreringsnummer", "regnummer", "regnr", "reg.nr", "reg nr", "registreringsskylt", "nummerplåt", "bil", "bilen",
        "bilens", "fordon", "fordonet", "fordonets", "registration", "licence plate", "license plate",
    ];

    // ISO 4217 codes a plate's letters could spell, for amounts such as "EUR 100"
    private static readonly HashSet<string> Currencies =
    [
        "AUD", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HUF", "JPY", "MXN", "NOK", "NZD", "PLN", "RUB", "SEK",
        "SGD", "THB", "TRY", "USD", "ZAR",
    ];

    /// <summary>The entity type it detects, <c>SE_VEHICLE_REGISTRATION</c>.</summary>
    public const string EntityType = "SE_VEHICLE_REGISTRATION";

    /// <summary>Initializes a new instance of the <see cref="SeVehicleRegistrationRecognizer"/> class.</summary>
    public SeVehicleRegistrationRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage,
            regexOptions: DefaultRegexOptions & ~RegexOptions.IgnoreCase, countryCode: "se")
    {
    }

    /// <summary>Drops an amount in a currency (<c>EUR 100</c>, <c>SEK 250</c>).</summary>
    public override bool? InvalidateResult(string patternText) =>
        patternText.Length >= 3 && Currencies.Contains(patternText[..3]) ? true : null;
}
