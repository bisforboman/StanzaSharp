using System.Text.RegularExpressions;
using TasmanianDevil;
using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish street addresses as TasmanianDevil's <c>ADDRESS</c>: a street name in its definite form
/// (<c>Storgatan</c>, <c>Västra Ringvägen</c>, <c>S:t Eriksgatan</c>) followed by a house number, an optional entrance
/// letter and an optional apartment number, e.g. <c>Storgatan 14 B</c> or <c>Kungsvägen 3, lgh 1102</c>.
/// </summary>
/// <remarks>
/// <para>The street name must end in a definite street suffix (<c>-gatan</c>, <c>-vägen</c>, <c>-torget</c>, ...) and be
/// capitalized, or written in capitals as in address blocks. The indefinite forms are left out because ordinary words end
/// in them (<c>Handlingsplan 2026</c>, <c>Arbetsplats 3</c>). A word before the name counts only when it is one of the usual
/// prefixes (<c>Västra</c>, <c>Stora</c>, <c>Gamla</c>, <c>Sankt</c>, ...), so a person's name before the street isn't
/// taken in.</para>
/// <para>The whole address is one span, so it replaces the shorter location <see cref="StanzaNerRecognizer"/> tags for the
/// street name alone (TasmanianDevil's anonymizer drops a span contained in another). It uses <see cref="PiiEntities.Address"/>,
/// the type TasmanianDevil's remote detectors report addresses as, so one operator covers both.</para>
/// </remarks>
public sealed class SeStreetAddressRecognizer : PatternRecognizer
{
    private static readonly string[] Suffixes =
    [
        "gatan", "vägen", "gränd", "gränden", "torget", "platsen", "allén", "backen", "stigen", "leden", "esplanaden", "kajen",
        "gången", "stråket", "promenaden", "parken", "slingan", "svängen", "kroken", "liden", "udden",
    ];

    private static readonly string[] Prefixes =
    [
        "Västra", "Östra", "Norra", "Södra", "Stora", "Lilla", "Gamla", "Nya", "Övre", "Nedre", "Inre", "Yttre", "Sankt", "S:t",
    ];

    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern("Swedish street address", Build(), 0.6),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "adress", "gatuadress", "bostadsadress", "besöksadress", "utdelningsadress", "folkbokföringsadress", "bor", "boende",
        "address",
    ];

    /// <summary>The entity type it detects, TasmanianDevil's <c>ADDRESS</c>.</summary>
    public const string EntityType = PiiEntities.Address;

    /// <summary>Initializes a new instance of the <see cref="SeStreetAddressRecognizer"/> class.</summary>
    public SeStreetAddressRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage,
            regexOptions: DefaultRegexOptions & ~RegexOptions.IgnoreCase, countryCode: "se")
    {
    }

    private static string Build()
    {
        var suffix = string.Join('|', Suffixes);
        var upperSuffix = string.Join('|', Suffixes.Select(s => s.ToUpperInvariant()));
        var prefix = string.Join('|', Prefixes.Concat(Prefixes.Select(p => p.ToUpperInvariant())).Select(Regex.Escape));

        // "Storgatan" or "STORGATAN" (a hyphen inside the name is allowed); a name without a definite suffix, such as
        // "Drottning Kristinas väg", is not covered.
        var name = $@"(?:\p{{Lu}}[\p{{Ll}}-]*(?:{suffix})|\p{{Lu}}[\p{{Lu}}-]*(?:{upperSuffix}))";
        return $@"(?<![\p{{L}}\d-])(?:(?:{prefix}) )?{name} [1-9]\d{{0,3}}(?:-\d{{1,4}})?"
            + @"(?:[A-Za-z]| [A-Z])?(?![\p{L}\d])"
            + @"(?:,? (?:lgh|lägenhet|LGH) ?\d{4})?"
            + @"(?![\p{L}\d])";
    }
}
