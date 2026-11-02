using System.Text.RegularExpressions;
using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish postal codes (postnummer): five digits written <c>NNN NN</c> or <c>NNNNN</c>, from 100 00 to 984 99,
/// e.g. <c>753 20</c>.
/// </summary>
/// <remarks>
/// <para>Postal codes have no check digit, so the shape decides. A code with the country prefix (<c>SE-753 20</c>) or
/// directly followed by its town (<c>753 20 Uppsala</c>, or <c>753 20 UPPSALA</c> as written in address blocks) gets a
/// score above TasmanianDevil's default threshold. A bare five-digit number keeps a low base score, so a context word
/// (<c>postnummer</c>, <c>adress</c>, <c>box</c>, ...) is needed, like TasmanianDevil's Dutch and German postal codes.</para>
/// <para>The span is the code with its prefix; the town is left to <see cref="StanzaNerRecognizer"/>, which tags it as a
/// location. Register it with <see cref="SeStreetAddressRecognizer"/> (for <c>sv</c> by default: pass the analysis
/// language).</para>
/// </remarks>
public sealed class SePostnummerRecognizer : PatternRecognizer
{
    // 100 00 to 984 99; a run of digits, a part of a number (1 234 567) or of a phone number never matches.
    private const string Code = @"[1-9]\d{2} ?\d{2}";
    private const string NotAfter = @"(?<![\d+-]|\d[ .])";
    private const string NotBefore = @"(?![\d-]|[ .]\d)";

    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern("Postnummer (SE- prefix)", @"(?<![\w-])SE-? ?" + Code + NotBefore, 0.6),
        new Pattern("Postnummer (followed by the town)", NotAfter + @"\b" + Code + @"(?= \p{Lu}[\p{L}-]+(?![\p{L}\d]))", 0.6),
        new Pattern("Postnummer (5 digits, context required)", NotAfter + @"\b" + Code + NotBefore, 0.05),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "postnummer", "postnr", "postadress", "adress", "utdelningsadress", "besöksadress", "gatuadress", "bostadsadress",
        "folkbokföringsadress", "postort", "ort", "box", "postbox", "postal code", "zip",
    ];

    /// <summary>The entity type it detects, <c>SE_POSTNUMMER</c>.</summary>
    public const string EntityType = "SE_POSTNUMMER";

    /// <summary>Initializes a new instance of the <see cref="SePostnummerRecognizer"/> class.</summary>
    public SePostnummerRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage,
            regexOptions: DefaultRegexOptions & ~RegexOptions.IgnoreCase, countryCode: "se")
    {
    }

    /// <summary>Rejects codes above 984 99, the highest Swedish postal code.</summary>
    public override bool? ValidateResult(string patternText)
    {
        var digits = new string([.. patternText.Where(char.IsAsciiDigit)]);
        return digits.Length == 5 && int.Parse(digits, System.Globalization.CultureInfo.InvariantCulture) <= 98499 ? null : false;
    }
}
