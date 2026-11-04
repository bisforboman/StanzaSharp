using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish bankgiro numbers: seven or eight digits written <c>NNN-NNNN</c> or <c>NNNN-NNNN</c>, whose last
/// digit is a Luhn check digit, e.g. <c>5050-1055</c>.
/// </summary>
/// <remarks>
/// <para>Validation drops a match whose check digit fails. The Luhn check passes for one number in ten, and the shape is
/// shared with year ranges and order numbers, so a valid number abstains and keeps its base score: a context word
/// (<c>bankgiro</c>, <c>bg</c>, ...) is needed to clear TasmanianDevil's default threshold. Written with its hyphen it
/// scores higher than a bare run of digits once the context word is found.</para>
/// <para>Register it next to <see cref="StanzaNerRecognizer"/> (for <c>sv</c> by default: pass the analysis
/// language).</para>
/// </remarks>
public sealed class SeBankgiroRecognizer : PatternRecognizer
{
    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern("Bankgiro (with hyphen, context required)", @"(?<![\d+-]|\d[ .])\d{3,4}-\d{4}(?![\d-]|[ .]\d)", 0.3),
        new Pattern("Bankgiro (7 or 8 digits, context required)", @"(?<![\d+-]|\d[ .])\d{7,8}(?![\d-]|[ .]\d)", 0.05),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "bankgiro", "bankgironummer", "bankgironr", "bankgirot", "bg", "bg-nr", "bg-nummer", "bank giro",
    ];

    /// <summary>The entity type it detects, <c>SE_BANKGIRO</c>.</summary>
    public const string EntityType = "SE_BANKGIRO";

    /// <summary>Initializes a new instance of the <see cref="SeBankgiroRecognizer"/> class.</summary>
    public SeBankgiroRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage, countryCode: "se")
    {
    }

    /// <summary>
    /// Rejects a value that is not seven or eight digits or whose Luhn check digit fails; abstains (returns
    /// <see langword="null"/>) for a valid number, which still needs context.
    /// </summary>
    public override bool? ValidateResult(string patternText)
    {
        var digits = SwedishCheckDigits.Digits(patternText);
        return digits.Length is 7 or 8 && SwedishCheckDigits.Mod10(digits) ? null : false;
    }
}
