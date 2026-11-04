using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish plusgiro numbers: up to eight digits whose last is a Luhn check digit, written with a hyphen before
/// the check digit and the others in pairs from the right, e.g. <c>12 34 56-7</c> or <c>1234567-8</c>.
/// </summary>
/// <remarks>
/// <para>Validation drops a match whose check digit fails. A plusgiro number has no distinctive shape (<c>45 67-8</c>
/// could be anything), so every match abstains and keeps a low base score: a context word (<c>plusgiro</c>, <c>pg</c>,
/// ...) is needed. Numbers of fewer than four digits are left out: they are written like ordinary numbers.</para>
/// <para>Register it next to <see cref="StanzaNerRecognizer"/> (for <c>sv</c> by default: pass the analysis
/// language).</para>
/// </remarks>
public sealed class SePlusgiroRecognizer : PatternRecognizer
{
    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern("Plusgiro (with hyphen, context required)", @"(?<![\d+-]|\d[ .])\d{1,2}(?: ?\d{2}){1,3}-\d(?![\d-]|[ .]\d)", 0.1),
        new Pattern("Plusgiro (4 to 8 digits, context required)", @"(?<![\d+-]|\d[ .])\d{4,8}(?![\d-]|[ .]\d)", 0.05),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "plusgiro", "plusgironummer", "plusgironr", "plusgirot", "postgiro", "pg", "pg-nr", "pg-nummer",
    ];

    /// <summary>The entity type it detects, <c>SE_PLUSGIRO</c>.</summary>
    public const string EntityType = "SE_PLUSGIRO";

    /// <summary>Initializes a new instance of the <see cref="SePlusgiroRecognizer"/> class.</summary>
    public SePlusgiroRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage, countryCode: "se")
    {
    }

    /// <summary>
    /// Rejects a value that is not four to eight digits or whose Luhn check digit fails; abstains (returns
    /// <see langword="null"/>) for a valid number, which still needs context.
    /// </summary>
    public override bool? ValidateResult(string patternText)
    {
        var digits = SwedishCheckDigits.Digits(patternText);
        return digits.Length is >= 4 and <= 8 && SwedishCheckDigits.Mod10(digits) ? null : false;
    }
}
