using TasmanianDevil.Analyzer;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// Recognizes Swedish bank account numbers: a clearing number (four digits, five for Swedbank's <c>8NNN-N</c>) and an
/// account number, validated by the bank's own rules, e.g. <c>5000-1234566</c> (SEB) or <c>8327-9, 914 123 456-7</c>
/// (Swedbank).
/// </summary>
/// <remarks>
/// <para>The clearing number picks the bank and its rule from Bankgirot's table "Bankernas kontonummeruppbyggnad" (the
/// 2024-02-22 edition; a clearing number outside it is dropped): the account's length, and a modulus 11 check over the
/// clearing and account numbers (type 1) or a modulus 10 or 11 check over the account number alone (type 2). Swedbank's
/// five-digit clearing numbers must also pass their own Luhn check. The few Swedbank accounts whose check digit doesn't
/// work are missed.</para>
/// <para>Even valid, a number abstains and keeps a low base score: the checks pass for about one number in ten, and most
/// four-digit prefixes are some bank's clearing number. A context word (<c>konto</c>, <c>kontonummer</c>,
/// <c>clearingnummer</c>, ...) is needed. The international form is TasmanianDevil's own <c>IBAN_CODE</c>.</para>
/// <para>Register it next to <see cref="StanzaNerRecognizer"/> (for <c>sv</c> by default: pass the analysis
/// language).</para>
/// </remarks>
public sealed class SeBankAccountRecognizer : PatternRecognizer
{
    private const string NotAfter = @"(?<![\d+-]|\d[ .])";
    private const string NotBefore = @"(?![\d-]|[ .]\d)";

    private static readonly IReadOnlyList<Pattern> DefaultPatterns =
    [
        new Pattern(
            "Bank account (clearing number separated, context required)",
            NotAfter + @"(?:8\d{3}-\d|\d{4,5})(?:,? | ?- ?)\d(?:[ -]?\d){1,11}" + NotBefore,
            0.1),
        new Pattern("Bank account (11 to 15 digits, context required)", NotAfter + @"\d{11,15}" + NotBefore, 0.05),
    ];

    private static readonly IReadOnlyList<string> DefaultContext =
    [
        "konto", "kontonummer", "kontonr", "bankkonto", "bankkontonummer", "lönekonto", "sparkonto", "privatkonto",
        "personkonto", "clearingnummer", "clearingnr", "clearing", "clnr", "bank account", "account number",
    ];

    // (first clearing number, last, type, comment, shortest account, longest account); type 1 accounts are 7 digits.
    private static readonly (int From, int To, int Type, int Comment, int MinLength, int MaxLength)[] Banks =
    [
        (1100, 1199, 1, 1, 7, 7), // Nordea
        (1200, 1399, 1, 1, 7, 7), // Danske Bank
        (1400, 2099, 1, 1, 7, 7), // Nordea
        (2300, 2399, 1, 2, 7, 7), // Ålandsbanken
        (2400, 2499, 1, 1, 7, 7), // Danske Bank
        (3000, 3299, 1, 1, 7, 7), // Nordea
        (3300, 3300, 2, 1, 10, 10), // Nordea personkonto
        (3301, 3399, 1, 1, 7, 7), // Nordea
        (3400, 3409, 1, 1, 7, 7), // Länsförsäkringar Bank
        (3410, 3781, 1, 1, 7, 7), // Nordea
        (3782, 3782, 2, 1, 10, 10), // Nordea personkonto
        (3783, 3999, 1, 1, 7, 7), // Nordea
        (4000, 4999, 1, 2, 7, 7), // Nordea
        (5000, 5999, 1, 1, 7, 7), // SEB
        (6000, 6999, 2, 2, 8, 9), // Handelsbanken
        (7000, 7999, 1, 1, 7, 7), // Swedbank
        (8000, 8999, 2, 3, 6, 10), // Swedbank (five-digit clearing number)
        (9020, 9029, 1, 2, 7, 7), // Länsförsäkringar Bank
        (9040, 9049, 1, 2, 7, 7), // Citibank
        (9060, 9069, 1, 1, 7, 7), // Länsförsäkringar Bank
        (9070, 9079, 1, 1, 7, 7), // Multitude Bank
        (9100, 9109, 1, 2, 7, 7), // Nordnet Bank
        (9120, 9124, 1, 1, 7, 7), // SEB
        (9130, 9149, 1, 1, 7, 7), // SEB
        (9150, 9169, 1, 2, 7, 7), // Skandiabanken
        (9170, 9179, 1, 1, 7, 7), // Ikano Bank
        (9180, 9189, 2, 1, 10, 10), // Danske Bank
        (9190, 9199, 1, 2, 7, 7), // DNB Bank
        (9230, 9239, 1, 1, 7, 7), // Marginalen Bank
        (9250, 9259, 1, 1, 7, 7), // SBAB
        (9260, 9269, 1, 2, 7, 7), // DNB Bank
        (9270, 9279, 1, 1, 7, 7), // ICA Banken
        (9280, 9289, 1, 1, 7, 7), // Resurs Bank
        (9300, 9349, 2, 1, 10, 10), // Swedbank (formerly Sparbanken Öresund)
        (9390, 9399, 1, 2, 7, 7), // Landshypotek
        (9460, 9469, 1, 1, 7, 7), // Santander Consumer Bank
        (9470, 9479, 1, 2, 7, 7), // BNP Paribas
        (9500, 9549, 2, 3, 2, 8), // Nordea (Plusgirot)
        (9550, 9569, 1, 2, 7, 7), // Avanza Bank
        (9570, 9579, 2, 1, 10, 10), // Sparbanken Syd
        (9580, 9589, 1, 1, 7, 7), // Aion Bank
        (9590, 9599, 1, 2, 7, 7), // Erik Penser
        (9630, 9639, 1, 1, 7, 7), // Lån & Spar Bank Sverige
        (9640, 9649, 1, 2, 7, 7), // NOBA Bank Group
        (9660, 9669, 1, 2, 7, 7), // Svea Bank
        (9670, 9679, 1, 2, 7, 7), // JAK Medlemsbank
        (9680, 9689, 1, 1, 7, 7), // BlueStep Finans
        (9700, 9709, 1, 2, 7, 7), // Ekobanken
        (9710, 9719, 1, 2, 7, 7), // Lunar Bank
        (9750, 9759, 1, 2, 7, 7), // Northmill Bank
        (9780, 9789, 1, 2, 7, 7), // Klarna Bank
        (9880, 9889, 1, 2, 7, 7), // Riksgälden
        (9890, 9899, 2, 1, 10, 10), // Riksgälden
        (9960, 9969, 2, 3, 2, 8), // Nordea (Plusgirot)
    ];

    /// <summary>The entity type it detects, <c>SE_BANK_ACCOUNT</c>.</summary>
    public const string EntityType = "SE_BANK_ACCOUNT";

    /// <summary>Initializes a new instance of the <see cref="SeBankAccountRecognizer"/> class.</summary>
    public SeBankAccountRecognizer(string supportedEntity = EntityType, string supportedLanguage = "sv")
        : base(supportedEntity, patterns: DefaultPatterns, context: DefaultContext, supportedLanguage: supportedLanguage, countryCode: "se")
    {
    }

    /// <summary>
    /// Rejects a value whose clearing number is no bank's, whose account number has the wrong length or whose check digit
    /// fails; abstains (returns <see langword="null"/>) for a valid number, which still needs context.
    /// </summary>
    public override bool? ValidateResult(string patternText) => IsValid(SwedishCheckDigits.Digits(patternText)) ? null : false;

    private static bool IsValid(string digits)
    {
        var clearingLength = digits.StartsWith('8') ? 5 : 4;
        if (digits.Length < clearingLength + 2 || digits.Length > 15)
        {
            return false;
        }

        var clearing = int.Parse(digits.AsSpan(0, 4), provider: System.Globalization.CultureInfo.InvariantCulture);
        var account = digits.AsSpan(clearingLength);
        foreach (var bank in Banks)
        {
            if (clearing < bank.From || clearing > bank.To)
            {
                continue;
            }

            if (account.Length < bank.MinLength || account.Length > bank.MaxLength
                || (clearingLength == 5 && !SwedishCheckDigits.Mod10(digits.AsSpan(0, 5))))
            {
                return false;
            }

            return (bank.Type, bank.Comment) switch
            {
                (1, 1) => SwedishCheckDigits.Mod11(digits.AsSpan(1)), // the clearing number's last three digits and the account
                (1, _) => SwedishCheckDigits.Mod11(digits), // the whole clearing number and the account
                (2, 2) => SwedishCheckDigits.Mod11(account),
                _ => SwedishCheckDigits.Mod10(account),
            };
        }

        return false;
    }
}
