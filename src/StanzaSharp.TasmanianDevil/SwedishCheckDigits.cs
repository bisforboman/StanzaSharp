namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// The check-digit schemes of Swedish payment numbers (Bankgirot's "Bankernas kontonummeruppbyggnad"): modulus 10 (Luhn)
/// for bankgiro, plusgiro and type 2 bank accounts, modulus 11 for type 1 accounts and Handelsbanken.
/// </summary>
internal static class SwedishCheckDigits
{
    /// <summary>The digits of <paramref name="text"/>, in order, without separators.</summary>
    public static string Digits(string text) => new([.. text.Where(char.IsAsciiDigit)]);

    /// <summary>The Luhn check from the rightmost digit (weights 1, 2, 1, 2, ...); false for an empty or all-zero number.</summary>
    public static bool Mod10(ReadOnlySpan<char> digits)
    {
        var sum = 0;
        for (var i = 0; i < digits.Length; i++)
        {
            var d = (digits[^(i + 1)] - '0') * (i % 2 == 0 ? 1 : 2);
            sum += (d / 10) + (d % 10);
        }

        return sum != 0 && sum % 10 == 0;
    }

    /// <summary>The modulus 11 check, weights 1, 2, ..., 10, 1 from the rightmost digit; false for an all-zero number.</summary>
    public static bool Mod11(ReadOnlySpan<char> digits)
    {
        var sum = 0;
        for (var i = 0; i < digits.Length; i++)
        {
            sum += (digits[^(i + 1)] - '0') * ((i % 10) + 1);
        }

        return sum != 0 && sum % 11 == 0;
    }
}
