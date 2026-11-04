using System.Globalization;
using StanzaSharp.TasmanianDevil;
using TasmanianDevil;
using TasmanianDevil.Analyzer;

namespace StanzaSharp.Tests;

/// <summary>
/// The Swedish payment-number recognizers of StanzaSharp.TasmanianDevil (bankgiro, plusgiro, bank accounts), without
/// models. The numbers are synthetic: made to pass their check digits, not looked up.
/// </summary>
public class SwedishPaymentRecognizerTests
{
    [Theory]
    [InlineData("Betala till bankgiro 5050-1055 senast fredag.", "5050-1055")]
    [InlineData("Bankgiro: 505-0109", "505-0109")]
    [InlineData("BG 50501055.", "50501055")]
    public void Bankgiro_WithContext_IsMasked(string text, string expected) =>
        Assert.Equal(expected, Single(text, SeBankgiroRecognizer.EntityType));

    [Theory]
    [InlineData("Betala till bankgiro 5050-1056 senast fredag.")]
    [InlineData("Ordernummer 5050-1055 skickades.")]
    [InlineData("Bankgiro 1234-5678-9 är inget bankgironummer.")]
    [InlineData("Säsongen 2019-2020 bankgiro.")]
    public void Bankgiro_WithBadCheckDigitOrNoContext_IsNotMasked(string text) => None(text, SeBankgiroRecognizer.EntityType);

    [Theory]
    [InlineData("Skänk en gåva till plusgiro 12 34 56-6.", "12 34 56-6")]
    [InlineData("PG 4567-4", "4567-4")]
    [InlineData("Plusgironummer 1234566.", "1234566")]
    public void Plusgiro_WithContext_IsMasked(string text, string expected) =>
        Assert.Equal(expected, Single(text, SePlusgiroRecognizer.EntityType));

    [Theory]
    [InlineData("Skänk en gåva till plusgiro 12 34 56-7.")]
    [InlineData("Artikel 12 34 56-6 finns i lager.")]
    [InlineData("Plusgiro 1-2 saknas.")]
    public void Plusgiro_WithBadCheckDigitOrNoContext_IsNotMasked(string text) => None(text, SePlusgiroRecognizer.EntityType);

    [Theory]
    [InlineData("Lönekonto: 5000-1234560 (SEB).", "5000-1234560")]
    [InlineData("Kontonummer 5000 123 45 60.", "5000 123 45 60")]
    [InlineData("Konto 4000-1234567 i Nordea.", "4000-1234567")]
    [InlineData("Handelsbanken, konto 6000-123 456 789.", "6000-123 456 789")]
    [InlineData("Clearingnummer och konto: 8327-9, 914 123 456-8.", "8327-9, 914 123 456-8")]
    [InlineData("Personkonto 3300-7804012388 i Nordea.", "3300-7804012388")]
    [InlineData("Konto 9180-1234567897.", "9180-1234567897")]
    [InlineData("Bankkonto 50001234560.", "50001234560")]
    public void BankAccount_WithContext_IsMasked(string text, string expected) =>
        Assert.Equal(expected, Single(text, SeBankAccountRecognizer.EntityType));

    [Theory]
    [InlineData("Lönekonto: 5000-1234561.")]
    [InlineData("Konto 1000-1234567.")]
    [InlineData("Konto 6000-1234567.")]
    [InlineData("Konto 8327-8, 914 123 456-8.")]
    [InlineData("Ordernummer 5000-1234560 skickades.")]
    [InlineData("Konto: ring 0701-23 45 67.")]
    public void BankAccount_WithBadNumberOrNoContext_IsNotMasked(string text) => None(text, SeBankAccountRecognizer.EntityType);

    [Theory]
    [InlineData("")]
    [InlineData("1")]
    [InlineData("-")]
    [InlineData("0000-0000000")]
    [InlineData("1234567890123456789")]
    public void Validate_ReturnsFalse_RatherThanThrow(string value)
    {
        Assert.False(new SeBankgiroRecognizer().ValidateResult(value));
        Assert.False(new SePlusgiroRecognizer().ValidateResult(value));
        Assert.False(new SeBankAccountRecognizer().ValidateResult(value));
    }

    // The checks pass for about one number in ten: order numbers of each length must not be masked without context.
    [Theory]
    [InlineData(1_000, 9_999, "{0}-{1}")]
    [InlineData(1_000_000, 99_999_999, "{0}")]
    [InlineData(1_000, 9_999, "{0}-{1}{1}")]
    [InlineData(10_000_000_000, 999_999_999_999_999, "{0}")]
    public void PiiEngine_LeavesArbitraryNumbers(long min, long max, string format)
    {
        using var engine = Engine();
        var random = new Random(42);
        var masked = Enumerable.Range(0, 500)
            .Select(_ => string.Format(
                CultureInfo.InvariantCulture, format, random.NextInt64(min, max + 1), random.Next(1_000, 10_000)))
            .Count(n => engine.Analyze($"Order {n} skickad.").Any(r => r.EntityType.StartsWith("SE_", StringComparison.Ordinal)));
        Assert.Equal(0, masked);
    }

    [Fact]
    public void PiiEngine_MasksPaymentNumbers()
    {
        using var engine = Engine();
        Assert.Equal(
            "Betala till bankgiro <SE_BANKGIRO> eller plusgiro <SE_PLUSGIRO>; lönen går till konto <SE_BANK_ACCOUNT>.",
            engine.Anonymize("Betala till bankgiro 5050-1055 eller plusgiro 12 34 56-6; lönen går till konto 5000-1234560.").Text);
    }

    // All of the package's Swedish recognizers together: each number gets its own type.
    [Fact]
    public void PiiEngine_WithEverySwedishRecognizer_TypesEachNumber()
    {
        using var engine = new PiiEngine(new PiiOptions { Language = "sv" }, extraRecognizers:
        [
            new SePersonnummerRecognizer(), new SeOrganisationsnummerRecognizer(), new SeStreetAddressRecognizer(),
            new SePostnummerRecognizer(), new SeBankgiroRecognizer(), new SePlusgiroRecognizer(), new SeBankAccountRecognizer(),
            new SeVehicleRegistrationRecognizer(), new SeDateOfBirthRecognizer(),
        ]);
        Assert.Equal(
            "Sökande, personnummer <SE_PERSONNUMMER>, född <DATE_OF_BIRTH>, bor på <ADDRESS>, <SE_POSTNUMMER> Uppsala och äger "
            + "bilen <SE_VEHICLE_REGISTRATION>. Lönen betalas till konto <SE_BANK_ACCOUNT> av bolaget (organisationsnummer "
            + "<SE_ORGANISATIONSNUMMER>, bankgiro <SE_BANKGIRO>).",
            engine.Anonymize(
                "Sökande, personnummer 19780401-2388, född 1978-04-01, bor på Storgatan 14 B, 753 20 Uppsala och äger "
                + "bilen ABC 123. Lönen betalas till konto 8327-9, 914 123 456-8 av bolaget (organisationsnummer "
                + "559900-1236, bankgiro 5050-1055).").Text);
    }

    private static string Single(string text, string entityType)
    {
        using var engine = Engine();
        var result = Assert.Single(engine.Analyze(text), r => r.EntityType == entityType);
        return text[result.Start..result.End];
    }

    private static void None(string text, string entityType)
    {
        using var engine = Engine();
        Assert.DoesNotContain(engine.Analyze(text), r => r.EntityType == entityType);
    }

    private static PiiEngine Engine() =>
        new(new PiiOptions { Language = "sv" }, extraRecognizers:
            [new SeBankgiroRecognizer(), new SePlusgiroRecognizer(), new SeBankAccountRecognizer()]);
}

/// <summary>
/// The Swedish vehicle-registration and date-of-birth recognizers of StanzaSharp.TasmanianDevil, without models. The plates
/// and dates are invented.
/// </summary>
public class SwedishVehicleAndBirthDateRecognizerTests
{
    [Theory]
    [InlineData("Bilen med registreringsnummer ABC 123 stod parkerad.", "ABC 123")]
    [InlineData("Regnr: GHJ12K", "GHJ12K")]
    [InlineData("Fordonet XYZ 98W besiktades i maj.", "XYZ 98W")]
    public void Plate_WithContext_IsMasked(string text, string expected) =>
        Assert.Equal(expected, Single(text, SeVehicleRegistrationRecognizer.EntityType));

    [Theory]
    [InlineData("Artikel ABC 123 finns i lager.")]
    [InlineData("Bilen kostade EUR 100 per dag.")]
    [InlineData("Bilen kostade SEK 250 per dag.")]
    [InlineData("Bilen abc 123 i gemener.")]
    [InlineData("Bilen ABI 123 har ett I.")]
    [InlineData("Bilen ABC 12O har ett O sist.")]
    [InlineData("Bilen ABCD 123 och ABC 1234.")]
    [InlineData("Bilen enligt SFS 2018:123.")]
    public void NotAPlateOrNoContext_IsNotMasked(string text) => None(text, SeVehicleRegistrationRecognizer.EntityType);

    [Theory]
    [InlineData("Hon är född 1985-03-12 i Uppsala.", "1985-03-12")]
    [InlineData("Hon föddes den 12 mars 1985 i Uppsala.", "den 12 mars 1985")]
    [InlineData("Född: 12/3-85", "12/3-85")]
    [InlineData("Karin Lindqvist, f. 19850312, bor i Uppsala.", "19850312")]
    [InlineData("Födelsedatum: 12.3.1985", "12.3.1985")]
    [InlineData("Han är född 12:e mars 1985.", "12:e mars 1985")]
    [InlineData("Född 29/2-00.", "29/2-00")]
    [InlineData("född 1 dec. 1999", "1 dec. 1999")]
    public void DateOfBirth_AfterFodd_HasFullConfidence(string text, string expected)
    {
        var result = Assert.Single(new SeDateOfBirthRecognizer().Analyze(text, []));
        Assert.Equal(
            (SeDateOfBirthRecognizer.EntityType, expected, EntityRecognizer.MaxScore),
            (result.EntityType, text[result.Start..result.End], result.Score));
    }

    [Theory]
    [InlineData("Hon är född 1985-02-30.")]
    [InlineData("Hon är född 2999-01-01.")]
    [InlineData("Mötet hölls 1985-03-12.")]
    [InlineData("Hon är född 1985 i Uppsala.")]
    [InlineData("Personnummer: född 19850312-1234.")]
    [InlineData("Avtalet ref. 12 mars 1985.")]
    [InlineData("Född 31/4-85.")]
    [InlineData("Ofödd 1985-03-12.")]
    public void NotADateOfBirth_IsNotFound(string text) => Assert.Empty(new SeDateOfBirthRecognizer().Analyze(text, []));

    [Fact]
    public void PiiEngine_MasksPlateAndDateOfBirth()
    {
        using var engine = Engine();
        Assert.Equal(
            "Föraren, född <DATE_OF_BIRTH>, körde bilen <SE_VEHICLE_REGISTRATION>.",
            engine.Anonymize("Föraren, född 1985-03-12, körde bilen ABC 123.").Text);
    }

    private static string Single(string text, string entityType)
    {
        using var engine = Engine();
        var result = Assert.Single(engine.Analyze(text), r => r.EntityType == entityType);
        return text[result.Start..result.End];
    }

    private static void None(string text, string entityType)
    {
        using var engine = Engine();
        Assert.DoesNotContain(engine.Analyze(text), r => r.EntityType == entityType);
    }

    private static PiiEngine Engine() =>
        new(new PiiOptions { Language = "sv" }, extraRecognizers:
            [new SeVehicleRegistrationRecognizer(), new SeDateOfBirthRecognizer()]);
}
