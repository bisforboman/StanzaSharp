using StanzaSharp.TasmanianDevil;
using TasmanianDevil;
using TasmanianDevil.Analyzer;

namespace StanzaSharp.Tests;

/// <summary>StanzaSharp.TasmanianDevil (#234): <see cref="StanzaNerRecognizer"/> and <see cref="StanzaTokenNormalizer"/>.</summary>
public class TasmanianDevilTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    private const string Text = "Barack Obama was born in Hawaii on August 4, 1961.";

    [ModelFact]
    public void Recognizer_MapsStanzasTypesToTasmanianDevils()
    {
        var recognizer = new StanzaNerRecognizer(pipelines.Get(processors: "tokenize,ner"));
        var results = recognizer.Analyze(Text, []);
        Assert.Equal(
            [("PERSON", "Barack Obama", "PERSON"), ("LOCATION", "Hawaii", "GPE"), ("DATE_TIME", "August 4, 1961", "DATE")],
            results.Select(r => (r.EntityType, Text[r.Start..r.End], (string)r.RecognitionMetadata!["stanzaType"])));
        Assert.All(results, r => Assert.Equal(0.85, r.Score));
        Assert.All(results, r => Assert.Equal(nameof(StanzaNerRecognizer), r.RecognitionMetadata![RecognizerResult.RecognizerNameKey]));
        Assert.Equal(["PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME"], recognizer.SupportedEntities);
    }

    [ModelFact]
    public void Recognizer_ReturnsOnlyTheRequestedTypes()
    {
        var recognizer = new StanzaNerRecognizer(pipelines.Get(processors: "tokenize,ner"), score: 0.6);
        Assert.Equal(["Barack Obama"], recognizer.Analyze(Text, [PiiEntities.Person]).Select(r => Text[r.Start..r.End]));
        Assert.Empty(recognizer.Analyze(Text, [PiiEntities.EmailAddress]));
        Assert.Empty(recognizer.Analyze("  ", []));
        Assert.All(recognizer.Analyze(Text, []), r => Assert.Equal(0.6, r.Score));
    }

    // In TasmanianDevil's engine its spans are anonymized with the regex ones.
    [ModelFact]
    public void PiiEngine_AnonymizesStanzasEntitiesWithItsOwn()
    {
        using var engine = new PiiEngine(new PiiOptions(), extraRecognizers: [new StanzaNerRecognizer(pipelines.Get(processors: "tokenize,ner"))]);
        var result = engine.Anonymize("Barack Obama met Angela Merkel in Berlin. Write to jane@contoso.com.");
        Assert.Equal("<PERSON> met <PERSON> in <LOCATION>. Write to <EMAIL_ADDRESS>.", result.Text);
    }

    [ModelFact]
    public void Normalizer_GivesLemmasStopWordsAndPunctuation()
    {
        var normalizer = new StanzaTokenNormalizer(pipelines.Get(processors: "tokenize,mwt,pos,lemma"));
        const string text = "The children were playing with their dogs, weren't they?";
        var tokens = normalizer.Tokenize(text);
        Assert.All(tokens, t => Assert.Equal(t.Token, text[t.Start..t.End]));
        Assert.Equal(["The", "children", "were", "playing", "with", "their", "dogs", ",", "weren't", "they", "?"], tokens.Select(t => t.Token));

        Assert.Equal(["the", "child", "be", "play", "with", "their", "dog", ",", "be not", "they", "?"], tokens.Select(t => normalizer.Normalize(t.Token)));
        Assert.Equal(
            ["The", "were", "with", "their", "weren't", "they"],
            tokens.Where(t => normalizer.IsStopWord(t.Token)).Select(t => t.Token));
        Assert.Equal([",", "?"], tokens.Where(t => normalizer.IsPunctuation(t.Token)).Select(t => t.Token));
    }

    // A word normalizes the same whether or not a text that contains it was tokenized first.
    [ModelFact]
    public void Normalizer_IsTheSameBeforeAndAfterTokenize()
    {
        var pipeline = pipelines.Get(processors: "tokenize,mwt,pos,lemma");
        string[] words = ["Cards", "saw", "leaves", "better", "U.S."];
        var fresh = words.Select(new StanzaTokenNormalizer(pipeline).Normalize).ToList();
        var warmed = new StanzaTokenNormalizer(pipeline);
        warmed.Tokenize("He saw the cards and the leaves fall in the U.S. better than before.");
        Assert.Equal(fresh, words.Select(warmed.Normalize));
    }

    [ModelFact]
    public void Adapters_NeedTheirProcessors()
    {
        Assert.Throws<ArgumentException>(() => new StanzaNerRecognizer(pipelines.Get(processors: "tokenize,mwt,pos,lemma")));
        Assert.Throws<ArgumentException>(() => new StanzaTokenNormalizer(pipelines.Get(processors: "tokenize,ner")));
    }
}

/// <summary>
/// The Swedish id-number recognizers of StanzaSharp.TasmanianDevil, without models. Personnummer are Skatteverket's
/// published test personnummer and the samordningsnummer one of its test samordningsnummer (open data, never issued);
/// the organisationsnummer is synthetic.
/// </summary>
public class SwedishIdRecognizerTests
{
    [Theory]
    [InlineData("19780401-2388")]
    [InlineData("780401-2388")]
    [InlineData("900101+9802")]
    [InlineData("20260161-2381")]
    public void Personnummer_WithSeparator_HasFullConfidence(string pnr)
    {
        var result = Assert.Single(new SePersonnummerRecognizer().Analyze($"Personnummer {pnr}.", []));
        Assert.Equal((SePersonnummerRecognizer.EntityType, EntityRecognizer.MaxScore), (result.EntityType, result.Score));
    }

    [Theory]
    [InlineData("19780401-2389")]
    [InlineData("781301-2382")]
    [InlineData("780400-2389")]
    [InlineData("559900-1236")]
    public void Personnummer_WithBadCheckDigitOrDate_IsDropped(string pnr) =>
        Assert.Empty(new SePersonnummerRecognizer().Analyze($"Personnummer {pnr}.", []));

    [Theory]
    [InlineData("559900-1236")]
    [InlineData("16559900-1236")]
    public void Organisationsnummer_WithHyphen_HasFullConfidence(string orgnr)
    {
        var result = Assert.Single(new SeOrganisationsnummerRecognizer().Analyze($"Organisationsnummer {orgnr}.", []));
        Assert.Equal((SeOrganisationsnummerRecognizer.EntityType, EntityRecognizer.MaxScore), (result.EntityType, result.Score));
    }

    [Theory]
    [InlineData("559900-1237")]
    [InlineData("780401-2388")]
    [InlineData("19559900-1236")]
    public void Organisationsnummer_WithBadThirdOrCheckDigit_IsDropped(string orgnr) =>
        Assert.Empty(new SeOrganisationsnummerRecognizer().Analyze($"Organisationsnummer {orgnr}.", []));

    [Theory]
    [InlineData("")]
    [InlineData("1")]
    [InlineData("-1234")]
    [InlineData("12345-6789")]
    [InlineData("123456789012345")]
    public void Validate_ReturnsFalse_RatherThanThrow(string value)
    {
        Assert.False(new SePersonnummerRecognizer().ValidateResult(value));
        Assert.False(new SeOrganisationsnummerRecognizer().ValidateResult(value));
    }

    // In TasmanianDevil's engine: the default engine leaves the personnummer and takes the organisationsnummer for a phone
    // number; with the recognizers both are found, and a bare ten-digit number needs a context word.
    [Fact]
    public void PiiEngine_MasksSwedishIdNumbers()
    {
        const string text = "Sökande, personnummer 19780401-2388, är anställd hos bolaget med organisationsnummer 559900-1236.";
        Assert.Contains("19780401-2388", new PiiEngine(new PiiOptions { Language = "sv" }).Anonymize(text).Text);

        using var engine = Engine();
        Assert.Equal(
            "Sökande, personnummer <SE_PERSONNUMMER>, är anställd hos bolaget med organisationsnummer <SE_ORGANISATIONSNUMMER>.",
            engine.Anonymize(text).Text);
        Assert.DoesNotContain(engine.Analyze("Order 7804012388 skickad."), r => r.EntityType == SePersonnummerRecognizer.EntityType);
        Assert.Contains(engine.Analyze("Personnummer 7804012388 enligt ansökan."), r => r.EntityType == SePersonnummerRecognizer.EntityType);
    }

    // The Luhn check passes for one number in ten: bare ten-digit order numbers must not be masked without context.
    [Fact]
    public void PiiEngine_LeavesArbitraryTenDigitNumbers()
    {
        using var engine = Engine();
        var random = new Random(42);
        var masked = Enumerable.Range(0, 500)
            .Select(_ => random.NextInt64(1_000_000_000, 10_000_000_000).ToString(System.Globalization.CultureInfo.InvariantCulture))
            .Count(n => engine.Analyze($"Order {n} skickad.").Any(r => r.EntityType.StartsWith("SE_", StringComparison.Ordinal)));
        Assert.Equal(0, masked);
    }

    private static PiiEngine Engine() =>
        new(new PiiOptions { Language = "sv" }, extraRecognizers: [new SePersonnummerRecognizer(), new SeOrganisationsnummerRecognizer()]);
}

/// <summary>The Swedish address recognizers of StanzaSharp.TasmanianDevil, without models. The addresses are invented.</summary>
public class SwedishAddressRecognizerTests
{
    [Theory]
    [InlineData("Hon bor på Storgatan 14 B i Uppsala.", "Storgatan 14 B")]
    [InlineData("Adress: Storgatan 14B, 753 20 Uppsala", "Storgatan 14B")]
    [InlineData("Ny adress Västra Ringvägen 7, lgh 1102.", "Västra Ringvägen 7, lgh 1102")]
    [InlineData("Besök oss på S:t Eriksgatan 112.", "S:t Eriksgatan 112")]
    [InlineData("STORGATAN 3\n753 20 UPPSALA", "STORGATAN 3")]
    [InlineData("Kontoret ligger på Hamngatan 10-12 sedan maj.", "Hamngatan 10-12")]
    [InlineData("Han flyttade till Storgatan 4 i Uppsala.", "Storgatan 4")]
    [InlineData("Karin Lindqvist, Kungsvägen 3", "Kungsvägen 3")]
    public void Street_IsFound(string text, string expected)
    {
        var result = Assert.Single(new SeStreetAddressRecognizer().Analyze(text, []));
        Assert.Equal((SeStreetAddressRecognizer.EntityType, expected), (result.EntityType, text[result.Start..result.End]));
    }

    [Theory]
    [InlineData("Handlingsplan 2026 antogs i maj.")]
    [InlineData("Arbetsplats 3 är stängd.")]
    [InlineData("Förändringen 2025 gällde alla.")]
    [InlineData("Avdelningen 4 har möte.")]
    [InlineData("Han bor på Storgatan.")]
    [InlineData("storgatan 14 är ett exempel i gemener.")]
    public void NotAStreetAddress_IsNotFound(string text) => Assert.Empty(new SeStreetAddressRecognizer().Analyze(text, []));

    [Theory]
    [InlineData("753 20 Uppsala", "753 20")]
    [InlineData("SE-753 20 Uppsala", "SE-753 20")]
    [InlineData("Postadress: Box 123, 75320 UPPSALA", "75320")]
    [InlineData("Storgatan 14 B\n753 20 Uppsala", "753 20")]
    public void Postnummer_WithTownOrPrefix_IsMasked(string text, string expected)
    {
        using var engine = Engine();
        var result = Assert.Single(engine.Analyze(text), r => r.EntityType == SePostnummerRecognizer.EntityType);
        Assert.Equal(expected, text[result.Start..result.End]);
    }

    [Theory]
    [InlineData("Ordernummer 753 20 skickades.")]
    [InlineData("Det kostade 125 000 kronor.")]
    [InlineData("Ring 08-123 45 67 i dag.")]
    [InlineData("Talet 1 234 567 är stort.")]
    [InlineData("Koden 99999 Uppsala finns inte.")]
    public void Postnummer_WithoutTownOrContext_IsNotMasked(string text)
    {
        using var engine = Engine();
        Assert.DoesNotContain(engine.Analyze(text), r => r.EntityType == SePostnummerRecognizer.EntityType);
    }

    [Fact]
    public void Postnummer_BareWithContextWord_IsMasked()
    {
        using var engine = Engine();
        Assert.Contains(engine.Analyze("Postnummer 75320 enligt blanketten."), r => r.EntityType == SePostnummerRecognizer.EntityType);
    }

    // Five-digit order numbers in ordinary text stay unmasked without a context word or town.
    [Fact]
    public void Postnummer_LeavesArbitraryFiveDigitNumbers()
    {
        using var engine = Engine();
        var random = new Random(42);
        var masked = Enumerable.Range(0, 500)
            .Select(_ => random.Next(10_000, 100_000).ToString(System.Globalization.CultureInfo.InvariantCulture))
            .Count(n => engine.Analyze($"Order {n} skickad.").Any(r => r.EntityType == SePostnummerRecognizer.EntityType));
        Assert.Equal(0, masked);
    }

    [Fact]
    public void PiiEngine_MasksTheWholeAddress()
    {
        using var engine = Engine();
        Assert.Equal(
            "Sökanden bor på <ADDRESS>, <SE_POSTNUMMER> Uppsala.",
            engine.Anonymize("Sökanden bor på Storgatan 14 B, 753 20 Uppsala.").Text);
    }

    private static PiiEngine Engine() =>
        new(new PiiOptions { Language = "sv" }, extraRecognizers: [new SeStreetAddressRecognizer(), new SePostnummerRecognizer()]);
}
