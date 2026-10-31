using StanzaSharp.Masking;

namespace StanzaSharp.Tests;

/// <summary>
/// The masking sample's patterns, merging and placeholders (samples/StanzaSharp.Masking), without models: the NER side
/// is Swedish's ner, whose output the languages golden checks (tests/golden/languages/sv).
/// </summary>
public class MaskingTests
{
    [Theory]
    [InlineData("7804122385", true)]
    [InlineData("7804122384", false)]
    [InlineData("5566778899", true)]
    [InlineData("1203154594", true)]
    public void Luhn_ChecksTheLastDigit(string digits, bool valid) => Assert.Equal(valid, Masker.Luhn(digits));

    [Theory]
    [InlineData("Personnummer 19780412-2385 enligt", "19780412-2385", "PNR")]
    [InlineData("pnr 780412-2385.", "780412-2385", "PNR")]
    [InlineData("född 7804122385", "7804122385", "PNR")]
    [InlineData("samordningsnummer 780472-2382", "780472-2382", "PNR")]
    [InlineData("organisationsnummer 556677-8899.", "556677-8899", "ORGNR")]
    [InlineData("ring 070-123 45 67 i dag", "070-123 45 67", "PHONE")]
    [InlineData("ring 08-555 123 45 i dag", "08-555 123 45", "PHONE")]
    [InlineData("ring +46 70 123 45 67 i dag", "+46 70 123 45 67", "PHONE")]
    [InlineData("mejla kristina.oberg@uppsala.example.se vid", "kristina.oberg@uppsala.example.se", "EMAIL")]
    [InlineData("bor på Storgatan 14 B, Uppsala", "Storgatan 14 B", "ADDRESS")]
    [InlineData("från Västra Ringvägen 7 i Örebro", "Västra Ringvägen 7", "ADDRESS")]
    [InlineData("Storgatan 14, 753 20 Uppsala", "753 20 Uppsala", "ADDRESS")]
    public void Patterns_FindTheNumberOrAddress(string text, string expected, string type)
    {
        var span = Assert.Single(Masker.FindPatterns(text), s => s.Text == expected);
        Assert.Equal(type, span.Type);
        Assert.Equal(expected, text.Substring(span.Start, span.Length));
    }

    [Theory]
    [InlineData("personnummer 19780412-2384")]        // fails the Luhn check
    [InlineData("ärende 781312-2384")]                // month 13
    [InlineData("beviljas 8 450 kronor för oktober")] // an amount
    [InlineData("4 kap. 1 § socialtjänstlagen")]
    [InlineData("år 2026, vecka 40")]
    public void Patterns_LeaveOtherNumbersAlone(string text) => Assert.Empty(Masker.FindPatterns(text));

    [Fact]
    public void Find_MergesOverlapsAndPrefersThePattern()
    {
        const string text = "Karin Lindqvist bor på Storgatan 14 B.";
        var document = Document("Karin Lindqvist", 0, "PER", "Storgatan", 23, "LOC");
        var spans = Masker.Find(text, document);
        Assert.Equal(["Karin Lindqvist", "Storgatan 14 B"], spans.Select(s => s.Text));
        Assert.Equal(["PER", "ADDRESS"], spans.Select(s => s.Type));
        Assert.Equal(["ner", "street"], spans.Select(s => s.Source));
    }

    [Fact]
    public void Find_KeepsOnlyTheChosenNerTypes()
    {
        const string text = "Karin kom den 3 september.";
        var document = Document("Karin", 0, "PER", "den 3 september", 10, "TME");
        Assert.Equal(["PER"], Masker.Find(text, document).Select(s => s.Type));
        Assert.Equal(["PER", "TME"], Masker.Find(text, document, new HashSet<string> { "PER", "TME" }).Select(s => s.Type));
    }

    [Fact]
    public void Apply_NumbersEachDistinctTextOnce()
    {
        const string text = "Karin och Elin. Karin ringde 070-123 45 67.";
        var document = Document("Karin", 0, "PER", "Elin", 10, "PER", "Karin", 16, "PER");
        var masked = Masker.Apply(text, Masker.Find(text, document));
        Assert.Equal("[PER-1] och [PER-2]. [PER-1] ringde [PHONE-1].", masked);
    }

    // A document holding only the given entities (text, start, type triples), as the pipeline's NER would give them.
    private static Document Document(params object[] entities)
    {
        var sentence = new Sentence();
        for (var i = 0; i < entities.Length; i += 3)
        {
            var text = (string)entities[i];
            var start = (int)entities[i + 1];
            sentence.Entities.Add(new Entity(text, (string)entities[i + 2], start, start + text.Length, []));
        }

        var document = new Document();
        document.Sentences.Add(sentence);
        return document;
    }
}
