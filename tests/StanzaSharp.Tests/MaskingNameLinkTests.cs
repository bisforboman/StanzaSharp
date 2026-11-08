using StanzaSharp.Masking;

namespace StanzaSharp.Tests;

/// <summary>
/// The masking sample's name linking (<c>--link</c>, samples/StanzaSharp.Masking/NameLinker.cs) and its labelled-text
/// scoring, without models: the entities are given as NER would give them.
/// </summary>
public class MaskingNameLinkTests
{
    [Fact]
    public void Link_GivesEachPersonOnePlaceholder()
    {
        const string text = "Karin Lindqvist och Elin Lindqvist kom. Karins ansökan. K. Lindqvist ringde. Elin stannade.";
        var document = Document("Karin Lindqvist", 0, "PER", "Elin Lindqvist", 20, "PER", "Karins", 40, "PER",
            "K. Lindqvist", 56, "PER", "Elin", 77, "PER");
        Assert.Equal("[PER-1] och [PER-2] kom. [PER-1] ansökan. [PER-1] ringde. [PER-2] stannade.", Mask(text, document));
        Assert.Equal("[PER-1] och [PER-2] kom. [PER-3] ansökan. [PER-4] ringde. [PER-5] stannade.",
            Masker.Apply(text, Masker.Find(text, document)));
    }

    [Fact]
    public void Link_KeepsASurnameSharedByTwoPeopleApart()
    {
        const string text = "Ingrid Holm och Björn Holm. Holm ringde.";
        var document = Document("Ingrid Holm", 0, "PER", "Björn Holm", 16, "PER", "Holm", 28, "PER");
        Assert.Equal("[PER-1] och [PER-2]. [PER-3] ringde.", Mask(text, document));
    }

    [Fact]
    public void Link_MasksTheMentionsNerMissed()
    {
        const string text = "Sökande är Sofia Nilsson. Nilssons dotter Sofia ringde, liksom S. Nilsson.";
        var document = Document("Sofia Nilsson", 11, "PER");
        var spans = Masker.Find(text, document, linkNames: true);
        Assert.Equal(["Sofia Nilsson", "Nilssons", "Sofia", "S. Nilsson"], spans.Select(s => s.Text));
        Assert.Equal(["ner", "name-link", "name-link", "name-link"], spans.Select(s => s.Source));
        Assert.Equal("Sökande är [PER-1]. [PER-1] dotter [PER-1] ringde, liksom [PER-1].", Masker.Apply(text, spans));
    }

    // Lowercase words and all-caps abbreviations are left: sten is a stone, SOL the social services act.
    [Fact]
    public void Link_LeavesLowercaseAndUnseenAllCaps()
    {
        const string text = "Petra Sol och Maria Sten. Enligt SOL lades en sten. STEN och Sol.";
        var document = Document("Petra Sol", 0, "PER", "Maria Sten", 14, "PER");
        Assert.Equal("[PER-1] och [PER-2]. Enligt SOL lades en sten. STEN och [PER-1].", Mask(text, document));
    }

    [Fact]
    public void Link_NeedsAWholeWord()
    {
        const string text = "Mentor Ek, klass EK1a, Eken och Ekbacken.";
        var document = Document("Ek", 7, "PER");
        Assert.Equal("Mentor [PER-1], klass EK1a, Eken och Ekbacken.", Mask(text, document));
    }

    // Given name + father's name + grandfather's: the father is another person, not a shorter form of the son.
    [Fact]
    public void Link_TellsAFathersNameFromHisSons()
    {
        const string text = "Yusuf Abdirahman Nur och fadern Abdirahman Nur. Yusuf och Abdirahman.";
        var document = Document("Yusuf Abdirahman Nur", 0, "PER", "Abdirahman Nur", 32, "PER", "Yusuf", 48, "PER",
            "Abdirahman", 58, "PER");
        Assert.Equal("[PER-1] och fadern [PER-2]. [PER-1] och [PER-2].", Mask(text, document));
    }

    [Fact]
    public void Link_JoinsAnAllCapsHeaderAndDropsTheFormLabel()
    {
        const string text = "LINDQVIST, KARIN\nPersonnummer: 19780401-2388\nKarin kom.";
        var document = Document("LINDQVIST", 0, "PER", "KARIN\nPersonnummer", 11, "PER", "Karin", 45, "PER");
        var spans = Masker.Find(text, document, linkNames: true);
        Assert.Equal(["LINDQVIST", "KARIN", "19780401-2388", "Karin"], spans.Select(s => s.Text));
        Assert.Equal("[PER-1], [PER-1]\nPersonnummer: [PNR-1]\n[PER-1] kom.", Masker.Apply(text, spans));
    }

    [Fact]
    public void Link_CutsSpansAtColumnsAndAtPatterns()
    {
        const string text = "Från: Jonas Ek <jonas.ek@example.se>\nAhmed Yusuf      Karin Lind\nMvh\nJonas";
        var document = Document("Jonas Ek <jonas.ek", 6, "PER", "Ahmed Yusuf      Karin Lind", 37, "PER", "Mvh\nJonas", 65, "PER");
        var spans = Masker.Find(text, document, linkNames: true);
        Assert.Equal(["Jonas Ek", "jonas.ek@example.se", "Ahmed Yusuf", "Karin Lind", "Jonas"], spans.Select(s => s.Text));
        Assert.Equal("Från: [PER-1] <[EMAIL-1]>\n[PER-2]      [PER-3]\nMvh\n[PER-1]", Masker.Apply(text, spans));
    }

    // NER sometimes types a genitive first name as an organisation; next to the full name it is the person.
    [Fact]
    public void Link_RetypesAPersonsNameTaggedAsAnOrganisation()
    {
        const string text = "Mikael Lindqvist arbetar. Mikaels arbetsgivare, Bergs Bygg, intygar.";
        var document = Document("Mikael Lindqvist", 0, "PER", "Mikaels", 26, "ORG", "Bergs Bygg", 48, "ORG");
        Assert.Equal("[PER-1] arbetar. [PER-1] arbetsgivare, [ORG-1], intygar.", Mask(text, document));
    }

    [Fact]
    public void Link_RepeatsPlacesAndOrganisationsByTheirText()
    {
        const string text = "Eleven går på Ekbackens skola i Sala. Ekbackens skola ligger i Sala.";
        var document = Document("Ekbackens skola", 14, "ORG", "Sala", 32, "LOC");
        Assert.Equal("Eleven går på [ORG-1] i [LOC-1]. [ORG-1] ligger i [LOC-1].", Mask(text, document));
    }

    [Fact]
    public void Parse_RemovesTheLabels()
    {
        var (text, spans) = Evaluation.Parse("Hej {PER#k:Karin Lindqvist}, ring {PHONE:08-465 004 00}. {PER#k:Karins} brev.");
        Assert.Equal("Hej Karin Lindqvist, ring 08-465 004 00. Karins brev.", text);
        Assert.Equal([("PER", "k", "Karin Lindqvist"), ("PHONE", null, "08-465 004 00"), ("PER", "k", "Karins")],
            spans.Select(s => (s.Type, s.Id, s.Text)));
        Assert.All(spans, s => Assert.Equal(s.Text, text.Substring(s.Start, s.Length)));
        Assert.Throws<FormatException>(() => Evaluation.Parse("{PER:Karin"));
    }

    [Fact]
    public void Add_CountsLeaksOverMasksAndPlaceholders()
    {
        var (text, gold) = Evaluation.Parse("{PER#k:Karin Lindqvist} och {PER#e:Elin}. {PER#k:Karin} kom. SV: hej");
        var masked = new List<MaskSpan>
        {
            new(0, 15, "PER", "ner", "Karin Lindqvist", "k"),
            new(26, 5, "PER", "ner", "Karin", "k"),
            new(37, 2, "ORG", "ner", "SV"),
        };
        var score = new Score();
        Evaluation.Add(score, "t", text, gold, masked);
        Assert.Equal(["t: Elin (PER)"], score.Leaks);
        Assert.Equal(["t: SV (ORG, ner)"], score.OverMasked);
        Assert.Equal((1, 0, 0), (score.SamePersonSamePlaceholder, score.SamePersonOtherPlaceholder, score.OtherPersonSamePlaceholder));
        Assert.Equal((1, 1), (score.People, score.PeopleWithOwnPlaceholder));
    }

    private static string Mask(string text, Document document) => Masker.Apply(text, Masker.Find(text, document, linkNames: true));

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
