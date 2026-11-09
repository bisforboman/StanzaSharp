using StanzaSharp.TasmanianDevil;
using TasmanianDevil;
using TasmanianDevil.Analyzer;
using TasmanianDevil.Anonymizer;
using TasmanianDevil.Anonymizer.Operators;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="StanzaNerRecognizer.LinkNames"/> and <see cref="StanzaNerRecognizer.PersonPlaceholders"/>: the linker on
/// spans as NER would give them (no models), and the recognizer end to end (English models).
/// </summary>
public class TasmanianDevilNameLinkTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    [Fact]
    public void Link_GivesEachPersonOneKey()
    {
        const string text = "Karin Lindqvist och Elin Lindqvist kom. Karins ansökan. K. Lindqvist ringde. Elin stannade.";
        var spans = Link(text, Person(text, "Karin Lindqvist"), Person(text, "Elin Lindqvist"), Person(text, "Karins"),
            Person(text, "K. Lindqvist"), Person(text, "Elin", 1));
        Assert.Equal(["A", "B", "A", "A", "B"], Letters(spans));
        Assert.All(spans, s => Assert.False(s.Added));
    }

    [Fact]
    public void Link_KeepsASurnameSharedByTwoPeopleApart()
    {
        const string text = "Ingrid Holm och Björn Holm. Holm ringde.";
        Assert.Equal(["A", "B", "C"], Letters(Link(text, Person(text, "Ingrid Holm"), Person(text, "Björn Holm"), Person(text, "Holm", 2))));
    }

    [Fact]
    public void Link_AddsTheMentionsNerMissed()
    {
        const string text = "Sökande är Sofia Nilsson. Nilssons dotter Sofia ringde, liksom S. Nilsson.";
        var spans = Link(text, Person(text, "Sofia Nilsson"));
        Assert.Equal(["Sofia Nilsson", "Nilssons", "Sofia", "S. Nilsson"], spans.Select(s => text[s.Start..s.End]));
        Assert.Equal([false, true, true, true], spans.Select(s => s.Added));
        Assert.Equal(["A", "A", "A", "A"], Letters(spans));
    }

    // Lowercase words, all-caps abbreviations and names inside words or e-mail addresses are left alone.
    [Fact]
    public void Link_LeavesWordsAbbreviationsAndAddresses()
    {
        const string text = "Petra Sol och Maria Sten. Enligt SOL lades en sten, klass SOL1a, Solgården. Skriv till Sten.Petra@example.se.";
        var spans = Link(text, Person(text, "Petra Sol"), Person(text, "Maria Sten"));
        Assert.Equal(["Petra Sol", "Maria Sten"], spans.Select(s => text[s.Start..s.End]));
    }

    // Given name + father's name + grandfather's: the father is another person, not a shorter form of the son.
    [Fact]
    public void Link_TellsAFathersNameFromHisSons()
    {
        const string text = "Yusuf Abdirahman Nur och fadern Abdirahman Nur. Yusuf och Abdirahman.";
        var spans = Link(text, Person(text, "Yusuf Abdirahman Nur"), Person(text, "Abdirahman Nur", 1), Person(text, "Yusuf", 1),
            Person(text, "Abdirahman", 2));
        Assert.Equal(["A", "B", "A", "B"], Letters(spans));
    }

    [Fact]
    public void Link_JoinsAnAllCapsHeaderAndCutsAtLabelsColumnsAndAddresses()
    {
        const string text = "LINDQVIST, KARIN\nPersonnummer: 19780401-2388\nFrån: Karin Lindqvist <karin@example.se>\n"
            + "Ahmed Yusuf      Karin Lindqvist\nMvh\nKarin";
        var spans = Link(text, Person(text, "LINDQVIST"), Person(text, "KARIN\nPersonnummer"),
            Person(text, "Karin Lindqvist <karin"), Person(text, "Ahmed Yusuf      Karin Lindqvist"), Person(text, "Mvh\nKarin"));
        Assert.Equal(["LINDQVIST", "KARIN", "Karin Lindqvist", "Ahmed Yusuf", "Karin Lindqvist", "Karin"],
            spans.Select(s => text[s.Start..s.End]));
        Assert.Equal(["A", "A", "A", "B", "A", "A"], Letters(spans));
    }

    [Fact]
    public void Link_RetypesAPersonsNameAndRepeatsPlacesAndOrganisations()
    {
        const string text = "Mikael Lindqvist arbetar på Bergs Bygg i Sala. Mikaels arbetsgivare Bergs Bygg finns i Sala sedan juni. Juni";
        var spans = Link(text, Person(text, "Mikael Lindqvist"), Span(text, "Bergs Bygg", PiiEntities.Organization, "ORG"),
            Span(text, "Sala", PiiEntities.Location, "LOC"), Span(text, "Mikaels", PiiEntities.Organization, "ORG"),
            Span(text, "juni", PiiEntities.DateTime, "TME"));
        Assert.Equal(
            [
                ("Mikael Lindqvist", "PERSON", false), ("Bergs Bygg", "ORGANIZATION", false), ("Sala", "LOCATION", false),
                ("Mikaels", "PERSON", false), ("Bergs Bygg", "ORGANIZATION", true), ("Sala", "LOCATION", true),
                ("juni", "DATE_TIME", false),
            ],
            spans.Select(s => (text[s.Start..s.End], s.Type, s.Added)));
        Assert.Equal(spans[0].Person, spans[3].Person);
    }

    [Fact]
    public void PersonPlaceholders_NumbersEachPersonInOrder()
    {
        const string text = "Karin och Elin. Karins bror Bo. Lisa ringde Lisa.";
        RecognizerResult[] results =
        [
            Result(text, "Karin", "person-1"), Result(text, "Elin", "person-0"), Result(text, "Karins", "person-1"),
            Result(text, "Bo", null), Result(text, "Lisa", null), Result(text, "Lisa", null, 1),
        ];
        var operators = StanzaNerRecognizer.PersonPlaceholders(text, results);
        var anonymized = new AnonymizerEngine().Anonymize(text, results, operators, ConflictResolutionStrategy.MergeSimilarOrContained, false);
        Assert.Equal("<PERSON_1> och <PERSON_2>. <PERSON_1> bror <PERSON_3>. <PERSON_4> ringde <PERSON_4>.", anonymized.Text);
    }

    [Fact]
    public void PersonPlaceholders_KeepsTheOtherOperators()
    {
        var mask = new OperatorConfig("redact", new Dictionary<string, object>());
        var operators = StanzaNerRecognizer.PersonPlaceholders("Karin", [Result("Karin", "Karin", "person-0")],
            new Dictionary<string, OperatorConfig> { ["EMAIL_ADDRESS"] = mask, ["PERSON"] = mask });
        Assert.Same(mask, operators["EMAIL_ADDRESS"]);
        Assert.Equal("custom", operators["PERSON"].OperatorName);
    }

    [ModelFact]
    public void Recognizer_LinksNamesEndToEnd()
    {
        const string text = "Barack Obama met Michelle Obama in Chicago. Later Barack thanked Michelle, and Obama left Chicago.";
        var pipeline = pipelines.Get(processors: "tokenize,ner");
        var plain = new StanzaNerRecognizer(pipeline).Analyze(text, []);
        Assert.All(plain, r => Assert.False(r.RecognitionMetadata!.ContainsKey(StanzaNerRecognizer.PersonKey)));

        var recognizer = new StanzaNerRecognizer(pipeline) { LinkNames = true };
        using var engine = new PiiEngine(new PiiOptions(), extraRecognizers: [recognizer]);
        var results = engine.Analyze(text);
        var anonymized = new AnonymizerEngine().Anonymize(text, results, StanzaNerRecognizer.PersonPlaceholders(text, results),
            ConflictResolutionStrategy.MergeSimilarOrContained, false);
        Assert.Equal("<PERSON_1> met <PERSON_2> in <LOCATION>. Later <PERSON_1> thanked <PERSON_2>, and <PERSON_3> left <LOCATION>.",
            anonymized.Text);
        Assert.Equal(["PERSON", "ORGANIZATION", "LOCATION", "DATE_TIME"], recognizer.SupportedEntities);
    }

    private static List<LinkedSpan> Link(string text, params LinkedSpan[] ner) => NameLinker.Link(text, ner);

    // Each person key as a letter in order of first appearance: A for the first person, B for the second, ...
    private static List<string> Letters(List<LinkedSpan> spans)
    {
        var letters = new Dictionary<string, string>();
        return [.. spans.Where(s => s.IsPerson).Select(s => letters.TryGetValue(s.Person!, out var l)
            ? l
            : letters[s.Person!] = ((char)('A' + letters.Count)).ToString())];
    }

    private static LinkedSpan Person(string text, string value, int occurrence = 0) => Span(text, value, PiiEntities.Person, "PER", occurrence);

    private static LinkedSpan Span(string text, string value, string type, string stanzaType, int occurrence = 0)
    {
        var start = Find(text, value, occurrence);
        return new LinkedSpan(start, start + value.Length, type, stanzaType);
    }

    private static RecognizerResult Result(string text, string value, string? person, int occurrence = 0)
    {
        var start = Find(text, value, occurrence);
        var metadata = new Dictionary<string, object>();
        if (person != null)
            metadata[StanzaNerRecognizer.PersonKey] = person;
        return new RecognizerResult(PiiEntities.Person, start, start + value.Length, 0.85, metadata);
    }

    // The offset of the occurrence-th whole-word occurrence of value.
    private static int Find(string text, string value, int occurrence)
    {
        var at = -1;
        for (var seen = -1; seen < occurrence;)
        {
            at = text.IndexOf(value, at + 1, StringComparison.Ordinal);
            Assert.True(at >= 0, $"'{value}' #{occurrence} not in the text");
            var end = at + value.Length;
            if ((at == 0 || !char.IsLetter(text[at - 1])) && (end == text.Length || !char.IsLetter(text[end])))
                seen++;
        }

        return at;
    }
}
