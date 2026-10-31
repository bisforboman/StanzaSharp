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
