using System.Text.Json.Nodes;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="MultilingualPipeline"/> against tests/golden/langid/multilingual.json (Stanza's
/// <c>MultilingualPipeline(restrict=True)</c> with English and German): each document's language and CoNLL-U byte for
/// byte, on both backends; and what an unconfigured language gives.
/// </summary>
public sealed class MultilingualPipelineTests
{
    [MultilingualTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_MatchesStanzasMultilingualPipeline(bool managed)
    {
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "langid", "multilingual.json")))!["documents"]!.AsArray();
        var backend = Repo.PipelineBackendFor(managed);
        using var nlp = MultilingualPipeline.Load(MultilingualTheoryAttribute.Root, new()
        {
            Languages = [new() { Language = "en", Backend = backend }, new() { Language = "de", Backend = backend }],
            Restrict = true,
            LanguageIdentifier = new() { Backend = backend },
        });
        Assert.Equal(["en", "de"], nlp.Languages);
        var docs = nlp.Process(golden.Select(d => d!["text"]!.GetValue<string>()));
        Assert.Equal(golden.Count, docs.Count);
        foreach (var (expected, doc) in golden.Zip(docs))
        {
            Assert.Equal(expected!["lang"]!.GetValue<string>(), doc.Lang);
            Assert.Equal(expected["text"]!.GetValue<string>(), doc.Text);
            Assert.Equal(expected["conllu"]!.GetValue<string>() + "\n", Conllu.Write(doc)); // Stanza's "{:C}" has no final newline
        }
    }

    [MultilingualFact]
    public void UnconfiguredLanguage_ComesBackWithOnlyItsLanguage()
    {
        using var nlp = MultilingualPipeline.Load(MultilingualTheoryAttribute.Root, new()
        {
            Languages = [new() { Language = "en", Processors = "tokenize" }],
        });

        // The README's example, plus PACKAGE.md's texts.
        var docs = nlp.Process(["The weather in Berlin was lovely yesterday. We walked along the river.",
            "Das Wetter in Berlin war gestern schön. Wir sind am Fluss entlang gegangen.", "Ceci est une phrase française sur la météo.",
            "I don't think she's coming to the meeting.", "Die Bundesregierung hat am Mittwoch neue Regeln beschlossen."]);
        Assert.Equal(["en", "de", "fr", "en", "de"], docs.Select(d => d.Lang));
        Assert.Equal([2, 0, 0, 1, 0], docs.Select(d => d.Sentences.Count));
        Assert.Equal("Ceci est une phrase française sur la météo.", docs[2].Text);

        var one = nlp.Process("Das ist ein deutscher Satz über das Wetter.");
        Assert.Equal("de", one.Lang);
        Assert.Empty(one.Sentences);
    }

    [MultilingualFact]
    public void Restrict_ProcessesEveryTextAsAConfiguredLanguage()
    {
        using var nlp = MultilingualPipeline.Load(MultilingualTheoryAttribute.Root, new()
        {
            Languages = [new() { Language = "en", Processors = "tokenize" }],
            Restrict = true,
        });
        var doc = nlp.Process("Ceci est une phrase française sur la météo.");
        Assert.Equal("en", doc.Lang);
        Assert.Equal(9, doc.Sentences[0].Tokens.Count);

        // The identifier's own subset wins over Restrict, as in Stanza.
        using var own = MultilingualPipeline.Load(MultilingualTheoryAttribute.Root, new()
        {
            Languages = [new() { Language = "en", Processors = "tokenize" }],
            Restrict = true,
            LanguageIdentifier = new() { Languages = ["en", "fr"] },
        });
        Assert.Equal("fr", own.Process("Ceci est une phrase française sur la météo.").Lang);
    }

    [MultilingualFact]
    public void EmptyText_Throws()
    {
        using var nlp = MultilingualPipeline.Load(MultilingualTheoryAttribute.Root, new() { Languages = [new() { Language = "en", Processors = "tokenize" }] });
        Assert.Throws<ArgumentException>(() => nlp.Process(["Hello there.", ""]));
    }

    [Theory]
    [InlineData("en,en", "2 entries for language 'en'")]
    [InlineData("", "Languages is empty")]
    public void BadLanguages_Throw(string languages, string message)
    {
        var e = Assert.Throws<ArgumentException>(() => MultilingualPipeline.Load("nowhere", new()
        {
            Languages = [.. languages.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(l => new PipelineOptions { Language = l })],
        }));
        Assert.Contains(message, e.Message);
    }
}

/// <summary>
/// A theory skipped where Stanza's English, German or multilingual folder next to <see cref="Repo.StanzaModels"/> is
/// missing (CI has all three in golden-shard 1 only; ci.yml's no-skip checks elsewhere exempt MultilingualPipelineTests).
/// </summary>
public sealed class MultilingualTheoryAttribute : TheoryAttribute
{
    /// <summary>The folder holding en/, de/ and multilingual/.</summary>
    public static readonly string Root = Path.GetFullPath(Path.Combine(Repo.StanzaModels, ".."));

    internal static string? SkipReason() =>
        LangIdTheoryAttribute.SkipReason()
        ?? (Directory.Exists(Path.Combine(Root, "en")) && Directory.Exists(Path.Combine(Root, "de")) ? null : $"{Root}/en or {Root}/de not found");

    public MultilingualTheoryAttribute() => Skip = SkipReason();
}

/// <summary>As <see cref="MultilingualTheoryAttribute"/>, for a fact.</summary>
public sealed class MultilingualFactAttribute : FactAttribute
{
    public MultilingualFactAttribute() => Skip = MultilingualTheoryAttribute.SkipReason();
}
