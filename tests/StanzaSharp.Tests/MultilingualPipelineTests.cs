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

    [MultilingualFact]
    public void Default_UsesEveryLanguageFolder_LoadingEachAtItsFirstText()
    {
        using var nlp = MultilingualPipeline.Load(MultilingualTheoryAttribute.Root);
        Assert.Contains("en", nlp.Languages);
        Assert.Contains("de", nlp.Languages);
        Assert.Equal([.. nlp.Languages.Order(StringComparer.Ordinal)], nlp.Languages);
        Assert.Empty(nlp.LoadedLanguages);

        var doc = nlp.Process("The weather in Berlin was lovely yesterday. We walked along the river.");
        Assert.Equal("en", doc.Lang);
        Assert.Equal(2, doc.Sentences.Count);
        Assert.NotNull(doc.Sentences[0].Words.First().Head); // the default package's processors
        Assert.Equal(["en"], nlp.LoadedLanguages);
    }

    private static readonly string[] Texts =
    [
        "The weather in Berlin was lovely yesterday. We walked along the river.",
        "Das Wetter in Berlin war gestern schön. Wir sind am Fluss entlang gegangen.",
        "I don't think she's coming to the meeting.",
        "Die Bundesregierung hat am Mittwoch neue Regeln beschlossen.",
    ];

    private static MultilingualPipeline LoadTokenizers(int maxLoaded) => MultilingualPipeline.Load(MultilingualTheoryAttribute.Root, new()
    {
        Languages = [new() { Language = "en", Processors = "tokenize,mwt" }, new() { Language = "de", Processors = "tokenize,mwt" }],
        MaxLoadedPipelines = maxLoaded,
    });

    [MultilingualFact]
    public void MaxLoadedPipelines_FreesTheLeastRecentlyUsed()
    {
        using var all = LoadTokenizers(10);
        var bulk = all.Process(Texts).Select(d => Conllu.Write(d)).ToList();
        Assert.Equal(["en", "de"], all.LoadedLanguages);
        var expected = Texts.Select(t => Conllu.Write(all.Process(t))).ToList();
        Assert.Equal(["en", "de"], all.LoadedLanguages); // de was used last
        all.Process(Texts[0]);
        Assert.Equal(["de", "en"], all.LoadedLanguages);

        using var one = LoadTokenizers(1);
        Assert.Equal(bulk, one.Process(Texts).Select(d => Conllu.Write(d))); // both languages in one call
        Assert.Equal(["de"], one.LoadedLanguages);
        for (int i = 0; i < Texts.Length; i++)
        {
            Assert.Equal(expected[i], Conllu.Write(one.Process([Texts[i]])[0]));
            Assert.Equal([i % 2 == 0 ? "en" : "de"], one.LoadedLanguages);
        }
    }

    [MultilingualFact]
    public void ConcurrentCalls_WithEvictions_GiveTheSameDocuments()
    {
        using var all = LoadTokenizers(10);
        var expected = Texts.Select(t => Conllu.Write(all.Process([t])[0])).ToList();

        // One pipeline for two languages: nearly every call frees the pipeline another call may still be using.
        using var one = LoadTokenizers(1);
        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, thread =>
        {
            for (int i = 0; i < 20; i++)
            {
                int k = (thread + i) % Texts.Length;
                Assert.Equal(expected[k], Conllu.Write(one.Process([Texts[k]])[0]));
            }
        });
        Assert.Single(one.LoadedLanguages);
    }

    [MultilingualFact]
    public void MissingModelFile_ThrowsAtTheLanguagesFirstText_AndAgainAtTheNext()
    {
        using var root = new ModelRoot("en");
        using var nlp = MultilingualPipeline.Load(root.Path);
        Assert.Equal(["en"], nlp.Languages);
        Assert.Throws<FileNotFoundException>(() => nlp.Process("The weather in Berlin was lovely yesterday."));
        Assert.Empty(nlp.LoadedLanguages);
        Assert.Throws<FileNotFoundException>(() => nlp.Process("The weather in Berlin was lovely yesterday."));
        Assert.Empty(nlp.Process("Das Wetter in Berlin war gestern schön.").Sentences); // no de/ folder: unprocessed
    }

    [MultilingualFact]
    public void Norwegian_IsRoutedToNb()
    {
        // The identifier says "no", Stanza's alias of nb; Restrict maps nb to "no" for its subset. The empty nb/ folder
        // throws at the first text, so the text was routed there.
        using var root = new ModelRoot("nb", "hyw");
        const string text = "Været i Oslo var nydelig i går. Vi gikk en lang tur langs elva.";
        using (var nlp = MultilingualPipeline.Load(root.Path, new() { Languages = [new() { Language = "nb" }], Restrict = true }))
            Assert.Throws<FileNotFoundException>(() => nlp.Process(text));

        // Without Languages, a language the identifier doesn't know (hyw) stays out of the subset.
        using var present = MultilingualPipeline.Load(root.Path, new() { Restrict = true });
        Assert.Equal(["hyw", "nb"], present.Languages);
        Assert.Throws<FileNotFoundException>(() => present.Process("The weather in Berlin was lovely yesterday."));
    }

    [Fact]
    public void BadOptions_ThrowAtLoad()
    {
        var root = Directory.CreateTempSubdirectory("stanzasharp-multilingual").FullName;
        try
        {
            Assert.Throws<DirectoryNotFoundException>(() => MultilingualPipeline.Load(root));
            Assert.Throws<DirectoryNotFoundException>(() => MultilingualPipeline.Load(root, new() { Languages = [new() { Language = "fr" }] }));
            Assert.Throws<ArgumentException>(() => MultilingualPipeline.Load(root, new() { Languages = [new() { Language = "xx" }] }));
            Assert.Throws<ArgumentException>(() => MultilingualPipeline.Load(root, new() { Languages = [new() { Language = "en", Package = "nope" }] }));
            Assert.Throws<ArgumentOutOfRangeException>(() => MultilingualPipeline.Load(root, new() { MaxLoadedPipelines = 0 }));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    /// <summary>A temporary model root: a copy of the identifier and an empty folder per language.</summary>
    private sealed class ModelRoot : IDisposable
    {
        public ModelRoot(params string[] languages)
        {
            Path = Directory.CreateTempSubdirectory("stanzasharp-multilingual").FullName;
            var langid = System.IO.Path.Combine(Path, "multilingual", "langid");
            Directory.CreateDirectory(langid);
            foreach (var file in Directory.GetFiles(System.IO.Path.Combine(LangIdTheoryAttribute.Directory, "langid"), "ud.*"))
                File.Copy(file, System.IO.Path.Combine(langid, System.IO.Path.GetFileName(file)));
            foreach (var language in languages)
                Directory.CreateDirectory(System.IO.Path.Combine(Path, language));
        }

        public string Path { get; }

        public void Dispose() => Directory.Delete(Path, recursive: true);
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
