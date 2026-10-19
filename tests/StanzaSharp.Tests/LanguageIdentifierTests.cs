using System.Text.Json.Nodes;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="LanguageIdentifier"/> against tests/golden/langid/langid.json (Stanza's langid processor): every language
/// exactly, unrestricted and with the en/de subset, and the summed scores within 1e-4 of Stanza's float32 or float64 sums
/// (relative above |100|, as for UPOS), on both backends.
/// </summary>
public sealed class LanguageIdentifierTests
{
    private static readonly JsonNode Golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "langid", "langid.json")))!;

    private static IEnumerable<JsonNode> Cases => Golden["cases"]!.AsArray().Select(c => c!);

    [LangIdTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Identify_MatchesStanza(bool managed)
    {
        var texts = Cases.Select(c => c["text"]!.GetValue<string>()).ToList();
        using var identifier = LanguageIdentifier.Load(LangIdTheoryAttribute.Directory, new() { Backend = Repo.PipelineBackendFor(managed) });
        Assert.Equal(Golden["languages"]!.AsArray().Select(l => l!.GetValue<string>()), identifier.Languages);
        Assert.Equal(Cases.Select(c => c["lang"]!.GetValue<string>()), identifier.Identify(texts));
        foreach (var c in Cases) // each alone, a batch of one, as Stanza's process gives (make_golden checks bulk equals it)
            Assert.Equal(c["lang"]!.GetValue<string>(), identifier.Identify(c["text"]!.GetValue<string>()));

        using var subset = LanguageIdentifier.Load(LangIdTheoryAttribute.Directory, new() { Backend = Repo.PipelineBackendFor(managed), Languages = ["en", "de"] });
        Assert.Equal(Cases.Select(c => c["lang_en_de"]!.GetValue<string>()), subset.Identify(texts));
    }

    [LangIdTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Scores_MatchStanzasSums(bool managed)
    {
        using var identifier = LanguageIdentifier.Load(LangIdTheoryAttribute.Directory, new() { Backend = Repo.PipelineBackendFor(managed) });
        identifier.Model.Identify([.. Cases.Select(c => c["text"]!.GetValue<string>())], out var scores);
        foreach (var (c, actual) in Cases.Zip(scores))
        {
            var f32 = c["scores"]!.AsArray().Select(v => v!.GetValue<double>()).ToArray();
            var f64 = c["scores64"]!.AsArray().Select(v => v!.GetValue<double>()).ToArray();
            for (int j = 0; j < actual.Length; j++)
            {
                double off = Math.Min(Math.Abs(actual[j] - f32[j]), Math.Abs(actual[j] - f64[j]));
                Assert.True(off <= TokenizerTests.Allowed(1e-4f, f64[j]), $"{c["text"]}: score {j} is {actual[j]}, Stanza {f32[j]} (float64 {f64[j]})");
            }
        }
    }

    [LangIdFact]
    public void EmptyText_Throws()
    {
        using var identifier = LanguageIdentifier.Load(LangIdTheoryAttribute.Directory);
        Assert.Throws<ArgumentException>(() => identifier.Identify(["Hello there.", ""]));
    }

    [LangIdFact]
    public void UnknownLanguage_Throws()
    {
        var e = Assert.Throws<ArgumentException>(() => LanguageIdentifier.Load(LangIdTheoryAttribute.Directory, new() { Languages = ["en", "xx"] }));
        Assert.Contains("Unknown language xx", e.Message);
    }

    [LangIdFact]
    public void VerifyChecksums_PassesStanzasFile() =>
        LanguageIdentifier.Load(LangIdTheoryAttribute.Directory, new() { VerifyChecksums = true }).Dispose();

    [Fact]
    public void MissingModel_NamesTheDownload()
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-langid-").FullName;
        try
        {
            var e = Assert.Throws<FileNotFoundException>(() => LanguageIdentifier.Load(dir));
            Assert.Contains("DownloadLanguageIdentifierAsync", e.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

/// <summary>A test that needs the language identifier's model (models/stanza/multilingual, next to the English models); skipped without it.</summary>
public sealed class LangIdTheoryAttribute : TheoryAttribute
{
    /// <summary>Stanza's multilingual folder next to <see cref="Repo.StanzaModels"/>.</summary>
    public static readonly string Directory = Path.GetFullPath(Path.Combine(Repo.StanzaModels, "..", "multilingual"));

    internal static string? SkipReason() => File.Exists(Path.Combine(Directory, "langid", "ud.pt")) || File.Exists(Path.Combine(Directory, "langid", "ud.json"))
        ? null
        : $"{Directory}/langid/ud.pt not found; run stanzasharp download {Directory} --lang multilingual";

    public LangIdTheoryAttribute() => Skip = SkipReason();
}

/// <summary>As <see cref="LangIdTheoryAttribute"/>, for a fact.</summary>
public sealed class LangIdFactAttribute : FactAttribute
{
    public LangIdFactAttribute() => Skip = LangIdTheoryAttribute.SkipReason();
}
