namespace StanzaSharp.Tests;

/// <summary>
/// The languages with only tests/golden/languages/&lt;lang&gt;/ (make_golden.py --languages-only; owner's decision,
/// 2026-10-30): each supported package on the language's hand-written text.txt, byte for byte on both backends. Their
/// models are downloaded only by the nightly run (nightly.yml's languages job), so every other job skips these rows and
/// its no-skip check exempts the class.
/// </summary>
public class LanguageGoldenTests
{
    [Theory]
    [Language("es", "default", false)]
    [Language("es", "default", true)]
    [Language("fr", "default", false)]
    [Language("fr", "default", true)]
    [Language("fr", "default_fast", false)]
    [Language("fr", "default_fast", true)]
    [Language("it", "default", false)]
    [Language("it", "default", true)]
    [Language("it", "default_fast", false)]
    [Language("it", "default_fast", true)]
    [Language("pt", "default", false)]
    [Language("pt", "default", true)]
    [Language("pt", "default_fast", false)]
    [Language("pt", "default_fast", true)]
    [Language("nl", "default", false)]
    [Language("nl", "default", true)]
    [Language("nl", "default_fast", false)]
    [Language("nl", "default_fast", true)]
    public void Process_MatchesGolden(string lang, string package, bool managed)
    {
        using var nlp = Pipeline.Load(Repo.ModelsFor(lang), new PipelineOptions { Language = lang, Package = package, Backend = Repo.PipelineBackendFor(managed) });
        var dir = Path.Combine(Repo.Golden, "languages", lang);
        var actual = Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(dir, "text.txt"))));
        Assert.Equal(File.ReadAllText(Path.Combine(dir, package + ".conllu")), actual);
    }

    /// <summary>Every package of these languages has its golden file and a row above.</summary>
    [Fact]
    public void EverySupportedPackage_HasGoldenData()
    {
        var rows = typeof(LanguageGoldenTests).GetMethod(nameof(Process_MatchesGolden))!.GetCustomAttributes(typeof(LanguageAttribute), false)
            .Cast<LanguageAttribute>().Select(a => $"{a.Lang}/{a.Package}").ToHashSet();
        foreach (var (lang, packages) in Pipeline.SupportedPackages.Where(l => l.Key is not ("en" or "de")))
        {
            foreach (var package in packages)
            {
                Assert.True(File.Exists(Path.Combine(Repo.Golden, "languages", lang, package + ".conllu")), $"{lang}/{package}: no golden file");
                Assert.Contains($"{lang}/{package}", rows);
            }
        }
    }
}

/// <summary>A row of <see cref="LanguageGoldenTests"/>, skipped without the language's models (models/stanza/&lt;lang&gt;).</summary>
public sealed class LanguageAttribute : Xunit.Sdk.DataAttribute
{
    public LanguageAttribute(string lang, string package, bool managed)
    {
        (Lang, Package, Managed) = (lang, package, managed);
        if (!Directory.Exists(Repo.ModelsFor(lang)))
            Skip = $"{Repo.ModelsFor(lang)} not found; run stanzasharp download --lang {lang} --package {package}";
    }

    public string Lang { get; }

    public string Package { get; }

    public bool Managed { get; }

    public override IEnumerable<object[]> GetData(System.Reflection.MethodInfo testMethod) => [[Lang, Package, Managed]];
}
