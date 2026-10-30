using System.Reflection;

namespace StanzaSharp.Tests;

/// <summary>
/// The languages with only tests/golden/languages/&lt;lang&gt;/ (make_golden.py --languages-only; owner's decisions,
/// 2026-10-30): each supported package on the language's text.txt, byte for byte on both backends. Their models are
/// downloaded only by the nightly run (nightly.yml's languages job, a few languages at a time), so every other job skips
/// these rows and its no-skip check exempts the class.
/// </summary>
public class LanguageGoldenTests
{
    [Theory]
    [LanguageGoldenData]
    public void Process_MatchesGolden(string lang, string package, bool managed)
    {
        using var nlp = Pipeline.Load(Repo.ModelsFor(lang), new PipelineOptions { Language = lang, Package = package, Backend = Repo.PipelineBackendFor(managed) });
        var dir = Path.Combine(Repo.Golden, "languages", lang);
        var actual = Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(dir, "text.txt"))));
        Assert.Equal(File.ReadAllText(Path.Combine(dir, package + ".conllu")), actual);
    }

    /// <summary>Every package of these languages has its text and golden file, and every golden folder a supported language.</summary>
    [Fact]
    public void EverySupportedPackage_HasGoldenData()
    {
        foreach (var (lang, package) in LanguageGoldenDataAttribute.Packages)
        {
            Assert.True(File.Exists(Path.Combine(Repo.Golden, "languages", lang, "text.txt")), $"{lang}: no text.txt");
            Assert.True(File.Exists(Path.Combine(Repo.Golden, "languages", lang, package + ".conllu")), $"{lang}/{package}: no golden file");
        }

        var folders = Directory.GetDirectories(Path.Combine(Repo.Golden, "languages")).Select(Path.GetFileName).Order();
        Assert.Equal(LanguageGoldenDataAttribute.Packages.Select(p => p.Lang).Distinct().Order(), folders);
    }
}

/// <summary>
/// The rows of <see cref="LanguageGoldenTests"/>: every package of every language but English and German
/// (<see cref="Pipeline.SupportedPackages"/>) on both backends, for the languages whose models are present
/// (models/stanza/&lt;lang&gt;) and, when STANZASHARP_LANGUAGES is set (comma-separated), only those. Skipped when none is.
/// </summary>
public sealed class LanguageGoldenDataAttribute : Xunit.Sdk.DataAttribute
{
    internal static readonly (string Lang, string Package)[] Packages =
        [.. Pipeline.SupportedPackages.Where(l => l.Key is not ("en" or "de")).SelectMany(l => l.Value.Select(p => (l.Key, p)))];

    private readonly (string Lang, string Package)[] _present;

    public LanguageGoldenDataAttribute()
    {
        var only = Environment.GetEnvironmentVariable("STANZASHARP_LANGUAGES") is { Length: > 0 } list ? list.Split(',') : null;
        _present = [.. Packages.Where(p => (only == null || only.Contains(p.Lang)) && Directory.Exists(Repo.ModelsFor(p.Lang)))];
        if (_present.Length == 0)
            Skip = "No other language's models found (models/stanza/<lang>); run stanzasharp download --lang <lang>";
    }

    // With none present, one row, which the Skip skips (a theory without rows fails).
    public override IEnumerable<object[]> GetData(MethodInfo testMethod) =>
        _present.Length == 0
            ? [["none", Pipeline.DefaultPackage, true]]
            : _present.SelectMany(p => new[] { false, true }.Select(managed => new object[] { p.Lang, p.Package, managed }));
}
