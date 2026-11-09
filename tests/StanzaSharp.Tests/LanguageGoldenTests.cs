using System.Reflection;
using System.Text.Json;
using StanzaSharp.Nn;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// The languages with only tests/golden/languages/&lt;lang&gt;/ (make_golden.py --languages-only; owner's decisions,
/// 2026-10-30): each supported package on the language's text.txt, byte for byte on both backends. Their models are
/// downloaded only by the nightly run (nightly.yml's languages job, a few languages at a time), so every other job skips
/// these rows and its no-skip check exempts the class.
/// </summary>
public class LanguageGoldenTests(ITestOutputHelper output)
{
    /// <summary>
    /// A package on its language's text.txt; first, every transformer tokenizer golden file of the language
    /// (<c>tokenizer_&lt;transformer&gt;.json</c>: Odia's MuRIL) against that transformer's tokenizer, so the check runs
    /// where the nightly job has the transformer, adding no row that skips elsewhere.
    /// </summary>
    [Theory]
    [LanguageGoldenData]
    public void Process_MatchesGolden(string lang, string package, bool managed)
    {
        var dir = Path.Combine(Repo.Golden, "languages", lang);
        foreach (var golden in Directory.GetFiles(dir, "tokenizer_*.json"))
        {
            using var meta = JsonDocument.Parse(File.ReadAllBytes(golden));
            var name = meta.RootElement.GetProperty("transformer").GetString()!;
            TransformerTokenizerTests.AssertGolden(TransformerEmbedder.LoadTokenizer(ModelDownloader.TransformerDirectory(Repo.ModelsFor(lang), name)),
                golden, minCases: 10, output);
        }

        using var nlp = Pipeline.Load(Repo.ModelsFor(lang), new PipelineOptions { Language = lang, Package = package, Backend = Repo.PipelineBackendFor(managed) });
        var actual = Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(dir, "text.txt"))));
        Assert.Equal(File.ReadAllText(Path.Combine(dir, package + ".conllu")), actual);
    }

    /// <summary>
    /// The <see cref="PipelineOptions.Models"/> choices of these languages' <see cref="Pipeline.Alternatives"/> that no
    /// package runs (make_golden.py's LANGUAGE_MODEL_CHOICES), each on its language's text.txt: every golden file
    /// <c>&lt;package&gt;+&lt;processor&gt;=&lt;model&gt;,...conllu</c>, as (lang, file name, package, models).
    /// </summary>
    internal static readonly (string Lang, string Name, string Package, Dictionary<string, string> Models)[] Choices =
    [
        .. Directory.GetFiles(Path.Combine(Repo.Golden, "languages"), "*+*.conllu", SearchOption.AllDirectories).Order().Select(file =>
        {
            var name = Path.GetFileNameWithoutExtension(file);
            var plus = name.IndexOf('+');
            return (Path.GetFileName(Path.GetDirectoryName(file))!, name, name[..plus],
                name[(plus + 1)..].Split(',').Select(m => m.Split('=')).ToDictionary(m => m[0], m => m[1]));
        }),
    ];

    /// <summary>A <see cref="Choices"/> row on its language's text.txt.</summary>
    [Theory]
    [LanguageChoiceData]
    public void Models_MatchGolden(string lang, string name, bool managed)
    {
        var (_, _, package, models) = Choices.Single(c => c.Lang == lang && c.Name == name);
        var dir = Path.Combine(Repo.Golden, "languages", lang);
        using var nlp = Pipeline.Load(Repo.ModelsFor(lang),
            new PipelineOptions { Language = lang, Package = package, Models = models, Backend = Repo.PipelineBackendFor(managed) });
        var actual = Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(dir, "text.txt"))));
        Assert.Equal(File.ReadAllText(Path.Combine(dir, name + ".conllu")), actual);
    }

    /// <summary>Every alternative of these languages is in a package or a <see cref="Choices"/> row, so each has golden data.</summary>
    [Fact]
    public void EveryAlternative_IsInAPackageOrAChoice()
    {
        foreach (var (lang, processors) in Pipeline.Alternatives.Where(l => l.Key is not ("en" or "de")))
        {
            foreach (var (processor, names) in processors)
            {
                foreach (var name in names)
                {
                    Assert.True(LanguageGoldenDataAttribute.Packages.Any(p => p.Lang == lang && Pipeline.Packages[lang][p.Package].GetValueOrDefault(processor) == name)
                        || Choices.Any(c => c.Lang == lang && c.Models.GetValueOrDefault(processor) == name), $"{lang} {processor}={name}: no golden data");
                }
            }
        }
    }

    /// <summary>
    /// KB-BERT's tokenizer has no model_max_length and Stanza doesn't set one, so Stanza never runs a sentence in windows and
    /// crashes on one over its 512 positions (a RuntimeError in the position embeddings): StanzaSharp throws instead
    /// (owner's decision, 2026-11-09).
    /// </summary>
    [SwedishKbBertTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void SwedishKbBert_SentenceOverItsPositions_Throws(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.ModelsFor("sv"), new PipelineOptions
        {
            Language = "sv",
            Package = "default_accurate",
            Processors = "tokenize,pos",
            Backend = Repo.PipelineBackendFor(managed),
        });
        var ex = Assert.Throws<ArgumentException>(() => nlp.Process(string.Join(' ', Enumerable.Repeat("och", 600)) + " ."));
        Assert.Contains("512 positions", ex.Message);
        Assert.Single(nlp.Process("Det här går bra.").Sentences); // the pipeline still runs after the throw
    }

    /// <summary>
    /// Every package of these languages has its text and golden file, every golden folder is a supported language, and every
    /// choice file names a package and alternatives of its language.
    /// </summary>
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
        Assert.NotEmpty(Choices);
        foreach (var (lang, name, package, models) in Choices)
        {
            Assert.Contains(package, Pipeline.SupportedPackages[lang]);
            Assert.All(models, m => Assert.Contains(m.Value, Pipeline.Alternatives[lang][m.Key]));
        }
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

/// <summary>
/// The rows of <see cref="LanguageGoldenTests.Models_MatchGolden"/>: every <see cref="LanguageGoldenTests.Choices"/> row on both
/// backends whose language's models are present (and, with STANZASHARP_LANGUAGES set, named there). Skipped when none is.
/// </summary>
public sealed class LanguageChoiceDataAttribute : Xunit.Sdk.DataAttribute
{
    private readonly (string Lang, string Name)[] _present;

    public LanguageChoiceDataAttribute()
    {
        var only = Environment.GetEnvironmentVariable("STANZASHARP_LANGUAGES") is { Length: > 0 } list ? list.Split(',') : null;
        _present = [.. LanguageGoldenTests.Choices.Where(c => (only == null || only.Contains(c.Lang)) && Directory.Exists(Repo.ModelsFor(c.Lang)))
            .Select(c => (c.Lang, c.Name))];
        if (_present.Length == 0)
            Skip = "No models found for a language with model choices (models/stanza/<lang>); run stanzasharp download --lang <lang>";
    }

    // With none present, one row, which the Skip skips (a theory without rows fails).
    public override IEnumerable<object[]> GetData(MethodInfo testMethod) =>
        _present.Length == 0
            ? [["none", "none", true]]
            : _present.SelectMany(c => new[] { false, true }.Select(managed => new object[] { c.Lang, c.Name, managed }));
}

/// <summary>Skips unless Swedish default_accurate's models and KBLab/bert-base-swedish-cased are in models/stanza/sv.</summary>
public sealed class SwedishKbBertTheoryAttribute : TheoryAttribute
{
    public SwedishKbBertTheoryAttribute()
    {
        var models = Repo.ModelsFor("sv");
        var missing = new[] { Path.Combine(models, "pos", "talbanken_kb-swedish-bert.pt"),
            Path.Combine(ModelDownloader.TransformerDirectory(models, "KBLab/bert-base-swedish-cased"), "pytorch_model.bin"), }
            .FirstOrDefault(f => !File.Exists(f));
        if (missing != null)
            Skip = $"{missing} not found; run stanzasharp download models/stanza/sv --lang sv --package default_accurate";
    }
}
