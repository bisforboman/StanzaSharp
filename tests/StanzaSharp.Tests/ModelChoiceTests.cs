using System.Reflection;
using System.Text.Json.Nodes;
using StanzaSharp.Tool;
using Xunit.Sdk;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="PipelineOptions.Models"/> (Stanza's <c>processors={"ner": "i2b2"}</c>): the selection rules, without models,
/// and each chosen model against tests/golden/models/&lt;name&gt; (make_golden.py --models-only), byte for byte on both backends.
/// </summary>
public class ModelChoiceTests
{
    [ModelTheory]
    [ModelChoice("en", "constituency=ptb3_charlm", false)]
    [ModelChoice("en", "constituency=ptb3_charlm", true)]
    [ModelChoice("en", "ner=ontonotes_charlm", false)]
    [ModelChoice("en", "ner=ontonotes_charlm", true)]
    [ModelChoice("en", "ner=ontonotes_nocharlm", false)]
    [ModelChoice("en", "ner=ontonotes_nocharlm", true)]
    [ModelChoice("en", "ner=combined_charlm", false)]
    [ModelChoice("en", "ner=combined_charlm", true)]
    [ModelChoice("en", "ner=i2b2", false)]
    [ModelChoice("en", "ner=i2b2", true)]
    [ModelChoice("en", "default+ner=i2b2", false)]
    [ModelChoice("en", "default+ner=i2b2", true)]
    [ModelChoice("de", "ner=conll03", false)]
    [ModelChoice("de", "ner=conll03", true)]
    public void Process_MatchesGolden(string name, bool managed, string lang) => AssertGolden(name, managed, lang, cache: true);

    /// <summary>i2b2 reads the mimic charlms, like the mimic tagger, so it reuses the tagger's outputs; without the cache it computes its own.</summary>
    [ModelTheory]
    [ModelChoice("en", "ner=i2b2", true)]
    [ModelChoice("en", "default+ner=i2b2", false)]
    [ModelChoice("en", "default+ner=i2b2", true)]
    public void Process_WithoutTheCharlmCache_MatchesGolden(string name, bool managed, string lang) => AssertGolden(name, managed, lang, cache: false);

    private static void AssertGolden(string name, bool managed, string lang, bool cache)
    {
        var choice = ModelChoiceAttribute.Choice(lang, name);
        var options = ModelChoiceAttribute.Options(choice, lang);
        using var nlp = Pipeline.Load(ModelChoiceAttribute.ModelsFor(lang), new PipelineOptions
        {
            Language = lang,
            Package = options.Package,
            Processors = options.Processors,
            Models = options.Models,
            Backend = Repo.PipelineBackendFor(managed),
            CharlmCache = new() { IsEnabled = cache },
        });
        var golden = Repo.GoldenFor(lang);
        var failures = new List<string>();
        foreach (var file in choice["texts"]!.AsArray().Select(t => t!.GetValue<string>()))
        {
            var text = File.ReadAllText(Path.Combine(golden, file));
            var stem = Path.GetFileName(file) == "corpus.txt" ? "corpus" : Path.GetFileNameWithoutExtension(file);
            var actual = Conllu.Write(nlp.Process(text));
            var expected = File.ReadAllText(Path.Combine(golden, "models", name, stem + ".conllu"));
            var e = expected.Split("\n\n");
            var g = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(e.Length, g.Length) && e[i] == g[i]) i++;
            if (i < e.Length || i < g.Length)
                failures.Add($"{stem}, sentence {i}:\n--- expected\n{e.ElementAtOrDefault(i)}\n--- actual\n{g.ElementAtOrDefault(i)}");
        }

        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    private static Dictionary<string, string> Models(params string[] pairs) => pairs.Select(p => p.Split('=')).ToDictionary(p => p[0], p => p[1]);

    [Fact]
    public void Models_ReplaceThePackagesModel_AndAddTheirProcessor()
    {
        var ptb3 = Pipeline.SelectModels("default", null, addRequired: false, "options", "en", Models("constituency=ptb3_charlm"));
        Assert.Equal(Pipeline.AllProcessors.Split(',').Order(), ptb3.Keys.Order());
        Assert.Equal("ptb3_charlm", ptb3["constituency"]);
        Assert.Equal("ontonotes-ww-multi_charlm", ptb3["ner"]);

        // mimic has no ner: Models adds it, with no requirement beyond tokenize.
        var mimic = Pipeline.SelectModels("mimic", null, addRequired: false, "options", "en", Models("ner=i2b2"));
        Assert.Equal(["depparse", "lemma", "ner", "pos", "tokenize"], mimic.Keys.Order());
        Assert.Equal("i2b2", mimic["ner"]);
        Assert.Equal("mimic_charlm", mimic["pos"]);

        // Constituency added to a treebank package brings what it requires (here all there already) and is checked.
        var ewt = Pipeline.SelectModels("ewt", null, addRequired: false, "options", "en", Models("constituency=ptb3_charlm"));
        Assert.Equal(["constituency", "depparse", "lemma", "mwt", "pos", "tokenize"], ewt.Keys.Order());

        // Keys and names are trimmed and lowercased, as in Stanza.
        Assert.Equal("ontonotes_charlm", Pipeline.SelectModels("default", "tokenize,ner", false, "options", "en", Models(" NER =OntoNotes_CharLM"))["ner"]);
    }

    [Fact]
    public void Models_WithProcessors_StayInsideThem()
    {
        // tokenize brings the package's mwt along, as in Stanza.
        var two = Pipeline.SelectModels("default", "tokenize,ner", addRequired: false, "options", "en", Models("ner=i2b2"));
        Assert.Equal(new Dictionary<string, string> { ["tokenize"] = "combined_nocharlm", ["mwt"] = "combined", ["ner"] = "i2b2" }, two);
        var e = Assert.Throws<ArgumentException>(() => Pipeline.SelectModels("default", "tokenize,mwt", false, "options", "en", Models("ner=i2b2")));
        Assert.Contains("Models names ner, which Processors", e.Message);
    }

    [Fact]
    public void Models_AcceptAPackageName_AndAReplacedTokenizerBringsItsMwt()
    {
        var pos = Pipeline.SelectModels("default", "tokenize,mwt,pos", false, "options", "en", Models("pos=ewt"));
        Assert.Equal("ewt_charlm", pos["pos"]);
        Assert.Equal("combined", pos["mwt"]);

        // Stanza's add_mwt: the tokenizer's own treebank's mwt, by package, model name or the name without _nocharlm.
        Assert.Equal("ewt", Pipeline.SelectModels("default", "tokenize,mwt", false, "options", "en", Models("tokenize=ewt"))["mwt"]);
        Assert.Equal("gum", Pipeline.SelectModels("default", "tokenize,mwt", false, "options", "en", Models("tokenize=gum_nocharlm"))["mwt"]);
        Assert.Equal("combined", Pipeline.SelectModels("default", "tokenize,mwt", false, "options", "en", Models("tokenize=craft"))["mwt"]);
        Assert.Equal("lines", Pipeline.SelectModels("default", "tokenize,mwt", false, "options", "en", Models("tokenize=ewt", "mwt=lines"))["mwt"]);
        Assert.Equal("identity", Pipeline.SelectModels("default", "tokenize,mwt,pos,lemma", false, "options", "en", Models("lemma=identity"))["lemma"]);
    }

    [Theory]
    [InlineData("nerr=i2b2", "Unknown processor 'nerr'")]
    [InlineData("ner=ontonotes", "Unknown or unsupported ner model 'ontonotes'")]
    [InlineData("ner=anatem", "Unknown or unsupported ner model 'anatem'")]
    [InlineData("pos=combined_electra-large", "a transformer model")]
    [InlineData("constituency=ptb3", "Supported: ptb3-revised_charlm, ptb3_charlm")]
    [InlineData("ner=", "Unknown or unsupported ner model ''")]
    public void Models_ThrowForUnknownAndUnsupportedNames(string pair, string message)
    {
        var models = new Dictionary<string, string> { [pair.Split('=')[0]] = pair.Split('=')[1] };
        var e = Assert.Throws<ArgumentException>(() => Pipeline.SelectModels("default", null, false, "options", "en", models));
        Assert.Contains(message, e.Message);
        Assert.Equal("options", e.ParamName);
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(null, "default", "options", "en", models));
    }

    [Fact]
    public void Models_MayReadSeveralCharlmPairs()
    {
        // default's tagger, parsers and sentiment read 1billion, i2b2 mimic (with its own pretrain): the pipeline loads both pairs.
        var models = Pipeline.SelectModels("default", null, false, "options", "en", Models("ner=i2b2"));
        Assert.Equal(8, models.Count);
        Assert.Equal("i2b2", models["ner"]);
        Assert.Equal(["pretrain/conll17", "pretrain/mimic", "forward_charlm/1billion", "forward_charlm/mimic", "backward_charlm/1billion", "backward_charlm/mimic"],
            Pipeline.SharedModels(models, "en"));
        var files = ModelDownloader.FilesFor(null, "default", "options", "en", Models("ner=i2b2")).Select(f => f.Path).ToList();
        Assert.Contains("ner/i2b2.pt", files);
        Assert.Contains("forward_charlm/mimic.pt", files);
        Assert.Contains("backward_charlm/1billion.pt", files);
        Assert.DoesNotContain("ner/ontonotes-ww-multi_charlm.pt", files);
    }

    /// <summary>
    /// With two pairs (1billion for pos, constituency, depparse and sentiment; mimic for ner i2b2) the cache serves only the
    /// tagger's, so each one-sentence call (CharlmCache's single-sentence table, depparse's backward continuation, NER
    /// taking sentiment's multi-word-token sentences) equals the uncached pipeline.
    /// </summary>
    [ModelTheory]
    [ModelChoice("en", "default+ner=i2b2", true)]
    public void OneSentenceDocuments_WithTwoCharlmPairs_EqualTheUncachedPipeline(string name, bool managed, string lang)
    {
        var (package, processors, models) = ModelChoiceAttribute.Options(ModelChoiceAttribute.Choice(lang, name), lang);
        PipelineOptions Options(bool cache) => new()
        {
            Language = lang,
            Package = package,
            Processors = processors,
            Models = models,
            Backend = Repo.PipelineBackendFor(managed),
            CharlmCache = new() { IsEnabled = cache },
        };
        using var cached = Pipeline.Load(ModelChoiceAttribute.ModelsFor(lang), Options(cache: true));
        using var uncached = Pipeline.Load(ModelChoiceAttribute.ModelsFor(lang), Options(cache: false));
        var texts = new[] { "validation.conllu", "validation_contractions.conllu" }
            .SelectMany(f => File.ReadLines(Path.Combine(Repo.GoldenFor(lang), f)).Where(l => l.StartsWith("# text = ")).Take(12))
            .Select(l => l["# text = ".Length..])
            .Concat(File.ReadLines(Path.Combine(Repo.GoldenFor(lang), "models", "clinical.txt")).Where(l => l.Length > 0))
            .ToList();
        int mwt = 0, entities = 0;
        foreach (var text in texts)
        {
            var doc = cached.Process(text);
            mwt += doc.Sentences.Count(s => s.Tokens.Any(t => t.Words.Count > 1));
            entities += doc.Entities.Count();
            Assert.Equal(Conllu.Write(uncached.Process(text)), Conllu.Write(doc));
        }

        Assert.True(mwt >= 5, $"only {mwt} of {texts.Count} texts have multi-word tokens");
        Assert.True(entities >= 20, $"only {entities} entities");
    }

    [Theory]
    [InlineData("en", "default", "tokenize,ner", "ner=ontonotes_charlm",
        "tokenize/combined_nocharlm.pt mwt/combined.pt ner/ontonotes_charlm.pt pretrain/fasttextcrawl.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("en", "default", "tokenize,ner", "ner=ontonotes_nocharlm", "tokenize/combined_nocharlm.pt mwt/combined.pt ner/ontonotes_nocharlm.pt pretrain/fasttextcrawl.pt")]
    [InlineData("en", "mimic", null, "ner=i2b2",
        "tokenize/mimic_nocharlm.pt pos/mimic_charlm.pt lemma/mimic_nocharlm.pt depparse/mimic_charlm.pt ner/i2b2.pt pretrain/mimic.pt forward_charlm/mimic.pt backward_charlm/mimic.pt")]
    [InlineData("en", "default", "constituency", "constituency=ptb3_charlm",
        "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt constituency/ptb3_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("de", "default", "tokenize,ner", "ner=conll03",
        "tokenize/combined_nocharlm.pt mwt/combined.pt ner/conll03.pt pretrain/fasttextwiki.pt forward_charlm/newswiki.pt backward_charlm/newswiki.pt")]
    public void FilesFor_IncludesTheChosenModelsAndTheirDependencies(string lang, string package, string? processors, string pair, string expected)
    {
        var files = ModelDownloader.FilesFor(processors, package, "options", lang, Models(pair)).Select(f => f.Path);
        Assert.Equal(expected.Split(' ').Order(), files.Order());
    }

    [Fact]
    public void EveryAlternative_HasItsFilesAndDependencies()
    {
        foreach (var (lang, processors) in Pipeline.Alternatives)
        {
            foreach (var (processor, names) in processors)
            {
                foreach (var name in names)
                {
                    Assert.True(Pipeline.Dependencies[lang].ContainsKey($"{processor}/{name}"), $"{lang} {processor}/{name}: no Dependencies entry");
                    var sole = new Dictionary<string, string> { [processor] = name };
                    foreach (var file in Pipeline.ModelFiles(sole, lang))
                        Assert.Contains(ModelDownloader.Files[lang], f => f.Path == file + ".pt");
                }
            }
        }
    }

    [Theory]
    [InlineData("ner=i2b2", true, "ner=i2b2")]
    [InlineData("ner=i2b2,constituency=ptb3_charlm", true, "ner=i2b2 constituency=ptb3_charlm")]
    [InlineData(" ner = i2b2 ", true, "ner=i2b2")]
    [InlineData("ner", false, "")]
    [InlineData("ner=", false, "")]
    [InlineData("=i2b2", false, "")]
    [InlineData("ner=a=b", false, "")]
    [InlineData("", false, "")]
    public void ToolModelOption_ParsesProcessorNamePairs(string value, bool ok, string expected)
    {
        var models = new Dictionary<string, string>();
        Assert.Equal(ok, DownloadCommand.AddModels(value, models));
        if (ok)
            Assert.Equal(expected.Split(' '), models.Select(m => $"{m.Key}={m.Value}"));
    }

    [Fact]
    public void ToolModelOption_RejectsAProcessorTwice()
    {
        var models = new Dictionary<string, string>();
        Assert.True(DownloadCommand.AddModels("ner=i2b2", models));
        Assert.False(DownloadCommand.AddModels("ner=conll03_charlm", models));
    }

    [Fact]
    public void ToolStanzaProcessors_IsTheResolvedMapWithModels()
    {
        Assert.Equal("tokenize,mwt", CompareCommand.StanzaProcessors("default", "tokenize,mwt", "en", null));
        var list = CompareCommand.ProcessorList("mimic", null, "en", Models("ner=i2b2"));
        Assert.Equal("tokenize,pos,lemma,depparse,ner", list);
        Assert.Equal("""{"tokenize":"mimic_nocharlm","pos":"mimic_charlm","lemma":"mimic_nocharlm","depparse":"mimic_charlm","ner":"i2b2"}""",
            CompareCommand.StanzaProcessors("mimic", list, "en", Models("ner=i2b2")));
    }
}

/// <summary>
/// A row of <see cref="ModelChoiceTests"/>: a choice of tests/golden/models/models.json (or de/models), plus whether the
/// managed backend runs it, plus the language. Skipped, not failed, where its .pt files are missing (CI caches them for the
/// golden shards only; the no-skip checks elsewhere exempt ModelChoiceTests).
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
public sealed class ModelChoiceAttribute : DataAttribute
{
    private readonly object?[] _data;

    public ModelChoiceAttribute(string lang, string name, bool managed)
    {
        _data = [name, managed, lang];
        var (package, processors, models) = Options(Choice(lang, name), lang);
        var dir = ModelsFor(lang);
        var missing = ModelDownloader.FilesFor(processors, package, "options", lang, models).Select(f => f.Path).Where(f => !File.Exists(Path.Combine(dir, f))).ToList();
        if (missing.Count > 0)
        {
            Skip = $"{name}'s models not found in {dir} ({missing[0]}); run stanzasharp download {dir} --lang {lang} --package {package} "
                + $"--processors {processors} --model {string.Join(",", models.Select(m => $"{m.Key}={m.Value}"))}";
        }
    }

    public override IEnumerable<object?[]> GetData(MethodInfo testMethod) => [_data];

    /// <summary>Stanza's .pt files of a language (models/stanza/&lt;lang&gt;): the chosen models are never converted.</summary>
    internal static string ModelsFor(string lang) => lang == "en" ? Repo.StanzaModels : Path.GetFullPath(Path.Combine(Repo.StanzaModels, "..", lang));

    internal static JsonNode Choice(string lang, string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "models", "models.json")))!.AsArray().Single(c => c!["name"]!.GetValue<string>() == name)!;

    internal static (string Package, string Processors, Dictionary<string, string> Models) Options(JsonNode choice, string lang) =>
        (choice["package"]!.GetValue<string>(), choice["processors"]!.GetValue<string>(),
            choice["models"]!.AsObject().ToDictionary(m => m.Key, m => m.Value!.GetValue<string>()));
}
