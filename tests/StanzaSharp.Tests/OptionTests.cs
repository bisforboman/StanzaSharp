using System.Reflection;
using System.Text.Json.Nodes;
using Xunit.Sdk;

namespace StanzaSharp.Tests;

/// <summary>
/// The Stanza options with a <see cref="PipelineOptions"/> property (<see cref="PipelineOptions.NerTagSet"/>,
/// <see cref="PipelineOptions.LemmaDictionaryOnly"/>, <see cref="PipelineOptions.MwtDictionaryOnly"/>) against
/// tests/golden/options/ (and de/options), Stanza run with ner_predict_tagset, lemma_dict_only and mwt_dict_only; byte for
/// byte on both backends.
/// </summary>
public class OptionTests
{
    [ModelTheory]
    [OptionCase("en", "ner_predict_tagset=1", false)]
    [OptionCase("en", "ner_predict_tagset=1", true)]
    [OptionCase("en", "ner=combined_charlm+ner_predict_tagset=2", false)]
    [OptionCase("en", "ner=combined_charlm+ner_predict_tagset=2", true)]
    [OptionCase("en", "lemma_dict_only", false)]
    [OptionCase("en", "lemma_dict_only", true)]
    [OptionCase("en", "lemma=combined_charlm+lemma_dict_only", false)]
    [OptionCase("en", "lemma=combined_charlm+lemma_dict_only", true)]
    [OptionCase("en", "mwt_dict_only", false)]
    [OptionCase("en", "mwt_dict_only", true)]
    [OptionCase("de", "mwt_dict_only", false)]
    [OptionCase("de", "mwt_dict_only", true)]
    [OptionCase("de", "lemma_dict_only", false)]
    [OptionCase("de", "lemma_dict_only", true)]
    public void Process_MatchesGolden(string name, bool managed, string lang)
    {
        var golden = Repo.GoldenFor(lang);
        var choice = OptionCaseAttribute.Case(lang, name);
        var options = OptionCaseAttribute.Options(choice, lang, managed);
        using var nlp = Pipeline.Load(ModelChoiceAttribute.ModelsFor(lang), options);
        var failures = new List<string>();
        foreach (var file in choice["texts"]!.AsArray().Select(t => t!.GetValue<string>()))
        {
            var text = File.ReadAllText(Path.Combine(golden, file));
            var stem = Path.GetFileName(file) == "corpus.txt" ? "corpus" : Path.GetFileNameWithoutExtension(file);
            var actual = Conllu.Write(nlp.Process(text));
            var expected = File.ReadAllText(Path.Combine(golden, "options", name, stem + ".conllu"));
            if (PipelineTests.HasNonBmp(text))
                (actual, expected) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(expected));
            var e = expected.Split("\n\n");
            var g = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(e.Length, g.Length) && e[i] == g[i]) i++;
            if (i < e.Length || i < g.Length)
                failures.Add($"{stem}, sentence {i}:\n--- expected\n{e.ElementAtOrDefault(i)}\n--- actual\n{g.ElementAtOrDefault(i)}");
        }

        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelFact]
    public void NerTagSet_TheModelLacks_Throws()
    {
        // ontonotes-ww-multi has tag sets 0 and 1; the error names them, before any text is processed.
        var e = Assert.Throws<ArgumentException>(() => Pipeline.Load(Repo.StanzaModels, new PipelineOptions { Processors = "tokenize,ner", NerTagSet = 2 }));
        Assert.Contains("2 tag sets (0, 1), not 2", e.Message);
    }
}

/// <summary>A row of <see cref="OptionTests"/>: a case of tests/golden/options/options.json, skipped where its models are missing.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
public sealed class OptionCaseAttribute : DataAttribute
{
    private readonly object?[] _data;

    public OptionCaseAttribute(string lang, string name, bool managed)
    {
        _data = [name, managed, lang];
        var options = Options(Case(lang, name), lang, managed);
        var dir = ModelChoiceAttribute.ModelsFor(lang);
        var missing = ModelDownloader.FilesFor(options.Processors, options.Package, "options", lang, options.Models).Select(f => f.Path)
            .Where(f => !File.Exists(Path.Combine(dir, f)))
            .ToList();
        if (missing.Count > 0)
            Skip = $"{name}'s models not found in {dir} ({missing[0]})";
    }

    public override IEnumerable<object?[]> GetData(MethodInfo testMethod) => [_data];

    internal static JsonNode Case(string lang, string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "options", "options.json")))!.AsArray().Single(c => c!["name"]!.GetValue<string>() == name)!;

    /// <summary>The case's PipelineOptions: Stanza's keyword arguments mapped to their properties.</summary>
    internal static PipelineOptions Options(JsonNode choice, string lang, bool managed)
    {
        var stanza = choice["options"]!.AsObject();
        foreach (var key in stanza.Select(kv => kv.Key))
        {
            if (key is not ("ner_predict_tagset" or "lemma_dict_only" or "mwt_dict_only"))
                throw new InvalidDataException($"options.json: no PipelineOptions property for Stanza's {key}");
        }

        return new PipelineOptions
        {
            Language = lang,
            Package = choice["package"]!.GetValue<string>(),
            Processors = choice["processors"]!.GetValue<string>(),
            Models = choice["models"]!.AsObject().ToDictionary(m => m.Key, m => m.Value!.GetValue<string>()),
            Backend = Repo.PipelineBackendFor(managed),
            NerTagSet = stanza["ner_predict_tagset"]?.GetValue<int>(),
            LemmaDictionaryOnly = stanza["lemma_dict_only"]?.GetValue<bool>() ?? false,
            MwtDictionaryOnly = stanza["mwt_dict_only"]?.GetValue<bool>() ?? false,
        };
    }
}
