using System.Reflection;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Xunit.Sdk;

namespace StanzaSharp.Tests;

/// <summary>
/// The Stanza options with a <see cref="PipelineOptions"/> property (<see cref="PipelineOptions.NerTagSet"/>,
/// <see cref="PipelineOptions.LemmaDictionaryOnly"/>, <see cref="PipelineOptions.MwtDictionaryOnly"/>, the batch sizes)
/// against tests/golden/options/ (and de/options), Stanza run with ner_predict_tagset, lemma_dict_only, mwt_dict_only and
/// the <c>{processor}_batch_size</c> options; byte for byte on both backends.
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
    [OptionCase("en", "sentiment_batch_size=20", false)]
    [OptionCase("en", "sentiment_batch_size=20", true)]
    [OptionCase("en", "batch_sizes", false)]
    [OptionCase("en", "batch_sizes", true)]
    [OptionCase("en", "default_fast+batch_sizes", false)]
    [OptionCase("en", "default_fast+batch_sizes", true)]
    [OptionCase("de", "batch_sizes", false)]
    [OptionCase("de", "batch_sizes", true)]
    [OptionCase("en", "lemma_beam_size=5", false)]
    [OptionCase("en", "lemma_beam_size=5", true)]
    [OptionCase("en", "lemma=combined_charlm+lemma_beam_size=3", false)]
    [OptionCase("en", "lemma=combined_charlm+lemma_beam_size=3", true)]
    [OptionCase("de", "lemma_beam_size=5", false)]
    [OptionCase("de", "lemma_beam_size=5", true)]
    [OptionCase("en", "pos_model_path", false)]
    [OptionCase("en", "pos_model_path", true)]
    [OptionCase("en", "ner_model_path", false)]
    [OptionCase("en", "ner_model_path", true)]
    [OptionCase("en", "default_fast+constituency_model_path", false)]
    [OptionCase("en", "default_fast+constituency_model_path", true)]
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

    [ModelTheory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(5, true)]
    public void TokenizeBatchSize_GivesTheSameTokens(int size, bool managed)
    {
        // Stanza's tokenizer ignores tokenize_batch_size, so there is no golden of its own: any size gives the tokens of
        // the model's 32 (pipeline.conllu, and tokenize_stress's paragraph over one window and its many paragraphs).
        using var nlp = Pipeline.Load(Repo.StanzaModels, new PipelineOptions
        {
            Processors = "tokenize,mwt",
            TokenizeBatchSize = size,
            Backend = Repo.PipelineBackendFor(managed),
        });
        foreach (var name in new[] { "tokenize_stress", "validation" })
        {
            var text = File.ReadAllText(Path.Combine(Repo.Golden, name + ".txt"));
            var expected = Tokens(Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, name + ".conllu"))));
            Assert.Equal(expected, Tokens(nlp.Process(text)));
        }

        static List<string> Tokens(Document doc) =>
            [.. doc.Sentences.Select(s => string.Join(" ", s.Tokens.Select(t => $"{t.Text}|{t.StartChar}|{t.EndChar}")))];
    }

    [Theory]
    [InlineData(nameof(PipelineOptions.TokenizeBatchSize))]
    [InlineData(nameof(PipelineOptions.PosBatchMaximumTokens))]
    [InlineData(nameof(PipelineOptions.DepparseMinLengthToBatchSeparately))]
    [InlineData(nameof(PipelineOptions.SentimentBatchSize))]
    public void BatchSize_BelowOne_Throws(string option)
    {
        // Checked before any model file is looked for.
        var options = option switch
        {
            nameof(PipelineOptions.TokenizeBatchSize) => new PipelineOptions { TokenizeBatchSize = 0 },
            nameof(PipelineOptions.PosBatchMaximumTokens) => new PipelineOptions { PosBatchMaximumTokens = -1 },
            nameof(PipelineOptions.DepparseMinLengthToBatchSeparately) => new PipelineOptions { DepparseMinLengthToBatchSeparately = 0 },
            _ => new PipelineOptions { SentimentBatchSize = 0 },
        };
        var e = Assert.Throws<ArgumentOutOfRangeException>(() => Pipeline.Load(Path.GetTempPath(), options));
        Assert.Contains(option, e.Message);
    }
}

/// <summary>A row of <see cref="OptionTests"/>: a case of tests/golden/options/options.json, skipped where its models are missing.</summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
public sealed partial class OptionCaseAttribute : DataAttribute
{
    [GeneratedRegex("^([a-z]+)_(model|pretrain|forward_charlm|backward_charlm)_path$")]
    private static partial Regex PathKey();

    private readonly object?[] _data;

    public OptionCaseAttribute(string lang, string name, bool managed)
    {
        _data = [name, managed, lang];
        var options = Options(Case(lang, name), lang, managed);
        var dir = ModelChoiceAttribute.ModelsFor(lang);
        var missing = ModelDownloader.FilesFor(options.Processors, options.Package, "options", lang, options.Models, modelPaths: options.ModelPaths).Select(f => f.Path)
            .Where(f => !File.Exists(Path.Combine(dir, f)))
            .ToList();
        if (missing.Count > 0)
            Skip = $"{name}'s models not found in {dir} ({missing[0]})";
    }

    public override IEnumerable<object?[]> GetData(MethodInfo testMethod) => [_data];

    internal static JsonNode Case(string lang, string name) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "options", "options.json")))!.AsArray().Single(c => c!["name"]!.GetValue<string>() == name)!;

    /// <summary>Stanza's batch-size keyword arguments (each <see cref="Options"/> maps to its property).</summary>
    private static readonly HashSet<string> BatchSizeKeys =
    [
        "mwt_batch_size", "pos_batch_size", "pos_batch_maximum_tokens", "lemma_batch_size", "constituency_batch_size",
        "depparse_batch_size", "depparse_min_length_to_batch_separately", "sentiment_batch_size", "ner_batch_size",
    ];

    /// <summary>The case's PipelineOptions: Stanza's keyword arguments mapped to their properties.</summary>
    internal static PipelineOptions Options(JsonNode choice, string lang, bool managed)
    {
        var stanza = choice["options"]!.AsObject();
        foreach (var key in stanza.Select(kv => kv.Key))
        {
            if (key is not ("ner_predict_tagset" or "lemma_dict_only" or "mwt_dict_only" or "lemma_beam_size") && !BatchSizeKeys.Contains(key) && !PathKey().IsMatch(key))
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
            MwtBatchSize = Size("mwt_batch_size"),
            PosBatchSize = Size("pos_batch_size"),
            PosBatchMaximumTokens = Size("pos_batch_maximum_tokens"),
            LemmaBatchSize = Size("lemma_batch_size"),
            LemmaBeamSize = Size("lemma_beam_size"),
            ConstituencyBatchSize = Size("constituency_batch_size"),
            DepparseBatchSize = Size("depparse_batch_size"),
            DepparseMinLengthToBatchSeparately = Size("depparse_min_length_to_batch_separately"),
            SentimentBatchSize = Size("sentiment_batch_size"),
            NerBatchSize = Size("ner_batch_size"),
            ModelPaths = Paths(),
        };

        int? Size(string key) => stanza[key]?.GetValue<int>();

        // {processor}_model_path and the dependency paths beside it, "{lang_dir}" being the folder the models load from.
        Dictionary<string, ModelPath>? Paths()
        {
            var paths = stanza.Select(kv => (Match: PathKey().Match(kv.Key), kv.Value))
                .Where(x => x.Match.Success)
                .Select(x => (x.Match, Value: x.Value!.GetValue<string>().Replace("{lang_dir}", Path.GetFullPath(ModelChoiceAttribute.ModelsFor(lang)))))
                .GroupBy(x => x.Match.Groups[1].Value)
                .ToDictionary(g => g.Key, g =>
                {
                    string? Of(string kind) => g.SingleOrDefault(x => x.Match.Groups[2].Value == kind).Value;
                    return new ModelPath(Of("model")!, Of("pretrain"), Of("forward_charlm"), Of("backward_charlm"));
                });
            return paths.Count == 0 ? null : paths;
        }
    }
}
