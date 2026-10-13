using System.Text.Json.Nodes;
using StanzaSharp.Depparse;
using StanzaSharp.Ner;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using StanzaSharp.Pos;
using TorchSharp;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>Stanza's <c>default_fast</c> package against tests/golden/fast and tests/golden/de/fast (make_golden.py --fast-only).</summary>
public class FastPackageTests(ITestOutputHelper output, SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    private static string GoldenDir(string lang) => Path.Combine(Repo.GoldenFor(lang), "fast");

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Process_MatchesEveryGoldenFile(bool managed, string lang)
    {
        var nlp = pipelines.Get("default_fast", managed, lang: lang);
        var failures = new List<string>();
        var files = new[] { Path.Combine(Repo.GoldenFor(lang), "corpus.txt") }.Concat(Directory.GetFiles(Repo.GoldenFor(lang), "validation*.txt").Order()).ToList();
        foreach (var txt in files)
        {
            var name = Path.GetFileName(txt) == "corpus.txt" ? "corpus" : Path.GetFileNameWithoutExtension(txt);
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(txt)));
            var golden = File.ReadAllText(Path.Combine(GoldenDir(lang), name + ".conllu"));

            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (PipelineTests.HasNonBmp(File.ReadAllText(txt)))
                (actual, golden) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(golden));
            var expected = golden.Split("\n\n");
            var got = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
            if (i < expected.Length || i < got.Length)
                failures.Add($"{name}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");
        }

        Assert.Equal(13, files.Count);
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    /// <remarks>German's fast package keeps the charlm NER (germeval2014), with its own pretrain.</remarks>
    [ModelTheory]
    [InlineData("en")]
    [German]
    public void Intermediates_MatchGolden(string lang)
    {
        // The nocharlm models need no charlms; they are given none.
        using var pretrain = Pretrain.Load(Repo.Model(lang, "pretrain/conll17"));
        using var tagger = PosTagger.Load(Repo.ModelOf(lang, "pos", "default_fast"), pretrain, null, null);
        using var parser = DependencyParser.Load(Repo.ModelOf(lang, "depparse", "default_fast"), pretrain, null, null);
        using var nerPretrain = lang == "en" ? null : Repo.LoadPretrainOf(managed: false, lang, "ner", "default_fast");
        using var forward = lang == "en" ? null : CharLanguageModel.Load(Repo.Charlm(lang, forward: true));
        using var backward = lang == "en" ? null : CharLanguageModel.Load(Repo.Charlm(lang, forward: false));
        using var ner = NerTagger.Load(Repo.ModelOf(lang, "ner", "default_fast"), nerPretrain ?? pretrain, forward, backward);
        var golden = SafeTensorFile.Load(Path.Combine(GoldenDir(lang), "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(GoldenDir(lang), "intermediates.json")))!["sentences"]!.AsArray();
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(GoldenDir(lang), "corpus.conllu")));
        Assert.Equal(3, index.Count);

        // One padded batch each; the golden values were computed one sentence at a time.
        var words = index.Select(e => (IReadOnlyList<string>)Strings(e!["words"])).ToList();
        var tags = tagger.Predict(words, out var uposLogits);
        var tokens = index.Select(e => (IReadOnlyList<string>)Strings(e!["tokens"])).ToList();
        var nerTags = ner.Predict(tokens, out var emissions);
        for (int i = 0; i < index.Count; i++)
        {
            TokenizerTests.AssertClose(golden.Read<float>($"s{i}.pos.upos_logits"), uposLogits[i], 1e-4f, $"s{i}.pos");
            Assert.Equal(Strings(index[i]!["xpos"]), tags[i].Select(t => t.Xpos));
            TokenizerTests.AssertClose(golden.Read<float>($"s{i}.ner.emissions"), emissions[i], 1e-4f, $"s{i}.ner");
            Assert.Equal(Strings(index[i]!["ner"]), nerTags[i]);

            var sentence = doc.Sentences[i].Words.ToList();
            Assert.Equal(words[i], sentence.Select(w => w.Text));
            DepparseTests.AssertScoresMatch(parser.Scores([sentence], labelScores: true),
                golden.Read<float>($"s{i}.depparse.unlabeled"), golden.Read<float>($"s{i}.depparse.deprel"), $"s{i}");
            Assert.Equal(index[i]!["heads"]!.AsArray().Select(h => h!.GetValue<int>()), parser.Parse([sentence])[0].Select(p => p.Head));
        }
    }

    [ModelTheory]
    [InlineData("en")]
    [German]
    public void ManagedIntermediates_MatchGolden(string lang)
    {
        using var pretrain = Pretrain.Load(Repo.Model(lang, "pretrain/conll17"));
        using var tagger = PosTagger.LoadManaged(Repo.ModelOf(lang, "pos", "default_fast"), pretrain, null, null);
        using var nerPretrain = lang == "en" ? null : Repo.LoadPretrainOf(managed: true, lang, "ner", "default_fast");
        using var ner = NerTagger.LoadManaged(Repo.ModelOf(lang, "ner", "default_fast"), nerPretrain ?? pretrain,
            lang == "en" ? null : ManagedCharLanguageModel.Load(Repo.Charlm(lang, forward: true)),
            lang == "en" ? null : ManagedCharLanguageModel.Load(Repo.Charlm(lang, forward: false)));
        using var parser = DependencyParser.LoadManaged(Repo.ModelOf(lang, "depparse", "default_fast"), pretrain, null, null);
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(GoldenDir(lang), "corpus.conllu")));
        var golden = SafeTensorFile.Load(Path.Combine(GoldenDir(lang), "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(GoldenDir(lang), "intermediates.json")))!["sentences"]!.AsArray();
        var words = index.Select(e => (IReadOnlyList<string>)Strings(e!["words"])).ToList();
        var tags = tagger.Predict(words, out var uposLogits);
        var tokens = index.Select(e => (IReadOnlyList<string>)Strings(e!["tokens"])).ToList();
        var nerTags = ner.Predict(tokens, out var emissions);
        for (int i = 0; i < index.Count; i++)
        {
            output.WriteLine($"s{i}: UPOS max |diff| {TokenizerTests.AssertClose(golden.Read<float>($"s{i}.pos.upos_logits"), uposLogits[i], 1e-4f, $"s{i}.pos"):E2}");
            Assert.Equal(Strings(index[i]!["xpos"]), tags[i].Select(t => t.Xpos));
            output.WriteLine($"s{i}: max |diff| {TokenizerTests.AssertClose(golden.Read<float>($"s{i}.ner.emissions"), emissions[i], 1e-4f, $"s{i}.ner"):E2}");
            Assert.Equal(Strings(index[i]!["ner"]), nerTags[i]);

            var sentence = doc.Sentences[i].Words.ToList();
            Assert.Equal(words[i], sentence.Select(w => w.Text));
            var (arcs, labels) = DepparseTests.AssertScoresMatch(parser.Scores([sentence], labelScores: true),
                golden.Read<float>($"s{i}.depparse.unlabeled"), golden.Read<float>($"s{i}.depparse.deprel"), $"s{i}");
            output.WriteLine($"s{i}: arc log-probs max |diff| {arcs:E2}, label log-probs {labels:E2}");
            Assert.Equal(index[i]!["heads"]!.AsArray().Select(h => h!.GetValue<int>()), parser.Parse([sentence])[0].Select(p => p.Head));
        }
    }

    private static List<string> Strings(JsonNode? array) => array!.AsArray().Select(x => x!.GetValue<string>()).ToList();

    [Fact]
    [Trait("Backend", "Managed")]
    public void SelectModels_UsesThePackagesModelsAndProcessors()
    {
        var fast = Pipeline.SelectModels("default_fast", null, addRequired: false, "options");
        Assert.Equal(new[] { "tokenize", "mwt", "pos", "lemma", "depparse", "sentiment", "ner" }.Order(), fast.Keys.Order());
        Assert.Equal("combined_nocharlm", fast["pos"]);
        Assert.Equal("combined_nocharlm", fast["depparse"]);
        Assert.Equal("ontonotes-ww-multi_nocharlm", fast["ner"]);

        // Only sentiment reads the charlms.
        Assert.Equal(["pretrain/conll17", "forward_charlm/1billion", "backward_charlm/1billion"], Pipeline.SharedModels(fast));
        fast.Remove("sentiment");
        Assert.Equal(["pretrain/conll17"], Pipeline.SharedModels(fast));
        Assert.Equal(Pipeline.AllProcessors.Split(',').Order(), Pipeline.SelectModels("default", null, false, "options").Keys.Order());
    }

    [Fact]
    [Trait("Backend", "Managed")]
    public void Load_RejectsConstituencyInTheFastPackageAndUnknownPackages()
    {
        var dir = Repo.Root; // validation happens before loading
        var ex = Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Package = "default_fast", Processors = "tokenize,mwt,pos,constituency" }));
        Assert.Contains("constituency", ex.Message);
        Assert.Contains("\"default\"", ex.Message);
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Package = "fast" }));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Package = "" }));
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor("constituency", "default_fast"));
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(null, "Default_Fast"));
    }
}
