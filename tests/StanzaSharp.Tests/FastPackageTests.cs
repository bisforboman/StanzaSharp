using Xunit.Abstractions;
using System.Text.Json.Nodes;
using StanzaSharp.Depparse;
using StanzaSharp.Ner;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using TorchSharp;

namespace StanzaSharp.Tests;

/// <summary>Stanza's English <c>default_fast</c> package against tests/golden/fast (make_golden.py --fast-only).</summary>
public class FastPackageTests(ITestOutputHelper output)
{
    private static readonly string Golden = Path.Combine(Repo.Golden, "fast");

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_MatchesEveryGoldenFile(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Package = "default_fast", Backend = Repo.Backend(managed) });
        var failures = new List<string>();
        var files = new[] { Path.Combine(Repo.Golden, "corpus.txt") }.Concat(Directory.GetFiles(Repo.Golden, "validation*.txt").Order()).ToList();
        foreach (var txt in files)
        {
            var name = Path.GetFileName(txt) == "corpus.txt" ? "corpus" : Path.GetFileNameWithoutExtension(txt);
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(txt)));
            var golden = File.ReadAllText(Path.Combine(Golden, name + ".conllu"));
            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (name == "validation_nonbmp")
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

    [ModelFact]
    public void Intermediates_MatchGolden()
    {
        // The nocharlm models need no charlms; they are given none.
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var tagger = PosTagger.Load(Repo.Model("pos/combined_nocharlm"), pretrain, null, null);
        using var parser = DependencyParser.Load(Repo.Model("depparse/combined_nocharlm"), pretrain, null, null);
        using var ner = NerTagger.Load(Repo.Model("ner/ontonotes-ww-multi_nocharlm"), pretrain, null, null);
        var golden = SafeTensorFile.Load(Path.Combine(Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Golden, "corpus.conllu")));
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

    [ModelFact]
    public void ManagedIntermediates_MatchGolden()
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var tagger = PosTagger.LoadManaged(Repo.Model("pos/combined_nocharlm"), pretrain, null, null);
        using var ner = NerTagger.LoadManaged(Repo.Model("ner/ontonotes-ww-multi_nocharlm"), pretrain, null, null);
        using var parser = DependencyParser.LoadManaged(Repo.Model("depparse/combined_nocharlm"), pretrain, null, null);
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Golden, "corpus.conllu")));
        var golden = SafeTensorFile.Load(Path.Combine(Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "intermediates.json")))!["sentences"]!.AsArray();
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

    private static float Finite(float x) => float.IsNegativeInfinity(x) ? 0 : x;

    private static List<string> Strings(JsonNode? array) => array!.AsArray().Select(x => x!.GetValue<string>()).ToList();

    [Fact]
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
