using System.Text.Json.Nodes;

namespace StanzaSharp.Tests;

/// <summary><see cref="PipelineOptions.SplitSentences"/> = false against Stanza's tokenize_no_ssplit (tests/golden/no_ssplit).</summary>
public class NoSsplitTests
{
    private static readonly Dictionary<string, string> Suffix = new() { ["default"] = "", ["default_fast"] = ".fast" };
    private static readonly string Dir = Path.Combine(Repo.Golden, "no_ssplit");

    [ModelTheory]
    [InlineData("default", false)]
    [InlineData("default_fast", false)]
    [InlineData("default", true)]
    public void NoSsplit_MatchesGolden(string package, bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Package = package, SplitSentences = false, Backend = Repo.PipelineBackendFor(managed) });
        var names = Directory.GetFiles(Dir, "*.conllu").Select(Path.GetFileName).Where(f => !f!.EndsWith(".fast.conllu")).ToList();
        Assert.Equal(7, names.Count);
        var failures = new List<string>();
        foreach (var name in names)
        {
            var stem = Path.GetFileNameWithoutExtension(name)!;
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, stem + ".txt"))));
            var golden = File.ReadAllText(Path.Combine(Dir, stem + Suffix[package] + ".conllu"));
            if (stem == "validation_nonbmp")
                (actual, golden) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(golden));
            var e = golden.Split("\n\n");
            var g = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(e.Length, g.Length) && e[i] == g[i]) i++;
            if (i < e.Length || i < g.Length)
                failures.Add($"{stem}, sentence {i}:\n--- expected\n{e.ElementAtOrDefault(i)}\n--- actual\n{g.ElementAtOrDefault(i)}");
        }
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    [ModelTheory]
    [InlineData("default", false)]
    [InlineData("default_fast", false)]
    [InlineData("default", true)]
    public void NoSsplit_Bulk_MatchesStanzasBulkProcess(string package, bool managed)
    {
        var cases = JsonNode.Parse(File.ReadAllText(Path.Combine(Dir, "bulk" + Suffix[package] + ".json")))!.AsArray();
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Package = package, SplitSentences = false, Backend = Repo.PipelineBackendFor(managed) });
        var docs = nlp.Process(cases.Select(c => c!["text"]!.GetValue<string>()).ToList());
        for (int i = 0; i < cases.Count; i++)
        {
            // Stanza's "{:C}": Conllu.Write adds a final newline, except to an empty document.
            var expected = cases[i]!["conllu"]!.GetValue<string>();
            Assert.Equal(expected.Length > 0 ? expected + "\n" : "", Conllu.Write(docs[i]));
        }
    }

    [ModelFact]
    [Trait("Backend", "Managed")]
    public void NoSsplit_OnePerParagraph_AndPretokenizedKeepsItsSentences()
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt", SplitSentences = false });
        var doc = nlp.Process("First one. Second one!\nStill the first paragraph.\n\nNew paragraph. Two sentences.");
        Assert.Equal(["First one. Second one!\nStill the first paragraph.", "New paragraph. Two sentences."], doc.Sentences.Select(s => s.Text));
        // As in Stanza, pretokenized input ignores the option (make_golden asserts it).
        Assert.Equal(2, nlp.Process([["A", "."], ["B", "."]]).Sentences.Count);
    }
}
