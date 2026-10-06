using System.Text.RegularExpressions;

namespace StanzaSharp.Tests;

public class PipelineTests
{
    [ModelFact]
    public void Process_ReproducesGoldenConlluExactly()
    {
        using var nlp = Pipeline.Load(Repo.Models);
        var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));
        Assert.Equal(golden, Conllu.Write(doc));
    }

    [ModelFact]
    public void Process_MatchesGoldenOnValidationCorpus()
    {
        using var nlp = Pipeline.Load(Repo.Models);
        var failures = new List<string>();
        var files = Directory.GetFiles(Repo.Golden, "validation*.txt").Order().ToList();
        Assert.NotEmpty(files);
        foreach (var txt in files)
        {
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(txt)));
            var golden = File.ReadAllText(Path.ChangeExtension(txt, ".conllu"));
            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (Path.GetFileName(txt) == "validation_nonbmp.txt")
                (actual, golden) = (StripOffsets(actual), StripOffsets(golden));

            // Compare sentence by sentence so a failure names the first sentence that differs.
            var expected = golden.Split("\n\n");
            var got = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
            if (i < expected.Length || i < got.Length)
                failures.Add($"{Path.GetFileName(txt)}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");
        }
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    internal static string StripOffsets(string conllu) =>
        Regex.Replace(Regex.Replace(conllu, @"\tstart_char=\d+\|end_char=\d+$", "\t_", RegexOptions.Multiline),
                      @"\|start_char=\d+\|end_char=\d+", "");

    [ModelFact]
    public void Load_RunsASubsetOfProcessors()
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt" });
        var doc = nlp.Process("I don't know.");
        var words = doc.Sentences.Single().Words.ToList();
        Assert.Equal(["I", "do", "n't", "know", "."], words.Select(w => w.Text));
        Assert.All(words, w => Assert.Null(w.Upos));
        Assert.Null(doc.Sentences[0].Constituency);
    }

    [Fact]
    public void Load_RejectsUnknownOrIncompleteProcessorLists()
    {
        var dir = Repo.Root; // any existing directory: validation happens before loading
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,lemma" }));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,constituency" }));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Processors = "" }));
        Assert.Throws<DirectoryNotFoundException>(() => Pipeline.Load(Path.Combine(dir, "no-such-dir")));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pipeline.Load(dir, new PipelineOptions { CharlmCache = new() { MaxWords = 0 } }));
    }

    [ModelFact]
    public void CharlmCacheSettings_DoNotChangeOutput()
    {
        // Off, and a cap small enough that most sentences are recomputed: same bytes as the default.
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"));
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));
        foreach (var cache in new CharlmCacheOptions[] { new() { IsEnabled = false, MaxWords = 0 }, new() { MaxWords = 10 } })
        {
            using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { CharlmCache = cache });
            Assert.Equal(golden, Conllu.Write(nlp.Process(text)));
        }
    }
}
