using System.Text.RegularExpressions;

namespace StanzaSharp.Tests;

public class PipelineTests
{
    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_ReproducesGoldenConlluExactly(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Backend = Repo.Backend(managed) });
        var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));
        Assert.Equal(golden, Conllu.Write(doc));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_MatchesGoldenOnValidationCorpus(bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Backend = Repo.Backend(managed) });
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

    /// <summary>Removes start_char/end_char from MISC wherever they sit (alone, first, or after other pieces).</summary>
    internal static string StripOffsets(string conllu)
    {
        conllu = Regex.Replace(conllu, @"\tstart_char=\d+\|end_char=\d+$", "\t_", RegexOptions.Multiline); // alone
        conllu = Regex.Replace(conllu, @"(?<=\t)start_char=\d+\|end_char=\d+\|", "");                      // first, e.g. before ner=
        return Regex.Replace(conllu, @"\|start_char=\d+\|end_char=\d+", "");                               // after SpaceAfter= etc.
    }

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

    /// <remarks>With <paramref name="managed"/>, the managed tagger fills the cache with arrays, which managed NER reads as
    /// they are and the TorchSharp constituency parser and sentiment classifier as tensors.</remarks>
    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void CharlmCacheSettings_DoNotChangeOutput(bool managed)
    {
        // Off, and a cap small enough that most sentences are recomputed: same bytes as the default.
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"));
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));
        foreach (var cache in new CharlmCacheOptions[] { new() { IsEnabled = false, MaxWords = 0 }, new() { MaxWords = 10 } })
        {
            using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { CharlmCache = cache, Backend = Repo.Backend(managed) });
            Assert.Equal(golden, Conllu.Write(nlp.Process(text)));
        }
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_AllEightProcessorsMatchEachGoldenFolder(bool managed)
    {
        // No golden file has all eight, so each field is checked against its own folder: word lines and
        // ner= against ner/ (tokenize..depparse,ner), trees against the default pipeline's files, and
        // sentiment against sentiment/. NER, constituency and sentiment share the tagger's CharlmCache.
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,pos,lemma,depparse,ner,sentiment,constituency", Backend = Repo.Backend(managed) });
        foreach (var (name, conllu) in new[] { ("corpus", "pipeline"), ("validation_news", "validation_news"), ("validation_contractions", "validation_contractions") })
        {
            var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, name + ".txt")));
            var written = Conllu.Write(doc);

            var withoutComments = string.Join('\n', written.Split('\n').Where(l => !l.StartsWith("# constituency = ") && !l.StartsWith("# sentiment = ")));
            Assert.Equal(File.ReadAllText(Path.Combine(Repo.Golden, "ner", name + ".conllu")), withoutComments);

            var trees = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, conllu + ".conllu"))).Sentences.Select(s => s.Constituency!.ToString());
            Assert.Equal(trees, doc.Sentences.Select(s => s.Constituency!.ToString()));

            var sentiment = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "sentiment", name + ".json")))!["sentences"]!.AsArray();
            Assert.Equal(sentiment.Select(s => s!["sentiment"]!.GetValue<int>()), doc.Sentences.Select(s => s.Sentiment!.Value));

            var entities = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "ner", name + ".json")))!.AsArray()
                .SelectMany(s => s!.AsArray()).Select(e => (e!["text"]!.GetValue<string>(), e["type"]!.GetValue<string>(), e["start_char"]!.GetValue<int>()));
            Assert.Equal(entities, doc.Sentences.SelectMany(s => s.Entities).Select(e => (e.Text, e.Type, e.StartChar!.Value)));

            // Comments in Stanza's order: sentiment last.
            Assert.Contains("\n# sentiment = ", written.Split("\n\n")[0]);
            Assert.True(written.IndexOf("# constituency = ") < written.IndexOf("# sentiment = "));
        }
    }
}
