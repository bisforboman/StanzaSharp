using System.Text.RegularExpressions;

namespace StanzaSharp.Tests;

public class PipelineTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    [ModelTheory]
    [Trait("Smoke", "TorchSharp")] // all eight processors on TorchSharp in a ready pull request's light golden run (ci.yml)
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Process_ReproducesGoldenConlluExactly(bool managed, string lang)
    {
        var nlp = pipelines.Get(managed: managed, lang: lang);
        var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "corpus.txt")));
        var golden = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "pipeline.conllu"));
        Assert.Equal(golden, Conllu.Write(doc));
    }

    /// <summary>Every processor runs managed, so a managed pipeline of either default package holds only the managed charlms.</summary>
    [ModelFact]
    [Trait("Backend", "Managed")]
    public void ManagedBackend_LoadsNoTorchSharpCharlms()
    {
        foreach (var package in new[] { Pipeline.DefaultPackage, "default_fast" })
        {
            var nlp = pipelines.Get(package);
            const System.Reflection.BindingFlags Private = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;

            // The charlms belong to the backend's models (BackendModels); a managed pipeline's are managed by construction.
            var models = Assert.IsType<ManagedModels>(typeof(Pipeline).GetField("_models", Private)!.GetValue(nlp));
            Assert.NotNull(typeof(ManagedModels).GetField("_forward", Private)!.GetValue(models));
            Assert.NotNull(typeof(ManagedModels).GetField("_backward", Private)!.GetValue(models));
        }
    }

    /// <summary>
    /// One sentence per call: the managed processors share every single-sentence charlm output they can
    /// (CharlmCache.TryGetAlone: the parser's backward pass, NER after sentiment on multi-word tokens). Computed alone,
    /// those are the same bits as computing them again, so the output equals the uncached pipeline's exactly.
    /// </summary>
    [ModelTheory]
    [InlineData("en")]
    [German]
    [Trait("Backend", "Managed")]
    public void OneSentenceDocuments_EqualTheUncachedPipeline(string lang)
    {
        var cached = pipelines.Get(lang: lang);
        using var uncached = Pipeline.Load(Repo.ModelsFor(lang), new PipelineOptions { Language = lang, CharlmCache = new CharlmCacheOptions { IsEnabled = false } });
        var texts = Directory.GetFiles(Repo.GoldenFor(lang), "validation*.conllu").Order().SelectMany(f => File.ReadLines(f).Where(l => l.StartsWith("# text = ")).Take(8))
            .Select(l => l["# text = ".Length..])
            .ToList();
        int mwt = 0;
        foreach (var text in texts)
        {
            var doc = cached.Process(text);
            mwt += doc.Sentences.Count(s => s.Tokens.Any(t => t.Words.Count > 1));
            Assert.Equal(Conllu.Write(uncached.Process(text)), Conllu.Write(doc));
        }

        Assert.True(mwt >= 5, $"only {mwt} of {texts.Count} texts have multi-word tokens");
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Process_MatchesGoldenOnValidationCorpus(bool managed, string lang)
    {
        var nlp = pipelines.Get(managed: managed, lang: lang);
        var failures = new List<string>();
        var files = Directory.GetFiles(Repo.GoldenFor(lang), "validation*.txt").Order().ToList();
        Assert.NotEmpty(files);
        foreach (var txt in files)
        {
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(txt)));
            var golden = File.ReadAllText(Path.ChangeExtension(txt, ".conllu"));

            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (HasNonBmp(File.ReadAllText(txt)))
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

    /// <summary>Whether <paramref name="text"/> has characters outside the BMP, after which C# and Stanza offsets differ.</summary>
    internal static bool HasNonBmp(string text) => text.Any(char.IsSurrogate);

    /// <summary>Removes start_char/end_char from MISC wherever they sit (alone, first, or after other pieces).</summary>
    internal static string StripOffsets(string conllu)
    {
        conllu = Regex.Replace(conllu, @"\tstart_char=\d+\|end_char=\d+$", "\t_", RegexOptions.Multiline); // alone
        conllu = Regex.Replace(conllu, @"(?<=\t)start_char=\d+\|end_char=\d+\|", "");                      // first, e.g. before ner=
        return Regex.Replace(conllu, @"\|start_char=\d+\|end_char=\d+", "");                               // after SpaceAfter= etc.
    }

    [ModelFact]
    [Trait("Backend", "Managed")]
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
    [Trait("Backend", "Managed")]
    public void Load_RejectsUnknownOrIncompleteProcessorLists()
    {
        var dir = Repo.Root; // any existing directory: validation happens before loading
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,lemma" }));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Processors = "tokenize,constituency" }));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, new PipelineOptions { Processors = "" }));
        Assert.Throws<DirectoryNotFoundException>(() => Pipeline.Load(Path.Combine(dir, "no-such-dir")));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pipeline.Load(dir, new PipelineOptions { CharlmCache = new() { MaxWords = 0 } }));
    }

    /// <remarks>With <paramref name="managed"/>, the managed tagger fills the cache with arrays, which managed constituency,
    /// sentiment and NER read as they are.</remarks>
    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void CharlmCacheSettings_DoNotChangeOutput(bool managed, string lang)
    {
        // Off, and a cap small enough that most sentences are recomputed: same bytes as the default.
        var text = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "corpus.txt"));
        var golden = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "pipeline.conllu"));
        foreach (var cache in new CharlmCacheOptions[] { new() { IsEnabled = false, MaxWords = 0 }, new() { MaxWords = 10 } })
        {
            using var nlp = Pipeline.Load(Repo.ModelsFor(lang), new PipelineOptions { Language = lang, CharlmCache = cache, Backend = Repo.PipelineBackendFor(managed) });
            Assert.Equal(golden, Conllu.Write(nlp.Process(text)));
        }
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Process_AllEightProcessorsMatchEachGoldenFolder(bool managed, string lang)
    {
        // Each field is checked against its own processor's folder (pipeline.conllu has all eight, but only corpus.txt): word lines and
        // ner= against ner/ (tokenize..depparse,ner), trees against the default pipeline's files, and
        // sentiment against sentiment/. NER, constituency and sentiment share the tagger's CharlmCache.
        var nlp = pipelines.Get(managed: managed, processors: "tokenize,mwt,pos,lemma,depparse,ner,sentiment,constituency", lang: lang);
        var golden = Repo.GoldenFor(lang);
        foreach (var (name, conllu) in new[] { ("corpus", "pipeline"), ("validation_news", "validation_news"), ("validation_contractions", "validation_contractions") })
        {
            var doc = nlp.Process(File.ReadAllText(Path.Combine(golden, name + ".txt")));
            var written = Conllu.Write(doc);

            var withoutComments = string.Join('\n', written.Split('\n').Where(l => !l.StartsWith("# constituency = ") && !l.StartsWith("# sentiment = ")));
            Assert.Equal(File.ReadAllText(Path.Combine(golden, "ner", name + ".conllu")), withoutComments);

            var trees = Conllu.Read(File.ReadAllText(Path.Combine(golden, conllu + ".conllu"))).Sentences.Select(s => s.Constituency!.ToString());
            Assert.Equal(trees, doc.Sentences.Select(s => s.Constituency!.ToString()));

            var sentiment = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(golden, "sentiment", name + ".json")))!["sentences"]!.AsArray();
            Assert.Equal(sentiment.Select(s => s!["sentiment"]!.GetValue<int>()), doc.Sentences.Select(s => s.Sentiment!.Value));

            var entities = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(golden, "ner", name + ".json")))!.AsArray()
                .SelectMany(s => s!.AsArray())
                .Select(e => (e!["text"]!.GetValue<string>(), e["type"]!.GetValue<string>(), e["start_char"]!.GetValue<int>()));
            Assert.Equal(entities, doc.Sentences.SelectMany(s => s.Entities).Select(e => (e.Text, e.Type, e.StartChar!.Value)));

            // Comments in Stanza's order: sentiment last.
            Assert.Contains("\n# sentiment = ", written.Split("\n\n")[0]);
            Assert.True(written.IndexOf("# constituency = ") < written.IndexOf("# sentiment = "));
        }
    }
}
