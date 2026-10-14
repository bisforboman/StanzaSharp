using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using StanzaSharp.Tokenize;

namespace StanzaSharp.Tests;

/// <summary>Pretokenized input and bulk processing (tests/golden/pretokenized and tests/golden/bulk).</summary>
public class InputModeTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    private static readonly Dictionary<string, string> Suffix = new() { ["default"] = "", ["default_fast"] = ".fast" };

    [ModelTheory]
    [Trait("Smoke", "TorchSharp")] // both packages on TorchSharp in a ready pull request's light golden run (ci.yml)
    [InlineData("default", false, "en")]
    [InlineData("default_fast", false, "en")]
    [InlineData("default", true, "en")]
    [InlineData("default_fast", true, "en")]
    [German("default", false)]
    [German("default_fast", false)]
    [German("default", true)]
    [German("default_fast", true)]
    public void Pretokenized_MatchesGolden(string package, bool managed, string lang)
    {
        var nlp = pipelines.Get(package, managed, lang: lang);
        var dir = Path.Combine(Repo.GoldenFor(lang), "pretokenized");
        var cases = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "inputs.json")))!.AsObject();
        Assert.Equal(7, cases.Count);
        foreach (var (name, sentences) in cases)
        {
            var input = sentences!.AsArray().Select(s => s!.AsArray().Select(t => t!.GetValue<string>()).ToList()).ToList();
            var actual = Conllu.Write(nlp.Process(input));
            var golden = File.ReadAllText(Path.Combine(dir, name + Suffix[package] + ".conllu"));

            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (input.Any(s => s.Any(PipelineTests.HasNonBmp)))
                (actual, golden) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(golden));
            Assert.True(golden == actual, $"{name}:\n--- expected\n{golden}\n--- actual\n{actual}");
        }
    }

    [ModelTheory]
    [InlineData("default", false, "en")]
    [InlineData("default_fast", false, "en")]
    [InlineData("default", true, "en")]
    [German("default", false)]
    [German("default_fast", false)]
    [German("default", true)]
    public void Bulk_MatchesStanzasBulkProcess(string package, bool managed, string lang)
    {
        // bulk/<package>.json stores each document either whole or as its output alone plus what bulk changes.
        var docs = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "bulk", package + ".json")))!["documents"]!.AsArray();
        var texts = docs.Select(d => d!["text"]?.GetValue<string>() ?? File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), d["file"]!.GetValue<string>()))).ToList();
        var nlp = pipelines.Get(package, managed, lang: lang);
        var results = nlp.Process(texts);
        Assert.Equal(texts.Count, results.Count);

        int changed = 0;
        var failures = new List<string>();
        for (int i = 0; i < docs.Count; i++)
        {
            var d = docs[i]!;
            Assert.Equal(texts[i], results[i].Text);
            var actual = Conllu.Write(results[i]);
            string expected;
            if (d["conllu"] is { } whole)
            {
                // Stanza's "{:C}": Conllu.Write adds a final newline, except to an empty document.
                expected = whole.GetValue<string>();
                if (expected.Length > 0)
                    expected += "\n";
            }
            else
            {
                int offset = d["sent_id_offset"]!.GetValue<int>();
                var sentences = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), d["alone"]!.GetValue<string>())).Split("\n\n");
                for (int j = 0; j < sentences.Length; j++)
                    sentences[j] = new Regex(@"# sent_id = \d+\n").Replace(sentences[j], $"# sent_id = {j + offset}\n", 1);
                foreach (var (j, sentence) in d["sentences"]!.AsObject())
                {
                    sentences[int.Parse(j)] = sentence!.GetValue<string>();
                    changed++;
                }

                expected = string.Join("\n\n", sentences);
            }

            if (PipelineTests.HasNonBmp(texts[i]))
                (actual, expected) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(expected));
            var e = expected.Split("\n\n");
            var g = actual.Split("\n\n");
            int k = 0;
            while (k < Math.Min(e.Length, g.Length) && e[k] == g[k]) k++;
            if (k < e.Length || k < g.Length)
                failures.Add($"document {i}, sentence {k}:\n--- expected\n{e.ElementAtOrDefault(k)}\n--- actual\n{g.ElementAtOrDefault(k)}");
        }

        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
        Assert.True(changed > 0, "bulk is expected to change some sentences, as it does in Stanza");
    }

    [ModelFact]
    [Trait("Backend", "Managed")]
    public void Bulk_EqualsProcessingAloneExceptBatchedProcessors()
    {
        // Without sentiment and depparse, which batch sentences across documents, bulk output equals one call per
        // text, apart from the sentence ids, which continue across the documents.
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,pos,lemma,constituency,ner" });
        string[] texts = ["  I don't know.  Do you?\n", "", "Barack Obama was born in Hawaii.\n\nHe's 😀 happy!", "Ok"];
        var bulk = nlp.Process(texts);
        int offset = 0;
        for (int i = 0; i < texts.Length; i++)
        {
            var alone = nlp.Process(texts[i]);
            foreach (var s in alone.Sentences)
                s.SentId = (int.Parse(s.SentId!) + offset).ToString();
            offset += alone.Sentences.Count;
            Assert.Equal(Conllu.Write(alone), Conllu.Write(bulk[i]));
            Assert.Equal(alone.Entities.Select(e => (e.Text, e.StartChar)), bulk[i].Entities.Select(e => (e.Text, e.StartChar)));
        }
    }

    [ModelFact]
    [Trait("Backend", "Managed")]
    public void InputModes_ValidateArguments()
    {
        var nlp = pipelines.Get(processors: "tokenize,mwt");
        Assert.Empty(nlp.Process(Array.Empty<string>()));
        Assert.Empty(nlp.Process(Array.Empty<string[]>()).Sentences);
        Assert.Throws<ArgumentNullException>(() => nlp.Process((string)null!));
        Assert.Throws<ArgumentNullException>(() => nlp.Process((IEnumerable<string>)null!));
        Assert.Throws<ArgumentNullException>(() => nlp.Process(["a", null!]));
        Assert.Throws<ArgumentNullException>(() => nlp.Process((IEnumerable<IEnumerable<string>>)null!));
        Assert.Throws<ArgumentNullException>(() => nlp.Process([["a"], null!]));
        Assert.Throws<ArgumentException>(() => nlp.Process([["a"], []]));
        Assert.Throws<ArgumentException>(() => nlp.Process([["a", ""]]));
        Assert.Throws<ArgumentException>(() => nlp.Process([["a", " "]]));
        Assert.Throws<ArgumentException>(() => nlp.Process([["a", null!]]));

        // Collection expressions pick the intended overload: strings are texts, lists of strings are sentences.
        Assert.Equal(2, nlp.Process(["One.", "Two."]).Count);
        Assert.Equal("Hello world . Bye", nlp.Process([["Hello", "world", "."], ["Bye"]]).Text);
        Assert.Equal(["do", "n't"], nlp.Process(["don't"])[0].Sentences[0].Words.Select(w => w.Text));
        Assert.Equal(["don't"], nlp.Process([["don't"]]).Sentences[0].Words.Select(w => w.Text)); // no MWT expansion
    }

    [ModelTheory]
    [InlineData("default", false, "en")]
    [InlineData("default_fast", false, "en")]
    [InlineData("default", true, "en")]
    [InlineData("default_fast", true, "en")]
    [German("default", false)]
    [German("default_fast", false)]
    [German("default", true)]
    [German("default_fast", true)]
    public void ProcessDocument_MatchesStanzasProcessConllu(string package, bool managed, string lang)
    {
        // process_document/input.conllu: tokenize,mwt's output for corpus.txt with every annotation wrong. Every
        // processor but tokenize and mwt overwrites its own; default_fast has no parser, so the input's tree stays.
        var dir = Path.Combine(Repo.GoldenFor(lang), "process_document");
        var input = Conllu.Read(File.ReadAllText(Path.Combine(dir, "input.conllu")));
        var tokens = Json.Write(input);
        var doc = pipelines.Get(package, managed, lang: lang).Process(input);
        Assert.Same(input, doc);
        var golden = File.ReadAllText(Path.Combine(dir, "corpus" + Suffix[package] + ".conllu"));
        var actual = Conllu.Write(doc);
        Assert.True(golden == actual, $"--- expected\n{golden}\n--- actual\n{actual}");
        Assert.Contains(doc.Entities, e => e.Type != "WRONG");
        Assert.DoesNotContain(doc.Entities, e => e.Type == "WRONG");
        Assert.NotEqual(tokens, Json.Write(doc));
    }

    [ModelFact]
    [Trait("Backend", "Managed")]
    public void ProcessDocument_SkipsTokenizeAndMwt()
    {
        // A document's tokens stay as they are: "don't" read as one word is not expanded, "Hello world" not split.
        var nlp = pipelines.Get(processors: "tokenize,mwt,pos");
        var doc = Conllu.Read("1\tHello world\t_\t_\t_\t_\t_\t_\t_\t_\n2\tdon't\t_\t_\t_\t_\t_\t_\t_\tSpaceAfter=No\n3\t!\t_\t_\t_\t_\t_\t_\t_\t_\n");
        nlp.Process(doc);
        Assert.Equal(["Hello world", "don't", "!"], doc.Sentences[0].Words.Select(w => w.Text));
        Assert.All(doc.Sentences[0].Words, w => Assert.NotNull(w.Upos));
        Assert.Null(doc.Text);
        Assert.Null(doc.Sentences[0].Words.First().Lemma);

        Assert.Empty(nlp.Process(new Document()).Sentences);
        Assert.Throws<ArgumentNullException>(() => nlp.Process((Document)null!));
        Assert.Throws<OperationCanceledException>(() => nlp.Process(doc, new CancellationToken(canceled: true)));
    }

    [Fact]
    [Trait("Backend", "Managed")]
    public void Pretokenized_JoinsTokensWithSpacesLikeStanza()
    {
        var doc = Tokenizer.Pretokenized([["Hello", "world", "."], ["New York", "😀"]]);
        Assert.Equal("Hello world . New York 😀", doc.Text);
        Assert.Equal(["Hello world .", "New York 😀"], doc.Sentences.Select(s => s.Text));
        Assert.Equal(["0", "1"], doc.Sentences.Select(s => s.SentId));
        var tokens = doc.Sentences.SelectMany(s => s.Tokens).ToList();
        Assert.Equal([0, 6, 12, 14, 23], tokens.Select(t => t.StartChar!.Value));
        Assert.Equal([5, 11, 13, 22, 25], tokens.Select(t => t.EndChar!.Value)); // UTF-16: the emoji is 2 units
        Assert.Equal([" ", " ", " ", " ", ""], tokens.Select(t => t.SpaceAfter));
        Assert.All(tokens, t => Assert.Equal(t.Text, Assert.Single(t.Words).Text));
    }

    [Fact]
    [Trait("Backend", "Managed")]
    public void Split_GivesEachTextItsSentencesAndOffsets()
    {
        // What the tokenizer makes of "A b.\n\n\n\n  \n\n C." (texts "A b.", "", "  ", " C."): two sentences.
        string[] texts = ["A b.", "", "  ", " C."];
        var combined = Tokenizer.Pretokenized([["A", "b."], ["C."]]);
        combined.Text = string.Join("\n\n", texts);
        var c = combined.Sentences[1].Tokens[0];
        (c.StartChar, c.EndChar) = (13, 15);
        (c.Words[0].StartChar, c.Words[0].EndChar) = (13, 15);

        var docs = Tokenizer.Split(combined, texts);
        Assert.Equal(texts, docs.Select(d => d.Text));
        Assert.Equal([1, 0, 0, 1], docs.Select(d => d.Sentences.Count));
        var b = docs[0].Sentences[0].Tokens[1];
        Assert.Equal((2, 4, ""), (b.StartChar!.Value, b.EndChar!.Value, b.SpaceAfter));
        Assert.Equal((1, 3, 1, 3), (c.StartChar!.Value, c.EndChar!.Value, c.Words[0].StartChar!.Value, c.Words[0].EndChar!.Value));
        Assert.Equal((" ", ""), (c.SpacesBefore, c.SpaceAfter));
        Assert.Equal("1", docs[3].Sentences[0].SentId); // ids continue across documents, as in Stanza
    }
}
