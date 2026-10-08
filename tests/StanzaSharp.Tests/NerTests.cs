using Xunit.Abstractions;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using StanzaSharp.Ner;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Tests;

public class NerTests(ITestOutputHelper output)
{
    private static readonly string Golden = Path.Combine(Repo.Golden, "ner");

    [Fact]
    public void Viterbi_BreaksTiesTowardTheLowestTagLikeNumpy()
    {
        // Paths 0,0 and 1,1 both score 1; np.argmax picks the first maximum at every step.
        Assert.Equal([0, 0], NerTagger.Viterbi([1, 0, 0, 1], [0, -5, -5, 0], 2));
        // 1,0 has the best emissions (3), but the transition 1 -> 0 costs 100, so 0,0 (2) wins.
        Assert.Equal([0, 0], NerTagger.Viterbi([0, 1, 2, 0], [0, 0, -100, 0], 2));
        Assert.Equal([1, 0], NerTagger.Viterbi([0, 1, 2, 0], [0, 0, 0, 0], 2));
    }

    [Fact]
    public void Viterbi_FindsTheBestPathByBruteForce()
    {
        var random = new Random(7);
        for (int trial = 0; trial < 50; trial++)
        {
            int n = random.Next(1, 6), k = random.Next(2, 5);
            var scores = Enumerable.Range(0, n * k).Select(_ => (float)random.NextDouble() * 4 - 2).ToArray();
            var trans = Enumerable.Range(0, k * k).Select(_ => (float)random.NextDouble() * 4 - 2).ToArray();
            int[] best = [];
            double bestScore = double.NegativeInfinity;
            for (int code = 0; code < Math.Pow(k, n); code++)
            {
                var path = Enumerable.Range(0, n).Select(t => code / (int)Math.Pow(k, t) % k).ToArray();
                double s = Enumerable.Range(0, n).Sum(t => scores[t * k + path[t]] + (t > 0 ? trans[path[t - 1] * k + path[t]] : 0));
                if (s > bestScore)
                    (best, bestScore) = (path, s);
            }
            Assert.Equal(best, NerTagger.Viterbi(scores, trans, k));
        }
    }

    [Theory]
    // Expected values from stanza.models.ner.trainer.fix_singleton_tags.
    [InlineData("I-PER", "S-PER")]
    [InlineData("B-ORG I-ORG", "B-ORG E-ORG")]
    [InlineData("O I-LOC I-LOC O", "O B-LOC E-LOC O")]
    [InlineData("B-PER O E-GPE", "S-PER O S-GPE")]
    [InlineData("B-ORG I-PER E-PER", "S-ORG B-PER E-PER")]
    [InlineData("E-ORG I-ORG B-ORG", "S-ORG S-ORG S-ORG")]
    [InlineData("I-A I-B E-B B-A", "S-A B-B E-B S-A")]
    [InlineData("O B-X I-X E-X S-Y", "O B-X I-X E-X S-Y")]
    public void FixSingletonTags_MatchesStanza(string tags, string expected) =>
        Assert.Equal(expected.Split(' '), NerTagger.FixSingletonTags(tags.Split(' ')));

    [Fact]
    public void BuildEntities_DecodesLikeStanza()
    {
        // decode_from_bioes: an I-/E- without B- still makes an entity, typed by its last tag.
        var sentence = new Sentence();
        foreach (var tag in "I-A E-B O B-C B-D I-D S-E E-F".Split(' '))
            sentence.Tokens.Add(new Token { Text = "t" + sentence.Tokens.Count, Ner = tag, SpaceAfter = " " });
        Entity.Build(sentence, null);
        Assert.Equal(["t0 t1/B", "t3/C", "t4 t5/D", "t6/E", "t7/F"], sentence.Entities.Select(e => $"{e.Text}/{e.Type}"));
    }

    [Fact]
    public void Conllu_RoundTripsNerTags()
    {
        var conllu = File.ReadAllText(Path.Combine(Golden, "validation_contractions.conllu"));
        var doc = Conllu.Read(conllu);
        Assert.Equal(conllu, Conllu.Write(doc));
        Assert.NotEmpty(doc.Entities);
        Assert.Contains(doc.Sentences.SelectMany(s => s.Tokens), t => t.IsMultiWord && t.Ner != null);
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Emissions_MatchGoldenIntermediates(bool managed)
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        using var backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        string model = Repo.Model("ner/ontonotes-ww-multi_charlm");
        using var ner = managed
            ? NerTagger.LoadManaged(model, pretrain, ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion")), ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion")))
            : NerTagger.Load(model, pretrain, forward, backward);
        var golden = SafeTensorFile.Load(Path.Combine(Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e => (IReadOnlyList<string>)Strings(e!["tokens"])).ToList();

        // One padded batch; the golden scores were computed one sentence at a time.
        var tags = ner.Predict(sentences, out var emissions);
        for (int i = 0; i < sentences.Count; i++)
        {
            output.WriteLine($"s{i}: max |diff| {TokenizerTests.AssertClose(golden.Read<float>($"s{i}.emissions"), emissions[i], 1e-4f, $"s{i}"):E2}");
            Assert.Equal(Strings(index[i]!["ner"]), tags[i]);
        }
    }

    public static readonly TheoryData<string, bool, bool> Configurations = new()
    {
        { "tokenize,mwt,pos,lemma,depparse,ner", true, false },
        { "tokenize,mwt,pos,lemma,depparse,ner", false, false }, // no CharlmCache: NER runs its own charlms
        { "tokenize,ner", true, false }, // NER reads only the tokens, so it needs nothing else
        // Managed NER: reading the TorchSharp tagger's cached charlm outputs, computing its own, and alone.
        { "tokenize,mwt,pos,lemma,depparse,ner", true, true },
        { "tokenize,mwt,pos,lemma,depparse,ner", false, true },
        { "tokenize,ner", true, true },
    };

    [ModelTheory]
    [MemberData(nameof(Configurations))]
    public void Pipeline_ReproducesGoldenTagsAndEntities(string processors, bool cache, bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = processors, CharlmCache = new() { IsEnabled = cache }, Backend = Repo.Backend(managed) });
        bool full = processors.Contains("depparse");
        var failures = new List<string>();
        var files = Directory.GetFiles(Golden, "*.conllu").Order().ToList();
        Assert.Equal(13, files.Count);
        foreach (var conllu in files)
        {
            var name = Path.GetFileNameWithoutExtension(conllu);
            var txt = Path.Combine(Repo.Golden, name == "corpus" ? "corpus.txt" : name + ".txt");
            var doc = nlp.Process(File.ReadAllText(txt));
            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            bool offsets = name != "validation_nonbmp";

            var golden = File.ReadAllText(conllu);
            if (full)
            {
                var actual = Conllu.Write(doc);
                if (!offsets)
                    (actual, golden) = (StripOffsets(actual), StripOffsets(golden));
                var expected = golden.Split("\n\n");
                var got = actual.Split("\n\n");
                int i = 0;
                while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i]) i++;
                if (i < expected.Length || i < got.Length)
                    failures.Add($"{name}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");
            }
            else
            {
                var expected = Conllu.Read(golden).Sentences.SelectMany(s => s.Tokens.Select(t => $"{t.Text} {t.Ner}"));
                var got = doc.Sentences.SelectMany(s => s.Tokens.Select(t => $"{t.Text} {t.Ner}"));
                if (!expected.SequenceEqual(got))
                    failures.Add($"{name}: token NER tags differ");
            }

            var ents = JsonNode.Parse(File.ReadAllText(Path.ChangeExtension(conllu, ".json")))!.AsArray();
            var expectedEnts = ents.Select(s => s!.AsArray().Select(e => Describe(e!["text"]!.GetValue<string>(), e["type"]!.GetValue<string>(),
                e["start_char"]!.GetValue<int>(), e["end_char"]!.GetValue<int>(), offsets)).ToList()).ToList();
            var gotEnts = doc.Sentences.Select(s => s.Entities.Select(e => Describe(e.Text, e.Type, e.StartChar, e.EndChar, offsets)).ToList()).ToList();
            Assert.Equal(expectedEnts.Count, gotEnts.Count);
            for (int i = 0; i < expectedEnts.Count; i++)
                if (!expectedEnts[i].SequenceEqual(gotEnts[i]))
                    failures.Add($"{name}, sentence {i} entities:\n--- expected\n{string.Join("\n", expectedEnts[i])}\n--- actual\n{string.Join("\n", gotEnts[i])}");
        }
        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    private static string StripOffsets(string conllu) => PipelineTests.StripOffsets(conllu);

    private static string Describe(string text, string type, int? start, int? end, bool offsets) =>
        offsets ? $"{text} {type} {start}-{end}" : $"{text} {type}";

    private static List<string> Strings(JsonNode? array) => array!.AsArray().Select(x => x!.GetValue<string>()).ToList();
}
