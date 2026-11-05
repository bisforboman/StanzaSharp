using System.Text.Json.Nodes;
using StanzaSharp.Coref;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// The English-only coref models with roberta-large (#210) against Stanza: CorefModel.run's steps on corpus.txt and the
/// pipeline's chains on corpus.txt and every validation text (tests/golden/coref/models/&lt;model&gt;/, make_golden.py
/// <c>--coref-models-only</c>). Their 1.8 GB of files only nightly.yml's coref-models job downloads (owner's decision,
/// 2026-11-05), so ci.yml's shards leave the class out (<c>SHARD_NIGHTLY</c>).
/// </summary>
public class RobertaCorefTests(ITestOutputHelper output)
{
    internal const string GumNoSpeakers = "gum-nospeakers_roberta-large-lora", GumSpeakers = "gum-speakers_roberta-large-lora",
        OntonotesSingletons = "ontonotes-singletons_roberta-large-lora";

    internal static string ModelPath(string model) => Path.Combine(Repo.StanzaModels, "coref", model);

    private static string Folder(string model) => Path.Combine(Repo.Golden, "coref", "models", model);

    internal static TransformerEncoderTests.Owned Embedder(bool managed) => new(TransformerEmbedder.Load(RobertaCorefTheoryAttribute.Directory,
        RobertaCorefTheoryAttribute.RobertaLarge, managed ? (config, ckpt) => ManagedTransformer.Load(config, ckpt) : (config, ckpt) => new TransformerNet(config, ckpt)));

    // Stanza's words of corpus.txt are the default model's (coref/corpus.conllu: the same tagger).
    [RobertaCorefTheory]
    [InlineData(GumNoSpeakers, true)]
    [InlineData(GumSpeakers, true)]
    [InlineData(OntonotesSingletons, true)]
    [InlineData(GumNoSpeakers, false)]
    [InlineData(GumSpeakers, false)]
    [InlineData(OntonotesSingletons, false)]
    public void Run_MatchesStanzasSteps(string model, bool managed)
    {
        using var transformer = Embedder(managed);
        using var resolver = managed ? CorefResolver.LoadManaged(ModelPath(model), transformer.Value) : CorefResolver.Load(ModelPath(model), transformer.Value);
        TransformerCorefTests.AssertSteps(Folder(model), Path.Combine(Repo.Golden, "coref", "corpus.conllu"), "corpus", resolver, output);
    }

    /// <summary>
    /// Reviewed near-ties (owner's sign-off, 2026-11-05): a (model, text, backend) whose chains through the listed sentences
    /// may differ from Stanza's; every other chain of the text must still be Stanza's. roberta-large gives the token "."
    /// (TOKENIZER_MAPS) almost the same vector everywhere (to about 1e-7), so which "." words reach a word's top 50 candidates,
    /// and in what order, is float32 noise, and start_map reads the candidates in that order: a start score moves by up to
    /// about 0.1 between backends. Measured in Stanza's scores: gum-nospeakers on validation_tech, sentence 46's "int" (start
    /// vs dummy 0.0299; TorchSharp +0.0796, managed −0.0092 against Stanza's −0.0299); gum-speakers on
    /// validation_contractions, sentences 118 and 120 ("IT": start 0.205; "He": antecedent 0.125). docs/decisions/scope.md.
    /// </summary>
    internal static readonly (string Model, string Text, bool Managed, int[] Sentences)[] KnownNearTies =
    [
        (GumNoSpeakers, "validation_tech", false, [46]),
        (GumSpeakers, "validation_contractions", false, [118, 120]),
    ];

    // tokenize,mwt,pos,coref with Models[coref] = model on the default package (make_golden.py's COREF_PROCESSORS): the
    // chains of corpus.txt and every validation text, but KnownNearTies' chains.
    [RobertaCorefTheory]
    [InlineData(GumNoSpeakers, true)]
    [InlineData(GumSpeakers, true)]
    [InlineData(OntonotesSingletons, true)]
    [InlineData(GumNoSpeakers, false)]
    [InlineData(GumSpeakers, false)]
    [InlineData(OntonotesSingletons, false)]
    public void Pipeline_ReproducesStanzasChains(string model, bool managed)
    {
        using var nlp = Pipeline.Load(Repo.StanzaModels, new PipelineOptions
        {
            Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos},{Processor.Coref}",
            Models = new Dictionary<string, string> { [Processor.Coref] = model },
            Backend = Repo.PipelineBackendFor(managed),
        });
        var failures = new List<string>();
        var texts = new[] { Path.Combine(Repo.Golden, "corpus.txt") }.Concat(Directory.GetFiles(Repo.Golden, "validation*.txt").Order());
        foreach (var txt in texts)
        {
            var name = Path.GetFileNameWithoutExtension(txt);
            var doc = nlp.Process(File.ReadAllText(txt));
            var expected = JsonNode.Parse(File.ReadAllText(Path.Combine(Folder(model), name + ".chains.json")))!.AsArray()
                .Select(c => (Mentions: c!["mentions"]!.AsArray().Select(CorefDocumentTests.Mention).ToList(),
                    Text: $"{c["index"]} {c["representative_text"]} {c["representative_index"]}"))
                .ToList();
            var actual = doc.Coref.Select(c => (Mentions: c.Mentions.ToList(), Text: $"{c.Index} {c.RepresentativeText} {c.RepresentativeIndex}")).ToList();
            var tie = KnownNearTies.FirstOrDefault(t => t.Model == model && t.Text == name && t.Managed == managed).Sentences;

            // Without a near-tie, the chains as they are; with one, those through none of its sentences, unnumbered.
            List<string> Lines(IEnumerable<(List<CorefMention> Mentions, string Text)> chains) => tie == null
                ? [.. chains.Select(c => $"{c.Text}: {string.Join(" ", c.Mentions)}")]
                : [.. chains.Where(c => !c.Mentions.Any(m => tie.Contains(m.Sentence)))
                    .Select(c => $"{c.Text[(c.Text.IndexOf(' ') + 1)..]}: {string.Join(" ", c.Mentions)}")];
            var e = Lines(expected);
            var a = Lines(actual);
            int i = 0;
            while (i < Math.Min(e.Count, a.Count) && e[i] == a[i])
                i++;
            bool same = i == e.Count && i == a.Count;
            if (!same)
                failures.Add($"{name}, chain {i}:\n--- expected\n{e.ElementAtOrDefault(i)}\n--- actual\n{a.ElementAtOrDefault(i)}");
            string near = tie == null ? "" : $" without the chains through sentences {string.Join(", ", tie)} (a known near-tie)";
            output.WriteLine($"{name}: {expected.Count} chains, {(same ? "identical" : "differ")}{near}");
        }

        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    // TOKENIZER_MAPS: roberta-large's coref models give ".", ",", "!", "?", ":", ";" and "'s" their own token (no Ġ prefix),
    // which the tokenizer alone would not; the other words keep the tokenizer's subwords.
    [RobertaCorefTheory]
    [InlineData(true)]
    public void Run_MapsPunctuationToTheirOwnTokens(bool managed)
    {
        using var transformer = Embedder(managed);
        using var resolver = managed ? CorefResolver.LoadManaged(ModelPath(GumNoSpeakers), transformer.Value) : CorefResolver.Load(ModelPath(GumNoSpeakers), transformer.Value);
        var bpe = (ByteLevelBpe)transformer.Value.Tokenizer;
        var words = new[] { "John", "'s", "dog", "barked", "." }.Select(t => new Word { Text = t }).ToList();
        var result = resolver.Run(words, [0, 0, 0, 0, 0]);
        int[] expected = [bpe.TokenId("<s>")!.Value, .. bpe.Encode(["John"]).Ids[1..^1], bpe.TokenId("'s")!.Value, .. bpe.Encode(["dog", "barked"]).Ids[1..^1],
            bpe.TokenId(".")!.Value, bpe.TokenId("</s>")!.Value];
        Assert.Equal(expected, result.Windows);
        Assert.NotEqual(bpe.TokenId(".")!.Value, bpe.Encode(["."]).Ids[1]); // the tokenizer alone: "Ġ."
    }
}

/// <summary>Skips unless roberta-large's weights and the three coref models are next to <see cref="Repo.StanzaModels"/>.</summary>
public sealed class RobertaCorefTheoryAttribute : TheoryAttribute
{
    public const string RobertaLarge = "roberta-large";

    public static string Directory => ModelDownloader.TransformerDirectory(Repo.StanzaModels, RobertaLarge);

    public RobertaCorefTheoryAttribute()
    {
        var missing = new[] { RobertaCorefTests.GumNoSpeakers, RobertaCorefTests.GumSpeakers, RobertaCorefTests.OntonotesSingletons }
            .Select(m => RobertaCorefTests.ModelPath(m) + ".pt")
            .Concat(new[] { "config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin" }.Select(f => Path.Combine(Directory, f)))
            .FirstOrDefault(f => !File.Exists(f));
        Skip = missing == null ? null : $"{missing} not found; run tools/ci-models.sh coref-models";
    }
}
