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

    // tokenize,mwt,pos,coref with Models[coref] = model on the default package (make_golden.py's COREF_PROCESSORS): the
    // chains of corpus.txt and every validation text.
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
                .Select(c => $"{c!["index"]} {c["representative_text"]} {c["representative_index"]}: {string.Join(" ", c["mentions"]!.AsArray().Select(CorefDocumentTests.Mention))}")
                .ToList();
            var actual = doc.Coref.Select(c => $"{c.Index} {c.RepresentativeText} {c.RepresentativeIndex}: {string.Join(" ", c.Mentions)}").ToList();
            int i = 0;
            while (i < Math.Min(expected.Count, actual.Count) && expected[i] == actual[i])
                i++;
            bool same = i == expected.Count && i == actual.Count;
            if (!same)
                failures.Add($"{name}, chain {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{actual.ElementAtOrDefault(i)}");
            output.WriteLine($"{name}: {expected.Count} chains, {(same ? "identical" : "differ")}");
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
