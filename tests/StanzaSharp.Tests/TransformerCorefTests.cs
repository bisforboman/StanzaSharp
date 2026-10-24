using System.Text.Json.Nodes;
using StanzaSharp.Coref;
using StanzaSharp.Nn;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// The coref processor (docs/coref-plan.md, PR c) against Stanza: CorefModel.run's steps on Stanza's own words
/// (tests/golden/coref/intermediates.*) and the whole pipeline's CoNLL-U (tests/golden/coref/*.conllu). Named
/// <c>Transformer*</c> so it runs in golden shard 4, which downloads XLM-R.
/// </summary>
public partial class TransformerCorefTests(ITestOutputHelper output)
{
    internal static string ModelPath => Path.Combine(Repo.StanzaModels, "coref", "udcoref_xlm-roberta-lora");

    private static readonly string Folder = Path.Combine(Repo.Golden, "coref");

    internal static CorefResolver Load(TransformerEmbedder transformer, bool managed) =>
        managed ? CorefResolver.LoadManaged(ModelPath, transformer) : CorefResolver.Load(ModelPath, transformer);

    // Stanza's run on <name>.txt, read back from its CoNLL-U: the same words and sentences the model saw.
    [CorefTheory]
    [InlineData("corpus", true)]
    [InlineData("validation_dialogue", true)]
    [InlineData("corpus", false)]
    [InlineData("validation_dialogue", false)]
    public void Run_MatchesStanzasSteps(string name, bool managed)
    {
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Folder, "intermediates.json")))!["texts"]![name]!;
        var tensors = SafeTensorFile.Load(Path.Combine(Folder, "intermediates.safetensors"));
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Folder, name + ".conllu")));
        var words = doc.Sentences.SelectMany(s => s.Words).ToList();
        var sentOf = doc.Sentences.SelectMany((s, i) => s.Words.Select(_ => i)).ToList();

        using var transformer = TransformerEncoderTests.XlmRobertaEmbedder(managed);
        using var resolver = Load(transformer.Value, managed);
        var result = resolver.Run(words, sentOf);

        Assert.Equal(golden["windows"]!.AsArray().SelectMany(w => w!.AsArray().Select(i => i!.GetValue<int>())), result.Windows);

        var words8 = tensors.Read<float>(name + ".words8");
        double wordDrift = words8.Select((v, i) => Math.Abs(v - result.WordsHead[i])).Max();
        output.WriteLine($"word vectors (first 8): max |diff| {wordDrift:g3}");

        // The top k: the same candidates; their order is nth_element's, which a near-tie between two rough scores (closer than
        // the float drift between torch and us) can rearrange. Pair scores are compared by candidate, start scores (start_map
        // over the candidates in that order) only where the order is torch's.
        var top = tensors.Read<int>(name + ".top_indices");
        var scores = tensors.Read<float>(name + ".coref_scores");
        int k = Math.Min(50, words.Count), rowsDiffering = 0;
        double scoreDrift = 0;
        for (int i = 0; i < words.Count; i++)
        {
            var theirs = top.AsSpan(i * k, k).ToArray();
            var ours = result.TopIndices.AsSpan(i * k, k).ToArray();
            Assert.Equal(theirs.Order(), ours.Order());
            bool sameOrder = theirs.SequenceEqual(ours);
            rowsDiffering += sameOrder ? 0 : 1;
            for (int c = 0; c < k; c++)
            {
                float expected = scores[(i * (k + 2)) + 2 + c], actual = result.Scores[(i * (k + 2)) + 2 + Array.IndexOf(ours, theirs[c])];
                if (!float.IsInfinity(expected))
                    scoreDrift = Math.Max(scoreDrift, Math.Abs(expected - actual));
            }

            if (sameOrder)
                scoreDrift = Math.Max(scoreDrift, Math.Abs(scores[i * (k + 2)] - result.Scores[i * (k + 2)]));
        }

        output.WriteLine($"top-k candidates: {rowsDiffering} of {words.Count} rows in another order than torch's (same candidates)");
        output.WriteLine($"coref scores: max |diff| {scoreDrift:g3}");

        // The decisions' margins in Stanza's scores: start vs the dummy (singletons), best vs second of [dummy, candidates]
        // (antecedents), against the drift above.
        double startMargin = double.MaxValue, antecedentMargin = double.MaxValue;
        for (int i = 0; i < words.Count; i++)
        {
            var row = scores.AsSpan(i * (k + 2), k + 2);
            startMargin = Math.Min(startMargin, Math.Abs(row[0] - row[1]));
            var best = row[1..].ToArray().Where(v => !float.IsInfinity(v)).OrderDescending().Take(2).ToArray();
            if (best.Length == 2)
                antecedentMargin = Math.Min(antecedentMargin, best[0] - best[1]);
        }

        output.WriteLine($"smallest margins: start vs dummy {startMargin:g3}, antecedent {antecedentMargin:g3}");

        Assert.Equal(golden["word_clusters"]!.AsArray().Select(c => c!.AsArray().Select(w => w!.GetValue<int>()).ToList()), result.WordClusters);
        Assert.Equal(golden["span_clusters"]!.AsArray().Select(c => c!.AsArray().Select(s => (s![0]!.GetValue<int>(), s[1]!.GetValue<int>())).ToList()), result.SpanClusters);
        var zeros = tensors.Read<float>(name + ".zero_scores");
        Assert.Equal(zeros.Length, result.ZeroScores.Length);
        output.WriteLine($"zero scores: max |diff| {zeros.Select((v, i) => Math.Abs(v - result.ZeroScores[i])).DefaultIfEmpty().Max():g3}");

        Assert.True(wordDrift < 1e-3, $"word vectors drift {wordDrift}");
        Assert.True(scoreDrift < 1e-3, $"coref scores drift {scoreDrift}");
    }
}

public partial class TransformerCorefTests
{
    // tokenize,mwt,pos,coref on the default package (make_golden.py's COREF_PROCESSORS) for corpus.txt and every validation
    // text, compared sentence by sentence (without offsets after non-BMP characters, as PipelineTests does). German reads
    // the same model, from models/stanza/de.
    [CorefTheory]
    [InlineData(true, "en")]
    [InlineData(false, "en")]
    [German(true)]
    [German(false)]
    public void Pipeline_ReproducesStanzasCoref(bool managed, string lang)
    {
        var models = Path.GetFullPath(Path.Combine(Repo.StanzaModels, "..", lang));
        if (!File.Exists(Path.Combine(models, "coref", "udcoref_xlm-roberta-lora.pt")))
            Assert.Fail($"{models}/coref not found; run stanzasharp download {models} --lang {lang} --processors tokenize,coref");
        using var nlp = Pipeline.Load(models, new PipelineOptions
        {
            Language = lang,
            Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos},{Processor.Coref}",
            Backend = Repo.PipelineBackendFor(managed),
        });
        var golden = Repo.GoldenFor(lang);
        var failures = new List<string>();
        var texts = new[] { Path.Combine(golden, "corpus.txt") }.Concat(Directory.GetFiles(golden, "validation*.txt").Order()).ToList();
        foreach (var txt in texts)
        {
            var name = Path.GetFileNameWithoutExtension(txt);
            var text = File.ReadAllText(txt);
            var actual = Conllu.Write(nlp.Process(text));
            var expectedText = File.ReadAllText(Path.Combine(golden, "coref", name + ".conllu"));
            if (PipelineTests.HasNonBmp(text))
                (actual, expectedText) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(expectedText));
            var expected = expectedText.Split("\n\n");
            var got = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(expected.Length, got.Length) && expected[i] == got[i])
                i++;
            if (i < expected.Length || i < got.Length)
                failures.Add($"{name}, sentence {i}:\n--- expected\n{expected.ElementAtOrDefault(i)}\n--- actual\n{got.ElementAtOrDefault(i)}");
            output.WriteLine($"{name}: {(i < expected.Length || i < got.Length ? "differs" : "identical")}");
        }

        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }
}

/// <summary>Skips unless the coref model and XLM-R's files are next to <see cref="Repo.StanzaModels"/>.</summary>
public sealed class CorefTheoryAttribute : TheoryAttribute
{
    public CorefTheoryAttribute()
    {
        var model = TransformerCorefTests.ModelPath + ".pt";
        var transformer = Path.Combine(XlmRobertaFactAttribute.Directory, "pytorch_model.bin");
        Skip = !File.Exists(model) ? $"{model} not found; run stanzasharp download --processors tokenize,coref"
            : !File.Exists(transformer) ? $"{transformer} not found; run StanzaSharp.Cli download-transformer models/stanza/en {XlmRobertaFactAttribute.XlmRoberta}"
            : null;
    }
}

public partial class TransformerCorefTests
{
    // The English model predicts no zero mentions on the golden texts, so coref/zeros/ is Stanza's run with make_golden.py's
    // COREF_ZERO_SHIFT added to the zeros predictor's output; here the same shift on our net's zero scores. Checks the empty
    // words (ids, numbering, CoNLL-U and JSON lines), the zero mentions' place in their chains and the representatives.
    private const float ZeroShift = 3.0f;

    [CorefTheory]
    [InlineData("validation_dialogue", true)]
    [InlineData("validation_news", true)]
    [InlineData("validation_dialogue", false)]
    [InlineData("validation_news", false)]
    public void Process_WithZeroMentions_MatchesStanza(string name, bool managed)
    {
        using var tagger = Pipeline.Load(Repo.StanzaModels, new PipelineOptions
        {
            Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos}",
            Backend = Repo.PipelineBackendFor(managed),
        });
        using var transformer = TransformerEncoderTests.XlmRobertaEmbedder(managed);
        using var resolver = new CorefResolver(Checkpoint.Load(ModelPath), transformer.Value,
            ckpt => new ShiftedZeros(managed ? new ManagedCorefNet(ckpt) : new CorefNet(ckpt), ZeroShift));
        var doc = tagger.Process(File.ReadAllText(Path.Combine(Repo.Golden, name + ".txt")));
        resolver.Process(doc);

        var folder = Path.Combine(Folder, "zeros");
        var expected = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, name + ".chains.json")))!.AsArray();
        Assert.Equal(expected.Select(c => (
            c!["index"]!.GetValue<int>(), c["representative_text"]!.GetValue<string>(), c["representative_index"]?.GetValue<int>(),
            string.Join(" ", c["mentions"]!.AsArray().Select(CorefDocumentTests.Mention)))),
            doc.Coref.Select(c => (c.Index, c.RepresentativeText, c.RepresentativeIndex, string.Join(" ", c.Mentions))));
        Assert.True(doc.Sentences.Sum(s => s.EmptyWords.Count) > 0, "no zero mentions: the shift no longer makes any");
        Assert.Equal(File.ReadAllText(Path.Combine(folder, name + ".conllu")), Conllu.Write(doc));
        Assert.Equal(File.ReadAllText(Path.Combine(folder, name + ".str.json")), Json.Write(doc));
    }

    private sealed class ShiftedZeros(ICorefNet net, float shift) : ICorefNet
    {
        public float[] Words(float[] subwords, int count, int[] starts, int[] ends) => net.Words(subwords, count, starts, ends);

        public void RoughScores(float[] words, int n, Action<int, int, float[], int> rows, CancellationToken ct) => net.RoughScores(words, n, rows, ct);

        public float[] Anaphoricity(float[] words, int n, int[] top, int[] buckets, int k, CancellationToken ct) =>
            net.Anaphoricity(words, n, top, buckets, k, ct);

        public (float[] Start, float[] End) SpanScores(float[] words, int n, int[] heads, int[] sentStart, int[] sentLength, int width, CancellationToken ct) =>
            net.SpanScores(words, n, heads, sentStart, sentLength, width, ct);

        public float[] Zeros(float[] words, int[] rows) => [.. net.Zeros(words, rows).Select(z => z + shift)];

        public void Dispose() => net.Dispose();
    }
}

public partial class TransformerCorefTests
{
    // ModelPaths for coref (#209): the model copied under a name no table knows, on the default package (whose optional coref
    // model it replaces) and on combined (which lists none: the path still reads XLM-R, as Models may choose coref there).
    // combined's tokenizer, mwt and tagger are the default package's, so both give Stanza's corpus.txt output.
    [CorefTheory]
    [InlineData("default", true)]
    [InlineData("combined", true)]
    [InlineData("default", false)]
    public void CopiedCorefModel_UnderAnotherName_GivesStanzasChains(string package, bool managed)
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-coref-path-").FullName;
        try
        {
            var copy = Path.Combine(dir, "my-coref");
            File.Copy(ModelPath + ".pt", copy + ".pt");
            var paths = new Dictionary<string, ModelPath> { [Processor.Coref] = copy };
            var processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos},{Processor.Coref}";
            Assert.DoesNotContain(ModelDownloader.FilesFor(processors, package, modelPaths: paths),
                f => f.Path.StartsWith("coref/", StringComparison.Ordinal));

            using var nlp = Pipeline.Load(Repo.StanzaModels, new PipelineOptions
            {
                Package = package,
                Processors = processors,
                ModelPaths = paths,
                Backend = Repo.PipelineBackendFor(managed),
            });
            var expected = File.ReadAllText(Path.Combine(Folder, "corpus.conllu"));
            Assert.Equal(expected, Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")))));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
