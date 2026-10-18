using StanzaSharp.Nn;

namespace StanzaSharp.Tests;

/// <summary>
/// default_accurate's frozen-transformer models as <see cref="PipelineOptions.Models"/> (docs/transformer-plan.md, PR 2):
/// pos, depparse, ner, constituency and sentiment (LoRA) <c>_electra-large</c> and the <c>default_accurate</c> package against
/// their tests/golden/models/ rows, byte for byte on both backends. Named Transformer* so that only golden shard 1, which downloads the transformer, runs it in CI.
/// </summary>
public class TransformerModelTests
{
    private const string Electra = "pos+depparse+ner=electra-large";
    private const string Constituency = "constituency=ptb3-revised_electra-large";
    private const string Sentiment = "sentiment=sstplus_electra-large";
    private const string AllFive = "electra-large";

    [ModelTheory]
    [ModelChoice("en", Electra, false)]
    [ModelChoice("en", Electra, true)]
    [ModelChoice("en", Constituency, false)]
    [ModelChoice("en", Constituency, true)]
    [ModelChoice("en", Sentiment, false)]
    [ModelChoice("en", Sentiment, true)]
    [ModelChoice("en", AllFive, false)]
    [ModelChoice("en", AllFive, true)]
    [ModelChoice("en", "default_accurate", false)]
    [ModelChoice("en", "default_accurate", true)]
    public void Process_MatchesGolden(string name, bool managed, string lang) => ModelChoiceTests.AssertGolden(name, managed, lang, cache: true);

    [Fact]
    public void SstplusElectra_DownloadsButDoesNotLoadTheCharlms()
    {
        var models = new Dictionary<string, string> { ["sentiment"] = "sstplus_electra-large" };
        Assert.Equal(["pretrain/conll17", "forward_charlm/1billion", "backward_charlm/1billion"], Pipeline.SharedModels(models, "en"));
        Assert.Equal(["pretrain/conll17"], Pipeline.SharedModels(models, "en", readOnly: true));
        Assert.Contains("forward_charlm/1billion.pt", ModelDownloader.FilesFor("tokenize,sentiment", "default", "options", "en",
            new Dictionary<string, string> { ["sentiment"] = "sstplus_electra-large" }).Select(f => f.Path));
    }

    [TransformerFact("config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin")]
    public void VerifyChecksums_ChecksTheTransformersSha256()
    {
        var models = new Dictionary<string, string> { ["sentiment"] = "sstplus_electra-large" };
        ModelDownloader.Verify(Repo.StanzaModels, models, "en");

        // The same files, linked into a folder whose config.json has one byte changed.
        var dir = Directory.CreateTempSubdirectory("stanzasharp-sha256-").FullName;
        try
        {
            foreach (var path in Pipeline.ModelFiles(models, "en", readOnly: true).Select(p => p + ".pt"))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(dir, path))!);
                File.CreateSymbolicLink(Path.Combine(dir, path), Path.Combine(Repo.StanzaModels, path));
            }

            var transformer = ModelDownloader.TransformerDirectory(dir, TransformerFactAttribute.Electra);
            Directory.CreateDirectory(transformer);
            foreach (var (file, _) in ModelDownloader.Transformers[TransformerFactAttribute.Electra].Files)
            {
                if (file != "config.json")
                    File.CreateSymbolicLink(Path.Combine(transformer, file), Path.Combine(TransformerFactAttribute.Directory, file));
            }

            File.WriteAllText(Path.Combine(transformer, "config.json"), File.ReadAllText(Path.Combine(TransformerFactAttribute.Directory, "config.json")) + " ");
            var e = Assert.Throws<InvalidDataException>(() => ModelDownloader.Verify(dir, models, "en"));
            Assert.Contains("config.json", e.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ElectraModels_ReadTheTransformerAndTheCharlms()
    {
        var chosen = new Dictionary<string, string> { ["pos"] = "combined_electra-large", ["depparse"] = "combined_electra-large", ["ner"] = "ontonotes-ww-multi_electra-large" };
        var models = Pipeline.SelectModels("default", "pos,ner,depparse", addRequired: true, "options", "en", chosen);
        Assert.Equal([TransformerFactAttribute.Electra], Pipeline.TransformersOf(models, "en"));
        Assert.Equal(["pretrain/conll17", "forward_charlm/1billion", "backward_charlm/1billion"], Pipeline.SharedModels(models, "en"));

        // The transformer comes from Hugging Face, so it isn't among the Stanza files (or CI's cache keys, `download --list`).
        var files = ModelDownloader.FilesFor("pos,ner,depparse", "default", "options", "en", chosen).Select(f => f.Path).ToList();
        Assert.Contains("pos/combined_electra-large.pt", files);
        Assert.DoesNotContain(files, f => f.StartsWith("transformer/", StringComparison.Ordinal));
        Assert.Empty(Pipeline.TransformersOf(Pipeline.SelectModels("default", null, addRequired: true, "options"), "en"));
    }

    [Fact]
    public void DefaultAccurate_ReadsTheTransformerWithAllEightProcessors()
    {
        var models = Pipeline.SelectModels("default_accurate", null, addRequired: true, "options");
        Assert.Equal(Pipeline.AllProcessors.Split(',').Order(), models.Keys.Order());
        Assert.Equal([TransformerFactAttribute.Electra], Pipeline.TransformersOf(models, "en"));
        var files = ModelDownloader.FilesFor(null, "default_accurate").Select(f => f.Path).ToList();
        Assert.Contains("sentiment/sstplus_electra-large.pt", files);
        Assert.Contains("tokenize/combined_charlm.pt", files);
        Assert.DoesNotContain(files, f => f.StartsWith("transformer/", StringComparison.Ordinal));

        // German's reads another transformer (german-nlp-electra), not ported.
        var e = Assert.Throws<ArgumentException>(() => Pipeline.SelectModels("default_accurate", null, addRequired: true, "options", "de"));
        Assert.Contains("transformer", e.Message);
    }

    [Fact]
    public void Load_WithoutTheTransformer_SaysHowToDownloadIt()
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-transformer").FullName;
        try
        {
            var options = new PipelineOptions { Processors = "tokenize,ner", Models = new Dictionary<string, string> { ["ner"] = "ontonotes-ww-multi_electra-large" } };
            foreach (var file in ModelDownloader.FilesFor(options.Processors, models: options.Models))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(dir, file.Path))!);
                File.WriteAllBytes(Path.Combine(dir, file.Path), []);
            }

            var e = Assert.Throws<FileNotFoundException>(() => Pipeline.Load(dir, options));
            Assert.Contains("Transformer file not found", e.Message);
            Assert.Contains("--model ner=ontonotes-ww-multi_electra-large", e.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}

/// <summary><see cref="TransformerCache"/> and <see cref="TransformerEmbedder.Embed"/>'s use of it, on a fake encoder (no models).</summary>
public class TransformerCacheTests
{
    /// <summary>States that depend on the ids only: layer k, piece p, column c is id·1000 + k·100 + c, and counts its calls.</summary>
    private sealed class FakeNet : ITransformerNet
    {
        public TransformerConfig Config { get; } = new(Hidden: 2, Heads: 1, Layers: 4, Intermediate: 4, MaxPositions: 512, LayerNormEps: 1e-12);

        public int Sentences { get; private set; }

        public float[][] Encode(IReadOnlyList<int[]> sentences, int lastLayers, CancellationToken ct = default)
        {
            Sentences += sentences.Count;
            return [.. sentences.Select(ids =>
            {
                var states = new float[lastLayers * ids.Length * 2];
                for (int k = 0; k < lastLayers; k++)
                {
                    for (int p = 0; p < ids.Length; p++)
                    {
                        for (int c = 0; c < 2; c++)
                            states[((k * ids.Length) + p) * 2 + c] = ids[p] * 1000 + k * 100 + c;
                    }
                }

                return states;
            })];
        }

        public ITransformerNet WithLora(LoraWeights lora) => new FakeNet();

        public void Dispose()
        {
        }
    }

    private static readonly Lazy<WordPiece> Tokenizer = new(() =>
    {
        // A tokenizer.json with just these sentences' words (lowercased) and the special tokens.
        string[] vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "cat", "sat", ".", "hello", "world"];
        var added = string.Join(",", vocab[..4].Select((t, i) =>
            "{\"id\":" + i + ",\"content\":\"" + t + "\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":true}"));
        var ids = string.Join(",", vocab.Select((t, i) => "\"" + t + "\":" + i));
        var json = "{\"added_tokens\":[" + added + "],\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"[UNK]\",\"continuing_subword_prefix\":\"##\","
            + "\"max_input_chars_per_word\":100,\"vocab\":{" + ids + "}}}";
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, json);
            return WordPiece.Load(path);
        }
        finally
        {
            File.Delete(path);
        }
    });

    private static (TransformerEmbedder, FakeNet) Embedder()
    {
        var net = new FakeNet();
        return (new TransformerEmbedder(net, Tokenizer.Value, 512, TransformerFactAttribute.Electra), net);
    }

    private static readonly string[][] Sentences = [["The", "cat", "sat", "."], ["Hello", "world"]];

    [Fact]
    public void SecondReader_GetsTheFirstsStates_AndMixesThemItsOwnWay()
    {
        var (embedder, net) = Embedder();
        var cache = new TransformerCache(net);
        var legacy = embedder.Embed(Sentences, keepEndpoints: false, layerMix: null, cache);
        Assert.Equal(2, net.Sentences);
        var mixed = embedder.Embed(Sentences, keepEndpoints: true, layerMix: [0.5f, 0, 0, 0], cache);
        Assert.Equal(2, net.Sentences); // all from the cache

        // Without the cache: the same values.
        var (fresh, _) = Embedder();
        Assert.Equal(fresh.Embed(Sentences, false, null), legacy);
        Assert.Equal(fresh.Embed(Sentences, true, [0.5f, 0, 0, 0]), mixed);
    }

    [Fact]
    public void Cache_KeepsAtMostMaxPieces_AndOnlyFourLayers()
    {
        var (embedder, net) = Embedder();
        var cache = new TransformerCache(net, maxPieces: 7); // "The cat sat ." is 6 pieces with [CLS] and [SEP]; "Hello world" 4
        embedder.Embed(Sentences, false, null, cache);
        embedder.Embed(Sentences, false, null, cache);
        Assert.Equal(3, net.Sentences); // only the first sentence fit

        // A mix of other than 4 layers neither reads nor fills it.
        embedder.Embed(Sentences, false, [1f, 1f], cache);
        Assert.Equal(5, net.Sentences);
    }

    [Fact]
    public void Cache_OfAnotherNet_IsIgnored()
    {
        // Sentiment's LoRA encoder gives other states for the same ids: it must neither read nor fill the frozen one's cache.
        var (embedder, net) = Embedder();
        var cache = new TransformerCache(net);
        embedder.Embed(Sentences, false, null, cache);
        var lora = (FakeNet)net.WithLora(new LoraWeights(new Dictionary<string, (float[], float[], int)>(), 2));
        var other = new TransformerEmbedder(lora, embedder.Tokenizer, 512, TransformerFactAttribute.Electra);
        other.Embed(Sentences, false, null, cache);
        Assert.Equal(2, lora.Sentences);
    }
}
