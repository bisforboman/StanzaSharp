using StanzaSharp.Nn;

namespace StanzaSharp.Tests;

/// <summary>
/// default_accurate's frozen-transformer models as <see cref="PipelineOptions.Models"/> (docs/transformer-plan.md, PR 2):
/// pos, depparse, ner, constituency and sentiment (LoRA) <c>_electra-large</c>, the treebanks' pos and depparse and ontonotes' ner
/// <c>_electra-large</c> (#168), conll03's roberta-base ner (#172, also next to the other four: two transformers) and the
/// <c>default_accurate</c> package against their tests/golden/models/ rows, byte for byte on both backends; German's (german-nlp-group/electra-base-german-uncased:
/// pos and depparse frozen, constituency and sentiment with their LoRA adapters) against tests/golden/de/models/. Named
/// Transformer* so that only golden shard 4, which downloads the transformers, runs it in CI.
/// </summary>
public class TransformerModelTests
{
    private const string Electra = "pos+depparse+ner=electra-large";
    private const string Constituency = "constituency=ptb3-revised_electra-large";
    private const string TopDown = "constituency=ptb3_electra-large";
    private const string Sentiment = "sentiment=sstplus_electra-large";
    private const string AllFive = "electra-large";
    private const string Roberta = "ner=conll03_electra-large";
    private const string TwoTransformers = "electra-large+ner=conll03_electra-large";
    private const string GermanFrozen = "pos+depparse=german-nlp-electra";
    private const string GermanConstituency = "constituency=spmrl_german-nlp-electra";
    private const string GermanSentiment = "sentiment=sb10k_german-nlp-electra";

    [ModelTheory]
    [ModelChoice("en", Electra, false)]
    [ModelChoice("en", Electra, true)]
    [ModelChoice("en", Constituency, false)]
    [ModelChoice("en", Constituency, true)]
    [ModelChoice("en", TopDown, false)]
    [ModelChoice("en", TopDown, true)]
    [ModelChoice("en", Sentiment, false)]
    [ModelChoice("en", Sentiment, true)]
    [ModelChoice("en", AllFive, false)]
    [ModelChoice("en", AllFive, true)]
    [ModelChoice("en", "default_accurate", false)]
    [ModelChoice("en", "default_accurate", true)]
    [ModelChoice("en", "pos+depparse=ewt_electra-large", false)]
    [ModelChoice("en", "pos+depparse=ewt_electra-large", true)]
    [ModelChoice("en", "pos+depparse=gum_electra-large", false)]
    [ModelChoice("en", "pos+depparse=gum_electra-large", true)]
    [ModelChoice("en", "pos+depparse=lines_electra-large", false)]
    [ModelChoice("en", "pos+depparse=lines_electra-large", true)]
    [ModelChoice("en", "pos+depparse=partut_electra-large", false)]
    [ModelChoice("en", "pos+depparse=partut_electra-large", true)]
    [ModelChoice("en", "pos+depparse=atis_electra-large", false)]
    [ModelChoice("en", "pos+depparse=atis_electra-large", true)]
    [ModelChoice("en", "pos+depparse=childes_electra-large", false)]
    [ModelChoice("en", "pos+depparse=childes_electra-large", true)]
    [ModelChoice("en", "pos+depparse=eslspok_electra-large", false)]
    [ModelChoice("en", "pos+depparse=eslspok_electra-large", true)]
    [ModelChoice("en", "ner=ontonotes_electra-large", false)]
    [ModelChoice("en", "ner=ontonotes_electra-large", true)]
    [ModelChoice("en", Roberta, false)]
    [ModelChoice("en", Roberta, true)]
    [ModelChoice("en", TwoTransformers, false)]
    [ModelChoice("en", TwoTransformers, true)]
    [ModelChoice("de", GermanFrozen, false)]
    [ModelChoice("de", GermanFrozen, true)]
    [ModelChoice("de", GermanConstituency, false)]
    [ModelChoice("de", GermanConstituency, true)]
    [ModelChoice("de", GermanSentiment, false)]
    [ModelChoice("de", GermanSentiment, true)]
    [ModelChoice("de", "default_accurate", false)]
    [ModelChoice("de", "default_accurate", true)]
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

        // German's reads german-nlp-group's base ELECTRA under four processors (ner is default's germeval2014).
        var german = Pipeline.SelectModels("default_accurate", null, addRequired: true, "options", "de");
        Assert.Equal(Pipeline.AllProcessors.Split(',').Order(), german.Keys.Order());
        Assert.Equal([GermanTransformerFactAttribute.Electra], Pipeline.TransformersOf(german, "de"));
        Assert.Equal("germeval2014", german["ner"]);
        var germanFiles = ModelDownloader.FilesFor(null, "default_accurate", language: "de").Select(f => f.Path).ToList();
        Assert.Contains("constituency/spmrl_german-nlp-electra.pt", germanFiles);
        Assert.Contains("lemma/combined_charlm.pt", germanFiles);
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

        public int JointCalls { get; private set; }

        // A LoRA net's states are offset by this, so they differ from the frozen net's for the same ids.
        public float Offset { get; init; }

        public FakeNet? Base { get; init; }

        public bool EncodesJointly { get; init; } = true;

        public float[][][]? EncodeJoint(IReadOnlyList<(ITransformerNet Net, IReadOnlyList<int[]> Sentences)> groups, int lastLayers, CancellationToken ct = default)
        {
            var root = Base ?? this;
            if (groups.Any(g => g.Net is not FakeNet f || (f.Base ?? f) != root))
                return null;
            root.JointCalls++;
            return [.. groups.Select(g => ((FakeNet)g.Net).Encode(g.Sentences, lastLayers, ct))];
        }

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
                            states[((k * ids.Length) + p) * 2 + c] = ids[p] * 1000 + k * 100 + c + Offset;
                    }
                }

                return states;
            })];
        }

        public ITransformerNet WithLora(LoraWeights lora) => new FakeNet { Base = this, Offset = 0.5f };

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
        var json = "{\"added_tokens\":[" + added + "],\"normalizer\":{\"type\":\"BertNormalizer\",\"clean_text\":true,\"handle_chinese_chars\":true,"
            + "\"strip_accents\":null,\"lowercase\":true},\"pre_tokenizer\":{\"type\":\"BertPreTokenizer\"},\"model\":{\"type\":\"WordPiece\",\"unk_token\":\"[UNK]\",\"continuing_subword_prefix\":\"##\","
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

    private static TransformerEmbedder Lora(TransformerEmbedder embedder) =>
        new((FakeNet)embedder.Net.WithLora(new LoraWeights(new Dictionary<string, (float[], float[], int)>(), 2)), embedder.Tokenizer, 512,
            TransformerFactAttribute.Electra);

    [Fact]
    public void Prefill_EncodesEveryReaderInOneSweep_IntoItsOwnNetsCache()
    {
        var (embedder, net) = Embedder();
        var lora = Lora(embedder);
        var caches = new Dictionary<ITransformerNet, TransformerCache>();
        TransformerCache CacheOf(ITransformerNet n) => caches.TryGetValue(n, out var c) ? c : caches[n] = new TransformerCache(n);

        // pos and ner on the frozen net (the same sentences, so encoded once), sentiment on its LoRA net.
        Assert.True(TransformerEmbedder.Prefill(
            [new(embedder, null, Sentences), new(embedder, [1f, 0, 0, 0], Sentences), new(lora, null, [Sentences[1]])], CacheOf));
        Assert.Equal(1, net.JointCalls);
        Assert.Equal(2, net.Sentences);
        Assert.Equal(1, ((FakeNet)lora.Net).Sentences);

        // Each reader now finds its states, mixed as without the prefill.
        var (fresh, _) = Embedder();
        Assert.Equal(fresh.Embed(Sentences, false, null), embedder.Embed(Sentences, false, null, caches[net]));
        Assert.Equal(Lora(fresh).Embed([Sentences[1]], false, null), lora.Embed([Sentences[1]], false, null, caches[lora.Net]));
        Assert.Equal(2, net.Sentences);
        Assert.Equal(1, ((FakeNet)lora.Net).Sentences);

        // The frozen cache never holds the LoRA states and vice versa.
        Assert.Equal(2, caches.Count);
        foreach (var cache in caches.Values)
            cache.Dispose();
    }

    [Fact]
    public void Prefill_DoesNothing_ForOneReader_OtherMixes_OrANetThatCant()
    {
        var (embedder, net) = Embedder();
        TransformerCache Fail(ITransformerNet n) => throw new InvalidOperationException("no cache should be made");
        Assert.False(TransformerEmbedder.Prefill([new(embedder, null, Sentences)], Fail));
        Assert.False(TransformerEmbedder.Prefill([new(embedder, null, Sentences), new(embedder, [1f, 1f], Sentences)], Fail)); // a 2-layer mix isn't cached

        var torch = new TransformerEmbedder(new FakeNet { EncodesJointly = false }, embedder.Tokenizer, 512, TransformerFactAttribute.Electra);
        Assert.False(TransformerEmbedder.Prefill([new(torch, null, Sentences), new(torch, null, Sentences)], Fail));
        Assert.Equal(0, net.JointCalls);
        Assert.Equal(0, net.Sentences);
    }

    [Fact]
    public void Prefill_DoesNothing_PastOneEncoderCall()
    {
        var (embedder, net) = Embedder();

        // 200 sentences of distinct lengths (so distinct ids), over 20,000 pieces: more than one call's BatchPieces.
        List<IReadOnlyList<string>> many = [.. Enumerable.Range(1, 200).Select(n => (IReadOnlyList<string>)[.. Enumerable.Repeat("cat", n)])];
        TransformerCache Fail(ITransformerNet n) => throw new InvalidOperationException("no cache should be made");
        Assert.False(TransformerEmbedder.Prefill([new(embedder, null, many), new(embedder, null, many)], Fail));
        Assert.Equal(0, net.Sentences);
    }
}
