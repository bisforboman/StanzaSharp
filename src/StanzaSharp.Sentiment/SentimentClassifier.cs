using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Sentiment;

/// <summary>
/// Labels each sentence 0 (negative), 1 (neutral) or 2 (positive). Port of
/// stanza/models/classifiers/cnn_classifier.py <c>CNNClassifier</c> for the English <c>sstplus_charlm</c>
/// configuration:
/// - input per token: pretrain vector (a learned vector for unknown words) + delta embedding, summed,
///   then the forward and backward charlm
/// - a 2-layer biLSTM, then full-width convolutions and one 2d convolution, each max-pooled over time
/// - fully connected layers with ReLU, and an argmax over the classes
/// The network is an <see cref="ISentimentNet"/> (per <see cref="Backend"/>); vocab lookups, batching and the argmax are here.
/// </summary>
internal sealed class SentimentClassifier : IDisposable
{
    // sentiment_processor.py DEFAULT_BATCH_SIZE, counted in tokens.
    internal const int BatchSize = 5000;
    private const int UnkId = 1; // vocab.UNK_ID; padding is PAD_ID 0 in both vocabs

    private readonly Pretrain _pretrain;
    private readonly Dictionary<string, int> _extraVocab = [];
    private readonly int _maxWindow;
    private readonly ISentimentNet _net;
    private readonly TransformerEmbedder? _transformer; // its own, with the model's LoRA adapters (sstplus_electra-large)
    private readonly ITransformerNet? _loraNet;
    private readonly float[]? _layerMix;

    /// <param name="transformer">The pipeline's transformer, for a <c>_electra-large</c> classifier: its LoRA adapters go onto a
    /// copy that shares the transformer's weights (applied at run time, owner's decision 2026-10-17); else ignored.</param>
    internal SentimentClassifier(Checkpoint ckpt, Pretrain pretrain, Func<Filter[], ISentimentNet> net, TransformerEmbedder? transformer = null)
    {
        _pretrain = pretrain;
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        CheckSupported(config);
        UsesCharlm = config["has_charlm_forward"]?.GetValue<bool>() == true;
        bool lora = config["use_peft"]?.GetValue<bool>() == true;
        _layerMix = TransformerEmbedder.LayerMix(ckpt, config, p["model"]!, transformer, "sentiment", lora);
        if (config["bert_model"] != null)
        {
            _loraNet = lora
                ? transformer!.Net.WithLora(LoraWeights.Read(ckpt, p["bert_lora"] ?? throw new InvalidDataException("A LoRA classifier without bert_lora"), config))
                : null;
            _transformer = _loraNet == null ? transformer : new TransformerEmbedder(_loraNet, transformer!.Tokenizer, transformer.MaxLength, transformer.Name);
        }

        // { word: i for i, word in enumerate(extra_vocab) }: a repeated word keeps its last index.
        var extra = Checkpoint.Strings(p["extra_vocab"]);
        for (int i = 0; i < extra.Length; i++)
            _extraVocab[extra[i]] = i;
        if (ckpt.Shape(p["model"]!["extra_embedding.weight"])[1] != pretrain.Dim)
            throw new NotSupportedException("SUM needs the extra embedding to match the pretrain's dimension");

        var filters = config["filter_sizes"]!["$tuple"]!.AsArray().Select(size => size is JsonValue
            ? new Filter(size.GetValue<int>(), 0)
            : new Filter(size!["$tuple"]![0]!.GetValue<int>(), size["$tuple"]![1]!.GetValue<int>())).ToArray();
        _maxWindow = filters.Max(f => f.Width == 0 ? f.Height : f.Width);
        _net = net(filters);
    }

    /// <summary>A convolution's filter: <see cref="Height"/> tokens × the biLSTM's full width (Width 0), or Height × Width with stride (1, Width).</summary>
    internal readonly record struct Filter(int Height, int Width);

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static SentimentClassifier LoadManaged(string basePath, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward,
        TransformerEmbedder? transformer = null)
    {
        if (charlmForward is { IsForward: false } || charlmBackward is { IsForward: true })
            throw new ArgumentException("Pass the forward charlm first, then the backward one");
        var ckpt = Checkpoint.Load(basePath);
        var (charlm, transformerDim) = Inputs(ckpt, transformer);
        if (charlm && (charlmForward == null || charlmBackward == null))
            throw new ArgumentException($"{basePath} reads the charlms; pass both");
        return new SentimentClassifier(ckpt, pretrain, filters => new ManagedSentimentNet(ckpt, filters, pretrain, charlm ? charlmForward : null,
            charlm ? charlmBackward : null, transformerDim), transformer);
    }

    /// <summary>What the model reads besides its word vectors: the charlms (has_charlm_forward), and the transformer's columns (its hidden size, else 0).</summary>
    internal static (bool Charlm, int TransformerDim) Inputs(Checkpoint ckpt, TransformerEmbedder? transformer)
    {
        var config = ckpt.Root["params"]!["config"]!;
        return (config["has_charlm_forward"]?.GetValue<bool>() == true, config["bert_model"] == null ? 0 : transformer?.Net.Config.Hidden ?? 0);
    }

    /// <summary>Whether this classifier reads the charlms (and so the tagger's <see cref="CharlmCache"/>); sstplus_electra-large doesn't.</summary>
    internal bool UsesCharlm { get; }

    /// <summary>The sentences this classifier will send its transformer in <paramref name="doc"/>, for <see cref="TransformerEmbedder.Prefill"/> (null without one).</summary>
    internal TransformerReader? TransformerReader(Document doc) => _transformer == null
        ? null
        : new TransformerReader(_transformer, _layerMix, [.. doc.Sentences.Select(s => (IReadOnlyList<string>)[.. s.Tokens.Select(t => t.Text)])]);

    /// <summary>Sets <see cref="Sentence.Sentiment"/> on every sentence. Reads only the tokens' text.</summary>
    /// <param name="charlms">Charlm representations the tagger kept, if any. They are used for sentences
    /// whose tokens are exactly the tagger's words (no multi-word tokens); the rest are computed.</param>
    /// <param name="transformers">The call's transformer states of this classifier's LoRA encoder
    /// (<see cref="TransformerEmbedder.Prefill"/>), if any.</param>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default, TransformerCache? transformers = null)
    {
        if (!UsesCharlm)
            charlms = null;
        var sentences = doc.Sentences.Select(s => (IReadOnlyList<string>)s.Tokens.Select(t => t.Text).ToList()).ToList();
        var keys = charlms == null ? null : doc.Sentences.Select(CacheKey).ToList();
        var labels = Classify(sentences, out _, charlms, keys, cancellationToken, transformers);
        for (int i = 0; i < labels.Length; i++)
            doc.Sentences[i].Sentiment = labels[i];
    }

    /// <summary>The sentence if its tokens are exactly the tagger's words (no multi-word tokens), so the tagger's cached charlm outputs apply; else null.</summary>
    internal static Sentence? CacheKey(Sentence sentence) =>
        sentence.Tokens.All(t => t.Words.Count == 1 && t.Words[0].Text == t.Text) ? sentence : null;

    /// <summary>
    /// BaseClassifier.label_sentences: sentences sorted longest first (stably), cut into batches of at most
    /// <see cref="BatchSize"/> tokens, each padded to its longest sentence. Padding changes the results
    /// (the LSTM and convolutions run over it), so batches must be Stanza's.
    /// </summary>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: charlm outputs already computed; sentence i's are looked up
    /// under its key if not null.</param>
    internal int[] Classify(IReadOnlyList<IReadOnlyList<string>> sentences, out float[][] logits, CharlmCache? charlms = null,
        IReadOnlyList<Sentence?>? cacheKeys = null, CancellationToken cancellationToken = default, TransformerCache? transformers = null)
    {
        var order = Enumerable.Range(0, sentences.Count).OrderByDescending(i => sentences[i].Count).ToArray();
        var labels = new int[sentences.Count];
        logits = new float[sentences.Count][];
        foreach (var (start, end) in Batches(order.Select(i => sentences[i].Count).ToArray()))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batch = order[start..end];
            var scores = Forward(batch.Select(i => sentences[i]).ToList(), charlms, cacheKeys == null ? null : batch.Select(i => cacheKeys[i]).ToList(), cancellationToken,
                transformers);
            for (int k = 0; k < batch.Length; k++)
            {
                logits[batch[k]] = scores[k];
                labels[batch[k]] = Array.IndexOf(scores[k], scores[k].Max()); // torch.argmax: first maximum
            }
        }

        return labels;
    }

    /// <summary>common/utils.py split_into_batches over sorted lengths: [start, end) intervals.</summary>
    internal static List<(int Start, int End)> Batches(int[] lengths)
    {
        var intervals = new List<(int, int)>();
        int start = 0, size = 0;
        for (int i = 0; i < lengths.Length; i++)
        {
            if (lengths[i] > BatchSize)
            {
                if (size > 0)
                    intervals.Add((start, i));
                intervals.Add((i, i + 1));
                (start, size) = (i + 1, 0);
            }
            else if (lengths[i] + size > BatchSize)
            {
                intervals.Add((start, i));
                (start, size) = (i, lengths[i]);
            }
            else
            {
                size += lengths[i];
            }
        }

        if (size > 0)
            intervals.Add((start, lengths.Length));
        return intervals;
    }

    /// <summary>CNNClassifier.map_word: the word as written, then without a trailing apostrophe, then lowercased.</summary>
    internal int MapWord(string word)
    {
        int id = _pretrain.UnitToId(word);
        if (id != _pretrain.UnkId)
            return id;
        if (word.Length > 1 && word[^1] == '\'')
        {
            id = _pretrain.UnitToId(word[..^1]);
            if (id != _pretrain.UnkId)
                return id;
        }

        return _pretrain.UnitToId(PyString.Lower(word));
    }

    /// <summary>CNNClassifier.forward in eval mode: one batch, padded at the end to its longest sentence (at least the widest filter).</summary>
    /// <returns>The class logits of each sentence.</returns>
    internal float[][] Forward(IReadOnlyList<IReadOnlyList<string>> batch, CharlmCache? charlms = null, IReadOnlyList<Sentence?>? cacheKeys = null,
        CancellationToken cancellationToken = default, TransformerCache? transformers = null)
    {
        int n = batch.Count, width = Math.Max(_maxWindow, batch.Max(s => s.Count));
        var ids = new long[n * width]; // padding: PAD (0) in both vocabs
        var extraIds = new long[n * width];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < batch[i].Count; j++)
            {
                var word = batch[i][j];
                ids[i * width + j] = MapWord(word);
                extraIds[i * width + j] = _extraVocab.TryGetValue(word, out var e) ? e : UnkId;
            }
        }

        // extract_bert_embeddings over the token texts, without the endpoints, through the LoRA encoder: Embed reads a cache
        // only for its own net, never the frozen transformer's, whose states differ.
        var transformer = _transformer?.Embed(batch, keepEndpoints: false, _layerMix, transformers, cancellationToken);
        var scores = _net.Forward(batch, ids, extraIds, width, UsesCharlm ? charlms : null, UsesCharlm ? cacheKeys : null, cancellationToken, transformer);
        int classes = scores.Length / n;
        return Enumerable.Range(0, n).Select(i => scores[(i * classes)..((i + 1) * classes)]).ToArray();
    }

    private static void CheckSupported(JsonNode config)
    {
        void Require(bool ok, string what)
        {
            if (!ok) throw new NotSupportedException($"Sentiment checkpoint uses {what}, which is not ported");
        }

        Require(config["model_type"]?.GetValue<string>() == "CNN", $"model type {config["model_type"]}");
        Require(config["extra_wordvec_method"]?.GetValue<string>() == "SUM", $"extra_wordvec_method {config["extra_wordvec_method"]}");
        Require(config["extra_wordvec_max_norm"] == null, "extra_wordvec_max_norm");
        Require(config["has_charlm_forward"]?.GetValue<bool>() == config["has_charlm_backward"]?.GetValue<bool>(), "one charlm without the other");
        Require(config["has_charlm_forward"]?.GetValue<bool>() == true || config["bert_model"] != null, "a configuration without charlms or a transformer");
        Require(config["charlm_projection"] == null, "charlm_projection");
        Require(config["use_elmo"]?.GetValue<bool>() != true, "ELMo");
        Require(config["bilstm"]?.GetValue<bool>() == true, "a configuration without the biLSTM");
        Require(config["maxpool_width"]?.GetValue<int>() == 1, $"maxpool_width {config["maxpool_width"]}");
        Require(config["filter_channels"] is JsonValue, "per-filter channel counts");
    }

    public void Dispose()
    {
        _net.Dispose();
        _loraNet?.Dispose();
    }
}
