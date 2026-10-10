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

    internal SentimentClassifier(Checkpoint ckpt, Pretrain pretrain, Func<Filter[], ISentimentNet> net)
    {
        _pretrain = pretrain;
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        CheckSupported(config);

        // { word: i for i, word in enumerate(extra_vocab) }: a repeated word keeps its last index.
        var extra = p["extra_vocab"]!.AsArray();
        for (int i = 0; i < extra.Count; i++)
            _extraVocab[extra[i]!.GetValue<string>()] = i;
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

    /// <summary><c>Load</c> (StanzaSharp.TorchSharp) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static SentimentClassifier LoadManaged(string basePath, Pretrain pretrain, ManagedCharLanguageModel charlmForward, ManagedCharLanguageModel charlmBackward)
    {
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");
        var ckpt = Checkpoint.Load(basePath);
        return new SentimentClassifier(ckpt, pretrain, filters => new ManagedSentimentNet(ckpt, filters, pretrain, charlmForward, charlmBackward));
    }

    /// <summary>Sets <see cref="Sentence.Sentiment"/> on every sentence. Reads only the tokens' text.</summary>
    /// <param name="charlms">Charlm representations the tagger kept, if any. They are used for sentences
    /// whose tokens are exactly the tagger's words (no multi-word tokens); the rest are computed.</param>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default)
    {
        var sentences = doc.Sentences.Select(s => (IReadOnlyList<string>)s.Tokens.Select(t => t.Text).ToList()).ToList();
        var keys = charlms == null ? null : doc.Sentences.Select(CacheKey).ToList();
        var labels = Classify(sentences, out _, charlms, keys, cancellationToken);
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
        IReadOnlyList<Sentence?>? cacheKeys = null, CancellationToken cancellationToken = default)
    {
        var order = Enumerable.Range(0, sentences.Count).OrderByDescending(i => sentences[i].Count).ToArray();
        var labels = new int[sentences.Count];
        logits = new float[sentences.Count][];
        foreach (var (start, end) in Batches(order.Select(i => sentences[i].Count).ToArray()))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batch = order[start..end];
            var scores = Forward(batch.Select(i => sentences[i]).ToList(), charlms, cacheKeys == null ? null : batch.Select(i => cacheKeys[i]).ToList(), cancellationToken);
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
                size += lengths[i];
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
        CancellationToken cancellationToken = default)
    {
        int n = batch.Count, width = Math.Max(_maxWindow, batch.Max(s => s.Count));
        var ids = new long[n * width]; // padding: PAD (0) in both vocabs
        var extraIds = new long[n * width];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < batch[i].Count; j++)
            {
                var word = batch[i][j];
                ids[i * width + j] = MapWord(word);
                extraIds[i * width + j] = _extraVocab.TryGetValue(word, out var e) ? e : UnkId;
            }
        var scores = _net.Forward(batch, ids, extraIds, width, charlms, cacheKeys, cancellationToken);
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
        Require(config["has_charlm_forward"]?.GetValue<bool>() == true && config["has_charlm_backward"]?.GetValue<bool>() == true, "a configuration without both charlms");
        Require(config["charlm_projection"] == null, "charlm_projection");
        Require(config["use_elmo"]?.GetValue<bool>() != true, "ELMo");
        Require(config["bert_model"] == null, "a transformer");
        Require(config["bilstm"]?.GetValue<bool>() == true, "a configuration without the biLSTM");
        Require(config["maxpool_width"]?.GetValue<int>() == 1, $"maxpool_width {config["maxpool_width"]}");
        Require(config["filter_channels"] is JsonValue, "per-filter channel counts");
    }

    public void Dispose() => _net.Dispose();
}
