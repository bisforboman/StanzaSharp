using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Sentiment;

/// <summary>
/// Labels each sentence 0 (negative), 1 (neutral) or 2 (positive). Port of
/// stanza/models/classifiers/cnn_classifier.py <c>CNNClassifier</c> for the English <c>sstplus_charlm</c>
/// configuration:
/// - input per token: pretrain vector (a learned vector for unknown words) + delta embedding, summed,
///   then the forward and backward charlm
/// - a 2-layer biLSTM, then full-width convolutions and one 2d convolution, each max-pooled over time
/// - fully connected layers with ReLU, and an argmax over the classes
/// </summary>
internal sealed class SentimentClassifier : IDisposable
{
    // sentiment_processor.py DEFAULT_BATCH_SIZE, counted in tokens.
    internal const int BatchSize = 5000;
    private const int PadId = 0, UnkId = 1; // vocab.PAD_ID / UNK_ID

    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Dictionary<string, int> _extraVocab = [];
    private readonly Tensor _unk;
    private readonly Embedding _extraEmb;
    private readonly LSTM _bilstm;
    private readonly (Conv2d Conv, bool FullWidth)[] _convs;
    private readonly Linear[] _fc;
    private readonly int _maxWindow;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    private SentimentClassifier(Checkpoint ckpt, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        _charlmForward = charlmForward;
        _charlmBackward = charlmBackward;
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");

        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        CheckSupported(config);
        var model = p["model"]!;

        // { word: i for i, word in enumerate(extra_vocab) }: a repeated word keeps its last index.
        var extra = p["extra_vocab"]!.AsArray();
        for (int i = 0; i < extra.Count; i++)
            _extraVocab[extra[i]!.GetValue<string>()] = i;

        _unk = ckpt.ToTensor(model["unk"]);
        var extraShape = ckpt.Shape(model["extra_embedding.weight"]);
        _extraEmb = nn.Embedding(extraShape[0], extraShape[1]).LoadFrom(ckpt, model, "extra_embedding.");
        if (extraShape[1] != pretrain.Dim)
            throw new NotSupportedException("SUM needs the extra embedding to match the pretrain's dimension");

        int inputSize = pretrain.Dim + charlmForward.HiddenDim + charlmBackward.HiddenDim;
        int hidden = config["bilstm_hidden_dim"]!.GetValue<int>();
        _bilstm = nn.LSTM(inputSize, hidden, numLayers: 2, bidirectional: true, batchFirst: true).LoadFrom(ckpt, model, "bilstm.");

        int convInput = hidden * 2, channels = config["filter_channels"]!.GetValue<int>();
        var convs = new List<(Conv2d, bool)>();
        foreach (var (size, i) in config["filter_sizes"]!["$tuple"]!.AsArray().Select((s, i) => (s!, i)))
        {
            var prefix = $"conv_layers.{i}.";
            if (size is JsonValue)
            {
                int height = size.GetValue<int>();
                _maxWindow = Math.Max(_maxWindow, height);
                convs.Add((nn.Conv2d(1, channels, (height, convInput)).LoadFrom(ckpt, model, prefix), true));
            }
            else
            {
                var hw = size["$tuple"]!.AsArray().Select(x => x!.GetValue<int>()).ToArray();
                _maxWindow = Math.Max(_maxWindow, hw[1]);
                int ch = Math.Max(1, channels / (convInput / hw[1]));
                convs.Add((nn.Conv2d(1, ch, (hw[0], hw[1]), stride: (1, hw[1])).LoadFrom(ckpt, model, prefix), false));
            }
        }
        _convs = [.. convs];

        var fc = new List<Linear>();
        for (int i = 0; model[$"fc_layers.{i}.weight"] is { } w; i++)
        {
            var shape = ckpt.Shape(w);
            fc.Add(nn.Linear(shape[1], shape[0]).LoadFrom(ckpt, model, $"fc_layers.{i}."));
        }
        _fc = [.. fc];
    }

    /// <summary>
    /// Loads e.g. <c>models/converted/en/sentiment/sstplus_charlm</c>. The pretrain and charlms are
    /// shared with the tagger and parsers, so the caller owns them.
    /// </summary>
    /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
    public static SentimentClassifier Load(string basePath, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward, Device? device = null) =>
        Weights.On(device, () => new SentimentClassifier(Checkpoint.Load(basePath), pretrain, charlmForward, charlmBackward));

    /// <summary>Sets <see cref="Sentence.Sentiment"/> on every sentence. Reads only the tokens' text.</summary>
    /// <param name="charlms">Charlm representations the tagger kept, if any. They are used for sentences
    /// whose tokens are exactly the tagger's words (no multi-word tokens); the rest are computed.</param>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default)
    {
        var sentences = doc.Sentences.Select(s => (IReadOnlyList<string>)s.Tokens.Select(t => t.Text).ToList()).ToList();
        var labels = Classify(sentences, out _, charlms == null ? null : i => Cached(charlms, doc.Sentences[i]), cancellationToken);
        for (int i = 0; i < labels.Length; i++)
            doc.Sentences[i].Sentiment = labels[i];
    }

    private static (Tensor, Tensor)? Cached(CharlmCache charlms, Sentence sentence) =>
        sentence.Tokens.All(t => t.Words.Count == 1 && t.Words[0].Text == t.Text) && charlms.TryGet(sentence, out var reps) ? reps : null;

    /// <summary>
    /// BaseClassifier.label_sentences: sentences sorted longest first (stably), cut into batches of at most
    /// <see cref="BatchSize"/> tokens, each padded to its longest sentence. Padding changes the results
    /// (the LSTM and convolutions run over it), so batches must be Stanza's.
    /// </summary>
    /// <param name="cached">Charlm representations to use for sentence i instead of computing them, if any.</param>
    internal int[] Classify(IReadOnlyList<IReadOnlyList<string>> sentences, out float[][] logits, Func<int, (Tensor, Tensor)?>? cached = null,
        CancellationToken cancellationToken = default)
    {
        var order = Enumerable.Range(0, sentences.Count).OrderByDescending(i => sentences[i].Count).ToArray();
        var labels = new int[sentences.Count];
        logits = new float[sentences.Count][];
        foreach (var (start, end) in Batches(order.Select(i => sentences[i].Count).ToArray()))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batch = order[start..end];
            var scores = Forward(batch.Select(i => sentences[i]).ToList(), cached == null ? null : batch.Select(cached).ToList());
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
    internal float[][] Forward(IReadOnlyList<IReadOnlyList<string>> batch, IReadOnlyList<(Tensor Forward, Tensor Backward)?>? cached = null)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        int n = batch.Count, width = Math.Max(_maxWindow, batch.Max(s => s.Count));

        var ids = new long[n * width];
        var extraIds = new long[n * width];
        var unknown = new bool[n * width];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < batch[i].Count; j++)
            {
                var word = batch[i][j];
                int k = i * width + j, id = MapWord(word);
                ids[k] = id;
                unknown[k] = id == _pretrain.UnkId;
                extraIds[k] = _extraVocab.TryGetValue(word, out var e) ? e : UnkId;
            }

        // Unknown words get the learned unk vector instead of the pretrain's; then the delta embedding is added (SUM).
        var pretrained = _pretrain.Embeddings[torch.tensor(ids, [n, width], device: _device)];
        var mask = torch.tensor(unknown, [n, width, 1], device: _device);
        var words = torch.where(mask, _unk, pretrained).add(_extraEmb.forward(torch.tensor(extraIds, [n, width], device: _device)), Scalars.One);

        var input = cat([words, CharReps(_charlmForward, batch, width, cached?.Select(c => c?.Forward).ToList()),
            CharReps(_charlmBackward, batch, width, cached?.Select(c => c?.Backward).ToList())], 2);
        var (output, _, _) = _bilstm.call(input); // not packed: like Stanza, the padding reaches the LSTM
        var x = output.unsqueeze(1);

        var pooled = new List<Tensor>(_convs.Length);
        foreach (var (conv, fullWidth) in _convs)
        {
            var c = conv.forward(x);
            c = fullWidth ? c.squeeze(3) : c.transpose(2, 3).flatten(1, 2);
            pooled.Add(F.relu(c).amax([2])); // max_pool2d over the whole length
        }
        var hiddenLayer = cat(pooled, 1);
        for (int i = 0; i < _fc.Length - 1; i++)
            hiddenLayer = F.relu(_fc[i].forward(hiddenLayer));
        var scores = _fc[^1].forward(hiddenLayer).ToArray<float>();
        int classes = scores.Length / n;
        return Enumerable.Range(0, n).Select(i => scores[(i * classes)..((i + 1) * classes)]).ToArray();
    }

    /// <summary>build_char_reps: [batch, width, dim], each sentence's representations at its start, zeros after.</summary>
    private Tensor CharReps(CharLanguageModel charlm, IReadOnlyList<IReadOnlyList<string>> batch, int width, List<Tensor?>? cached)
    {
        var missing = Enumerable.Range(0, batch.Count).Where(i => cached?[i] is null).ToList();
        var computed = charlm.BuildCharRepresentation(missing.Select(i => batch[i]).ToList());
        var reps = cached?.ToArray() ?? new Tensor?[batch.Count];
        for (int k = 0; k < missing.Count; k++)
            reps[missing[k]] = computed[k];

        var result = torch.zeros([batch.Count, width, charlm.HiddenDim], device: _device);
        for (int i = 0; i < batch.Count; i++)
            if (batch[i].Count > 0)
                result[i].narrow(0, 0, batch[i].Count).copy_(reps[i]!);
        return result;
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

    public void Dispose()
    {
        nn.Module[] modules = [_extraEmb, _bilstm, .. _convs.Select(c => c.Conv), .. _fc];
        foreach (var m in modules)
            m.Dispose();
        _unk.Dispose();
    }
}
