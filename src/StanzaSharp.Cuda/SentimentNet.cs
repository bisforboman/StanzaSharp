using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;
using Filter = StanzaSharp.Sentiment.SentimentClassifier.Filter;

namespace StanzaSharp.Sentiment;

/// <summary>cnn_classifier.py <c>CNNClassifier.forward</c> on TorchSharp (today's code).</summary>
internal sealed class SentimentNet : ISentimentNet
{
    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Tensor _unk;
    private readonly Embedding _extraEmb;
    private readonly LSTM _bilstm;
    private readonly (Conv2d Conv, bool FullWidth)[] _convs;
    private readonly Linear[] _fc;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public SentimentNet(Checkpoint ckpt, Filter[] filters, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        var model = p["model"]!;

        _unk = ckpt.ToTensor(model["unk"]);
        var extraShape = ckpt.Shape(model["extra_embedding.weight"]);
        _extraEmb = nn.Embedding(extraShape[0], extraShape[1]).LoadFrom(ckpt, model, "extra_embedding.");

        int inputSize = pretrain.Dim + charlmForward.HiddenDim + charlmBackward.HiddenDim;
        int hidden = config["bilstm_hidden_dim"]!.GetValue<int>();
        _bilstm = nn.LSTM(inputSize, hidden, numLayers: 2, bidirectional: true, batchFirst: true).LoadFrom(ckpt, model, "bilstm.");

        int convInput = hidden * 2, channels = config["filter_channels"]!.GetValue<int>();
        _convs = filters.Select((f, i) => f.Width == 0
            ? (nn.Conv2d(1, channels, (f.Height, convInput)).LoadFrom(ckpt, model, $"conv_layers.{i}."), true)
            : (nn.Conv2d(1, Math.Max(1, channels / (convInput / f.Width)), (f.Height, f.Width), stride: (1, f.Width)).LoadFrom(ckpt, model, $"conv_layers.{i}."), false)).ToArray();

        var fc = new List<Linear>();
        for (int i = 0; model[$"fc_layers.{i}.weight"] is { } w; i++)
        {
            var shape = ckpt.Shape(w);
            fc.Add(nn.Linear(shape[1], shape[0]).LoadFrom(ckpt, model, $"fc_layers.{i}."));
        }
        _fc = [.. fc];
    }

    public float[] Forward(IReadOnlyList<IReadOnlyList<string>> batch, long[] ids, long[] extraIds, int width,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        int n = batch.Count;
        var unknown = ids.Select(id => id == _pretrain.UnkId).ToArray();

        // Unknown words get the learned unk vector instead of the pretrain's; then the delta embedding is added (SUM).
        var pretrained = _pretrain.Embeddings[torch.tensor(ids, [n, width], device: _device)];
        var mask = torch.tensor(unknown, [n, width, 1], device: _device);
        var words = torch.where(mask, _unk, pretrained).add(_extraEmb.forward(torch.tensor(extraIds, [n, width], device: _device)), Scalars.One);

        var cached = cacheKeys?.Select(key => key != null && charlms!.TryGet(key, out var reps, _device) ? reps : ((Tensor, Tensor)?)null).ToList();
        var forward = CharReps(_charlmForward, batch, width, cached?.Select(c => c?.Item1).ToList());
        ct.ThrowIfCancellationRequested();
        var backward = CharReps(_charlmBackward, batch, width, cached?.Select(c => c?.Item2).ToList());
        ct.ThrowIfCancellationRequested();
        var input = cat([words, forward, backward], 2);
        var (output, _, _) = _bilstm.call(input); // not packed: like Stanza, the padding reaches the LSTM
        var x = output.unsqueeze(1);

        var pooled = new List<Tensor>(_convs.Length);
        foreach (var (conv, fullWidth) in _convs)
        {
            ct.ThrowIfCancellationRequested();
            var c = conv.forward(x);
            c = fullWidth ? c.squeeze(3) : c.transpose(2, 3).flatten(1, 2);
            pooled.Add(F.relu(c).amax([2])); // max_pool2d over the whole length
        }
        var hiddenLayer = cat(pooled, 1);
        for (int i = 0; i < _fc.Length - 1; i++)
            hiddenLayer = F.relu(_fc[i].forward(hiddenLayer));
        return _fc[^1].forward(hiddenLayer).ToArray<float>();
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

    public void Dispose()
    {
        nn.Module[] modules = [_extraEmb, _bilstm, .. _convs.Select(c => c.Conv), .. _fc];
        foreach (var m in modules)
            m.Dispose();
        _unk.Dispose();
    }
}

/// <summary>Loads a <see cref="SentimentClassifier"/> on TorchSharp.</summary>
internal static class SentimentClassifierLoad
{
    extension(SentimentClassifier)
    {
        /// <summary>
        /// Loads e.g. <c>models/converted/en/sentiment/sstplus_charlm</c> on TorchSharp. The pretrain and charlms are
        /// shared with the tagger and parsers, so the caller owns them.
        /// </summary>
        /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
        public static SentimentClassifier Load(string basePath, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward, Device? device = null) =>
            Weights.On(device, () =>
            {
                if (!charlmForward.IsForward || charlmBackward.IsForward)
                    throw new ArgumentException("Pass the forward charlm first, then the backward one");
                var ckpt = Checkpoint.Load(basePath);
                return new SentimentClassifier(ckpt, pretrain, filters => new SentimentNet(ckpt, filters, pretrain, charlmForward, charlmBackward));
            });
    }
}
