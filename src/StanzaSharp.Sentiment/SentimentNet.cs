using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;
using Filter = StanzaSharp.Sentiment.SentimentClassifier.Filter;

namespace StanzaSharp.Sentiment;

/// <summary>The sentiment classifier's network, per <see cref="Backend"/>: <see cref="SentimentNet"/> or <see cref="ManagedSentimentNet"/>.</summary>
internal interface ISentimentNet : IDisposable
{
    /// <summary>
    /// CNNClassifier.forward in eval mode on one batch: embeddings, charlms, the biLSTM over the padded batch (padding
    /// included, as in Stanza), the convolutions max-pooled over every position, and the fully connected layers.
    /// </summary>
    /// <param name="batch">The batch's token texts.</param>
    /// <param name="ids">[batch, width] pretrain ids (map_word), padded with 0.</param>
    /// <param name="extraIds">[batch, width] delta ids, padded with 0.</param>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: charlm outputs to reuse; row i's key, if not null, is looked up.</param>
    /// <param name="ct">Checked after each charlm pass, inside the LSTM and between the convolutions.</param>
    /// <returns>[batch, classes] logits, row-major.</returns>
    float[] Forward(IReadOnlyList<IReadOnlyList<string>> batch, long[] ids, long[] extraIds, int width,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct);
}

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

        var cached = cacheKeys?.Select(key => key != null && charlms!.TryGet(key, out var reps) ? reps : ((Tensor, Tensor)?)null).ToList();
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

/// <summary>
/// Managed twin of <see cref="SentimentNet"/> (<see cref="Backend.Managed"/>). Every row of the batch runs the full
/// padded width through the biLSTM, as the unpacked <c>nn.LSTM</c> does. A full-width convolution is one GEMM whose
/// A rows are overlapping windows of the batch-major LSTM output (row t starts at token t and reads Height tokens),
/// so nothing is copied; windows that cross into the next sentence are computed and ignored.
/// </summary>
internal sealed unsafe class ManagedSentimentNet : ISentimentNet
{
    private readonly Pretrain _pretrain;
    private readonly ManagedCharLanguageModel _charlmForward, _charlmBackward;
    private readonly float[] _unk, _extraEmb;
    private readonly int _dim, _inSize, _pooledSize;
    private readonly ManagedLstm _lstm;
    private readonly Conv[] _convs;
    private readonly (PackedMatrix W, float[] Bias)[] _fc;

    /// <summary>One convolution. Full width (Width 0): <see cref="Packed"/> [channels, Height·lstm width]. Otherwise
    /// <see cref="Weights"/> [channels, Height, Width] with stride (1, Width), giving <see cref="Columns"/> outputs per channel.</summary>
    private sealed record Conv(int Height, int Width, int Channels, int Columns, int Offset, PackedMatrix? Packed, float[] Weights, float[] Bias);

    public ManagedSentimentNet(Checkpoint ckpt, Filter[] filters, Pretrain pretrain, ManagedCharLanguageModel charlmForward, ManagedCharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        var model = p["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));

        _dim = pretrain.Dim;
        _unk = T("unk");
        _extraEmb = T("extra_embedding.weight");
        _inSize = _dim + charlmForward.HiddenDim + charlmBackward.HiddenDim;
        int hidden = config["bilstm_hidden_dim"]!.GetValue<int>();
        _lstm = new ManagedLstm(ckpt, model, "bilstm.", _inSize, hidden, 2, bidirectional: true);

        int convInput = _lstm.OutputSize, channels = config["filter_channels"]!.GetValue<int>();
        _convs = new Conv[filters.Length];
        for (int i = 0, offset = 0; i < filters.Length; i++)
        {
            var (height, width) = filters[i];
            var w = T($"conv_layers.{i}.weight");
            var b = T($"conv_layers.{i}.bias");
            if (width == 0)
            {
                var packed = new PackedMatrix(w, channels, height * convInput);
                var bias = new float[packed.PaddedN];
                b.CopyTo(bias, 0);
                _convs[i] = new Conv(height, 0, channels, 1, offset, packed, [], bias);
                offset += channels;
            }
            else
            {
                int ch = Math.Max(1, channels / (convInput / width)), columns = (convInput - width) / width + 1;
                _convs[i] = new Conv(height, width, ch, columns, offset, null, w, b);
                offset += ch * columns;
            }
            _pooledSize = offset;
        }

        var fc = new List<(PackedMatrix, float[])>();
        for (int i = 0; model[$"fc_layers.{i}.weight"] is { } w; i++)
        {
            var shape = ckpt.Shape(w);
            var packed = new PackedMatrix(T($"fc_layers.{i}.weight"), (int)shape[0], (int)shape[1]);
            var bias = new float[packed.PaddedN];
            T($"fc_layers.{i}.bias").CopyTo(bias, 0);
            fc.Add((packed, bias));
        }
        _fc = [.. fc];
    }

    public float[] Forward(IReadOnlyList<IReadOnlyList<string>> batch, long[] ids, long[] extraIds, int width,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct)
    {
        int n = batch.Count, rows = n * width, h2 = _lstm.OutputSize, inSize = _inSize;
        var x = ArrayPool<float>.Shared.Rent(rows * inSize); // time-major: token (i, t) is row t·n + i
        var y = ArrayPool<float>.Shared.Rent(rows * h2);
        var z = ArrayPool<float>.Shared.Rent(rows * h2);     // the same, batch-major: row i·width + t
        var pooled = new float[n * _pooledSize];
        try
        {
            // Words: the pretrain vector (unk's for unknown words; PAD's row for padding) + the delta embedding.
            var vectors = _pretrain.CpuVectors();
            for (int i = 0; i < n; i++)
                for (int t = 0; t < width; t++)
                {
                    int k = i * width + t;
                    var dst = x.AsSpan((t * n + i) * inSize, inSize);
                    ReadOnlySpan<float> pre = ids[k] == _pretrain.UnkId ? _unk : vectors.Slice((int)ids[k] * _dim, _dim);
                    var delta = _extraEmb.AsSpan((int)extraIds[k] * _dim, _dim);
                    for (int d = 0; d < _dim; d++)
                        dst[d] = pre[d] + delta[d];
                    if (t >= batch[i].Count)
                        dst[_dim..].Clear(); // build_char_reps pads with zeros
                }
            CharlmFeatures(batch, charlms, cacheKeys, x, n, ct);

            // The unpacked LSTM: every row runs the full width.
            var batchSizes = new int[width];
            Array.Fill(batchSizes, n);
            _lstm.ForwardPacked(x, inSize, batchSizes, y, ct);
            for (int t = 0; t < width; t++)
                for (int i = 0; i < n; i++)
                    y.AsSpan((t * n + i) * h2, h2).CopyTo(z.AsSpan((i * width + t) * h2));

            foreach (var conv in _convs)
            {
                ct.ThrowIfCancellationRequested();
                if (conv.Packed != null)
                    FullWidth(conv, z, n, width, pooled, ct);
                else
                    Strided(conv, z, n, width, pooled);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(y);
            ArrayPool<float>.Shared.Return(z);
        }

        // Fully connected layers, ReLU between.
        var hidden = pooled;
        int ld = _pooledSize;
        for (int l = 0; l < _fc.Length; l++)
        {
            var (w, bias) = _fc[l];
            var next = new float[n * w.PaddedN];
            fixed (float* a = hidden, pb = bias, c = next)
                Gemm.Run(a, n, ld, w, pb, c, w.PaddedN, ct);
            if (l < _fc.Length - 1)
                for (int k = 0; k < next.Length; k++)
                    next[k] = Math.Max(next[k], 0);
            (hidden, ld) = (next, w.PaddedN);
        }
        int classes = _fc[^1].W.N;
        var logits = new float[n * classes];
        for (int i = 0; i < n; i++)
            hidden.AsSpan(i * ld, classes).CopyTo(logits.AsSpan(i * classes));
        return logits;
    }

    /// <summary>A Height × full-width convolution as one GEMM over overlapping windows, then ReLU and the max over each sentence's windows.</summary>
    private void FullWidth(Conv conv, float[] z, int n, int width, float[] pooled, CancellationToken ct)
    {
        var w = conv.Packed!;
        int h2 = _lstm.OutputSize, ldc = w.PaddedN, windows = width - conv.Height + 1, m = n * width - conv.Height + 1;
        var c = ArrayPool<float>.Shared.Rent(m * ldc);
        try
        {
            fixed (float* pz = z, pb = conv.Bias, pc = c)
                Gemm.Run(pz, m, h2, w, pb, pc, ldc, ct); // row r reads z[r·h2, r·h2 + Height·h2)
            for (int i = 0; i < n; i++)
            {
                var o = pooled.AsSpan(i * _pooledSize + conv.Offset, conv.Channels);
                o.Clear(); // ReLU then max = max(0, max over windows)
                for (int t = 0; t < windows; t++)
                {
                    var row = c.AsSpan((i * width + t) * ldc, conv.Channels);
                    for (int ch = 0; ch < row.Length; ch++)
                        o[ch] = Math.Max(o[ch], row[ch]);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(c);
        }
    }

    /// <summary>
    /// The Height × Width filter with stride (1, Width): output (channel, column) at window t sums Height·Width products.
    /// Pooled as Stanza flattens it: channel·Columns + column.
    /// </summary>
    // ponytail: scalar loops (8 channels × 120 columns × 25 products per window, ~2% of the full-width convolutions' work); a GEMM if it ever shows.
    private void Strided(Conv conv, float[] z, int n, int width, float[] pooled)
    {
        int h2 = _lstm.OutputSize, windows = width - conv.Height + 1, area = conv.Height * conv.Width;
        ManagedThreads.For(n, i =>
        {
            var o = pooled.AsSpan(i * _pooledSize + conv.Offset, conv.Channels * conv.Columns);
            o.Clear();
            for (int t = 0; t < windows; t++)
                for (int ch = 0; ch < conv.Channels; ch++)
                {
                    var wc = conv.Weights.AsSpan(ch * area, area);
                    for (int col = 0; col < conv.Columns; col++)
                    {
                        float s = conv.Bias[ch];
                        for (int a = 0; a < conv.Height; a++)
                        {
                            var src = z.AsSpan((i * width + t + a) * h2 + col * conv.Width, conv.Width);
                            var wr = wc.Slice(a * conv.Width, conv.Width);
                            for (int b = 0; b < src.Length; b++)
                                s += wr[b] * src[b];
                        }
                        ref float m = ref o[ch * conv.Columns + col];
                        m = Math.Max(m, s);
                    }
                }
        });
    }

    /// <summary>The forward and backward charlm columns of each real token's row: from the cache where it has them, else computed.</summary>
    private void CharlmFeatures(IReadOnlyList<IReadOnlyList<string>> batch, CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys,
        float[] x, int n, CancellationToken ct)
    {
        int dim = _charlmForward.HiddenDim, forwardCol = _dim, backwardCol = _dim + dim;
        void Put(int i, ReadOnlySpan<float> forward, ReadOnlySpan<float> backward)
        {
            for (int t = 0; t < batch[i].Count; t++)
            {
                int r = (t * n + i) * _inSize;
                forward.Slice(t * dim, dim).CopyTo(x.AsSpan(r + forwardCol, dim));
                backward.Slice(t * dim, dim).CopyTo(x.AsSpan(r + backwardCol, dim));
            }
        }

        var missing = new List<int>();
        for (int i = 0; i < batch.Count; i++)
            if (cacheKeys?[i] is { } key && charlms!.TryGetArrays(key, out var reps))
                Put(i, reps.Forward, reps.Backward);
            else
                missing.Add(i);
        if (missing.Count == 0)
            return;

        var texts = missing.Select(i => batch[i]).ToList();
        int words = texts.Sum(s => s.Count);
        var forwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        var backwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        try
        {
            // A single sentence (with multi-word tokens, say) is kept for NER, which reads the same tokens.
            ManagedCharLanguageModel.BuildBoth(_charlmForward, _charlmBackward, texts, forwardReps, backwardReps, charlms, ct);
            ct.ThrowIfCancellationRequested();
            int offset = 0;
            foreach (int i in missing)
            {
                int len = batch[i].Count * dim;
                Put(i, forwardReps.AsSpan(offset, len), backwardReps.AsSpan(offset, len));
                offset += len;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(forwardReps);
            ArrayPool<float>.Shared.Return(backwardReps);
        }
    }

    public void Dispose()
    {
    }
}
