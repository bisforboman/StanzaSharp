using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using Filter = StanzaSharp.Sentiment.SentimentClassifier.Filter;

namespace StanzaSharp.Sentiment;

/// <summary>The sentiment classifier's network, per <see cref="Backend"/>: <c>SentimentNet</c> or <see cref="ManagedSentimentNet"/>.</summary>
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

/// <summary>
/// Managed twin of <c>SentimentNet</c> (<see cref="Backend.Managed"/>). Every row of the batch runs the full
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
        int n = batch.Count, rows = n * width, h2 = _lstm.OutputSize, inSize = _inSize, dim = _charlmForward.HiddenDim;
        var z = ArrayPool<float>.Shared.Rent(rows * h2); // the LSTM output, batch-major: row i·width + t
        var pooled = new float[n * _pooledSize];
        var rented = new List<float[]>();
        try
        {
            var reps = CharlmFeatures(batch, charlms, cacheKeys, rented, ct);

            // The input rows (time-major: token (i, t) is row t·n + i), a chunk at a time: the pretrain vector (unk's
            // for unknown words; PAD's row for padding) + the delta embedding, then the charlms (zeros for padding, as
            // build_char_reps pads).
            void Fill(int r0, int r1, float* x)
            {
                var vectors = _pretrain.CpuVectors();
                for (int r = r0; r < r1; r++)
                {
                    int i = r % n, t = r / n, k = i * width + t;
                    var dst = new Span<float>(x + (long)(r - r0) * inSize, inSize);
                    ReadOnlySpan<float> pre = ids[k] == _pretrain.UnkId ? _unk : vectors.Slice((int)ids[k] * _dim, _dim);
                    var delta = _extraEmb.AsSpan((int)extraIds[k] * _dim, _dim);
                    for (int d = 0; d < _dim; d++)
                        dst[d] = pre[d] + delta[d];
                    if (t < batch[i].Count)
                    {
                        var (forward, backward, offset) = reps[i];
                        forward.AsSpan((offset + t) * dim, dim).CopyTo(dst[_dim..]);
                        backward.AsSpan((offset + t) * dim, dim).CopyTo(dst[(_dim + dim)..]);
                    }
                    else
                        dst[_dim..].Clear();
                }
            }

            // The unpacked LSTM: every row runs the full width.
            _lstm.ForwardUnpacked(n, width, Fill, z, ct);

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
            ArrayPool<float>.Shared.Return(z);
            foreach (var r in rented)
                ArrayPool<float>.Shared.Return(r);
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

    /// <summary>
    /// A Height × full-width convolution as one GEMM over overlapping windows, a chunk of rows at a time
    /// (<see cref="Gemm.RunRows"/>), then ReLU and the max over each sentence's windows.
    /// </summary>
    private void FullWidth(Conv conv, float[] z, int n, int width, float[] pooled, CancellationToken ct)
    {
        const int ChunkRows = 2048;
        var w = conv.Packed!;
        int h2 = _lstm.OutputSize, ldc = w.PaddedN, windows = width - conv.Height + 1, m = n * width - conv.Height + 1;
        var c = ArrayPool<float>.Shared.Rent(Math.Min(m, ChunkRows) * ldc);
        for (int i = 0; i < n; i++)
            pooled.AsSpan(i * _pooledSize + conv.Offset, conv.Channels).Clear(); // ReLU then max = max(0, max over windows)
        try
        {
            for (int r0 = 0; r0 < m; r0 += ChunkRows)
            {
                int r1 = Math.Min(m, r0 + ChunkRows);
                fixed (float* pz = z, pb = conv.Bias, pc = c)
                    Gemm.RunRows(pz + (long)r0 * h2, r0, r1, m, h2, w, pb, pc, ldc, ct); // row r reads z[r·h2, r·h2 + Height·h2)
                for (int r = r0; r < r1; r++)
                {
                    int i = r / width, t = r % width;
                    if (t >= windows)
                        continue;
                    var o = pooled.AsSpan(i * _pooledSize + conv.Offset, conv.Channels);
                    var row = c.AsSpan((r - r0) * ldc, conv.Channels);
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

    /// <summary>
    /// Each sentence's forward and backward charlm rows (token t at row offset + t of the arrays): from the cache where
    /// it has them, else computed into arrays rented here and added to <paramref name="rented"/>.
    /// </summary>
    private (float[] Forward, float[] Backward, int Offset)[] CharlmFeatures(IReadOnlyList<IReadOnlyList<string>> batch, CharlmCache? charlms,
        IReadOnlyList<Sentence?>? cacheKeys, List<float[]> rented, CancellationToken ct)
    {
        int dim = _charlmForward.HiddenDim;
        var reps = new (float[], float[], int)[batch.Count];
        var missing = new List<int>();
        for (int i = 0; i < batch.Count; i++)
            if (cacheKeys?[i] is { } key && charlms!.TryGetArrays(key, out var cached))
                reps[i] = (cached.Forward, cached.Backward, 0);
            else
                missing.Add(i);
        if (missing.Count == 0)
            return reps;

        var texts = missing.Select(i => batch[i]).ToList();
        int words = texts.Sum(s => s.Count);
        var forwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        rented.Add(forwardReps);
        var backwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        rented.Add(backwardReps);
        // A single sentence (with multi-word tokens, say) is kept for NER, which reads the same tokens.
        ManagedCharLanguageModel.BuildBoth(_charlmForward, _charlmBackward, texts, forwardReps, backwardReps, charlms, ct);
        ct.ThrowIfCancellationRequested();
        int offset = 0;
        foreach (int i in missing)
        {
            reps[i] = (forwardReps, backwardReps, offset);
            offset += batch[i].Count;
        }
        return reps;
    }

    public void Dispose()
    {
    }
}
