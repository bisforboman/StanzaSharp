using System.Buffers;
using System.Runtime.CompilerServices;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// <see cref="ITransformerNet"/> on the managed kernels (docs/transformer-plan.md). The dense layers are float32
/// <see cref="Gemm"/> runs over every sentence's pieces at once, row-invariant (no single-row kernel), so a sentence's
/// states have the same bits in any call; attention runs per sentence and head without padding. Attention, LayerNorm
/// and GELU are scalar code in double (one version for every <see cref="KernelPath"/>; only the GEMMs differ by path).
/// </summary>
internal sealed unsafe class ManagedTransformer : ITransformerNet
{
    private sealed record Layer(PackedMatrix Qkv, float[] QkvBias, PackedMatrix Out, float[] OutBias, float[] Ln1G, float[] Ln1B,
        PackedMatrix Up, float[] UpBias, PackedMatrix Down, float[] DownBias, float[] Ln2G, float[] Ln2B);

    // A LoRA adapter (LoraWeights): A [rank, in], B [out, rank], added to the output's columns from Column on.
    private sealed record Adapter(PackedMatrix A, PackedMatrix B, int Column);

    // Per layer, the adapters on the fused QKV (query, value), the attention output, the intermediate and the output dense.
    private sealed record LayerAdapters(Adapter[] Qkv, Adapter? Out, Adapter? Up, Adapter? Down);

    private readonly float[] _words, _positions, _type0, _embG, _embB;
    private readonly Layer[] _layers;
    private readonly LayerAdapters[]? _adapters;
    private readonly float _scaling;

    public TransformerConfig Config { get; }

    private ManagedTransformer(TransformerConfig config, float[] words, float[] positions, float[] type0, float[] embG, float[] embB, Layer[] layers,
        LayerAdapters[]? adapters = null, float scaling = 0)
    {
        Config = config;
        (_words, _positions, _type0, _embG, _embB, _layers) = (words, positions, type0, embG, embB, layers);
        (_adapters, _scaling) = (adapters, scaling);
    }

    public ITransformerNet WithLora(LoraWeights lora)
    {
        int h = Config.Hidden, inter = Config.Intermediate;
        Adapter? Make(int layer, string module, int input, int output, int column)
        {
            if (lora.Of(layer, module) is not { } w)
                return null;
            if (w.A.Length != w.Rank * input || w.B.Length != output * w.Rank)
                throw new InvalidDataException($"LoRA adapter on layer {layer} {module} has the wrong shape");
            return new Adapter(new PackedMatrix(w.A, w.Rank, input), new PackedMatrix(w.B, output, w.Rank), column);
        }

        var adapters = new LayerAdapters[_layers.Length];
        for (int l = 0; l < _layers.Length; l++)
        {
            adapters[l] = new LayerAdapters(
                [.. new[] { Make(l, "attention.self.query", h, h, 0), Make(l, "attention.self.value", h, h, 2 * h) }.OfType<Adapter>()],
                Make(l, "attention.output.dense", h, h, 0), Make(l, "intermediate.dense", h, inter, 0), Make(l, "output.dense", inter, h, 0));
        }

        return new ManagedTransformer(Config, _words, _positions, _type0, _embG, _embB, _layers, adapters, lora.Scaling);
    }

    /// <summary>Loads a transformer directory (<c>config.json</c>, <c>pytorch_model.bin</c>; ModelDownloader.TransformerDirectory).</summary>
    public static ManagedTransformer Load(string directory)
    {
        var config = TransformerConfig.Load(Path.Combine(directory, "config.json"));
        return Load(config, Checkpoint.Load(Path.Combine(directory, "pytorch_model.bin")));
    }

    /// <param name="prefix">The state dict's prefix of the encoder (transformers' ElectraForPreTraining: <c>electra.</c>).</param>
    public static ManagedTransformer Load(TransformerConfig config, Checkpoint checkpoint, string prefix = "electra.")
    {
        var root = checkpoint.Root;
        float[] T(string name, params long[] shape)
        {
            var node = root[prefix + name] ?? throw new InvalidDataException($"No tensor {prefix}{name}");
            if (!checkpoint.Shape(node).SequenceEqual(shape))
                throw new InvalidDataException($"{prefix}{name} is [{string.Join(", ", checkpoint.Shape(node))}], not [{string.Join(", ", shape)}]");
            return checkpoint.Tensor<float>(node);
        }

        int h = config.Hidden, inter = config.Intermediate;
        var layers = new Layer[config.Layers];
        for (int l = 0; l < config.Layers; l++)
        {
            string p = $"encoder.layer.{l}.";
            var qkv = new float[3 * h * h];
            T(p + "attention.self.query.weight", h, h).CopyTo(qkv, 0);
            T(p + "attention.self.key.weight", h, h).CopyTo(qkv, h * h);
            T(p + "attention.self.value.weight", h, h).CopyTo(qkv, 2 * h * h);
            float[] qkvBias = [.. T(p + "attention.self.query.bias", h), .. T(p + "attention.self.key.bias", h), .. T(p + "attention.self.value.bias", h)];
            layers[l] = new Layer(
                new PackedMatrix(qkv, 3 * h, h), Padded(qkvBias),
                new PackedMatrix(T(p + "attention.output.dense.weight", h, h), h, h), Padded(T(p + "attention.output.dense.bias", h)),
                T(p + "attention.output.LayerNorm.weight", h), T(p + "attention.output.LayerNorm.bias", h),
                new PackedMatrix(T(p + "intermediate.dense.weight", inter, h), inter, h), Padded(T(p + "intermediate.dense.bias", inter)),
                new PackedMatrix(T(p + "output.dense.weight", h, inter), h, inter), Padded(T(p + "output.dense.bias", h)),
                T(p + "output.LayerNorm.weight", h), T(p + "output.LayerNorm.bias", h));
        }

        var words = root[prefix + "embeddings.word_embeddings.weight"] ?? throw new InvalidDataException($"No tensor {prefix}embeddings.word_embeddings.weight");
        var vocab = checkpoint.Shape(words)[0];
        return new ManagedTransformer(config, T("embeddings.word_embeddings.weight", vocab, h), T("embeddings.position_embeddings.weight", config.MaxPositions, h),
            T("embeddings.token_type_embeddings.weight", 2, h)[..h], T("embeddings.LayerNorm.weight", h), T("embeddings.LayerNorm.bias", h), layers);
    }

    private static float[] Padded(float[] bias) => bias.Length % PackedMatrix.NR == 0 ? bias : [.. bias, .. new float[PackedMatrix.NR - (bias.Length % PackedMatrix.NR)]];

    public float[][] Encode(IReadOnlyList<int[]> sentences, int lastLayers, CancellationToken ct = default)
    {
        if (lastLayers < 1 || lastLayers > _layers.Length + 1)
            throw new ArgumentOutOfRangeException(nameof(lastLayers));
        int h = Config.Hidden, inter = _layers[0].Up.PaddedN, rows = 0, vocab = _words.Length / h;
        var starts = new int[sentences.Count + 1];
        for (int s = 0; s < sentences.Count; s++)
        {
            if (sentences[s].Length > Config.MaxPositions)
                throw new ArgumentException($"Sentence {s} has {sentences[s].Length} pieces, over {Config.MaxPositions}");
            if (sentences[s].Any(id => (uint)id >= (uint)vocab))
                throw new ArgumentException($"Sentence {s} has an id outside the vocabulary of {vocab}");
            starts[s] = rows;
            rows += sentences[s].Length;
        }

        starts[^1] = rows;
        var result = new float[sentences.Count][];
        for (int s = 0; s < sentences.Count; s++)
            result[s] = new float[lastLayers * (starts[s + 1] - starts[s]) * h];
        if (rows == 0)
            return result;

        var pool = ArrayPool<float>.Shared;
        float[] x = pool.Rent(rows * h), qkv = pool.Rent(rows * 3 * h), context = pool.Rent(rows * h), tmp = pool.Rent(rows * h), up = pool.Rent(rows * inter);
        int rank = _adapters?.SelectMany(a => a.Qkv.Concat([a.Out, a.Up, a.Down])).OfType<Adapter>().Max(a => a.A.PaddedN) ?? 0;
        float[]? low = _adapters == null ? null : pool.Rent(rows * rank), delta = _adapters == null ? null : pool.Rent(rows * inter);
        try
        {
            for (int s = 0; s < sentences.Count; s++)
            {
                var ids = sentences[s];
                for (int i = 0; i < ids.Length; i++)
                {
                    var row = x.AsSpan((starts[s] + i) * h, h);
                    var word = _words.AsSpan(ids[i] * h, h);
                    var position = _positions.AsSpan(i * h, h);
                    for (int j = 0; j < h; j++)
                        row[j] = word[j] + _type0[j] + position[j];
                }
            }

            fixed (float* px = x, pq = qkv, pc = context, pt = tmp, pu = up)
            {
                LayerNorm(px, null, rows, _embG, _embB);
                Keep(px, 0);
                for (int l = 0; l < _layers.Length; l++)
                {
                    ct.ThrowIfCancellationRequested();
                    var layer = _layers[l];
                    var adapters = _adapters?[l];
                    fixed (float* b1 = layer.QkvBias, b2 = layer.OutBias, b3 = layer.UpBias, b4 = layer.DownBias)
                    {
                        Gemm.Run(px, rows, h, layer.Qkv, b1, pq, 3 * h, ct, rowInvariant: true);
                        foreach (var a in adapters?.Qkv ?? [])
                            AddLora(a, px, rows, h, pq, 3 * h, low!, delta!, ct);
                        Attention(pq, pc, starts);
                        Gemm.Run(pc, rows, h, layer.Out, b2, pt, h, ct, rowInvariant: true);
                        AddLora(adapters?.Out, pc, rows, h, pt, h, low, delta, ct);
                        LayerNorm(px, pt, rows, layer.Ln1G, layer.Ln1B);
                        Gemm.Run(px, rows, h, layer.Up, b3, pu, inter, ct, rowInvariant: true);
                        AddLora(adapters?.Up, px, rows, h, pu, inter, low, delta, ct);
                        Gelu(pu, rows * inter);
                        Gemm.Run(pu, rows, inter, layer.Down, b4, pt, h, ct, rowInvariant: true);
                        AddLora(adapters?.Down, pu, rows, inter, pt, h, low, delta, ct);
                        LayerNorm(px, pt, rows, layer.Ln2G, layer.Ln2B);
                    }

                    Keep(px, l + 1);
                }
            }
        }
        finally
        {
            foreach (var buffer in new[] { x, qkv, context, tmp, up, low, delta })
            {
                if (buffer != null)
                    pool.Return(buffer);
            }
        }

        return result;

        // hidden_states[layer] (0 = the embeddings) into its slot when it is one of the last lastLayers.
        void Keep(float* states, int layer)
        {
            int k = layer - (_layers.Length + 1 - lastLayers);
            if (k < 0)
                return;
            for (int s = 0; s < sentences.Count; s++)
            {
                int n = starts[s + 1] - starts[s];
                new ReadOnlySpan<float>(states + ((long)starts[s] * h), n * h).CopyTo(result[s].AsSpan(k * n * h, n * h));
            }
        }
    }

    // peft's LoRA Linear in eval mode: output += B(A(input))·scaling, after the base layer's bias (float32, B's product
    // scaled, then added, as torch does). Both GEMMs are row-invariant like the base layers.
    private void AddLora(Adapter? adapter, float* input, int rows, int k, float* output, int ldOutput, float[]? low, float[]? delta, CancellationToken ct)
    {
        if (adapter == null)
            return;
        int rank = adapter.A.PaddedN, ld = adapter.B.PaddedN, n = adapter.B.N, column = adapter.Column;
        float scaling = _scaling;
        fixed (float* pl = low, pd = delta)
        {
            Gemm.Run(input, rows, k, adapter.A, null, pl, rank, ct, rowInvariant: true);
            Gemm.Run(pl, rows, adapter.A.N, adapter.B, null, pd, ld, ct, rowInvariant: true);
            nint po = (nint)output, pdd = (nint)pd;
            ManagedThreads.For((rows + 63) / 64, block =>
            {
                for (int r = block * 64; r < Math.Min(rows, (block * 64) + 64); r++)
                {
                    var o = (float*)po + ((long)r * ldOutput) + column;
                    var d = (float*)pdd + ((long)r * ld);
                    for (int j = 0; j < n; j++)
                        o[j] += d[j] * scaling;
                }
            });
        }
    }

    // context = softmax(q·kᵀ / √d) · v per sentence and head; q, k, v are the thirds of each qkv row. Scores and sums in
    // double; a task's arithmetic doesn't depend on how many threads run the tasks.
    private void Attention(float* qkv, float* context, int[] starts)
    {
        int h = Config.Hidden, heads = Config.Heads, d = h / heads, sentences = starts.Length - 1;
        nint pq = (nint)qkv, pc = (nint)context;
        double scale = 1 / Math.Sqrt(d);
        ManagedThreads.For(sentences * heads, task =>
        {
            int s = task / heads, head = task % heads, r0 = starts[s], n = starts[s + 1] - r0;
            var q_ = (float*)pq;
            var c_ = (float*)pc;
            var p = ArrayPool<double>.Shared.Rent(n + d);
            try
            {
                var acc = p.AsSpan(n, d);
                for (int i = 0; i < n; i++)
                {
                    var qi = new ReadOnlySpan<float>(q_ + ((long)(r0 + i) * 3 * h) + (head * d), d);
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < n; j++)
                    {
                        var kj = new ReadOnlySpan<float>(q_ + ((long)(r0 + j) * 3 * h) + h + (head * d), d);
                        p[j] = Act.Dot(qi, kj) * scale;
                        max = Math.Max(max, p[j]);
                    }

                    double sum = 0;
                    for (int j = 0; j < n; j++)
                        sum += p[j] = Math.Exp(p[j] - max);
                    acc.Clear();
                    for (int j = 0; j < n; j++)
                    {
                        var vj = q_ + ((long)(r0 + j) * 3 * h) + (2 * h) + (head * d);
                        double pj = p[j];
                        for (int e = 0; e < d; e++)
                            acc[e] += pj * vj[e];
                    }

                    var ci = c_ + ((long)(r0 + i) * h) + (head * d);
                    for (int e = 0; e < d; e++)
                        ci[e] = (float)(acc[e] / sum);
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(p);
            }
        });
    }

    // x = LayerNorm(x + residual) per row (residual null: x alone), statistics in double.
    private void LayerNorm(float* x, float* residual, int rows, float[] gamma, float[] beta)
    {
        int h = Config.Hidden;
        double eps = Config.LayerNormEps;
        nint px = (nint)x, pr = (nint)residual;
        ManagedThreads.For((rows + 63) / 64, block =>
        {
            for (int r = block * 64; r < Math.Min(rows, (block * 64) + 64); r++)
            {
                var row = new Span<float>((float*)px + ((long)r * h), h);
                if (pr != 0)
                {
                    var add = new ReadOnlySpan<float>((float*)pr + ((long)r * h), h);
                    for (int j = 0; j < h; j++)
                        row[j] = add[j] + row[j];
                }

                double mean = 0, variance = 0;
                for (int j = 0; j < h; j++)
                    mean += row[j];
                mean /= h;
                for (int j = 0; j < h; j++)
                    variance += (row[j] - mean) * (row[j] - mean);
                double inv = 1 / Math.Sqrt((variance / h) + eps);
                for (int j = 0; j < h; j++)
                    row[j] = ((float)((row[j] - mean) * inv) * gamma[j]) + beta[j];
            }
        });
    }

    // Exact GELU, x·Φ(x) = x/2·(1 + erf(x/√2)).
    private static void Gelu(float* x, int n)
    {
        nint px = (nint)x;
        ManagedThreads.For((n + 4095) / 4096, block =>
        {
            var v = (float*)px;
            for (int i = block * 4096; i < Math.Min(n, (block * 4096) + 4096); i++)
                v[i] = (float)(0.5 * v[i] * (1 + Erf(v[i] * 0.70710678118654752)));
        });
    }

    /// <summary>erf in double from Numerical Recipes' erfc (relative error under 1.2e-7, the order of a float ulp).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Erf(double x)
    {
        double z = Math.Abs(x), t = 1 / (1 + (0.5 * z));
        double poly = -0.82215223 + (t * 0.17087277);
        poly = 1.48851587 + (t * poly);
        poly = -1.13520398 + (t * poly);
        poly = 0.27886807 + (t * poly);
        poly = -0.18628806 + (t * poly);
        poly = 0.09678418 + (t * poly);
        poly = 0.37409196 + (t * poly);
        poly = 1.00002368 + (t * poly);
        double erfc = t * Math.Exp((-z * z) - 1.26551223 + (t * poly));
        return x >= 0 ? 1 - erfc : erfc - 1;
    }

    public void Dispose()
    {
    }
}
