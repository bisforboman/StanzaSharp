using System.Runtime.CompilerServices;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// A BERT-style transformer encoder (transformers' <c>ElectraModel</c> with embedding_size = hidden_size, absolute
/// positions, exact GELU, post-LayerNorm) on the managed kernels. Spike code (docs/transformer-plan.md): float32
/// weights packed for <see cref="Gemm"/>, attention per sentence and head without padding (a masked key contributes
/// an exact 0 in transformers too), LayerNorm and attention sums in double.
/// </summary>
internal sealed unsafe class ManagedElectra
{
    private sealed record Layer(PackedMatrix Qkv, float[] QkvBias, PackedMatrix Out, float[] OutBias, float[] Ln1G, float[] Ln1B,
        PackedMatrix Up, float[] UpBias, PackedMatrix Down, float[] DownBias, float[] Ln2G, float[] Ln2B);

    private readonly float[] _words, _positions, _type0, _embG, _embB;
    private readonly Layer[] _layers;
    private readonly float _eps;

    public int Hidden { get; }
    public int Heads { get; }
    public int Layers => _layers.Length;
    public int MaxPositions { get; }

    private ManagedElectra(float[] words, float[] positions, float[] type0, float[] embG, float[] embB, Layer[] layers, int hidden, int heads,
        int maxPositions, float eps)
    {
        (_words, _positions, _type0, _embG, _embB, _layers) = (words, positions, type0, embG, embB, layers);
        (Hidden, Heads, MaxPositions, _eps) = (hidden, heads, maxPositions, eps);
    }

    /// <summary>Loads a transformers state dict (<c>pytorch_model.bin</c> read as a <c>.pt</c>) whose keys start with <paramref name="prefix"/>.</summary>
    public static ManagedElectra Load(Checkpoint checkpoint, int heads, float eps = 1e-12f, string prefix = "electra.")
    {
        var root = checkpoint.Root;
        float[] T(string name) => checkpoint.Tensor<float>(root[prefix + name] ?? throw new InvalidDataException($"No tensor {prefix}{name}"));
        long[] S(string name) => checkpoint.Shape(root[prefix + name]);

        var wordShape = S("embeddings.word_embeddings.weight");
        int hidden = (int)wordShape[1];
        var positions = T("embeddings.position_embeddings.weight");
        var types = T("embeddings.token_type_embeddings.weight");
        var layers = new List<Layer>();
        for (int l = 0; root[$"{prefix}encoder.layer.{l}.attention.self.query.weight"] != null; l++)
        {
            string p = $"encoder.layer.{l}.";
            var qkv = new float[3 * hidden * hidden];
            T(p + "attention.self.query.weight").CopyTo(qkv, 0);
            T(p + "attention.self.key.weight").CopyTo(qkv, hidden * hidden);
            T(p + "attention.self.value.weight").CopyTo(qkv, 2 * hidden * hidden);
            var qkvBias = new float[3 * hidden];
            T(p + "attention.self.query.bias").CopyTo(qkvBias, 0);
            T(p + "attention.self.key.bias").CopyTo(qkvBias, hidden);
            T(p + "attention.self.value.bias").CopyTo(qkvBias, 2 * hidden);
            int inter = (int)S(p + "intermediate.dense.weight")[0];
            layers.Add(new Layer(
                new PackedMatrix(qkv, 3 * hidden, hidden), Padded(qkvBias),
                new PackedMatrix(T(p + "attention.output.dense.weight"), hidden, hidden), Padded(T(p + "attention.output.dense.bias")),
                T(p + "attention.output.LayerNorm.weight"), T(p + "attention.output.LayerNorm.bias"),
                new PackedMatrix(T(p + "intermediate.dense.weight"), inter, hidden), Padded(T(p + "intermediate.dense.bias")),
                new PackedMatrix(T(p + "output.dense.weight"), hidden, inter), Padded(T(p + "output.dense.bias")),
                T(p + "output.LayerNorm.weight"), T(p + "output.LayerNorm.bias")));
        }

        return new ManagedElectra(T("embeddings.word_embeddings.weight"), positions, types[..hidden], T("embeddings.LayerNorm.weight"),
            T("embeddings.LayerNorm.bias"), [.. layers], hidden, heads, positions.Length / hidden, eps);
    }

    private static float[] Padded(float[] bias) => bias.Length % PackedMatrix.NR == 0 ? bias : [.. bias, .. new float[PackedMatrix.NR - bias.Length % PackedMatrix.NR]];

    /// <summary>
    /// Runs every sentence's ids (at most <see cref="MaxPositions"/>) and returns, per sentence, hidden_states[L] for each L
    /// in <paramref name="keep"/> (0 = the embeddings, L = after layer L), each [pieces, Hidden].
    /// </summary>
    public float[][][] Forward(IReadOnlyList<int[]> sentences, IReadOnlyList<int> keep, CancellationToken ct = default, bool rowInvariant = false)
    {
        int h = Hidden, rows = 0;
        var starts = new int[sentences.Count + 1];
        for (int s = 0; s < sentences.Count; s++)
        {
            if (sentences[s].Length > MaxPositions)
                throw new ArgumentException($"Sentence {s} has {sentences[s].Length} pieces, over {MaxPositions}");
            starts[s] = rows;
            rows += sentences[s].Length;
        }

        starts[^1] = rows;
        int inter = _layers.Length > 0 ? _layers[0].Up.PaddedN : h;
        var x = GC.AllocateUninitializedArray<float>(rows * h, pinned: true);
        var qkv = GC.AllocateUninitializedArray<float>(rows * 3 * h, pinned: true);
        var context = GC.AllocateUninitializedArray<float>(rows * h, pinned: true);
        var tmp = GC.AllocateUninitializedArray<float>(rows * h, pinned: true);
        var up = GC.AllocateUninitializedArray<float>(rows * inter, pinned: true);
        var result = new float[sentences.Count][][];
        for (int s = 0; s < sentences.Count; s++)
            result[s] = new float[keep.Count][];

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

        LayerNorm(x, null, rows, _embG, _embB);
        Keep(0);
        fixed (float* px = x, pq = qkv, pc = context, pt = tmp, pu = up)
        {
            for (int l = 0; l < _layers.Length; l++)
            {
                var layer = _layers[l];
                fixed (float* b1 = layer.QkvBias, b2 = layer.OutBias, b3 = layer.UpBias, b4 = layer.DownBias)
                {
                    Gemm.Run(px, rows, h, layer.Qkv, b1, pq, 3 * h, ct, rowInvariant);
                    Attention(pq, pc, starts);
                    Gemm.Run(pc, rows, h, layer.Out, b2, pt, h, ct, rowInvariant);
                    LayerNorm(x, tmp, rows, layer.Ln1G, layer.Ln1B);
                    Gemm.Run(px, rows, h, layer.Up, b3, pu, inter, ct, rowInvariant);
                    Gelu(up, rows * inter);
                    Gemm.Run(pu, rows, inter, layer.Down, b4, pt, h, ct, rowInvariant);
                    LayerNorm(x, tmp, rows, layer.Ln2G, layer.Ln2B);
                }

                Keep(l + 1);
            }
        }

        return result;

        void Keep(int layer)
        {
            int k = -1;
            for (int i = 0; i < keep.Count; i++)
            {
                if (keep[i] == layer)
                    k = i;
            }

            if (k < 0)
                return;
            for (int s = 0; s < sentences.Count; s++)
                result[s][k] = x.AsSpan(starts[s] * h, (starts[s + 1] - starts[s]) * h).ToArray();
        }
    }

    // context = softmax(q·kᵀ / √d) · v per sentence and head; q, k, v are the thirds of each qkv row.
    private void Attention(float* qkv, float* context, int[] starts)
    {
        int h = Hidden, d = h / Heads, sentences = starts.Length - 1;
        nint pq = (nint)qkv, pc = (nint)context;
        float scale = 1f / MathF.Sqrt(d);
        ManagedThreads.For(sentences * Heads, task =>
        {
            int s = task / Heads, head = task % Heads, r0 = starts[s], n = starts[s + 1] - r0;
            var q_ = (float*)pq;
            var c_ = (float*)pc;
            var p = n <= 512 ? stackalloc double[n] : new double[n];
            var acc = stackalloc double[d];
            for (int i = 0; i < n; i++)
            {
                var qi = new ReadOnlySpan<float>(q_ + (long)(r0 + i) * 3 * h + head * d, d);
                double max = double.NegativeInfinity;
                for (int j = 0; j < n; j++)
                {
                    var kj = new ReadOnlySpan<float>(q_ + (long)(r0 + j) * 3 * h + h + head * d, d);
                    p[j] = (float)Act.Dot(qi, kj) * scale;
                    max = Math.Max(max, p[j]);
                }

                double sum = 0;
                for (int j = 0; j < n; j++)
                    sum += p[j] = Math.Exp(p[j] - max);
                for (int e = 0; e < d; e++)
                    acc[e] = 0;
                for (int j = 0; j < n; j++)
                {
                    var vj = q_ + (long)(r0 + j) * 3 * h + 2 * h + head * d;
                    double pj = p[j];
                    for (int e = 0; e < d; e++)
                        acc[e] += pj * vj[e];
                }

                var ci = c_ + (long)(r0 + i) * h + head * d;
                for (int e = 0; e < d; e++)
                    ci[e] = (float)(acc[e] / sum);
            }
        });
    }

    // x = LayerNorm(x + residual) per row (residual null: x alone), statistics in double.
    private void LayerNorm(float[] x, float[]? residual, int rows, float[] gamma, float[] beta)
    {
        int h = Hidden;
        double eps = _eps;
        ManagedThreads.For((rows + 63) / 64, block =>
        {
            for (int r = block * 64; r < Math.Min(rows, block * 64 + 64); r++)
            {
                var row = x.AsSpan(r * h, h);
                if (residual != null)
                {
                    var add = residual.AsSpan(r * h, h);
                    for (int j = 0; j < h; j++)
                        row[j] = add[j] + row[j];
                }

                double mean = 0, variance = 0;
                for (int j = 0; j < h; j++)
                    mean += row[j];
                mean /= h;
                for (int j = 0; j < h; j++)
                    variance += (row[j] - mean) * (row[j] - mean);
                double inv = 1 / Math.Sqrt(variance / h + eps);
                for (int j = 0; j < h; j++)
                    row[j] = (float)((row[j] - mean) * inv) * gamma[j] + beta[j];
            }
        });
    }

    // Exact GELU, x·Φ(x) = x/2·(1 + erf(x/√2)).
    private static void Gelu(float[] x, int n) =>
        ManagedThreads.For((n + 4095) / 4096, block =>
        {
            for (int i = block * 4096; i < Math.Min(n, block * 4096 + 4096); i++)
                x[i] = (float)(0.5 * x[i] * (1 + Erf(x[i] * 0.70710678118654752)));
        });

    /// <summary>erf in double from Numerical Recipes' erfc (relative error under 1.2e-7, the order of a float ulp).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Erf(double x)
    {
        double z = Math.Abs(x), t = 1 / (1 + 0.5 * z);
        double erfc = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
            + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1 - erfc : erfc - 1;
    }
}
