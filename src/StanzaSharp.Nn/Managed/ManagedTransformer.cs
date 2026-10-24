using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// <see cref="ITransformerNet"/> on the managed kernels (docs/transformer-plan.md). The dense layers are float32
/// <see cref="Gemm"/> runs over every sentence's pieces at once, row-invariant (no single-row kernel), so a sentence's
/// states have the same bits in any call; attention runs per sentence and head without padding. Attention, LayerNorm
/// and GELU are in double and vectorized lane by lane (round 45), so every <see cref="KernelPath"/> and vector width
/// gives them the same bits; only the GEMMs differ by path.
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

    /// <summary>The outputs are rented (round 47): 16 KB a piece with ELECTRA-large, per call.</summary>
    public bool RentsOutputs => true;

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

    /// <summary>Loads the encoder from a transformer's state dict, its weights under <see cref="TransformerConfig.Prefix"/>.</summary>
    public static ManagedTransformer Load(TransformerConfig config, Checkpoint checkpoint)
    {
        var root = checkpoint.Root;
        var prefix = config.Prefix;
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

        long Rows(string name) => checkpoint.Shape(root[prefix + name] ?? throw new InvalidDataException($"No tensor {prefix}{name}"))[0];

        // Position rows from PositionOffset on (RoBERTa numbers a sentence's pieces from 2), token type 0 only.
        var positions = T("embeddings.position_embeddings.weight", config.MaxPositions + config.PositionOffset, h)[(config.PositionOffset * h)..];
        return new ManagedTransformer(config, T("embeddings.word_embeddings.weight", Rows("embeddings.word_embeddings.weight"), h), positions,
            T("embeddings.token_type_embeddings.weight", Rows("embeddings.token_type_embeddings.weight"), h)[..h], T("embeddings.LayerNorm.weight", h),
            T("embeddings.LayerNorm.bias", h), layers);
    }

    private static float[] Padded(float[] bias) => bias.Length % PackedMatrix.NR == 0 ? bias : [.. bias, .. new float[PackedMatrix.NR - (bias.Length % PackedMatrix.NR)]];

    public float[][] Encode(IReadOnlyList<int[]> sentences, int lastLayers, CancellationToken ct = default) =>
        EncodeGroups([(this, sentences)], lastLayers, ct)[0];

    /// <summary>Every net made from one loaded transformer (it and its <see cref="WithLora"/> nets) encodes in one sweep.</summary>
    public bool EncodesJointly => true;

    public float[][][]? EncodeJoint(IReadOnlyList<(ITransformerNet Net, IReadOnlyList<int[]> Sentences)> groups, int lastLayers, CancellationToken ct = default)
    {
        var nets = groups.Select(g => g.Net as ManagedTransformer).ToList();
        return nets.All(n => n != null && ReferenceEquals(n._layers, _layers))
            ? EncodeGroups([.. groups.Select((g, i) => (nets[i]!, g.Sentences))], lastLayers, ct)
            : null;
    }

    // One sweep over the layers for every group's sentences: the base layers run over all rows (row-invariant, so a row's
    // bits don't depend on the others), and each group's adapters over its own rows only, in WithLora's order (base and
    // bias, then the adapter), so each sentence gets the bits its own net gives it alone.
    private float[][][] EncodeGroups(IReadOnlyList<(ManagedTransformer Net, IReadOnlyList<int[]> Sentences)> groups, int lastLayers, CancellationToken ct)
    {
        if (lastLayers < 1 || lastLayers > _layers.Length + 1)
            throw new ArgumentOutOfRangeException(nameof(lastLayers));
        var sentences = groups.SelectMany(g => g.Sentences).ToList();
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

        // Each group's first row and row count, and its sentences' place in the flat list.
        var firstRow = new int[groups.Count];
        var groupRows = new int[groups.Count];
        for (int g = 0, first = 0; g < groups.Count; first += groups[g++].Sentences.Count)
        {
            firstRow[g] = starts[first];
            groupRows[g] = starts[first + groups[g].Sentences.Count] - starts[first];
        }

        var pool = ArrayPool<float>.Shared;
        var flat = new float[sentences.Count][];
        for (int s = 0; s < sentences.Count; s++)
            flat[s] = pool.Rent(lastLayers * (starts[s + 1] - starts[s]) * h);
        var result = new float[groups.Count][][];
        for (int g = 0, first = 0; g < groups.Count; first += groups[g++].Sentences.Count)
            result[g] = flat[first..(first + groups[g].Sentences.Count)];
        if (rows == 0)
            return result;

        float[] x = pool.Rent(rows * h), qkv = pool.Rent(rows * 3 * h), context = pool.Rent(rows * h), tmp = pool.Rent(rows * h), up = pool.Rent(rows * inter);
        int rank = groups.SelectMany(g => g.Net._adapters ?? []).SelectMany(a => a.Qkv.Concat([a.Out, a.Up, a.Down])).OfType<Adapter>()
            .Select(a => a.A.PaddedN).DefaultIfEmpty(0).Max();
        bool lora = rank > 0;
        float[]? low = lora ? pool.Rent(rows * rank) : null, delta = lora ? pool.Rent(rows * inter) : null;
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

                    // adapter(input, its row stride, output, its row stride) for every group that has one at this point.
                    void Adapt(Func<LayerAdapters, Adapter?> pick, float* input, int k, float* output, int ldOutput)
                    {
                        for (int g = 0; g < groups.Count; g++)
                        {
                            if (groups[g].Net._adapters?[l] is { } adapters && groupRows[g] > 0)
                            {
                                AddLora(pick(adapters), input + ((long)firstRow[g] * k), groupRows[g], k, output + ((long)firstRow[g] * ldOutput), ldOutput,
                                    groups[g].Net._scaling, low, delta, ct);
                            }
                        }
                    }

                    fixed (float* b1 = layer.QkvBias, b2 = layer.OutBias, b3 = layer.UpBias, b4 = layer.DownBias)
                    {
                        Gemm.Run(px, rows, h, layer.Qkv, b1, pq, 3 * h, ct, rowInvariant: true);
                        for (int q = 0; q < 2; q++)
                            Adapt(a => q < a.Qkv.Length ? a.Qkv[q] : null, px, h, pq, 3 * h);
                        Attention(pq, pc, starts);
                        Gemm.Run(pc, rows, h, layer.Out, b2, pt, h, ct, rowInvariant: true);
                        Adapt(a => a.Out, pc, h, pt, h);
                        LayerNorm(px, pt, rows, layer.Ln1G, layer.Ln1B);
                        Gemm.Run(px, rows, h, layer.Up, b3, pu, inter, ct, rowInvariant: true);
                        Adapt(a => a.Up, px, h, pu, inter);
                        Gelu(pu, rows * inter);
                        Gemm.Run(pu, rows, inter, layer.Down, b4, pt, h, ct, rowInvariant: true);
                        Adapt(a => a.Down, pu, inter, pt, h);
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
                new ReadOnlySpan<float>(states + ((long)starts[s] * h), n * h).CopyTo(flat[s].AsSpan(k * n * h, n * h));
            }
        }
    }

    // peft's LoRA Linear in eval mode: output += B(A(input))·scaling, after the base layer's bias (float32, B's product
    // scaled, then added, as torch does). Both GEMMs are row-invariant like the base layers.
    private static void AddLora(Adapter? adapter, float* input, int rows, int k, float* output, int ldOutput, float scaling, float[]? low, float[]? delta,
        CancellationToken ct)
    {
        if (adapter == null)
            return;
        int rank = adapter.A.PaddedN, ld = adapter.B.PaddedN, n = adapter.B.N, column = adapter.Column;
        fixed (float* pl = low, pd = delta)
        {
            Gemm.Run(input, rows, k, adapter.A, null, pl, rank, ct, rowInvariant: true);
            Gemm.Run(pl, rows, rank, adapter.B, null, pd, ld, ct, rowInvariant: true);
            nint po = (nint)output, pdd = (nint)pd;
            ManagedThreads.For((rows + 63) / 64, block =>
            {
                for (int r = block * 64; r < Math.Min(rows, (block * 64) + 64); r++)
                    AddScaled((float*)po + ((long)r * ldOutput) + column, (float*)pdd + ((long)r * ld), n, scaling);
            });
        }
    }

    // o[j] += d[j]·scaling for j < n: a multiply then an add per lane (never fused), so every width gives the scalar
    // loop's bits (round 49: vectorized).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AddScaled(float* o, float* d, int n, float scaling)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var s = new Vector<float>(scaling);
            for (; j + Vector<float>.Count <= n; j += Vector<float>.Count)
                (Vector.Load(o + j) + (Vector.Load(d + j) * s)).Store(o + j);
        }

        for (; j < n; j++)
            o[j] += d[j] * scaling;
    }

    // context = softmax(q·kᵀ / √d) · v per sentence and head; q, k, v are the thirds of each qkv row. Scores and sums in
    // double; a task's arithmetic doesn't depend on how many threads run the tasks. Blocked (round 49): a head's keys and
    // values widened to double once, four query rows at a time through register-blocked kernels (ScoreBlock,
    // WeightedBlock) that give every element Act.Dot's and WeightedAdd's own operations in their order, so the bits are
    // the per-row loop's (kept for the Scalar path and a head width not a multiple of 8).
    private void Attention(float* qkv, float* context, int[] starts)
    {
        int h = Config.Hidden, heads = Config.Heads, d = h / heads, sentences = starts.Length - 1;
        nint pq = (nint)qkv, pc = (nint)context;
        double scale = 1 / Math.Sqrt(d);
        bool blocked = d % 8 == 0 && Gemm.Path != KernelPath.Scalar;
        ManagedThreads.For(sentences * heads, task =>
        {
            int s = task / heads, head = task % heads, r0 = starts[s], n = starts[s + 1] - r0;
            if (blocked)
                AttentionBlocked((float*)pq + ((long)r0 * 3 * h) + (head * d), (float*)pc + ((long)r0 * h) + (head * d), n, h, d, scale);
            else
                AttentionRows((float*)pq + ((long)r0 * 3 * h) + (head * d), (float*)pc + ((long)r0 * h) + (head * d), n, h, d, scale);
        });
    }

    // One sentence's head, a query row at a time: q at qkv, k at qkv + h, v at qkv + 2h (row stride 3h); context row
    // stride h.
    internal static void AttentionRows(float* qkv, float* context, int n, int h, int d, double scale)
    {
        var p = ArrayPool<double>.Shared.Rent(n + d);
        try
        {
            var acc = p.AsSpan(n, d);
            for (int i = 0; i < n; i++)
            {
                var qi = new ReadOnlySpan<float>(qkv + ((long)i * 3 * h), d);
                double max = double.NegativeInfinity;
                for (int j = 0; j < n; j++)
                {
                    var kj = new ReadOnlySpan<float>(qkv + ((long)j * 3 * h) + h, d);
                    p[j] = Act.Dot(qi, kj) * scale;
                    max = Math.Max(max, p[j]);
                }

                double sum = 0;
                for (int j = 0; j < n; j++)
                    sum += p[j] = Math.Exp(p[j] - max);
                acc.Clear();
                for (int j = 0; j < n; j++)
                    WeightedAdd(acc, p[j], qkv + ((long)j * 3 * h) + (2 * h));

                var ci = context + ((long)i * h);
                for (int e = 0; e < d; e++)
                    ci[e] = (float)(acc[e] / sum);
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(p);
        }
    }

    private const int QueryBlock = 4;

    // AttentionRows' arithmetic, QueryBlock query rows at a time (the last block's missing rows are zeros, computed and
    // dropped). Per row and key: the dot's four accumulators (lane q mod 4, as Act.Dot; a widened float product is exact
    // in double, so a fused multiply-add rounds like Dot's add), then (s0 + s1) + (s2 + s3), times scale; the maximum and
    // the sum of exponentials in key order; each context element summed over the keys in order, a multiply then an add
    // (WeightedAdd's), then divided by the sum.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void AttentionBlocked(float* qkv, float* context, int n, int h, int d, double scale)
    {
        var pool = ArrayPool<double>.Shared;
        var buffer = pool.Rent((2 * n * d) + (QueryBlock * (n + d)));
        try
        {
            // The rented buffer only through spans (no pointer arithmetic on it), each part pinned on its own.
            Span<double> keys = buffer.AsSpan(0, n * d), values = buffer.AsSpan(n * d, n * d), queries = buffer.AsSpan(2 * n * d, QueryBlock * d),
                scores = buffer.AsSpan((2 * n * d) + (QueryBlock * d), QueryBlock * n);
            for (int j = 0; j < n; j++)
            {
                Widen(qkv + ((long)j * 3 * h) + h, keys.Slice(j * d, d));
                Widen(qkv + ((long)j * 3 * h) + (2 * h), values.Slice(j * d, d));
            }

            Span<double> max = stackalloc double[QueryBlock], sum = stackalloc double[QueryBlock];
            fixed (double* k = keys, v = values, q = queries, p = scores)
            {
                for (int i0 = 0; i0 < n; i0 += QueryBlock)
                {
                    int rows = Math.Min(QueryBlock, n - i0);
                    for (int r = 0; r < QueryBlock; r++)
                    {
                        if (r < rows)
                            Widen(qkv + ((long)(i0 + r) * 3 * h), queries.Slice(r * d, d));
                        else
                            queries.Slice(r * d, d).Clear();
                    }

                    ScoreBlock(q, k, p, n, d, scale);
                    for (int r = 0; r < QueryBlock; r++)
                    {
                        double m = double.NegativeInfinity;
                        foreach (double x in scores.Slice(r * n, n))
                            m = Math.Max(m, x);
                        max[r] = m;
                        sum[r] = 0;
                    }

                    for (int j = 0; j < n; j++)
                    {
                        for (int r = 0; r < QueryBlock; r++)
                            sum[r] += scores[(r * n) + j] = Math.Exp(scores[(r * n) + j] - max[r]);
                    }

                    WeightedBlock(p, v, n, d, sum, context + ((long)i0 * h), h, rows);
                }
            }
        }
        finally
        {
            pool.Return(buffer);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Widen(float* source, Span<double> target)
    {
        for (int e = 0; e < target.Length; e++)
            target[e] = source[e];
    }

    // p[r·n + j] = dot(q_r, k_j)·scale for the QueryBlock rows of q [QueryBlock, d] and every key row of k [n, d].
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void ScoreBlock(double* q, double* k, double* p, int n, int d, double scale)
    {
        double* q0 = q, q1 = q + d, q2 = q + (2 * d), q3 = q + (3 * d);
        if (Gemm.Path is KernelPath.Vector512 or KernelPath.Vector256)
        {
            int j = 0;
            for (; j + 2 <= n; j += 2)
            {
                double* kj = k + ((long)j * d), kl = kj + d;
                Vector256<double> a0 = Vector256<double>.Zero, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                for (int c = 0; c < d; c += 4)
                {
                    var (x, y) = (Vector256.Load(kj + c), Vector256.Load(kl + c));
                    var w = Vector256.Load(q0 + c);
                    (a0, b0) = (Vector256.MultiplyAddEstimate(w, x, a0), Vector256.MultiplyAddEstimate(w, y, b0));
                    w = Vector256.Load(q1 + c);
                    (a1, b1) = (Vector256.MultiplyAddEstimate(w, x, a1), Vector256.MultiplyAddEstimate(w, y, b1));
                    w = Vector256.Load(q2 + c);
                    (a2, b2) = (Vector256.MultiplyAddEstimate(w, x, a2), Vector256.MultiplyAddEstimate(w, y, b2));
                    w = Vector256.Load(q3 + c);
                    (a3, b3) = (Vector256.MultiplyAddEstimate(w, x, a3), Vector256.MultiplyAddEstimate(w, y, b3));
                }

                (p[j], p[j + 1]) = (Sum(a0) * scale, Sum(b0) * scale);
                (p[n + j], p[n + j + 1]) = (Sum(a1) * scale, Sum(b1) * scale);
                (p[(2 * n) + j], p[(2 * n) + j + 1]) = (Sum(a2) * scale, Sum(b2) * scale);
                (p[(3 * n) + j], p[(3 * n) + j + 1]) = (Sum(a3) * scale, Sum(b3) * scale);
            }

            for (; j < n; j++)
            {
                var kj = k + ((long)j * d);
                Vector256<double> a0 = Vector256<double>.Zero, a1 = a0, a2 = a0, a3 = a0;
                for (int c = 0; c < d; c += 4)
                {
                    var x = Vector256.Load(kj + c);
                    a0 = Vector256.MultiplyAddEstimate(Vector256.Load(q0 + c), x, a0);
                    a1 = Vector256.MultiplyAddEstimate(Vector256.Load(q1 + c), x, a1);
                    a2 = Vector256.MultiplyAddEstimate(Vector256.Load(q2 + c), x, a2);
                    a3 = Vector256.MultiplyAddEstimate(Vector256.Load(q3 + c), x, a3);
                }

                p[j] = Sum(a0) * scale;
                p[n + j] = Sum(a1) * scale;
                p[(2 * n) + j] = Sum(a2) * scale;
                p[(3 * n) + j] = Sum(a3) * scale;
            }
        }
        else
        {
            for (int j = 0; j < n; j++)
            {
                var kj = k + ((long)j * d);
                Vector128<double> a01 = Vector128<double>.Zero, a23 = a01, b01 = a01, b23 = a01, c01 = a01, c23 = a01, e01 = a01, e23 = a01;
                for (int c = 0; c < d; c += 4)
                {
                    var (x01, x23) = (Vector128.Load(kj + c), Vector128.Load(kj + c + 2));
                    a01 = Vector128.MultiplyAddEstimate(Vector128.Load(q0 + c), x01, a01);
                    a23 = Vector128.MultiplyAddEstimate(Vector128.Load(q0 + c + 2), x23, a23);
                    b01 = Vector128.MultiplyAddEstimate(Vector128.Load(q1 + c), x01, b01);
                    b23 = Vector128.MultiplyAddEstimate(Vector128.Load(q1 + c + 2), x23, b23);
                    c01 = Vector128.MultiplyAddEstimate(Vector128.Load(q2 + c), x01, c01);
                    c23 = Vector128.MultiplyAddEstimate(Vector128.Load(q2 + c + 2), x23, c23);
                    e01 = Vector128.MultiplyAddEstimate(Vector128.Load(q3 + c), x01, e01);
                    e23 = Vector128.MultiplyAddEstimate(Vector128.Load(q3 + c + 2), x23, e23);
                }

                p[j] = ((a01[0] + a01[1]) + (a23[0] + a23[1])) * scale;
                p[n + j] = ((b01[0] + b01[1]) + (b23[0] + b23[1])) * scale;
                p[(2 * n) + j] = ((c01[0] + c01[1]) + (c23[0] + c23[1])) * scale;
                p[(3 * n) + j] = ((e01[0] + e01[1]) + (e23[0] + e23[1])) * scale;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double Sum(Vector256<double> s) => (s[0] + s[1]) + (s[2] + s[3]);

    // context row r (stride h), element e = (float)((Σ_j p[r·n + j]·v[j·d + e]) / sum[r]) for the first rows rows: the
    // sum from 0 over j in order, each term multiplied, then added.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void WeightedBlock(double* p, double* v, int n, int d, ReadOnlySpan<double> sum, float* context, int h, int rows)
    {
        double* p0 = p, p1 = p + n, p2 = p + (2 * n), p3 = p + (3 * n);
        Span<double> acc = stackalloc double[QueryBlock * 8];
        for (int e = 0; e < d; e += 8)
        {
            if (Gemm.Path is KernelPath.Vector512 or KernelPath.Vector256)
            {
                Vector256<double> a0 = Vector256<double>.Zero, b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
                for (int j = 0; j < n; j++)
                {
                    var vj = v + ((long)j * d) + e;
                    var (x, y) = (Vector256.Load(vj), Vector256.Load(vj + 4));
                    var w = Vector256.Create(p0[j]);
                    (a0, b0) = (a0 + (w * x), b0 + (w * y));
                    w = Vector256.Create(p1[j]);
                    (a1, b1) = (a1 + (w * x), b1 + (w * y));
                    w = Vector256.Create(p2[j]);
                    (a2, b2) = (a2 + (w * x), b2 + (w * y));
                    w = Vector256.Create(p3[j]);
                    (a3, b3) = (a3 + (w * x), b3 + (w * y));
                }

                fixed (double* pa = acc)
                {
                    a0.Store(pa);
                    b0.Store(pa + 4);
                    a1.Store(pa + 8);
                    b1.Store(pa + 12);
                    a2.Store(pa + 16);
                    b2.Store(pa + 20);
                    a3.Store(pa + 24);
                    b3.Store(pa + 28);
                }
            }
            else
            {
                // Two rows at a time: eight accumulators of two lanes.
                for (int r = 0; r < QueryBlock; r += 2)
                {
                    double* pr = p + ((long)r * n), ps = pr + n;
                    Vector128<double> a0 = Vector128<double>.Zero, a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                    for (int j = 0; j < n; j++)
                    {
                        var vj = v + ((long)j * d) + e;
                        var (x0, x1, x2, x3) = (Vector128.Load(vj), Vector128.Load(vj + 2), Vector128.Load(vj + 4), Vector128.Load(vj + 6));
                        var w = Vector128.Create(pr[j]);
                        (a0, a1, a2, a3) = (a0 + (w * x0), a1 + (w * x1), a2 + (w * x2), a3 + (w * x3));
                        w = Vector128.Create(ps[j]);
                        (b0, b1, b2, b3) = (b0 + (w * x0), b1 + (w * x1), b2 + (w * x2), b3 + (w * x3));
                    }

                    fixed (double* pa = acc)
                    {
                        double* ra = pa + (r * 8), rb = ra + 8;
                        a0.Store(ra);
                        a1.Store(ra + 2);
                        a2.Store(ra + 4);
                        a3.Store(ra + 6);
                        b0.Store(rb);
                        b1.Store(rb + 2);
                        b2.Store(rb + 4);
                        b3.Store(rb + 6);
                    }
                }
            }

            for (int r = 0; r < rows; r++)
            {
                var ci = context + ((long)r * h) + e;
                for (int t = 0; t < 8; t++)
                    ci[t] = (float)(acc[(r * 8) + t] / sum[r]);
            }
        }
    }

    // acc[e] += weight · v[e] for e < acc.Length, in double: a multiply then an add per lane (never fused), so every
    // vector width gives the scalar loop's bits.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void WeightedAdd(Span<double> acc, double weight, float* v)
    {
        int e = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int lanes = Vector<float>.Count;
            var w = new Vector<double>(weight);
            ref double a0 = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(acc);
            for (; e + lanes <= acc.Length; e += lanes)
            {
                Vector.Widen(Vector.Load(v + e), out var lo, out var hi);
                ref double a = ref Unsafe.Add(ref a0, e);
                var accLo = Vector.LoadUnsafe(ref a);
                var accHi = Vector.LoadUnsafe(ref a, (nuint)Vector<double>.Count);
                Vector.StoreUnsafe(accLo + (w * lo), ref a);
                Vector.StoreUnsafe(accHi + (w * hi), ref a, (nuint)Vector<double>.Count);
            }
        }

        for (; e < acc.Length; e++)
            acc[e] += weight * v[e];
    }

    // x = LayerNorm(x + residual) per row (residual null: x alone), statistics in double with four accumulators each
    // (LayerNormRow), so every path gives the same bits.
    private void LayerNorm(float* x, float* residual, int rows, float[] gamma, float[] beta)
    {
        int h = Config.Hidden;
        double eps = Config.LayerNormEps;
        nint px = (nint)x, pr = (nint)residual;
        ManagedThreads.For((rows + 63) / 64, block =>
        {
            fixed (float* g = gamma, b = beta)
            {
                for (int r = block * 64; r < Math.Min(rows, (block * 64) + 64); r++)
                    LayerNormRow((float*)px + ((long)r * h), pr == 0 ? null : (float*)pr + ((long)r * h), h, g, b, eps);
            }
        });
    }

    // One row: row += add (float), mean = Σ row / n and variance = Σ (row − mean)² / n in double, each sum over four
    // accumulators s0..s3 (q mod 4, the tail into s0, then (s0 + s1) + (s2 + s3), as Act.Dot), whose SIMD lanes are the
    // accumulators; then row = (float)((row − mean)·inv)·γ + β. Every step is the same per lane on every path.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void LayerNormRow(float* row, float* add, int n, float* gamma, float* beta, double eps)
    {
        bool wide = Gemm.Path is KernelPath.Vector512 or KernelPath.Vector256, narrow = Gemm.Path == KernelPath.Vector128;
        int q = 0;
        if (add != null)
        {
            if (wide || narrow)
            {
                for (; q <= n - 4; q += 4)
                    (Vector128.Load(add + q) + Vector128.Load(row + q)).Store(row + q);
            }

            for (; q < n; q++)
                row[q] = add[q] + row[q];
        }

        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        q = 0;
        if (wide)
        {
            var s = Vector256<double>.Zero;
            for (; q <= n - 4; q += 4)
                s += Vector256.WidenLower(Vector128.Load(row + q).ToVector256Unsafe());
            (s0, s1, s2, s3) = (s[0], s[1], s[2], s[3]);
        }
        else if (narrow)
        {
            Vector128<double> s01 = Vector128<double>.Zero, s23 = Vector128<double>.Zero;
            for (; q <= n - 4; q += 4)
            {
                var x = Vector128.Load(row + q);
                s01 += Vector128.WidenLower(x);
                s23 += Vector128.WidenUpper(x);
            }

            (s0, s1, s2, s3) = (s01[0], s01[1], s23[0], s23[1]);
        }
        else
        {
            for (; q <= n - 4; q += 4)
            {
                s0 += row[q];
                s1 += row[q + 1];
                s2 += row[q + 2];
                s3 += row[q + 3];
            }
        }

        for (; q < n; q++)
            s0 += row[q];
        double mean = ((s0 + s1) + (s2 + s3)) / n;

        (s0, s1, s2, s3) = (0, 0, 0, 0);
        q = 0;
        if (wide)
        {
            Vector256<double> s = Vector256<double>.Zero, m = Vector256.Create(mean);
            for (; q <= n - 4; q += 4)
            {
                var d = Vector256.WidenLower(Vector128.Load(row + q).ToVector256Unsafe()) - m;
                s += d * d;
            }

            (s0, s1, s2, s3) = (s[0], s[1], s[2], s[3]);
        }
        else if (narrow)
        {
            Vector128<double> s01 = Vector128<double>.Zero, s23 = Vector128<double>.Zero, m = Vector128.Create(mean);
            for (; q <= n - 4; q += 4)
            {
                var x = Vector128.Load(row + q);
                var (lo, hi) = (Vector128.WidenLower(x) - m, Vector128.WidenUpper(x) - m);
                s01 += lo * lo;
                s23 += hi * hi;
            }

            (s0, s1, s2, s3) = (s01[0], s01[1], s23[0], s23[1]);
        }
        else
        {
            for (; q <= n - 4; q += 4)
            {
                double d0 = row[q] - mean, d1 = row[q + 1] - mean, d2 = row[q + 2] - mean, d3 = row[q + 3] - mean;
                s0 += d0 * d0;
                s1 += d1 * d1;
                s2 += d2 * d2;
                s3 += d3 * d3;
            }
        }

        for (; q < n; q++)
            s0 += (row[q] - mean) * (row[q] - mean);
        double inv = 1 / Math.Sqrt((((s0 + s1) + (s2 + s3)) / n) + eps);

        q = 0;
        if (wide || narrow)
        {
            Vector128<double> m = Vector128.Create(mean), iv = Vector128.Create(inv);
            for (; q <= n - 4; q += 4)
            {
                var x = Vector128.Load(row + q);
                var y = Vector128.Narrow((Vector128.WidenLower(x) - m) * iv, (Vector128.WidenUpper(x) - m) * iv);
                ((y * Vector128.Load(gamma + q)) + Vector128.Load(beta + q)).Store(row + q);
            }
        }

        for (; q < n; q++)
            row[q] = ((float)((row[q] - mean) * inv) * gamma[q]) + beta[q];
    }

    // Exact GELU, x·Φ(x) = x/2·(1 + erf(x/√2)), in double; lanes of every vector width and the scalar tail run the same
    // operations (GeluLanes / Gelu(double)), so a value's result doesn't depend on the CPU.
    private static void Gelu(float* x, int n)
    {
        nint px = (nint)x;
        ManagedThreads.For((n + 4095) / 4096, block => GeluBlock((float*)px + (block * 4096), Math.Min(4096, n - (block * 4096))));
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void GeluBlock(float* v, int n)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
            {
                Vector.Widen(Vector.Load(v + i), out var lo, out var hi);
                Vector.Narrow(GeluLanes(lo), GeluLanes(hi)).Store(v + i);
            }
        }

        for (; i < n; i++)
            v[i] = (float)Gelu((double)v[i]);
    }

    /// <summary>GELU of one value: <see cref="GeluLanes"/>'s operations on a scalar.</summary>
    internal static double Gelu(double x)
    {
        double z = Math.Abs(x * 0.70710678118654752), t = 1 / (1 + (0.5 * z));
        double erfc = t * Exp((-z * z) - 1.26551223 + (t * ErfPoly(t)));
        double erf = x >= 0 ? 1 - erfc : erfc - 1;
        return 0.5 * x * (1 + erf);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> GeluLanes(Vector<double> x)
    {
        var z = Vector.Abs(x * new Vector<double>(0.70710678118654752));
        var t = Vector<double>.One / (Vector<double>.One + (new Vector<double>(0.5) * z));
        var erfc = t * Exp((-z * z) - new Vector<double>(1.26551223) + (t * ErfPoly(t)));
        var erf = Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, Vector<double>.Zero), Vector<double>.One - erfc, erfc - Vector<double>.One);
        return new Vector<double>(0.5) * x * (Vector<double>.One + erf);
    }

    // Numerical Recipes' erfc polynomial (relative error under 1.2e-7, the order of a float ulp), Horner in t.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ErfPoly(double t)
    {
        double poly = -0.82215223 + (t * 0.17087277);
        poly = 1.48851587 + (t * poly);
        poly = -1.13520398 + (t * poly);
        poly = 0.27886807 + (t * poly);
        poly = -0.18628806 + (t * poly);
        poly = 0.09678418 + (t * poly);
        poly = 0.37409196 + (t * poly);
        return 1.00002368 + (t * poly);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> ErfPoly(Vector<double> t)
    {
        var poly = new Vector<double>(-0.82215223) + (t * new Vector<double>(0.17087277));
        poly = new Vector<double>(1.48851587) + (t * poly);
        poly = new Vector<double>(-1.13520398) + (t * poly);
        poly = new Vector<double>(0.27886807) + (t * poly);
        poly = new Vector<double>(-0.18628806) + (t * poly);
        poly = new Vector<double>(0.09678418) + (t * poly);
        poly = new Vector<double>(0.37409196) + (t * poly);
        return new Vector<double>(1.00002368) + (t * poly);
    }

    // e^x for x ≤ 0 (the erfc argument; below -708 it is clamped, giving about 1e-308): x = n·ln 2 + r with n rounded to
    // nearest by adding and subtracting 1.5·2^52, then a degree-12 Taylor polynomial in |r| ≤ ln 2 / 2 (truncation under
    // 1e-16) times 2^n. Only multiplies and adds, never fused, so the scalar and every vector width give the same bits.
    private const double Log2E = 1.4426950408889634, Ln2Hi = 0.693147180369123816490, Ln2Lo = 1.90821492927058770002e-10,
        RoundMagic = 6755399441055744.0;

    private static readonly double[] ExpCoefficients = [1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664, 0.008333333333333333, 0.001388888888888889, 0.0001984126984126984, 2.48015873015873e-05, 2.7557319223985893e-06, 2.755731922398589e-07, 2.505210838544172e-08, 2.08767569878681e-09];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static double Exp(double x)
    {
        x = Math.Max(x, -708.0);
        double n = ((x * Log2E) + RoundMagic) - RoundMagic;
        double r = (x - (n * Ln2Hi)) - (n * Ln2Lo);
        double p = ExpCoefficients[12];
        for (int i = 11; i >= 0; i--)
            p = ExpCoefficients[i] + (r * p);
        return p * BitConverter.Int64BitsToDouble(((long)n + 1023) << 52);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> Exp(Vector<double> x)
    {
        x = Vector.Max(x, new Vector<double>(-708.0));
        var n = ((x * new Vector<double>(Log2E)) + new Vector<double>(RoundMagic)) - new Vector<double>(RoundMagic);
        var r = (x - (n * new Vector<double>(Ln2Hi))) - (n * new Vector<double>(Ln2Lo));
        var p = new Vector<double>(ExpCoefficients[12]);
        for (int i = 11; i >= 0; i--)
            p = new Vector<double>(ExpCoefficients[i]) + (r * p);
        var scale = Vector.ShiftLeft(Vector.ConvertToInt64(n) + new Vector<long>(1023), 52);
        return p * Vector.AsVectorDouble(scale);
    }

    public void Dispose()
    {
    }
}
