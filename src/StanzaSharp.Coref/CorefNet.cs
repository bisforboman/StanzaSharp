using System.Buffers;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Coref;

/// <summary>
/// The coref head's networks (stanza/models/coref), arrays in and out: <see cref="ManagedCorefNet"/> and <c>CorefNet</c>
/// (StanzaSharp.Cuda). Vectors are row-major with the transformer's hidden size H per row.
/// </summary>
internal interface ICorefNet : IDisposable
{
    /// <summary>WordEncoder: each word the softmax-weighted sum of its subwords [starts[i], ends[i]) (attn Linear H → 1), [n, H].</summary>
    float[] Words(float[] subwords, int count, int[] starts, int[] ends);

    /// <summary>
    /// RoughScorer's bilinear scores (W·m_i + b)·m_j of every pair, a block of rows at a time: <paramref name="rows"/>(first,
    /// count, scores, ld) with row r's n scores at scores[r·ld..] (the callee may overwrite them).
    /// </summary>
    void RoughScores(float[] words, int n, Action<int, int, float[], int> rows, CancellationToken ct);

    /// <summary>
    /// AnaphoricityScorer._ffnn's pair scores (before the rough score is added): for word i and its c-th candidate
    /// top[i·k + c], out(LeakyReLU(hidden([a, b, a·b, distance_emb[bucket]]))), [n, k].
    /// </summary>
    float[] Anaphoricity(float[] words, int n, int[] top, int[] buckets, int k, CancellationToken ct);

    /// <summary>
    /// SpanPredictor.forward's start and end scores, [heads, width] each: head h's candidates are the words of its sentence
    /// [sentStart[h], sentStart[h] + sentLength[h]), each input [head, word, emb(head − word)] through the FFNN; the rows
    /// past a sentence's length hold ffnn(0) (Stanza pads with zeros), then both convolutions run over all
    /// <paramref name="width"/> positions. Positions past a sentence's length are left unspecified.
    /// </summary>
    (float[] Start, float[] End) SpanScores(float[] words, int n, int[] heads, int[] sentStart, int[] sentLength, int width, CancellationToken ct);

    /// <summary>The zeros predictor (Linear, ReLU, Linear → 1) on the words <paramref name="rows"/>.</summary>
    float[] Zeros(float[] words, int[] rows);
}

/// <summary>
/// <see cref="ICorefNet"/> on the managed kernels (<see cref="Gemm"/>). The two concatenated inputs are split by their parts:
/// the anaphoricity layer's W·[a, b, a·b, pw] as Wa·a + Wb·b per word, Wp·pw per bucket and Wc·(a·b) per pair, and the span
/// predictor's first layer as per-head, per-word and per-distance parts, so only the products are per pair (a different
/// summation order than torch's one GEMM, as any managed GEMM has).
/// </summary>
internal sealed unsafe class ManagedCorefNet : ICorefNet
{
    private const int RoughChunk = 256, PairChunk = 128, HeadChunk = 32;
    private const float LeakySlope = 0.01f; // torch.nn.LeakyReLU's default

    private readonly int _hidden;
    private readonly float[] _attnWeight;
    private readonly float _attnBias;
    private readonly PackedMatrix _bilinear;
    private readonly float[] _bilinearBias;
    private readonly PackedMatrix _pairA, _pairB, _pairProduct;
    private readonly float[] _pairDistance; // [9, H]: Wp·distance_emb + the hidden layer's bias
    private readonly float[] _pairOut;
    private readonly float _pairOutBias;
    private readonly PackedMatrix _spanHead, _spanWord, _span3, _span6;
    private readonly float[] _spanDistance; // [128, H]: W·emb + the first layer's bias
    private readonly float[] _span3Bias, _span6Bias;
    private readonly float[] _spanPadding; // ffnn(0)
    private readonly float[] _conv0, _conv0Bias, _conv1, _conv1Bias;
    private readonly int _conv0Out, _conv1Out, _spanOut;
    private readonly PackedMatrix _zeros0;
    private readonly float[] _zeros0Bias, _zerosOut;
    private readonly float _zerosOutBias;

    public ManagedCorefNet(Checkpoint ckpt)
    {
        var root = ckpt.Root;
        float[] T(string module, string key) => ckpt.Tensor<float>(root[module]![key] ?? throw new InvalidDataException($"No {module}.{key}"));
        int[] Shape(string module, string key) => [.. ckpt.Shape(root[module]![key]).Select(d => (int)d)];

        _attnWeight = T("we", "attn.weight");
        _hidden = _attnWeight.Length;
        _attnBias = T("we", "attn.bias")[0];
        _bilinear = new PackedMatrix(T("rough_scorer", "bilinear.weight"), _hidden, _hidden);
        _bilinearBias = Pad(T("rough_scorer", "bilinear.bias"), _bilinear.PaddedN);

        int h = _hidden;
        var hidden = T("a_scorer", "hidden.0.weight");
        int inputs = Shape("a_scorer", "hidden.0.weight")[1], outputs = Shape("a_scorer", "hidden.0.weight")[0];
        var distance = T("pw", "distance_emb.weight");
        int emb = Shape("pw", "distance_emb.weight")[1], buckets = Shape("pw", "distance_emb.weight")[0];
        if (inputs != (3 * h) + emb)
            throw new InvalidDataException($"The anaphoricity layer reads {inputs} inputs, not 3 x {h} + {emb}");
        _pairA = Columns(hidden, outputs, inputs, 0, h);
        _pairB = Columns(hidden, outputs, inputs, h, h);
        _pairProduct = Columns(hidden, outputs, inputs, 2 * h, h);
        var pairBias = T("a_scorer", "hidden.0.bias");
        _pairDistance = Gemm.Run(distance, buckets, Columns(hidden, outputs, inputs, 3 * h, emb), Pad(pairBias, _pairA.PaddedN));
        _pairOut = T("a_scorer", "out.weight");
        _pairOutBias = T("a_scorer", "out.bias")[0];

        var span0 = T("sp", "ffnn.0.weight");
        int spanInputs = Shape("sp", "ffnn.0.weight")[1], spanHidden = Shape("sp", "ffnn.0.weight")[0];
        var spanEmb = T("sp", "emb.weight");
        int spanEmbDim = Shape("sp", "emb.weight")[1], distances = Shape("sp", "emb.weight")[0];
        _spanHead = Columns(span0, spanHidden, spanInputs, 0, h);
        _spanWord = Columns(span0, spanHidden, spanInputs, h, h);
        var span0Bias = Pad(T("sp", "ffnn.0.bias"), _spanHead.PaddedN);
        _spanDistance = Gemm.Run(spanEmb, distances, Columns(span0, spanHidden, spanInputs, 2 * h, spanEmbDim), span0Bias);
        int s3 = Shape("sp", "ffnn.3.weight")[0], s6 = Shape("sp", "ffnn.6.weight")[0];
        _span3 = new PackedMatrix(T("sp", "ffnn.3.weight"), s3, spanHidden);
        _span3Bias = Pad(T("sp", "ffnn.3.bias"), _span3.PaddedN);
        _span6 = new PackedMatrix(T("sp", "ffnn.6.weight"), s6, s3);
        _span6Bias = Pad(T("sp", "ffnn.6.bias"), _span6.PaddedN);
        _spanOut = s6;
        _conv0 = T("sp", "conv.0.weight");
        _conv0Bias = T("sp", "conv.0.bias");
        _conv0Out = _conv0Bias.Length;
        _conv1 = T("sp", "conv.1.weight");
        _conv1Bias = T("sp", "conv.1.bias");
        _conv1Out = _conv1Bias.Length;
        if (_conv1Out != 2 || Shape("sp", "conv.0.weight")[2] != 3 || Shape("sp", "conv.1.weight")[2] != 3)
            throw new NotSupportedException("The span predictor's convolutions must be kernel 3 and end in 2 channels (start, end)");

        // The padding rows: ffnn of a zero input.
        var h1 = span0Bias.AsSpan(0, spanHidden).ToArray();
        Relu(h1);
        var h2 = Gemm.Run(h1, 1, _span3, _span3Bias);
        Relu(h2.AsSpan(0, s3));
        _spanPadding = Gemm.Run(h2, 1, _span6, _span6Bias).AsSpan(0, s6).ToArray();

        int z = Shape("zeros_predictor", "0.weight")[0];
        _zeros0 = new PackedMatrix(T("zeros_predictor", "0.weight"), z, h);
        _zeros0Bias = Pad(T("zeros_predictor", "0.bias"), _zeros0.PaddedN);
        _zerosOut = T("zeros_predictor", "2.weight");
        _zerosOutBias = T("zeros_predictor", "2.bias")[0];
    }

    // Columns [first, first + count) of W [rows, cols], packed.
    private static PackedMatrix Columns(float[] w, int rows, int cols, int first, int count)
    {
        var part = new float[rows * count];
        for (int r = 0; r < rows; r++)
            Array.Copy(w, (r * cols) + first, part, r * count, count);
        return new PackedMatrix(part, rows, count);
    }

    private static float[] Pad(float[] v, int length)
    {
        var padded = new float[length];
        v.CopyTo(padded, 0);
        return padded;
    }

    private static void Relu(Span<float> v)
    {
        for (int i = 0; i < v.Length; i++)
            v[i] = Math.Max(v[i], 0);
    }

    public float[] Words(float[] subwords, int count, int[] starts, int[] ends)
    {
        int h = _hidden;
        var scores = new float[count];
        for (int t = 0; t < count; t++)
            scores[t] = (float)Act.Dot(subwords.AsSpan(t * h, h), _attnWeight) + _attnBias;
        var words = new float[starts.Length * h];
        for (int i = 0; i < starts.Length; i++)
        {
            float max = float.NegativeInfinity, sum = 0;
            for (int t = starts[i]; t < ends[i]; t++)
                max = Math.Max(max, scores[t]);
            for (int t = starts[i]; t < ends[i]; t++)
                sum += MathF.Exp(scores[t] - max);
            var row = words.AsSpan(i * h, h);
            for (int t = starts[i]; t < ends[i]; t++)
            {
                float p = MathF.Exp(scores[t] - max) / sum;
                var x = subwords.AsSpan(t * h, h);
                for (int q = 0; q < h; q++)
                    row[q] += p * x[q];
            }
        }

        return words;
    }

    public void RoughScores(float[] words, int n, Action<int, int, float[], int> rows, CancellationToken ct)
    {
        var bilinear = Gemm.Run(words, n, _bilinear, _bilinearBias);
        var others = new PackedMatrix(words, n, _hidden);
        int ld = others.PaddedN, chunk = Math.Min(RoughChunk, n);
        var scores = ArrayPool<float>.Shared.Rent(chunk * ld);
        try
        {
            fixed (float* pa = bilinear, pc = scores)
            {
                for (int first = 0; first < n; first += chunk)
                {
                    int count = Math.Min(chunk, n - first);
                    Gemm.Run(pa + ((long)first * _bilinear.PaddedN), count, _bilinear.PaddedN, others, null, pc, ld, ct);
                    rows(first, count, scores, ld);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scores);
        }
    }

    public float[] Anaphoricity(float[] words, int n, int[] top, int[] buckets, int k, CancellationToken ct)
    {
        int h = _hidden, ld = _pairA.PaddedN, width = _pairA.N;
        var a = Gemm.Run(words, n, _pairA, null);
        var b = Gemm.Run(words, n, _pairB, null);
        var result = new float[n * k];
        int chunk = Math.Min(PairChunk, n);
        var products = ArrayPool<float>.Shared.Rent(chunk * k * h);
        var hidden = ArrayPool<float>.Shared.Rent(chunk * k * ld);
        try
        {
            for (int first = 0; first < n; first += chunk)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(chunk, n - first), pairs = count * k;
                ManagedThreads.For(count, r =>
                {
                    int i = first + r;
                    var mention = words.AsSpan(i * h, h);
                    for (int c = 0; c < k; c++)
                    {
                        var other = words.AsSpan(top[(i * k) + c] * h, h);
                        var product = products.AsSpan(((r * k) + c) * h, h);
                        for (int q = 0; q < h; q++)
                            product[q] = mention[q] * other[q];
                    }
                });
                fixed (float* pa = products, pc = hidden)
                    Gemm.Run(pa, pairs, h, _pairProduct, null, pc, ld, ct);
                ManagedThreads.For(count, r =>
                {
                    int i = first + r;
                    for (int c = 0; c < k; c++)
                    {
                        int j = top[(i * k) + c];
                        var row = hidden.AsSpan(((r * k) + c) * ld, width);
                        var ai = a.AsSpan(i * ld, width);
                        var bj = b.AsSpan(j * ld, width);
                        var pw = _pairDistance.AsSpan(buckets[(i * k) + c] * ld, width);
                        for (int q = 0; q < width; q++)
                        {
                            float v = row[q] + ai[q] + bj[q] + pw[q];
                            row[q] = v >= 0 ? v : v * LeakySlope;
                        }

                        result[(i * k) + c] = (float)Act.Dot(row, _pairOut) + _pairOutBias;
                    }
                });
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(products);
            ArrayPool<float>.Shared.Return(hidden);
        }

        return result;
    }

    public (float[] Start, float[] End) SpanScores(float[] words, int n, int[] heads, int[] sentStart, int[] sentLength, int width, CancellationToken ct)
    {
        int h = _hidden, ld0 = _spanHead.PaddedN, ld3 = _span3.PaddedN, ld6 = _span6.PaddedN, h0 = _spanHead.N;
        var headRows = new float[heads.Length * h];
        for (int i = 0; i < heads.Length; i++)
            Array.Copy(words, heads[i] * h, headRows, i * h, h);
        var headPart = Gemm.Run(headRows, heads.Length, _spanHead, null);
        var wordPart = Gemm.Run(words, n, _spanWord, null);
        int distances = _spanDistance.Length / ld0;

        var start = new float[heads.Length * width];
        var end = new float[heads.Length * width];
        var x0 = ArrayPool<float>.Shared.Rent(HeadChunk * width * ld0);
        var x3 = ArrayPool<float>.Shared.Rent(HeadChunk * width * ld3);
        var x6 = ArrayPool<float>.Shared.Rent(HeadChunk * width * ld6);
        try
        {
            for (int first = 0; first < heads.Length; first += HeadChunk)
            {
                ct.ThrowIfCancellationRequested();
                int count = Math.Min(HeadChunk, heads.Length - first);

                // The real candidates, packed head after head; padding rows get ffnn(0) below.
                var offsets = new int[count + 1];
                for (int r = 0; r < count; r++)
                    offsets[r + 1] = offsets[r] + sentLength[first + r];
                int pairs = offsets[count];
                ManagedThreads.For(count, r =>
                {
                    int hh = first + r;
                    for (int t = 0; t < sentLength[hh]; t++)
                    {
                        int word = sentStart[hh] + t, id = heads[hh] - word + 63;
                        if (id < 0 || id > 126)
                            id = distances - 1; // "too far"
                        var row = x0.AsSpan((offsets[r] + t) * ld0, h0);
                        var hp = headPart.AsSpan(hh * ld0, h0);
                        var wp = wordPart.AsSpan(word * ld0, h0);
                        var dp = _spanDistance.AsSpan(id * ld0, h0);
                        for (int q = 0; q < h0; q++)
                            row[q] = Math.Max(hp[q] + wp[q] + dp[q], 0);
                    }
                });
                fixed (float* p0 = x0, p3 = x3, p6 = x6, b3 = _span3Bias, b6 = _span6Bias)
                {
                    Gemm.Run(p0, pairs, ld0, _span3, b3, p3, ld3, ct);
                    for (int i = 0; i < pairs; i++)
                    {
                        for (int q = 0; q < _span3.N; q++)
                            p3[((long)i * ld3) + q] = Math.Max(p3[((long)i * ld3) + q], 0);
                    }

                    Gemm.Run(p3, pairs, ld3, _span6, b6, p6, ld6, ct);
                }

                ManagedThreads.For(count, r => Convolve(x6, offsets[r], ld6, sentLength[first + r], width, start.AsSpan((first + r) * width, width),
                    end.AsSpan((first + r) * width, width)));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x0);
            ArrayPool<float>.Shared.Return(x3);
            ArrayPool<float>.Shared.Return(x6);
        }

        return (start, end);
    }

    // conv.0 then conv.1 (Conv1d, kernel 3, padding 1) over one head's width positions: its candidates' ffnn outputs, then
    // ffnn(0) up to the batch's width; the output's channels 0 and 1 are the start and end scores.
    private void Convolve(float[] x6, int offset, int ld6, int length, int width, Span<float> start, Span<float> end)
    {
        int c0 = _spanOut, c1 = _conv0Out;
        var input = new float[width * c0];
        for (int t = 0; t < width; t++)
        {
            var src = t < length ? x6.AsSpan((offset + t) * ld6, c0) : _spanPadding.AsSpan();
            src.CopyTo(input.AsSpan(t * c0, c0));
        }

        var middle = Conv(input, width, c0, _conv0, _conv0Bias, c1);
        var output = Conv(middle, width, c1, _conv1, _conv1Bias, _conv1Out);
        for (int t = 0; t < width; t++)
            (start[t], end[t]) = (output[t * 2], output[(t * 2) + 1]);
    }

    // y[t][o] = b[o] + Σ_c Σ_d w[o][c][d]·x[t + d − 1][c], zero outside [0, width).
    private static float[] Conv(float[] x, int width, int inChannels, float[] w, float[] bias, int outChannels)
    {
        var y = new float[width * outChannels];
        for (int t = 0; t < width; t++)
        {
            for (int o = 0; o < outChannels; o++)
            {
                float sum = bias[o];
                for (int c = 0; c < inChannels; c++)
                {
                    for (int d = 0; d < 3; d++)
                    {
                        int s = t + d - 1;
                        if (s >= 0 && s < width)
                            sum += w[(((o * inChannels) + c) * 3) + d] * x[(s * inChannels) + c];
                    }
                }

                y[(t * outChannels) + o] = sum;
            }
        }

        return y;
    }

    public float[] Zeros(float[] words, int[] rows)
    {
        int h = _hidden, ld = _zeros0.PaddedN;
        var input = new float[rows.Length * h];
        for (int i = 0; i < rows.Length; i++)
            Array.Copy(words, rows[i] * h, input, i * h, h);
        var hidden = Gemm.Run(input, rows.Length, _zeros0, _zeros0Bias);
        var scores = new float[rows.Length];
        for (int i = 0; i < rows.Length; i++)
        {
            var row = hidden.AsSpan(i * ld, _zeros0.N);
            Relu(row);
            scores[i] = (float)Act.Dot(row, _zerosOut) + _zerosOutBias;
        }

        return scores;
    }

    public void Dispose()
    {
    }
}
