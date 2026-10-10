using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Constituency;

/// <summary>One sentence's input to <see cref="IConstituencyNet.EncodeWords"/>: its words and their vocab ids.</summary>
/// <param name="CacheKey">The sentence whose charlm outputs a <see cref="CharlmCache"/> may hold, or null.</param>
internal sealed record WordInput(IReadOnlyList<string> Words, long[] PretrainIds, long[] DeltaIds, long[] TagIds, Sentence? CacheKey);

/// <summary>
/// The parser's network, per <see cref="Backend"/>: <c>ConstituencyNet</c> or <see cref="ManagedConstituencyNet"/>.
/// It makes opaque handles (word vectors, constituent vectors, stack LSTM states) that only it reads; the parser keeps them
/// in its states and disposes those that are <see cref="IDisposable"/> when the sentence is done. Every method computes each
/// state on its own, so the batch a state is in doesn't change its result.
/// </summary>
internal interface IConstituencyNet : IDisposable
{
    /// <summary>The transition stack's state after its start embedding (owned by the net).</summary>
    object TransitionStart { get; }

    /// <summary>The constituent stack's state after its start embedding (owned by the net).</summary>
    object ConstituentStart { get; }

    /// <summary>initial_word_queues: per sentence a handle to its word vectors, row 0 the start sentinel, then one per word,
    /// then the end sentinel (word_lstm, then relu(word_to_constituent)).</summary>
    object[] EncodeWords(IReadOnlyList<WordInput> sentences, CharlmCache? charlms, CancellationToken ct);

    /// <summary>Row <paramref name="position"/> of a sentence's word vectors, as a constituent vector (a Shift's).</summary>
    object Word(object words, int position);

    /// <summary>LSTMModel.forward: [states, transitions] scores from each state's word vector at its position and the tops of
    /// its transition and constituent stacks; ReLU before every output layer, including the first.</summary>
    float[] Score(IReadOnlyList<(object Words, int Position, object Transition, object Constituent)> states, CancellationToken ct);

    /// <summary>The Open markers' vectors: rows of the dummy (constituent open) embedding.</summary>
    object[] Open(int[] openIndices);

    /// <summary>MAX composition: relu(reduce_linear(elementwise max over the children's vectors)), per close.</summary>
    object[] Compose(IReadOnlyList<IReadOnlyList<object>> children, CancellationToken ct);

    /// <summary>Runs the transition LSTM one step from each parent state with the transition's embedding.</summary>
    object[] PushTransitions(IReadOnlyList<object> parents, int[] transitions, CancellationToken ct);

    /// <summary>Runs the constituent LSTM one step from each parent state with the constituent's vector.</summary>
    object[] PushConstituents(IReadOnlyList<object> parents, IReadOnlyList<object> inputs, CancellationToken ct);
}

/// <summary>
/// Managed twin of <c>ConstituencyNet</c> (<see cref="Backend.Managed"/>). Handles are float arrays: a word or
/// constituent vector is a [hidden] array, a sentence's words an array of them, a stack state its [layers, hidden] h and c.
/// The word encoder is a <see cref="ManagedLstm"/>; every per-step layer (output layers, reduce_linear, each stack LSTM
/// layer as one GEMM over <c>[x | h]</c>) runs with <c>rowInvariant</c>, so a state's arithmetic is the same in any batch.
/// </summary>
internal sealed unsafe class ManagedConstituencyNet : IConstituencyNet
{
    /// <summary>Study only (tools/constituency_divergence.py): per-step layers whose sums run in double. Set before loading.</summary>
    [Flags]
    internal enum DoubleSums { None = 0, Output = 1, LastOutput = 2, Stacks = 4, Reduce = 8 }

    internal static DoubleSums Double;

    /// <summary>Row r of <paramref name="y"/> (stride <paramref name="ldy"/>): bias + w·x_r, summed in double. w is
    /// [n, k] row-major, column j of the result reading w's row <paramref name="order"/>[j] (row j when null).</summary>
    private static void DoubleLinear(float[] w, int n, int k, float[] bias, float* x, int ldx, int rows, float* y, int ldy, int[]? order = null)
    {
        ManagedThreads.For(rows * n, rj =>
        {
            int r = rj / n, j = rj % n, src = order?[j] ?? j;
            double s = bias[j];
            var wr = w.AsSpan(src * k, k);
            float* xr = x + (long)r * ldx;
            for (int q = 0; q < k; q++)
                s += (double)wr[q] * xr[q];
            y[(long)r * ldy + j] = (float)s;
        });
    }

    /// <summary>A stack node's LSTM state: [layers, hidden] h and c. The node's output is the last layer's h.</summary>
    private sealed class StackState(float[] h, float[] c)
    {
        public readonly float[] H = h, C = c;
    }

    /// <summary>A multi-layer nn.LSTM stepped once per call, each row from its own state.</summary>
    private sealed class StackLstm
    {
        private readonly (PackedMatrix W, float[] Bias)[] _layers; // [W_ih | W_hh], gate rows in PackedLstm.GateOrder
        private readonly float[][]? _raw; // Double.Stacks: [W_ih | W_hh] rows in the checkpoint's order
        private readonly int[] _order;
        private readonly int _input;

        public int Hidden { get; }
        public int Layers => _layers.Length;

        public StackLstm(Func<string, float[]> weights, string prefix, int input, int hidden, int layers)
        {
            if (hidden % 4 != 0)
                throw new NotSupportedException("The managed parser needs LSTM hidden sizes that are a multiple of 4");
            (_input, Hidden) = (input, hidden);
            var order = _order = PackedLstm.GateOrder(hidden);
            _layers = new (PackedMatrix, float[])[layers];
            if ((Double & DoubleSums.Stacks) != 0)
                _raw = new float[layers][];
            for (int l = 0; l < layers; l++)
            {
                int inl = l == 0 ? input : hidden, k = inl + hidden;
                var (wih, whh) = (weights($"{prefix}weight_ih_l{l}"), weights($"{prefix}weight_hh_l{l}"));
                var (bih, bhh) = (weights($"{prefix}bias_ih_l{l}"), weights($"{prefix}bias_hh_l{l}"));
                var w = new float[4 * hidden * k];
                for (int j = 0; j < 4 * hidden; j++)
                {
                    wih.AsSpan(j * inl, inl).CopyTo(w.AsSpan(j * k));
                    whh.AsSpan(j * hidden, hidden).CopyTo(w.AsSpan(j * k + inl));
                }
                if (_raw != null)
                    _raw[l] = w;
                var packed = new PackedMatrix(w, 4 * hidden, k, order);
                var bias = new float[packed.PaddedN];
                for (int j = 0; j < order.Length; j++)
                    bias[j] = bih[order[j]] + bhh[order[j]];
                _layers[l] = (packed, bias);
            }
        }

        public StackState Zero => new(new float[_layers.Length * Hidden], new float[_layers.Length * Hidden]);

        /// <summary>Row r: one step from <paramref name="parents"/>[r] with input <paramref name="inputs"/>[r] ([input] floats).</summary>
        public StackState[] Push(IReadOnlyList<StackState> parents, IReadOnlyList<float[]> inputs, CancellationToken ct)
        {
            int n = parents.Count, h = Hidden, kMax = Math.Max(_input, h) + h, ldg = _layers[0].W.PaddedN;
            var result = new StackState[n];
            for (int r = 0; r < n; r++)
                result[r] = new StackState(new float[_layers.Length * h], (float[])parents[r].C.Clone());
            var xh = ArrayPool<float>.Shared.Rent(n * kMax);
            var gates = ArrayPool<float>.Shared.Rent(n * ldg);
            try
            {
                for (int l = 0; l < _layers.Length; l++)
                {
                    int inl = l == 0 ? _input : h, k = inl + h;
                    for (int r = 0; r < n; r++)
                    {
                        var x = l == 0 ? inputs[r].AsSpan(0, inl) : result[r].H.AsSpan((l - 1) * h, h);
                        x.CopyTo(xh.AsSpan(r * k));
                        parents[r].H.AsSpan(l * h, h).CopyTo(xh.AsSpan(r * k + inl));
                    }
                    var (w, bias) = _layers[l];
                    fixed (float* pxh = xh, pg = gates, pb = bias)
                    {
                        if (_raw != null)
                            DoubleLinear(_raw[l], 4 * h, k, bias, pxh, k, n, pg, ldg, _order);
                        else
                            Gemm.Run(pxh, n, k, w, pb, pg, ldg, ct, rowInvariant: true);
                        for (int r = 0; r < n; r++)
                            fixed (float* c = result[r].C, hOut = result[r].H)
                                for (int p = 0; p < h / 4; p++)
                                    Act.LstmCell(pg + r * ldg + p * 16, c + l * h + p * 4, hOut + l * h + p * 4);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(xh);
                ArrayPool<float>.Shared.Return(gates);
            }
            return result;
        }
    }

    private readonly Pretrain _pretrain;
    private readonly ManagedCharLanguageModel _charlmForward, _charlmBackward;
    private readonly float[] _deltaEmb, _tagEmb, _transitionEmb, _wordStart, _wordEnd;
    private readonly float[][] _dummy; // the Open markers' vectors, shared (read-only)
    private readonly int _deltaDim, _tagDim, _transitionDim, _inSize, _hidden;
    private readonly ManagedLstm _wordLstm;
    private readonly PackedMatrix _wordToConstituent, _reduce;
    private readonly float[] _wordToConstituentBias, _reduceBias;
    private readonly (PackedMatrix W, float[] Bias)[] _outputLayers;
    private readonly float[]?[] _outputRaw; // Double.Output / LastOutput: the layer's [n, k] weights
    private readonly float[]? _reduceRaw;
    private readonly StackLstm _transitionLstm, _constituentLstm;
    private readonly StackState _transitionStart, _constituentStart;

    public object TransitionStart => _transitionStart;
    public object ConstituentStart => _constituentStart;

    public ManagedConstituencyNet(Checkpoint ckpt, Pretrain pretrain, ManagedCharLanguageModel charlmForward, ManagedCharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        var model = p["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        int Dim(string key) => (int)ckpt.Shape(model[key])[1];
        int hidden = _hidden = config["hidden_size"]!.GetValue<int>();
        int layers = config["num_lstm_layers"]!.GetValue<int>();
        int transitionHidden = config["transition_hidden_size"]!.GetValue<int>();

        (_deltaEmb, _deltaDim) = (T("delta_embedding.weight"), Dim("delta_embedding.weight"));
        (_tagEmb, _tagDim) = (T("tag_embedding.weight"), Dim("tag_embedding.weight"));
        (_transitionEmb, _transitionDim) = (T("transition_embedding.weight"), Dim("transition_embedding.weight"));
        var dummy = T("dummy_embedding.weight");
        _dummy = Enumerable.Range(0, dummy.Length / hidden).Select(i => dummy.AsSpan(i * hidden, hidden).ToArray()).ToArray();

        _inSize = pretrain.Dim + _deltaDim + _tagDim + charlmForward.HiddenDim + charlmBackward.HiddenDim;
        if (_inSize != Dim("word_lstm.weight_ih_l0"))
            throw new NotSupportedException("The parser's word input isn't pretrain + delta + tag + both charlms");
        _wordLstm = new ManagedLstm(ckpt, model, "word_lstm.", _inSize, hidden, layers, bidirectional: true);
        _wordStart = T("word_start_embedding");
        _wordEnd = T("word_end_embedding");
        (_wordToConstituent, _wordToConstituentBias) = Linear(T, "word_to_constituent.", hidden, 2 * hidden);
        (_reduce, _reduceBias) = Linear(T, "reduce_linear.", hidden, hidden);
        if ((Double & DoubleSums.Reduce) != 0)
            _reduceRaw = T("reduce_linear.weight");
        var outputs = new List<(PackedMatrix, float[])>();
        for (int i = 0; model[$"output_layers.{i}.weight"] is { } w; i++)
        {
            var shape = ckpt.Shape(w);
            outputs.Add(Linear(T, $"output_layers.{i}.", (int)shape[0], (int)shape[1]));
        }
        _outputLayers = [.. outputs];
        _outputRaw = new float[]?[_outputLayers.Length];
        for (int i = 0; i < _outputRaw.Length; i++)
            if ((Double & DoubleSums.Output) != 0 || (Double & DoubleSums.LastOutput) != 0 && i == _outputRaw.Length - 1)
                _outputRaw[i] = T($"output_layers.{i}.weight");

        _transitionLstm = new StackLstm(T, "transition_stack.lstm.", _transitionDim, transitionHidden, layers);
        _constituentLstm = new StackLstm(T, "constituent_stack.lstm.", hidden, hidden, layers);
        _transitionStart = _transitionLstm.Push([_transitionLstm.Zero], [T("transition_stack.start_embedding")], default)[0];
        _constituentStart = _constituentLstm.Push([_constituentLstm.Zero], [T("constituent_stack.start_embedding")], default)[0];
    }

    private static (PackedMatrix W, float[] Bias) Linear(Func<string, float[]> weights, string prefix, int n, int k)
    {
        var w = new PackedMatrix(weights(prefix + "weight"), n, k);
        var bias = new float[w.PaddedN];
        weights(prefix + "bias").CopyTo(bias, 0);
        return (w, bias);
    }

    public object[] EncodeWords(IReadOnlyList<WordInput> sentences, CharlmCache? charlms, CancellationToken ct)
    {
        int n = sentences.Count, width = sentences.Max(s => s.Words.Count) + 2, inSize = _inSize, h2 = _wordLstm.OutputSize;
        int pre = _pretrain.Dim, deltaCol = pre, tagCol = pre + _deltaDim, forwardCol = tagCol + _tagDim;
        int charDim = _charlmForward.HiddenDim, backwardCol = forwardCol + charDim;
        int ldz = _wordToConstituent.PaddedN;
        var x = ArrayPool<float>.Shared.Rent(n * width * inSize); // [n, width, inSize], batch-first
        var y = ArrayPool<float>.Shared.Rent(n * width * h2);
        var z = ArrayPool<float>.Shared.Rent(n * width * ldz);
        try
        {
            var vectors = _pretrain.CpuVectors();
            for (int i = 0; i < n; i++)
            {
                var s = sentences[i];
                int len = s.Words.Count;
                _wordStart.CopyTo(x.AsSpan(i * width * inSize, inSize));
                _wordEnd.CopyTo(x.AsSpan((i * width + len + 1) * inSize, inSize));
                for (int t = 0; t < len; t++)
                {
                    var row = x.AsSpan((i * width + t + 1) * inSize, inSize);
                    vectors.Slice((int)s.PretrainIds[t] * pre, pre).CopyTo(row);
                    _deltaEmb.AsSpan((int)s.DeltaIds[t] * _deltaDim, _deltaDim).CopyTo(row[deltaCol..]);
                    _tagEmb.AsSpan((int)s.TagIds[t] * _tagDim, _tagDim).CopyTo(row[tagCol..]);
                }
            }

            // The charlm columns: the tagger's outputs where the cache has them, the rest computed in one batch.
            void Put(int i, ReadOnlySpan<float> forward, ReadOnlySpan<float> backward)
            {
                for (int t = 0; t < sentences[i].Words.Count; t++)
                {
                    int r = (i * width + t + 1) * inSize;
                    forward.Slice(t * charDim, charDim).CopyTo(x.AsSpan(r + forwardCol, charDim));
                    backward.Slice(t * charDim, charDim).CopyTo(x.AsSpan(r + backwardCol, charDim));
                }
            }
            var missing = new List<int>();
            for (int i = 0; i < n; i++)
                if (sentences[i].CacheKey is { } key && charlms!.TryGetArrays(key, out var reps))
                    Put(i, reps.Forward, reps.Backward);
                else
                    missing.Add(i);
            if (missing.Count > 0)
            {
                var texts = missing.Select(i => sentences[i].Words).ToList();
                int words = texts.Sum(s => s.Count);
                var forwardReps = ArrayPool<float>.Shared.Rent(words * charDim);
                var backwardReps = ArrayPool<float>.Shared.Rent(words * charDim);
                try
                {
                    _charlmForward.BuildCharRepresentation(texts, forwardReps, charDim, ct);
                    _charlmBackward.BuildCharRepresentation(texts, backwardReps, charDim, ct);
                    int offset = 0;
                    foreach (int i in missing)
                    {
                        int len = sentences[i].Words.Count * charDim;
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

            var lengths = sentences.Select(s => (long)s.Words.Count + 2).ToArray();
            _wordLstm.ForwardPadded(x, n, width, lengths, y, ct);
            fixed (float* py = y, pz = z, pb = _wordToConstituentBias)
                Gemm.Run(py, n * width, h2, _wordToConstituent, pb, pz, ldz, ct);

            var result = new object[n];
            for (int i = 0; i < n; i++)
            {
                var rows = new float[lengths[i]][];
                for (int t = 0; t < rows.Length; t++)
                {
                    var row = rows[t] = z.AsSpan((i * width + t) * ldz, _hidden).ToArray();
                    Relu(row);
                }
                result[i] = rows;
            }
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(y);
            ArrayPool<float>.Shared.Return(z);
        }
    }

    public object Word(object words, int position) => ((float[][])words)[position];

    public float[] Score(IReadOnlyList<(object Words, int Position, object Transition, object Constituent)> states, CancellationToken ct)
    {
        int n = states.Count, th = _transitionLstm.Hidden, ch = _constituentLstm.Hidden;
        int k0 = _hidden + th + ch;
        int ld = Math.Max(k0, _outputLayers.Max(l => l.W.PaddedN));
        var a = ArrayPool<float>.Shared.Rent(n * ld);
        var b = ArrayPool<float>.Shared.Rent(n * ld);
        try
        {
            for (int r = 0; r < n; r++)
            {
                var (words, position, transition, constituent) = states[r];
                var row = a.AsSpan(r * ld, k0);
                ((float[][])words)[position].CopyTo(row);
                ((StackState)transition).H.AsSpan((_transitionLstm.Layers - 1) * th, th).CopyTo(row[_hidden..]);
                ((StackState)constituent).H.AsSpan((_constituentLstm.Layers - 1) * ch, ch).CopyTo(row[(_hidden + th)..]);
            }
            int k = k0;
            for (int i = 0; i < _outputLayers.Length; i++)
            {
                var (w, bias) = _outputLayers[i];
                for (int r = 0; r < n; r++)
                    Relu(a.AsSpan(r * ld, k)); // nonlinearity before every layer, including the first
                fixed (float* pa = a, pb = b, pBias = bias)
                    if (_outputRaw[i] is { } raw)
                        DoubleLinear(raw, w.N, k, bias, pa, ld, n, pb, ld);
                    else
                        Gemm.Run(pa, n, ld, w, pBias, pb, ld, ct, rowInvariant: true);
                (a, b) = (b, a);
                k = w.N;
            }
            var result = new float[n * k];
            for (int r = 0; r < n; r++)
                a.AsSpan(r * ld, k).CopyTo(result.AsSpan(r * k));
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(a);
            ArrayPool<float>.Shared.Return(b);
        }
    }

    public object[] Open(int[] openIndices) => openIndices.Select(i => (object)_dummy[i]).ToArray();

    public object[] Compose(IReadOnlyList<IReadOnlyList<object>> children, CancellationToken ct)
    {
        int n = children.Count, h = _hidden, ld = _reduce.PaddedN;
        var pooled = ArrayPool<float>.Shared.Rent(n * h);
        var output = ArrayPool<float>.Shared.Rent(n * ld);
        try
        {
            for (int r = 0; r < n; r++)
            {
                var row = pooled.AsSpan(r * h, h);
                ((float[])children[r][0]).CopyTo(row);
                for (int c = 1; c < children[r].Count; c++)
                {
                    var x = (float[])children[r][c];
                    for (int j = 0; j < h; j++)
                        row[j] = Math.Max(row[j], x[j]);
                }
            }
            fixed (float* pp = pooled, po = output, pb = _reduceBias)
                if (_reduceRaw != null)
                    DoubleLinear(_reduceRaw, h, h, _reduceBias, pp, h, n, po, ld);
                else
                    Gemm.Run(pp, n, h, _reduce, pb, po, ld, ct, rowInvariant: true);
            var result = new object[n];
            for (int r = 0; r < n; r++)
            {
                var v = output.AsSpan(r * ld, h).ToArray();
                Relu(v);
                result[r] = v;
            }
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(pooled);
            ArrayPool<float>.Shared.Return(output);
        }
    }

    public object[] PushTransitions(IReadOnlyList<object> parents, int[] transitions, CancellationToken ct)
    {
        var inputs = transitions.Select(t => _transitionEmb.AsSpan(t * _transitionDim, _transitionDim).ToArray()).ToList();
        return _transitionLstm.Push(parents.Cast<StackState>().ToList(), inputs, ct);
    }

    public object[] PushConstituents(IReadOnlyList<object> parents, IReadOnlyList<object> inputs, CancellationToken ct) =>
        _constituentLstm.Push(parents.Cast<StackState>().ToList(), inputs.Cast<float[]>().ToList(), ct);

    private static void Relu(Span<float> x)
    {
        for (int j = 0; j < x.Length; j++)
            x[j] = Math.Max(x[j], 0f);
    }

    public void Dispose()
    {
    }
}
