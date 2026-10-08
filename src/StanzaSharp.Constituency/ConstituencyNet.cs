using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Constituency;

/// <summary>One sentence's input to <see cref="IConstituencyNet.EncodeWords"/>: its words and their vocab ids.</summary>
/// <param name="CacheKey">The sentence whose charlm outputs a <see cref="CharlmCache"/> may hold, or null.</param>
internal sealed record WordInput(IReadOnlyList<string> Words, long[] PretrainIds, long[] DeltaIds, long[] TagIds, Sentence? CacheKey);

/// <summary>
/// The parser's network, per <see cref="Backend"/>: <see cref="ConstituencyNet"/> or <see cref="ManagedConstituencyNet"/>.
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

/// <summary>The parser's network on TorchSharp (the code from before the seam).</summary>
internal sealed class ConstituencyNet : IConstituencyNet
{
    /// <summary>A stack node's LSTM state: [layers, hidden] h and c, and the last layer's output [hidden] (views).</summary>
    private sealed class StackState(Tensor hx, Tensor cx, Tensor output) : IDisposable
    {
        public readonly Tensor Hx = hx, Cx = cx, Output = output;

        public void Dispose()
        {
            Hx.Dispose();
            Cx.Dispose();
            Output.Dispose();
        }
    }

    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Embedding _deltaEmbedding, _tagEmbedding, _transitionEmbedding, _dummyEmbedding;
    private readonly LSTM _wordLstm, _transitionLstm, _constituentLstm;
    private readonly Linear _wordToConstituent, _reduceLinear;
    private readonly Linear[] _outputLayers;
    private readonly Tensor _wordStart, _wordEnd;
    private readonly StackState _transitionStart, _constituentStart;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public object TransitionStart => _transitionStart;
    public object ConstituentStart => _constituentStart;

    public ConstituencyNet(Checkpoint ckpt, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        var model = p["model"]!;
        int hidden = config["hidden_size"]!.GetValue<int>();
        int layers = config["num_lstm_layers"]!.GetValue<int>();
        int transitionHidden = config["transition_hidden_size"]!.GetValue<int>();

        Embedding Emb(string name) =>
            nn.Embedding(ckpt.Shape(model[name + ".weight"])[0], ckpt.Shape(model[name + ".weight"])[1]).LoadFrom(ckpt, model, name + ".");

        _deltaEmbedding = Emb("delta_embedding");
        _tagEmbedding = Emb("tag_embedding");
        _transitionEmbedding = Emb("transition_embedding");
        _dummyEmbedding = Emb("dummy_embedding"); // same weights as constituent_open_embedding (combined_dummy_embedding)

        long wordInput = ckpt.Shape(model["word_lstm.weight_ih_l0"])[1];
        _wordLstm = nn.LSTM(wordInput, hidden, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "word_lstm.");
        _wordToConstituent = nn.Linear(hidden * 2, hidden).LoadFrom(ckpt, model, "word_to_constituent.");
        _wordStart = ckpt.ToTensor(model["word_start_embedding"]);
        _wordEnd = ckpt.ToTensor(model["word_end_embedding"]);

        _transitionLstm = nn.LSTM(ckpt.Shape(model["transition_embedding.weight"])[1], transitionHidden, numLayers: layers).LoadFrom(ckpt, model, "transition_stack.lstm.");
        _constituentLstm = nn.LSTM(hidden, hidden, numLayers: layers).LoadFrom(ckpt, model, "constituent_stack.lstm.");
        _reduceLinear = nn.Linear(hidden, hidden).LoadFrom(ckpt, model, "reduce_linear.");

        var outputs = new List<Linear>();
        for (int i = 0; model[$"output_layers.{i}.weight"] is { } w; i++)
        {
            var shape = ckpt.Shape(w);
            outputs.Add(nn.Linear(shape[1], shape[0]).LoadFrom(ckpt, model, $"output_layers.{i}."));
        }
        _outputLayers = outputs.ToArray();

        // In a dispose scope, so no temporary tensor is left to a finalizer, which could free it during a native call.
        using (torch.no_grad())
        using (var scope = NewDisposeScope())
        {
            _transitionStart = InitialStack(_transitionLstm, ckpt.ToTensor(model["transition_stack.start_embedding"]));
            _constituentStart = InitialStack(_constituentLstm, ckpt.ToTensor(model["constituent_stack.start_embedding"]));
            scope.Detach((IEnumerable<IDisposable>)[_transitionStart.Hx, _transitionStart.Cx, _transitionStart.Output,
                _constituentStart.Hx, _constituentStart.Cx, _constituentStart.Output]);
        }
    }

    public object[] EncodeWords(IReadOnlyList<WordInput> sentences, CharlmCache? charlms, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        // Charlm outputs the tagger kept, computing only the missing ones.
        var charsForward = new List<Tensor?>();
        var charsBackward = new List<Tensor?>();
        foreach (var s in sentences)
        {
            (Tensor Forward, Tensor Backward) r = default;
            bool cached = s.CacheKey != null && charlms!.TryGet(s.CacheKey, out r);
            charsForward.Add(cached ? r.Forward : null);
            charsBackward.Add(cached ? r.Backward : null);
        }
        var missing = Enumerable.Range(0, sentences.Count).Where(k => charsForward[k] is null).ToList();
        if (missing.Count > 0)
        {
            var words = missing.Select(k => sentences[k].Words).ToList();
            var forward = _charlmForward.BuildCharRepresentation(words);
            var backward = _charlmBackward.BuildCharRepresentation(words);
            for (int m = 0; m < missing.Count; m++)
                (charsForward[missing[m]], charsBackward[missing[m]]) = (forward[m], backward[m]);
        }

        var inputs = new List<Tensor>(sentences.Count);
        for (int i = 0; i < sentences.Count; i++)
        {
            var wordInput = cat([
                _pretrain.Embeddings[torch.tensor(sentences[i].PretrainIds, device: _device)],
                _deltaEmbedding.forward(torch.tensor(sentences[i].DeltaIds, device: _device)),
                _tagEmbedding.forward(torch.tensor(sentences[i].TagIds, device: _device)),
                charsForward[i]!,
                charsBackward[i]!,
            ], 1);
            inputs.Add(cat([_wordStart.unsqueeze(0), wordInput, _wordEnd.unsqueeze(0)], 0));
        }

        var lengths = sentences.Select(s => (long)s.Words.Count + 2).ToArray();
        var output = Rnn.RunPacked(_wordLstm, Rnn.PadSequence(inputs), lengths);
        var wordHx = F.relu(_wordToConstituent.forward(output));
        return Enumerable.Range(0, sentences.Count).Select(i => (object)scope.Detach(wordHx[i])).ToArray();
    }

    public object Word(object words, int position)
    {
        using var scope = NewDisposeScope();
        return scope.Detach(((Tensor)words)[position]);
    }

    public float[] Score(IReadOnlyList<(object Words, int Position, object Transition, object Constituent)> states, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        var word = stack(states.Select(s => ((Tensor)s.Words)[s.Position]).ToArray());
        var transition = stack(states.Select(s => ((StackState)s.Transition).Output).ToArray());
        var constituent = stack(states.Select(s => ((StackState)s.Constituent).Output).ToArray());
        var hx = cat([word, transition, constituent], 1);
        foreach (var layer in _outputLayers)
            hx = layer.forward(F.relu(hx)); // nonlinearity before every layer, including the first
        return hx.ToArray<float>();
    }

    public object[] Open(int[] openIndices)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        var hx = _dummyEmbedding.forward(torch.tensor(openIndices.Select(i => (long)i).ToArray(), device: _device)).unbind(0);
        return hx.Select(t => (object)scope.Detach(t)).ToArray();
    }

    public object[] Compose(IReadOnlyList<IReadOnlyList<object>> children, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        var pooled = stack(children.Select(c => stack(c.Select(x => (Tensor)x).ToArray()).max(0).values).ToArray());
        var hx = F.relu(_reduceLinear.forward(pooled)).unbind(0);
        return hx.Select(t => (object)scope.Detach(t)).ToArray();
    }

    public object[] PushTransitions(IReadOnlyList<object> parents, int[] transitions, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        var input = _transitionEmbedding.forward(torch.tensor(transitions.Select(i => (long)i).ToArray(), device: _device));
        return Push(_transitionLstm, parents, input, scope);
    }

    public object[] PushConstituents(IReadOnlyList<object> parents, IReadOnlyList<object> inputs, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        return Push(_constituentLstm, parents, stack(inputs.Select(x => (Tensor)x).ToArray()), scope);
    }

    // ----- LSTM stacks (lstm_tree_stack.py) -----

    private static StackState InitialStack(LSTM lstm, Tensor startEmbedding)
    {
        var (output, hx, cx) = lstm.forward(startEmbedding.view(1, 1, -1));
        return new StackState(hx.squeeze(1), cx.squeeze(1), output[0, 0]);
    }

    /// <summary>Runs the LSTM one step for each stack from its own state; the new states' tensors leave <paramref name="scope"/>.</summary>
    private static object[] Push(LSTM lstm, IReadOnlyList<object> parents, Tensor inputs, DisposeScope scope)
    {
        var stacks = parents.Cast<StackState>().ToList();
        var hx = torch.stack(stacks.Select(s => s.Hx).ToArray(), 1);
        var cx = torch.stack(stacks.Select(s => s.Cx).ToArray(), 1);
        var (output, hn, cn) = lstm.forward(inputs.unsqueeze(0), (hx, cx));
        // One unbind per tensor instead of a view op per stack: the per-step op count is the parser's overhead.
        var (hs, cs, outputs) = (hn.unbind(1), cn.unbind(1), output[0].unbind(0));
        var result = new object[stacks.Count];
        for (int i = 0; i < result.Length; i++)
        {
            // Views of hn/cn/output; disposing a view leaves the storage to the others.
            scope.Detach((IEnumerable<IDisposable>)[hs[i], cs[i], outputs[i]]);
            result[i] = new StackState(hs[i], cs[i], outputs[i]);
        }
        return result;
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _deltaEmbedding, _tagEmbedding, _transitionEmbedding, _dummyEmbedding,
                     _wordLstm, _transitionLstm, _constituentLstm, _wordToConstituent, _reduceLinear })
            m.Dispose();
        foreach (var l in _outputLayers)
            l.Dispose();
        _wordStart.Dispose();
        _wordEnd.Dispose();
        _transitionStart.Dispose();
        _constituentStart.Dispose();
    }
}

/// <summary>
/// Managed twin of <see cref="ConstituencyNet"/> (<see cref="Backend.Managed"/>). Handles are float arrays: a word or
/// constituent vector is a [hidden] array, a sentence's words an array of them, a stack state its [layers, hidden] h and c.
/// The word encoder is a <see cref="ManagedLstm"/>; every per-step layer (output layers, reduce_linear, each stack LSTM
/// layer as one GEMM over <c>[x | h]</c>) runs with <c>rowInvariant</c>, so a state's arithmetic is the same in any batch.
/// </summary>
internal sealed unsafe class ManagedConstituencyNet : IConstituencyNet
{
    /// <summary>A stack node's LSTM state: [layers, hidden] h and c. The node's output is the last layer's h.</summary>
    private sealed class StackState(float[] h, float[] c)
    {
        public readonly float[] H = h, C = c;
    }

    /// <summary>A multi-layer nn.LSTM stepped once per call, each row from its own state.</summary>
    private sealed class StackLstm
    {
        private readonly (PackedMatrix W, float[] Bias)[] _layers; // [W_ih | W_hh], gate rows in PackedLstm.GateOrder
        private readonly int _input;

        public int Hidden { get; }
        public int Layers => _layers.Length;

        public StackLstm(Func<string, float[]> weights, string prefix, int input, int hidden, int layers)
        {
            if (hidden % 4 != 0)
                throw new NotSupportedException("The managed parser needs LSTM hidden sizes that are a multiple of 4");
            (_input, Hidden) = (input, hidden);
            var order = PackedLstm.GateOrder(hidden);
            _layers = new (PackedMatrix, float[])[layers];
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
        var outputs = new List<(PackedMatrix, float[])>();
        for (int i = 0; model[$"output_layers.{i}.weight"] is { } w; i++)
        {
            var shape = ckpt.Shape(w);
            outputs.Add(Linear(T, $"output_layers.{i}.", (int)shape[0], (int)shape[1]));
        }
        _outputLayers = [.. outputs];

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
            foreach (var (w, bias) in _outputLayers)
            {
                for (int r = 0; r < n; r++)
                    Relu(a.AsSpan(r * ld, k)); // nonlinearity before every layer, including the first
                fixed (float* pa = a, pb = b, pBias = bias)
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
