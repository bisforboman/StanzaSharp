using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Constituency;

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
            bool cached = s.CacheKey != null && charlms!.TryGet(s.CacheKey, out r, _device);
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

/// <summary>Loads a <see cref="ConstituencyParser"/> on TorchSharp.</summary>
internal static class ConstituencyParserLoad
{
    extension(ConstituencyParser)
    {
        /// <summary>
        /// Loads e.g. <c>models/converted/en/constituency/ptb3-revised_charlm</c> on TorchSharp. The pretrain and charlms
        /// are shared with the tagger, so the caller owns them.
        /// </summary>
        /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
        public static ConstituencyParser Load(string basePath, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward, Device? device = null) =>
            Weights.On(device, () =>
            {
                if (!charlmForward.IsForward || charlmBackward.IsForward)
                    throw new ArgumentException("Pass the forward charlm first, then the backward one");
                var ckpt = Checkpoint.Load(basePath);
                return new ConstituencyParser(ckpt, pretrain, () => new ConstituencyNet(ckpt, pretrain, charlmForward, charlmBackward));
            });
    }
}
