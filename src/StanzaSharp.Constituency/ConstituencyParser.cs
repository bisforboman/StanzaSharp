using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Constituency;

/// <summary>
/// Constituency parser: an in-order transition parser scored by LSTMs, ported from
/// stanza/models/constituency/lstm_model.py <c>LSTMModel</c> and base_model.py for the English
/// configuration:
/// - IN_ORDER transitions
/// - LSTM transition and constituent stacks
/// - MAX composition
/// - charlm, no attention, no transformer
/// </summary>
public sealed class ConstituencyParser : IDisposable
{
    private const int BatchSize = 50;

    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Dictionary<string, int> _deltaMap, _tagMap;
    private readonly Transition[] _transitions;
    private readonly HashSet<string> _rootLabels;
    private readonly int _unaryLimit;
    private readonly bool _usesXpos;

    private readonly Embedding _deltaEmbedding, _tagEmbedding, _transitionEmbedding, _dummyEmbedding;
    private readonly LSTM _wordLstm, _transitionLstm, _constituentLstm;
    private readonly Linear _wordToConstituent, _reduceLinear;
    private readonly Linear[] _outputLayers;
    private readonly Tensor _wordStart, _wordEnd;
    private readonly StackNode<Transition?> _initialTransitions;
    private readonly StackNode<Constituent> _initialConstituents;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    private ConstituencyParser(Checkpoint ckpt, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        _charlmForward = charlmForward;
        _charlmBackward = charlmBackward;
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");

        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        var model = p["model"]!;
        CheckSupported(p, config);

        // LSTMModel maps words and tags to i + 2 (0 = PAD, 1 = UNK) over sorted lists; the saved lists are sorted.
        _deltaMap = Strings(p["words"]).Select((w, i) => (w, i)).ToDictionary(x => x.w, x => x.i + 2);
        _tagMap = Strings(p["tags"]).Select((t, i) => (t, i)).ToDictionary(x => x.t, x => x.i + 2);
        _rootLabels = Strings(p["root_labels"]).ToHashSet();
        _unaryLimit = p["unary_limit"]!.GetValue<int>();
        _usesXpos = config["retag_method"]?.GetValue<string>() == "xpos";
        var opens = p["constituent_opens"]!.AsArray().Select(OpenLabel).ToList();
        _transitions = Strings(p["transitions"]).Select((t, i) => ParseTransition(t, i, opens)).ToArray();

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

        _transitionLstm = nn.LSTM(_transitionEmbedding.weight!.shape[1], transitionHidden, numLayers: layers).LoadFrom(ckpt, model, "transition_stack.lstm.");
        _constituentLstm = nn.LSTM(hidden, hidden, numLayers: layers).LoadFrom(ckpt, model, "constituent_stack.lstm.");
        _reduceLinear = nn.Linear(hidden, hidden).LoadFrom(ckpt, model, "reduce_linear.");

        var outputs = new List<Linear>();
        for (int i = 0; model[$"output_layers.{i}.weight"] is { } w; i++)
        {
            var shape = ckpt.Shape(w);
            outputs.Add(nn.Linear(shape[1], shape[0]).LoadFrom(ckpt, model, $"output_layers.{i}."));
        }
        _outputLayers = outputs.ToArray();

        using (torch.no_grad())
        {
            _initialTransitions = InitialStack<Transition?>(_transitionLstm, ckpt.ToTensor(model["transition_stack.start_embedding"]), null);
            _initialConstituents = InitialStack(_constituentLstm, ckpt.ToTensor(model["constituent_stack.start_embedding"]), new Constituent(null, null, null));
        }
    }

    /// <summary>
    /// Loads e.g. <c>models/converted/en/constituency/ptb3-revised_charlm</c>. The pretrain and charlms
    /// are shared with the tagger, so the caller owns them.
    /// </summary>
    /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
    public static ConstituencyParser Load(string basePath, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward, Device? device = null) =>
        Weights.On(device, () => new ConstituencyParser(Checkpoint.Load(basePath), pretrain, charlmForward, charlmBackward));

    /// <summary>Sets <see cref="Sentence.Constituency"/> on every sentence. Needs XPOS (or UPOS) tags from the tagger.</summary>
    /// <param name="charlms">Charlm representations the tagger kept, if any; sentences missing from it are computed.</param>
    public void Process(Document doc, CharlmCache? charlms = null)
    {
        var sentences = doc.Sentences.Where(s => s.Tokens.Count > 0).ToList();
        var tagged = sentences.Select(s => (IReadOnlyList<(string, string)>)s.Words.Select(w =>
            (w.Text, (_usesXpos ? w.Xpos : w.Upos) ?? throw new InvalidOperationException("Run the POS tagger before the parser"))).ToList()).ToList();
        var reps = charlms == null ? null
            : sentences.Select(s => charlms.TryGet(s, out var r) ? r : ((Tensor, Tensor)?)null).ToList();
        var trees = Parse(tagged, charlmReps: reps);
        for (int i = 0; i < sentences.Count; i++)
            sentences[i].Constituency = trees[i];
    }

    /// <summary>
    /// Parses (word, tag) sentences. A sentence the parser gets stuck on gives null, as Stanza drops it.
    /// <paramref name="scores"/>, if given, receives each sentence's output-layer rows per step (for tests).
    /// </summary>
    /// <remarks>
    /// Scheduled like Stanza (ConstituencyProcessor + parse_sentences): sentences sorted longest first,
    /// <see cref="BatchSize"/> states in flight, and each finished state replaced by the next one, whose
    /// word queues are built <see cref="BatchSize"/> sentences at a time. Every state is computed
    /// independently, so this only decides how much work each step does.
    /// </remarks>
    internal List<Tree?> Parse(IReadOnlyList<IReadOnlyList<(string Word, string Tag)>> sentences, List<List<float[]>>? scores = null,
        IReadOnlyList<(Tensor Forward, Tensor Backward)?>? charlmReps = null)
    {
        using var noGrad = torch.no_grad();
        var order = Enumerable.Range(0, sentences.Count).OrderByDescending(i => sentences[i].Count).ToArray();
        scores?.AddRange(sentences.Select(_ => new List<float[]>()));
        var result = new Tree?[sentences.Count];
        var batch = new List<ParserState>();
        var horizon = new Queue<ParserState>();
        int built = 0;
        try
        {
            while (true)
            {
                while (batch.Count < BatchSize)
                {
                    if (horizon.Count == 0)
                    {
                        if (built == order.Length)
                            break;
                        var chunk = order[built..Math.Min(built + BatchSize, order.Length)];
                        built += chunk.Length;
                        foreach (var s in InitialStates(chunk, sentences, charlmReps))
                            horizon.Enqueue(s);
                    }
                    batch.Add(horizon.Dequeue());
                }
                if (batch.Count == 0)
                    break;

                // Tensors a state keeps are detached from this scope into ParserState.Owned (see Apply).
                using var step = NewDisposeScope();
                var logits = Forward(batch);
                int n = _transitions.Length;
                var chosen = new Transition?[batch.Count];
                for (int k = 0; k < batch.Count; k++)
                {
                    var row = logits[(k * n)..((k + 1) * n)];
                    scores?[batch[k].Index].Add(row);
                    chosen[k] = Choose(batch[k], row);
                }
                Apply(batch, chosen, step);

                batch.RemoveAll(s =>
                {
                    if (!s.Broken && !s.Finished(_rootLabels))
                        return false;
                    if (!s.Broken)
                        result[s.Index] = s.Constituents.Value.Tree;
                    s.Dispose();
                    return true;
                });
            }
        }
        finally
        {
            foreach (var s in batch.Concat(horizon))
                s.Dispose();
        }
        return result.ToList();
    }

    // ----- initial state: the word queue (initial_word_queues) -----

    private List<ParserState> InitialStates(int[] indices, IReadOnlyList<IReadOnlyList<(string Word, string Tag)>> all,
        IReadOnlyList<(Tensor Forward, Tensor Backward)?>? charlmReps)
    {
        using var scope = NewDisposeScope();
        var sentences = indices.Select(i => all[i]).ToList();
        // Charlm outputs the tagger kept, computing only the missing ones.
        var charsForward = indices.Select(i => charlmReps?[i]?.Forward).ToList();
        var charsBackward = indices.Select(i => charlmReps?[i]?.Backward).ToList();
        var missing = Enumerable.Range(0, indices.Length).Where(k => charsForward[k] is null).ToList();
        if (missing.Count > 0)
        {
            var words = missing.Select(k => (IReadOnlyList<string>)sentences[k].Select(x => x.Word).ToList()).ToList();
            var forward = _charlmForward.BuildCharRepresentation(words);
            var backward = _charlmBackward.BuildCharRepresentation(words);
            for (int m = 0; m < missing.Count; m++)
                (charsForward[missing[m]], charsBackward[missing[m]]) = (forward[m], backward[m]);
        }

        var inputs = new List<Tensor>(sentences.Count);
        for (int i = 0; i < sentences.Count; i++)
        {
            var pretrainIds = sentences[i].Select(x =>
            {
                int id = _pretrain.UnitToId(x.Word);
                return (long)(id != _pretrain.UnkId ? id : _pretrain.UnitToId(PyString.Lower(x.Word)));
            }).ToArray();
            var deltaIds = sentences[i].Select(x => (long)_deltaMap.GetValueOrDefault(x.Word, 1)).ToArray();
            var tagIds = sentences[i].Select(x => (long)_tagMap.GetValueOrDefault(x.Tag, 1)).ToArray();
            var wordInput = cat([
                _pretrain.Embeddings[torch.tensor(pretrainIds, device: _device)],
                _deltaEmbedding.forward(torch.tensor(deltaIds, device: _device)),
                _tagEmbedding.forward(torch.tensor(tagIds, device: _device)),
                charsForward[i]!,
                charsBackward[i]!,
            ], 1);
            inputs.Add(cat([_wordStart.unsqueeze(0), wordInput, _wordEnd.unsqueeze(0)], 0));
        }

        var lengths = sentences.Select(s => (long)s.Count + 2).ToArray();
        var output = Rnn.RunPacked(_wordLstm, Rnn.PadSequence(inputs), lengths);
        var wordHx = F.relu(_wordToConstituent.forward(output));

        return sentences.Select((s, i) => new ParserState
        {
            Index = indices[i],
            SentenceLength = s.Count,
            Preterminals = s.Select(x => new Tree(x.Tag, [new Tree(x.Word)])).ToArray(),
            WordHx = scope.Detach(wordHx[i]),
            Transitions = _initialTransitions,
            Constituents = _initialConstituents,
        }).ToList();
    }

    // ----- scoring and choosing a transition -----

    /// <summary>LSTMModel.forward: [batch * transitions] scores from the next word, transition and constituent stacks.</summary>
    private float[] Forward(List<ParserState> states)
    {
        var word = stack(states.Select(s => s.WordHx[s.WordPosition + 1]).ToArray());
        var transition = stack(states.Select(s => s.Transitions.Output).ToArray());
        var constituent = stack(states.Select(s => s.Constituents.Output).ToArray());
        var hx = cat([word, transition, constituent], 1);
        foreach (var layer in _outputLayers)
            hx = layer.forward(F.relu(hx)); // nonlinearity before every layer, including the first
        return hx.ToArray<float>();
    }

    /// <summary>The best-scoring transition, or the best legal one if that is illegal (LSTMModel.predict).</summary>
    private Transition? Choose(ParserState state, float[] row)
    {
        int best = 0;
        for (int j = 1; j < row.Length; j++)
            if (row[j] > row[best])
                best = j;
        if (state.IsLegal(_transitions[best], _rootLabels, _unaryLimit))
            return _transitions[best];
        return Enumerable.Range(0, row.Length).OrderByDescending(j => row[j])
            .Select(j => _transitions[j]).FirstOrDefault(t => state.IsLegal(t, _rootLabels, _unaryLimit));
    }

    // ----- applying transitions (bulk_apply) -----

    private void Apply(List<ParserState> states, Transition?[] transitions, DisposeScope step)
    {
        var applied = new List<(ParserState State, Transition Transition, StackNode<Constituent> Base)>();
        var newConstituents = new List<Constituent>();
        var opens = new List<(int Slot, string Label, int OpenIndex)>();
        var closes = new List<(int Slot, string Label, List<Constituent> Children)>();

        for (int k = 0; k < states.Count; k++)
        {
            var s = states[k];
            var t = transitions[k];
            // bulk_apply: no legal transition, or far too many transitions, breaks the state.
            if (t == null || s.NumTransitions >= (s.SentenceLength + 2) * 20)
            {
                s.Broken = true;
                continue;
            }

            int slot = newConstituents.Count;
            switch (t.Kind)
            {
                case TransitionKind.Shift:
                    newConstituents.Add(new Constituent(s.Preterminals[s.WordPosition], null, s.WordHx[s.WordPosition + 1]));
                    applied.Add((s, t, s.Constituents));
                    s.WordPosition++;
                    break;

                case TransitionKind.Open:
                    newConstituents.Add(null!);
                    opens.Add((slot, t.Label!, t.OpenIndex));
                    applied.Add((s, t, s.Constituents));
                    s.NumOpens++;
                    break;

                case TransitionKind.Close:
                    // In-order: children above the open marker, the marker, then the first child below it.
                    var children = new List<Constituent>();
                    var node = s.Constituents;
                    while (!node.Value.IsOpenMarker)
                    {
                        children.Add(node.Value);
                        node = node.Parent!;
                    }
                    var label = node.Value.OpenLabel!;
                    node = node.Parent!;
                    children.Add(node.Value);
                    node = node.Parent!;
                    children.Reverse();
                    newConstituents.Add(null!);
                    closes.Add((slot, label, children));
                    applied.Add((s, t, node));
                    s.NumOpens--;
                    break;
            }
        }
        if (applied.Count == 0)
            return;

        if (opens.Count > 0)
        {
            var hx = _dummyEmbedding.forward(torch.tensor(opens.Select(o => (long)o.OpenIndex).ToArray(), device: _device));
            for (int i = 0; i < opens.Count; i++)
                newConstituents[opens[i].Slot] = new Constituent(null, opens[i].Label, hx[i]);
        }
        if (closes.Count > 0)
        {
            // MAX composition: elementwise max over the children, then relu(reduce_linear(.)).
            var pooled = stack(closes.Select(c => stack(c.Children.Select(x => x.Hx!).ToArray()).max(0).values).ToArray());
            var hx = F.relu(_reduceLinear.forward(pooled));
            for (int i = 0; i < closes.Count; i++)
            {
                var tree = new Tree(closes[i].Label, closes[i].Children.Select(c => c.Tree!).ToList());
                newConstituents[closes[i].Slot] = new Constituent(tree, null, hx[i]);
            }
        }

        var transitionInput = _transitionEmbedding.forward(torch.tensor(applied.Select(a => (long)a.Transition.Index).ToArray(), device: _device));
        var newTransitions = Push(_transitionLstm, applied.Select(a => a.State.Transitions).ToList(),
            applied.Select(a => (Transition?)a.Transition).ToList(), transitionInput);
        var constituentInput = stack(newConstituents.Select(c => c.Hx!).ToArray());
        var newStacks = Push(_constituentLstm, applied.Select(a => a.Base).ToList(), newConstituents, constituentInput);

        for (int i = 0; i < applied.Count; i++)
        {
            var (t, c) = (newTransitions[i], newStacks[i]);
            applied[i].State.Transitions = t;
            applied[i].State.Constituents = c;
            // Shifted words' vectors are views of WordHx; disposing a view leaves the storage to the others.
            Tensor[] kept = [t.Hx, t.Cx, t.Output, c.Hx, c.Cx, c.Output, c.Value.Hx!];
            step.Detach((IEnumerable<IDisposable>)kept);
            applied[i].State.Owned.AddRange(kept);
        }
    }

    // ----- LSTM stacks (lstm_tree_stack.py) -----

    private static StackNode<T> InitialStack<T>(LSTM lstm, Tensor startEmbedding, T value)
    {
        var (output, hx, cx) = lstm.forward(startEmbedding.view(1, 1, -1));
        return new StackNode<T>(value, null, hx.squeeze(1), cx.squeeze(1), output[0, 0]);
    }

    /// <summary>Runs the LSTM one step for each stack from its own state and pushes the results.</summary>
    private static List<StackNode<T>> Push<T>(LSTM lstm, List<StackNode<T>> stacks, List<T> values, Tensor inputs)
    {
        var hx = torch.stack(stacks.Select(s => s.Hx).ToArray(), 1);
        var cx = torch.stack(stacks.Select(s => s.Cx).ToArray(), 1);
        var (output, hn, cn) = lstm.forward(inputs.unsqueeze(0), (hx, cx));
        return stacks.Select((s, i) => new StackNode<T>(values[i], s, hn.select(1, i), cn.select(1, i), output[0, i])).ToList();
    }

    // ----- loading -----

    private static readonly Regex OpenRepr = new(@"^OpenConstituent\(\('([^']*)',\)\)$");

    private static Transition ParseTransition(string repr, int index, List<string> opens)
    {
        if (repr == "Shift")
            return new Transition(index, TransitionKind.Shift);
        if (repr == "CloseConstituent")
            return new Transition(index, TransitionKind.Close);
        var m = OpenRepr.Match(repr);
        if (!m.Success)
            throw new NotSupportedException($"Transition {repr} is not ported");
        var label = m.Groups[1].Value;
        return new Transition(index, TransitionKind.Open, label, opens.IndexOf(label));
    }

    private static string OpenLabel(JsonNode? node)
    {
        var parts = node!["$tuple"]!.AsArray();
        if (parts.Count != 1)
            throw new NotSupportedException("Compound open labels are not ported");
        return parts[0]!.GetValue<string>();
    }

    private static void CheckSupported(JsonNode p, JsonNode config)
    {
        void Require(string key, string expected)
        {
            var actual = config[key]?.ToString();
            if (actual != expected)
                throw new NotSupportedException($"Parser checkpoint has {key} = {actual ?? "null"}; only {expected} is ported");
        }
        if (p["model_type"]?.GetValue<string>() != "LSTM")
            throw new NotSupportedException("Only LSTM parser models are ported");
        Require("transition_scheme", "IN_ORDER");
        Require("constituency_composition", "MAX");
        Require("transition_stack", "LSTM");
        Require("constituent_stack", "LSTM");
        Require("sentence_boundary_vectors", "EVERYTHING");
        Require("nonlinearity", "relu");
        Require("num_tree_lstm_layers", "1");
        Require("reversed", "false");
        if (config["bert_model"] != null)
            throw new NotSupportedException("Parser checkpoints with a transformer are not ported");
        if (config["pattn_num_layers"]?.GetValue<int>() > 0 && config["pattn_num_heads"]?.GetValue<int>() > 0)
            throw new NotSupportedException("Partitioned attention is not ported");
        if (config["use_lattn"]?.GetValue<bool>() == true && config["lattn_d_proj"]?.GetValue<int>() > 0)
            throw new NotSupportedException("Label attention is not ported");
        if (config["use_rattn"]?.GetValue<bool>() == true)
            throw new NotSupportedException("Relative attention is not ported");
        if (config["maxout_k"]?.GetValue<int>() > 0)
            throw new NotSupportedException("Maxout output layers are not ported");
    }

    private static List<string> Strings(JsonNode? array) => array!.AsArray().Select(x => x!.GetValue<string>()).ToList();

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _deltaEmbedding, _tagEmbedding, _transitionEmbedding, _dummyEmbedding,
                     _wordLstm, _transitionLstm, _constituentLstm, _wordToConstituent, _reduceLinear })
            m.Dispose();
        foreach (var l in _outputLayers)
            l.Dispose();
        foreach (var t in new[] { _wordStart, _wordEnd, _initialTransitions.Hx, _initialTransitions.Cx, _initialTransitions.Output,
                     _initialConstituents.Hx, _initialConstituents.Cx, _initialConstituents.Output })
            t.Dispose();
    }
}
