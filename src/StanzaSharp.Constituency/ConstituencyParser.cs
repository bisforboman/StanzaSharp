using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Constituency;

/// <summary>
/// Constituency parser: an in-order transition parser scored by LSTMs, ported from
/// stanza/models/constituency/lstm_model.py <c>LSTMModel</c> and base_model.py for the English
/// configuration:
/// - IN_ORDER transitions
/// - LSTM transition and constituent stacks
/// - MAX composition
/// - charlm, no attention, no transformer
/// The network is an <see cref="IConstituencyNet"/> (per <see cref="Backend"/>); the transition system, the parser
/// states, the vocab lookups and the scheduling are here.
/// </summary>
internal sealed class ConstituencyParser : IDisposable
{
    private const int BatchSize = 50;

    /// <summary>The model's transitions, in score column order.</summary>
    internal IReadOnlyList<Transition> Transitions => _transitions;

    private readonly Pretrain _pretrain;
    private readonly Dictionary<string, int> _deltaMap, _tagMap;
    private readonly Transition[] _transitions;
    private readonly HashSet<string> _rootLabels;
    private readonly int _unaryLimit;
    private readonly bool _usesXpos;
    private readonly IConstituencyNet _net;
    private readonly StackNode<Transition?> _initialTransitions;
    private readonly StackNode<Constituent> _initialConstituents;

    internal ConstituencyParser(Checkpoint ckpt, Pretrain pretrain, Func<IConstituencyNet> net)
    {
        _pretrain = pretrain;
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        CheckSupported(p, config);

        // LSTMModel maps words and tags to i + 2 (0 = PAD, 1 = UNK) over sorted lists; the saved lists are sorted.
        _deltaMap = Strings(p["words"]).Select((w, i) => (w, i)).ToDictionary(x => x.w, x => x.i + 2);
        _tagMap = Strings(p["tags"]).Select((t, i) => (t, i)).ToDictionary(x => x.t, x => x.i + 2);
        _rootLabels = Strings(p["root_labels"]).ToHashSet();
        _unaryLimit = p["unary_limit"]!.GetValue<int>();
        _usesXpos = config["retag_method"]?.GetValue<string>() == "xpos";
        var opens = p["constituent_opens"]!.AsArray().Select(OpenLabel).ToList();
        _transitions = Strings(p["transitions"]).Select((t, i) => ParseTransition(t, i, opens)).ToArray();

        _net = net();
        _initialTransitions = new StackNode<Transition?>(null, null, _net.TransitionStart);
        _initialConstituents = new StackNode<Constituent>(new Constituent(null, null, null), null, _net.ConstituentStart);
    }

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static ConstituencyParser LoadManaged(string basePath, Pretrain pretrain, ManagedCharLanguageModel charlmForward, ManagedCharLanguageModel charlmBackward)
    {
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");
        var ckpt = Checkpoint.Load(basePath);
        return new ConstituencyParser(ckpt, pretrain, () => new ManagedConstituencyNet(ckpt, pretrain, charlmForward, charlmBackward));
    }

    /// <summary>Sets <see cref="Sentence.Constituency"/> on every sentence. Needs XPOS (or UPOS) tags from the tagger.</summary>
    /// <param name="charlms">Charlm representations the tagger kept, if any; sentences missing from it are computed.</param>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default)
    {
        var sentences = doc.Sentences.Where(s => s.Tokens.Count > 0).ToList();
        var tagged = sentences.Select(s => (IReadOnlyList<(string, string)>)s.Words.Select(w =>
            (w.Text, (_usesXpos ? w.Xpos : w.Upos) ?? throw new InvalidOperationException("Run the POS tagger before the parser"))).ToList()).ToList();
        var trees = Parse(tagged, charlms: charlms, cacheKeys: charlms == null ? null : sentences, cancellationToken: cancellationToken);
        for (int i = 0; i < sentences.Count; i++)
            sentences[i].Constituency = trees[i];
    }

    /// <summary>
    /// Parses (word, tag) sentences. A sentence the parser gets stuck on gives null, as Stanza drops it.
    /// <paramref name="scores"/>, if given, receives each sentence's output-layer rows per step (for tests), and
    /// <paramref name="onStep"/> each step's sentence index, row, and which transitions were legal.
    /// </summary>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: charlm outputs already computed; sentence i's are looked up
    /// under its key.</param>
    /// <remarks>
    /// Scheduled like Stanza (ConstituencyProcessor + parse_sentences): sentences sorted longest first,
    /// <see cref="BatchSize"/> states in flight, and each finished state replaced by the next one, whose
    /// word queues are built <see cref="BatchSize"/> sentences at a time. Every state is computed
    /// independently, so this only decides how much work each step does.
    /// </remarks>
    internal List<Tree?> Parse(IReadOnlyList<IReadOnlyList<(string Word, string Tag)>> sentences, List<List<float[]>>? scores = null,
        CharlmCache? charlms = null, IReadOnlyList<Sentence>? cacheKeys = null, CancellationToken cancellationToken = default,
        Action<int, float[], bool[]>? onStep = null)
    {
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
                // Every step: a parse takes about two steps per word, each one forward pass over the batch.
                // The finally below disposes the states in flight.
                cancellationToken.ThrowIfCancellationRequested();
                while (batch.Count < BatchSize)
                {
                    if (horizon.Count == 0)
                    {
                        if (built == order.Length)
                            break;
                        var chunk = order[built..Math.Min(built + BatchSize, order.Length)];
                        built += chunk.Length;
                        foreach (var s in InitialStates(chunk, sentences, charlms, cacheKeys, cancellationToken))
                            horizon.Enqueue(s);
                    }
                    batch.Add(horizon.Dequeue());
                }
                if (batch.Count == 0)
                    break;

                var logits = _net.Score(batch.Select(s => (s.WordHx, s.WordPosition + 1, s.Transitions.State, s.Constituents.State)).ToList(),
                    cancellationToken);
                int n = _transitions.Length;
                var chosen = new Transition?[batch.Count];
                for (int k = 0; k < batch.Count; k++)
                {
                    var row = logits[(k * n)..((k + 1) * n)];
                    scores?[batch[k].Index].Add(row);
                    onStep?.Invoke(batch[k].Index, row, _transitions.Select(t => batch[k].IsLegal(t, _rootLabels, _unaryLimit)).ToArray());
                    chosen[k] = Choose(batch[k], row);
                }
                Apply(batch, chosen, cancellationToken);

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
        CharlmCache? charlms, IReadOnlyList<Sentence>? cacheKeys, CancellationToken ct)
    {
        var sentences = indices.Select(i => all[i]).ToList();
        var inputs = sentences.Select((s, k) => new WordInput(
            s.Select(x => x.Word).ToList(),
            s.Select(x =>
            {
                int id = _pretrain.UnitToId(x.Word);
                return (long)(id != _pretrain.UnkId ? id : _pretrain.UnitToId(PyString.Lower(x.Word)));
            }).ToArray(),
            s.Select(x => (long)_deltaMap.GetValueOrDefault(x.Word, 1)).ToArray(),
            s.Select(x => (long)_tagMap.GetValueOrDefault(x.Tag, 1)).ToArray(),
            cacheKeys?[indices[k]])).ToList();
        var words = _net.EncodeWords(inputs, charlms, ct);

        return sentences.Select((s, i) => new ParserState
        {
            Index = indices[i],
            SentenceLength = s.Count,
            Preterminals = s.Select(x => new Tree(x.Tag, [new Tree(x.Word)])).ToArray(),
            WordHx = words[i],
            Transitions = _initialTransitions,
            Constituents = _initialConstituents,
        }).ToList();
    }

    // ----- choosing a transition -----

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

    private void Apply(List<ParserState> states, Transition?[] transitions, CancellationToken ct)
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
                    newConstituents.Add(new Constituent(s.Preterminals[s.WordPosition], null, _net.Word(s.WordHx, s.WordPosition + 1)));
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
            var hx = _net.Open(opens.Select(o => o.OpenIndex).ToArray());
            for (int i = 0; i < opens.Count; i++)
                newConstituents[opens[i].Slot] = new Constituent(null, opens[i].Label, hx[i]);
        }
        if (closes.Count > 0)
        {
            // MAX composition: elementwise max over the children, then relu(reduce_linear(.)).
            var hx = _net.Compose(closes.Select(c => (IReadOnlyList<object>)c.Children.Select(x => x.Hx!).ToList()).ToList(), ct);
            for (int i = 0; i < closes.Count; i++)
            {
                var tree = new Tree(closes[i].Label, closes[i].Children.Select(c => c.Tree!).ToList());
                newConstituents[closes[i].Slot] = new Constituent(tree, null, hx[i]);
            }
        }

        var newTransitions = _net.PushTransitions(applied.Select(a => a.State.Transitions.State).ToList(),
            applied.Select(a => a.Transition.Index).ToArray(), ct);
        var newStacks = _net.PushConstituents(applied.Select(a => a.Base.State).ToList(), newConstituents.Select(c => c.Hx!).ToList(), ct);

        for (int i = 0; i < applied.Count; i++)
        {
            var (s, t, below) = applied[i];
            s.Transitions = new StackNode<Transition?>(t, s.Transitions, newTransitions[i]);
            s.Constituents = new StackNode<Constituent>(newConstituents[i], below, newStacks[i]);
            s.Own(newTransitions[i]);
            s.Own(newStacks[i]);
            s.Own(newConstituents[i].Hx);
        }
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

    public void Dispose() => _net.Dispose();
}
