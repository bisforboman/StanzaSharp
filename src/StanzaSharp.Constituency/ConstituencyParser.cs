using System.Buffers;
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
    private readonly IConstituencyNet _net;
    private readonly StackNode<Transition?> _initialTransitions;
    private readonly StackNode<Constituent> _initialConstituents;
    private readonly TransformerEmbedder? _transformer; // a LoRA parser's own (spmrl_german-nlp-electra), else the pipeline's
    private readonly ITransformerNet? _loraNet;
    private readonly float[]? _layerMix;

    /// <summary>Whether the parser reads XPOS (<c>retag_method: xpos</c>, English) rather than UPOS (German).</summary>
    internal bool UsesXpos { get; }

    /// <param name="transformer">The transformer a transformer parser reads (the pipeline's, shared); else ignored. A
    /// LoRA-adapted parser (<c>use_peft</c>, German <c>spmrl_german-nlp-electra</c>) puts its adapters (the checkpoint's
    /// top-level <c>bert_lora</c>) onto a copy that shares the transformer's weights, like the sentiment classifier.</param>
    internal ConstituencyParser(Checkpoint ckpt, Pretrain pretrain, Func<IConstituencyNet> net, TransformerEmbedder? transformer = null)
    {
        _pretrain = pretrain;
        var p = ckpt.Root["params"]!;
        var config = p["config"]!;
        CheckSupported(p, config);
        bool lora = config["use_peft"]?.GetValue<bool>() == true;
        _layerMix = TransformerEmbedder.LayerMix(ckpt, config, p["model"]!, transformer, "constituency", lora);
        if (config["bert_model"] != null)
        {
            _loraNet = lora
                ? transformer!.Net.WithLora(LoraWeights.Read(ckpt, ckpt.Root["bert_lora"] ?? throw new InvalidDataException("A LoRA parser without bert_lora"), config))
                : null;
            _transformer = _loraNet == null ? transformer : new TransformerEmbedder(_loraNet, transformer!.Tokenizer, transformer.MaxLength, transformer.Name);
        }

        // LSTMModel maps words and tags to i + 2 (0 = PAD, 1 = UNK) over sorted lists; the saved lists are sorted.
        _deltaMap = Strings(p["words"]).Select((w, i) => (w, i)).ToDictionary(x => x.w, x => x.i + 2);
        _tagMap = Strings(p["tags"]).Select((t, i) => (t, i)).ToDictionary(x => x.t, x => x.i + 2);
        _rootLabels = Strings(p["root_labels"]).ToHashSet();
        _unaryLimit = p["unary_limit"]!.GetValue<int>();
        UsesXpos = config["retag_method"]?.GetValue<string>() == "xpos";
        var opens = p["constituent_opens"]!.AsArray().Select(OpenLabel).ToList();
        _transitions = Strings(p["transitions"]).Select((t, i) => ParseTransition(t, i, opens)).ToArray();

        _net = net();
        _initialTransitions = new StackNode<Transition?>(null, null, _net.TransitionStart);
        _initialConstituents = new StackNode<Constituent>(new Constituent(null, null, null), null, _net.ConstituentStart);
    }

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static ConstituencyParser LoadManaged(string basePath, Pretrain pretrain, ManagedCharLanguageModel charlmForward, ManagedCharLanguageModel charlmBackward,
        TransformerEmbedder? transformer = null)
    {
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");
        var ckpt = Checkpoint.Load(basePath);
        int transformerDim = TransformerDim(ckpt, transformer);
        return new ConstituencyParser(ckpt, pretrain, () => new ManagedConstituencyNet(ckpt, pretrain, charlmForward, charlmBackward, transformerDim), transformer);
    }

    /// <summary>The input columns a parser gives <paramref name="transformer"/>'s vectors: its hidden size for a <c>_electra-large</c> parser, else 0.</summary>
    internal static int TransformerDim(Checkpoint ckpt, TransformerEmbedder? transformer) =>
        ckpt.Root["params"]!["config"]!["bert_model"] == null ? 0 : transformer?.Net.Config.Hidden ?? 0;

    /// <summary>Whether this parser reads a transformer (<c>_electra-large</c>).</summary>
    internal bool UsesTransformer => _transformer != null;

    /// <summary>The sentences this parser will send its transformer in <paramref name="doc"/>, for <see cref="TransformerEmbedder.Prefill"/> (null without one).</summary>
    internal TransformerReader? TransformerReader(Document doc) => _transformer == null
        ? null
        : new TransformerReader(_transformer, _layerMix, [.. doc.Sentences.Where(s => s.Tokens.Count > 0).Select(s => (IReadOnlyList<string>)[.. s.Words.Select(w => w.Text)])]);

    /// <summary>Sets <see cref="Sentence.Constituency"/> on every sentence. Needs XPOS (or UPOS) tags from the tagger.</summary>
    /// <param name="charlms">Charlm representations the tagger kept, if any; sentences missing from it are computed.</param>
    /// <param name="transformers">The call's transformer states (a <c>_electra-large</c> parser), shared with the other readers.</param>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default, TransformerCache? transformers = null)
    {
        var sentences = doc.Sentences.Where(s => s.Tokens.Count > 0).ToList();
        var tagged = sentences.Select(s => (IReadOnlyList<(string, string)>)s.Words.Select(w =>
            (w.Text, (UsesXpos ? w.Xpos : w.Upos) ?? throw new InvalidOperationException("Run the POS tagger before the parser"))).ToList()).ToList();
        var trees = Parse(tagged, charlms: charlms, cacheKeys: charlms == null ? null : sentences, cancellationToken: cancellationToken,
            transformers: transformers);
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
        Action<int, float[], bool[]>? onStep = null, TransformerCache? transformers = null)
    {
        var order = Enumerable.Range(0, sentences.Count).OrderByDescending(i => sentences[i].Count).ToArray();
        scores?.AddRange(sentences.Select(_ => new List<float[]>()));
        var result = new Tree?[sentences.Count];
        var batch = new List<ParserState>();
        var horizon = new Queue<ParserState>();
        var scoreInputs = new List<(object, int, object, object)>(BatchSize); // refilled every step
        var steps = new StepBuffers();
        int n = _transitions.Length;
        var logits = ArrayPool<float>.Shared.Rent(BatchSize * n); // [batch, transitions], written by Score each step
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
                        foreach (var s in InitialStates(chunk, sentences, charlms, cacheKeys, cancellationToken, transformers))
                            horizon.Enqueue(s);
                    }

                    batch.Add(horizon.Dequeue());
                }

                if (batch.Count == 0)
                    break;

                scoreInputs.Clear();
                foreach (var s in batch)
                    scoreInputs.Add((s.WordHx, s.WordPosition + 1, s.Transitions.State, s.Constituents.State));
                _net.Score(scoreInputs, logits.AsSpan(0, batch.Count * n), cancellationToken);
                var chosen = steps.Chosen;
                for (int k = 0; k < batch.Count; k++)
                {
                    var row = logits.AsSpan(k * n, n);
                    scores?[batch[k].Index].Add(row.ToArray());
                    onStep?.Invoke(batch[k].Index, row.ToArray(), _transitions.Select(t => batch[k].IsLegal(t, _rootLabels, _unaryLimit)).ToArray());
                    chosen[k] = Choose(batch[k], row);
                }

                Apply(batch, chosen, steps, cancellationToken);

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
            ArrayPool<float>.Shared.Return(logits);
        }

        return result.ToList();
    }

    // ----- initial state: the word queue (initial_word_queues) -----
    private List<ParserState> InitialStates(int[] indices, IReadOnlyList<IReadOnlyList<(string Word, string Tag)>> all,
        CharlmCache? charlms, IReadOnlyList<Sentence>? cacheKeys, CancellationToken ct, TransformerCache? transformers = null)
    {
        var sentences = indices.Select(i => all[i]).ToList();

        // initial_word_queues: extract_bert_embeddings over the words as written, with the endpoints ([CLS] next to the start
        // sentinel, [SEP] next to the end one).
        var transformer = _transformer?.Embed([.. sentences.Select(s => (IReadOnlyList<string>)[.. s.Select(x => x.Word)])], keepEndpoints: true,
            _layerMix, transformers, ct);
        var inputs = sentences.Select((s, k) => new WordInput(
            s.Select(x => x.Word).ToList(),
            s.Select(x =>
            {
                int id = _pretrain.UnitToId(x.Word);
                return (long)(id != _pretrain.UnkId ? id : _pretrain.UnitToId(PyString.Lower(x.Word)));
            }).ToArray(),
            s.Select(x => (long)_deltaMap.GetValueOrDefault(x.Word, 1)).ToArray(),
            s.Select(x => (long)_tagMap.GetValueOrDefault(x.Tag, 1)).ToArray(),
            cacheKeys?[indices[k]], transformer?[k])).ToList();
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
    private Transition? Choose(ParserState state, ReadOnlySpan<float> scores)
    {
        int best = 0;
        for (int j = 1; j < scores.Length; j++)
        {
            if (scores[j] > scores[best])
                best = j;
        }

        return state.IsLegal(_transitions[best], _rootLabels, _unaryLimit) ? _transitions[best] : BestLegal(state, scores.ToArray());
    }

    /// <summary>The best legal transition (separate from <see cref="Choose"/>, so the common path allocates no closure).</summary>
    private Transition? BestLegal(ParserState state, float[] row) =>
        Enumerable.Range(0, row.Length).OrderByDescending(j => row[j])
            .Select(j => _transitions[j])
            .FirstOrDefault(t => state.IsLegal(t, _rootLabels, _unaryLimit));

    // ----- applying transitions (bulk_apply) -----

    /// <summary>One <see cref="Parse"/> call's per-step lists, cleared and refilled each step instead of made anew.</summary>
    private sealed class StepBuffers
    {
        public readonly Transition?[] Chosen = new Transition?[BatchSize];
        public readonly List<(ParserState State, Transition Transition, StackNode<Constituent> Base)> Applied = new(BatchSize);
        public readonly List<Constituent> NewConstituents = new(BatchSize);
        public readonly List<(int Slot, string Label, int OpenIndex)> Opens = new(BatchSize);
        public readonly List<(int Slot, string Label, List<Constituent> Children)> Closes = new(BatchSize);
        public readonly List<object> Parents = new(BatchSize), Inputs = new(BatchSize), Bases = new(BatchSize);
    }

    private void Apply(List<ParserState> states, Transition?[] transitions, StepBuffers steps, CancellationToken ct)
    {
        var (applied, newConstituents, opens, closes) = (steps.Applied, steps.NewConstituents, steps.Opens, steps.Closes);
        applied.Clear();
        newConstituents.Clear();
        opens.Clear();
        closes.Clear();

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
                var tree = new Tree(closes[i].Label, closes[i].Children.Select(c => c.Tree!).ToArray()); // an array: no List object per node
                newConstituents[closes[i].Slot] = new Constituent(tree, null, hx[i]);
            }
        }

        int count = applied.Count;
        var (parents, inputs, bases) = (steps.Parents, steps.Inputs, steps.Bases);
        parents.Clear();
        inputs.Clear();
        bases.Clear();
        var indices = new int[count];
        for (int i = 0; i < count; i++)
        {
            parents.Add(applied[i].State.Transitions.State);
            indices[i] = applied[i].Transition.Index;
            inputs.Add(newConstituents[i].Hx!);
            bases.Add(applied[i].Base.State);
        }

        var newTransitions = _net.PushTransitions(parents, indices, ct);
        var newStacks = _net.PushConstituents(bases, inputs, ct);

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
        if (config["pattn_num_layers"]?.GetValue<int>() > 0 && config["pattn_num_heads"]?.GetValue<int>() > 0)
            throw new NotSupportedException("Partitioned attention is not ported");
        if (config["use_lattn"]?.GetValue<bool>() == true && config["lattn_d_proj"]?.GetValue<int>() > 0)
            throw new NotSupportedException("Label attention is not ported");
        if (config["use_rattn"]?.GetValue<bool>() == true)
            throw new NotSupportedException("Relative attention is not ported");
        if (config["maxout_k"]?.GetValue<int>() > 0)
            throw new NotSupportedException("Maxout output layers are not ported");
    }

    private static string[] Strings(JsonNode? array) => Checkpoint.Strings(array); // "words" is long: a JsonValue in a .pt

    public void Dispose()
    {
        _net.Dispose();
        _loraNet?.Dispose();
    }
}
