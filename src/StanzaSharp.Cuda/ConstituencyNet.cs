using System.Buffers;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Constituency;

/// <summary>The parser's network on TorchSharp.</summary>
/// <remarks>
/// Every vector and stack state the parser holds is a row of a <see cref="Bank"/>, and a handle is its row number, so a
/// step reads and writes each bank with one <c>index_select</c> / <c>index_copy_</c> whatever the number of states (it
/// took a view per state and a stack of them before round 25). The banks belong to the calling thread, since Parse is
/// synchronous, so concurrent calls never share one. Copying rows moves no bits: the results are the same.
/// </remarks>
internal sealed class ConstituencyNet : IConstituencyNet
{
    /// <summary>A row of a bank that something else owns (a word row, a stack's start state).</summary>
    private class Row(int slot)
    {
        public readonly int Slot = slot;
    }

    /// <summary>A row the parser owns: disposing it gives the row back.</summary>
    private sealed class OwnedRow(Bank bank, int slot) : Row(slot), IDisposable
    {
        public void Dispose() => bank.Free(Slot);
    }

    /// <summary>A sentence's word vectors (<see cref="EncodeWords"/>): one vector-bank row per position.</summary>
    private sealed class WordRows(Bank bank, int[] slots) : IDisposable
    {
        public readonly int[] Slots = slots;

        public void Dispose()
        {
            foreach (var slot in Slots)
                bank.Free(slot);
        }
    }

    /// <summary>
    /// Rows of one shape in one growing tensor, with a free list. Row 0 may be a fixed start row. The tensor is freed when
    /// the last row given out comes back (the end of a Parse call), and made again on the next use.
    /// </summary>
    private sealed class Bank(long[] rowShape, Tensor? start, Device device)
    {
        private readonly Stack<int> _free = new();
        private int _next, _live;
        private Tensor? _data;

        public Tensor Data
        {
            get
            {
                if (_data is null)
                {
                    _data = empty([64, .. rowShape], device: device).DetachFromDisposeScope();
                    if (start is not null)
                        _data[0].copy_(start);
                    _next = start is null ? 0 : 1;
                }

                return _data;
            }
        }

        /// <summary>Writes <paramref name="rows"/> ([n, ..row]) to n free rows.</summary>
        public int[] Add(Tensor rows)
        {
            var data = Data;
            var slots = new int[rows.shape[0]];
            for (int i = 0; i < slots.Length; i++)
                slots[i] = _free.Count > 0 ? _free.Pop() : _next++;
            if (_next > data.shape[0])
            {
                var grown = empty([Math.Max(2 * data.shape[0], _next), .. rowShape], device: device).DetachFromDisposeScope();
                grown.narrow(0, 0, data.shape[0]).copy_(data);
                data.Dispose();
                _data = data = grown;
            }

            _live += slots.Length;
            data.index_copy_(0, Index(slots, device), rows);
            return slots;
        }

        public void Free(int slot)
        {
            _free.Push(slot);
            if (--_live == 0)
            {
                _data!.Dispose();
                _data = null;
                _free.Clear();
            }
        }
    }

    /// <summary>The calling thread's banks for this net.</summary>
    private sealed class Banks(ConstituencyNet net)
    {
        public readonly Bank Vectors = new([net._hidden], null, net._device);
        public readonly Bank Transitions = new(net._transitionStart.shape, net._transitionStart, net._device);
        public readonly Bank Constituents = new(net._constituentStart.shape, net._constituentStart, net._device);
    }

    private static readonly Row StartRow = new(0);

    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Embedding _deltaEmbedding, _tagEmbedding, _transitionEmbedding, _dummyEmbedding;
    private readonly LSTM _wordLstm, _transitionLstm, _constituentLstm;
    private readonly Linear _wordToConstituent, _reduceLinear;
    private readonly Linear[] _outputLayers;
    private readonly Tensor _wordStart, _wordEnd;
    private readonly Tensor _transitionStart, _constituentStart; // [2, layers, hidden]: h and c after the start embedding
    private readonly int _hidden;
    private readonly ThreadLocal<Banks> _banks;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public object TransitionStart => StartRow;
    public object ConstituentStart => StartRow;

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
        _hidden = hidden;
        _banks = new(() => new Banks(this));

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
            scope.Detach((IEnumerable<IDisposable>)[_transitionStart, _constituentStart]);
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

        // Each sentence's real rows (not the padding) go to the vector bank.
        long width = wordHx.shape[1];
        var rows = new List<long>();
        for (int i = 0; i < lengths.Length; i++)
        {
            for (int p = 0; p < lengths[i]; p++)
                rows.Add(i * width + p);
        }

        var bank = _banks.Value!.Vectors;
        var slots = bank.Add(wordHx.view(-1, _hidden).index_select(0, torch.tensor(rows.ToArray(), device: _device)));
        var result = new object[lengths.Length];
        for (int i = 0, first = 0; i < result.Length; first += (int)lengths[i++])
            result[i] = new WordRows(bank, slots[first..(first + (int)lengths[i])]);
        return result;
    }

    public object Word(object words, int position) => new Row(((WordRows)words).Slots[position]);

    public void Score(IReadOnlyList<(object Words, int Position, object Transition, object Constituent)> states, Span<float> output, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        var banks = _banks.Value!;
        var word = banks.Vectors.Data.index_select(0, Index(states.Select(s => ((WordRows)s.Words).Slots[s.Position]), _device));
        var hx = cat([word, Top(banks.Transitions, states.Select(s => s.Transition)), Top(banks.Constituents, states.Select(s => s.Constituent))], 1);
        foreach (var layer in _outputLayers)
            hx = layer.forward(F.relu(hx)); // nonlinearity before every layer, including the first
        hx.ToArray<float>().CopyTo(output);
    }

    public object[] Open(int[] openIndices)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        return Owned(_banks.Value!.Vectors, _dummyEmbedding.forward(torch.tensor(openIndices.Select(i => (long)i).ToArray(), device: _device)));
    }

    public object[] Compose(IReadOnlyList<IReadOnlyList<object>> children, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();

        // Every close's children as one [closes, most children] gather, short lists padded with their first child
        // (a repeated value doesn't change a max).
        int most = children.Max(c => c.Count);
        var index = new long[children.Count * most];
        for (int i = 0; i < children.Count; i++)
        {
            for (int k = 0; k < most; k++)
                index[i * most + k] = ((Row)children[i][k < children[i].Count ? k : 0]).Slot;
        }

        var bank = _banks.Value!.Vectors;
        var pooled = bank.Data.index_select(0, torch.tensor(index, device: _device)).view(children.Count, most, _hidden).max(1).values;
        return Owned(bank, F.relu(_reduceLinear.forward(pooled)));
    }

    public object[] PushTransitions(IReadOnlyList<object> parents, int[] transitions, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        var input = _transitionEmbedding.forward(torch.tensor(transitions.Select(i => (long)i).ToArray(), device: _device));
        return Push(_transitionLstm, _banks.Value!.Transitions, parents, input);
    }

    public object[] PushConstituents(IReadOnlyList<object> parents, IReadOnlyList<object> inputs, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        var banks = _banks.Value!;
        return Push(_constituentLstm, banks.Constituents, parents, banks.Vectors.Data.index_select(0, Index(inputs.Select(x => ((Row)x).Slot), _device)));
    }

    private static Tensor Index(IEnumerable<int> slots, Device device) => torch.tensor(slots.Select(i => (long)i).ToArray(), device: device);

    private static object[] Owned(Bank bank, Tensor rows) => bank.Add(rows).Select(slot => (object)new OwnedRow(bank, slot)).ToArray();

    // ----- LSTM stacks (lstm_tree_stack.py) -----

    /// <returns>[2, layers, hidden]: h and c after the start embedding.</returns>
    private static Tensor InitialStack(LSTM lstm, Tensor startEmbedding)
    {
        var (_, hx, cx) = lstm.forward(startEmbedding.view(1, 1, -1));
        return stack([hx.squeeze(1), cx.squeeze(1)]);
    }

    /// <summary>The stacks' outputs, [states, hidden]: the last layer's h (a one-step LSTM's output is its last h).</summary>
    private static Tensor Top(Bank bank, IEnumerable<object> states)
    {
        var data = bank.Data;
        return data.index_select(0, Index(states.Select(s => ((Row)s).Slot), data.device)).select(1, 0).select(1, data.shape[2] - 1);
    }

    /// <summary>Runs the LSTM one step for each stack from its own state, into new rows of <paramref name="bank"/>.</summary>
    private static object[] Push(LSTM lstm, Bank bank, IReadOnlyList<object> parents, Tensor inputs)
    {
        var data = bank.Data;
        var states = data.index_select(0, Index(parents.Select(s => ((Row)s).Slot), data.device)).transpose(0, 2); // [layers, 2, stacks, hidden]
        var (_, hn, cn) = lstm.forward(inputs.unsqueeze(0), (states.select(1, 0).contiguous(), states.select(1, 1).contiguous()));
        return Owned(bank, stack([hn, cn], 1).transpose(0, 2));
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _deltaEmbedding, _tagEmbedding, _transitionEmbedding, _dummyEmbedding,
                     _wordLstm, _transitionLstm, _constituentLstm, _wordToConstituent, _reduceLinear, })
        {
            m.Dispose();
        }

        foreach (var l in _outputLayers)
            l.Dispose();
        _wordStart.Dispose();
        _wordEnd.Dispose();
        _transitionStart.Dispose();
        _constituentStart.Dispose();
        _banks.Dispose();
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
