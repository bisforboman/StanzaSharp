using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Text.Json.Nodes;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// Managed twin of <see cref="CharLanguageModel.BuildCharRepresentation"/>: the same character ids, packed batch
/// and outputs, as float arrays. The input projection is a table: the input is an embedding, so
/// emb·W_ihᵀ + b_ih + b_hh is computed once per character of the vocabulary at load.
/// </summary>
internal sealed unsafe class ManagedCharLanguageModel
{
    private const string Start = "\n", End = " ";

    private readonly Dictionary<string, int> _vocab;
    private readonly int _unkId, _endId;
    private readonly float[] _table; // [vocab, 4H], gates in PackedLstm.GateOrder
    private readonly PackedMatrix _whh;
    private readonly float[] _h0, _c0;

    public bool IsForward { get; }
    public int HiddenDim { get; }

    public ManagedCharLanguageModel(Checkpoint ckpt)
    {
        var args = ckpt.Root["args"]!;
        var state = ckpt.Root["state_dict"]!;
        if (args["char_num_layers"]!.GetValue<int>() != 1)
            throw new NotSupportedException("Only 1-layer charlms");
        IsForward = ckpt.Root["is_forward_lm"]!.GetValue<bool>();
        HiddenDim = args["char_hidden_dim"]!.GetValue<int>();
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        _unkId = _vocab["<UNK>"];
        _endId = _vocab[End];

        int h = HiddenDim;
        var embShape = ckpt.Shape(state["char_emb.weight"]);
        int vocab = (int)embShape[0], dim = (int)embShape[1];
        var emb = ckpt.Tensor<float>(state["char_emb.weight"]);
        var (wih, bias) = PackedLstm.PackInput(h, dim, [ckpt.Tensor<float>(state["charlstm.lstm.weight_ih_l0"])],
            [ckpt.Tensor<float>(state["charlstm.lstm.bias_ih_l0"])], [ckpt.Tensor<float>(state["charlstm.lstm.bias_hh_l0"])]);
        _table = GC.AllocateArray<float>(vocab * 4 * h, pinned: true);
        fixed (float* e = emb, b = bias)
            Gemm.Run(e, vocab, dim, wih, b, (float*)Unsafe.AsPointer(ref _table[0]), 4 * h);
        _whh = new PackedMatrix(ckpt.Tensor<float>(state["charlstm.lstm.weight_hh_l0"]), 4 * h, h, PackedLstm.GateOrder(h));
        _h0 = ckpt.Tensor<float>(state["charlstm_h_init"]);
        _c0 = ckpt.Tensor<float>(state["charlstm_c_init"]);
    }

    public static ManagedCharLanguageModel Load(string basePath) => new(Checkpoint.Load(basePath));

    /// <summary>For each sentence, [words, HiddenDim] row-major: the LSTM state at the space after each word.</summary>
    public List<float[]> BuildCharRepresentation(IReadOnlyList<IReadOnlyList<string>> sentences)
    {
        int n = sentences.Count, h = HiddenDim;
        var ids = new int[n][];
        var wordAt = new int[n][]; // position → word index, or -1
        var result = new List<float[]>(n);
        for (int i = 0; i < n; i++)
        {
            var words = sentences[i];
            var chars = new List<int> { Id(Start) };
            var ends = new List<int>();
            foreach (var word in IsForward ? words : words.Reverse())
            {
                var runes = word.EnumerateRunes().Select(r => r.ToString());
                foreach (var rune in IsForward ? runes : runes.Reverse())
                    chars.Add(Id(rune));
                chars.Add(_endId);
                ends.Add(chars.Count - 1);
            }
            ids[i] = chars.ToArray();
            wordAt[i] = new int[chars.Count];
            Array.Fill(wordAt[i], -1);
            for (int w = 0; w < ends.Count; w++)
                wordAt[i][ends[w]] = IsForward ? w : ends.Count - 1 - w;
            result.Add(new float[words.Count * h]);
        }
        if (n == 0)
            return result;

        // Packed: sequences sorted longest first, so the rows still running at step t are a prefix.
        var order = Enumerable.Range(0, n).OrderByDescending(i => ids[i].Length).ToArray();
        var pointers = GC.AllocateArray<nint>(4 * n, pinned: true);
        var buffers = GC.AllocateArray<float>(3 * n * h, pinned: true); // h ping-pong, c
        float** ptr = (float**)Unsafe.AsPointer(ref pointers[0]);
        float* hA = (float*)Unsafe.AsPointer(ref buffers[0]), hB = hA + n * h, c = hB + n * h;
        var work = new StepRows { Whh = _whh, HPrev = ptr, Init = ptr + n, C = ptr + 2 * n, HOut = ptr + 3 * n };
        var works = new[] { work };
        fixed (float* table = _table, h0 = _h0)
        {
            for (int r = 0; r < n; r++)
                _c0.CopyTo(new Span<float>(c + (long)r * h, h));
            int m = n;
            for (int t = 0; m > 0; t++)
            {
                while (m > 0 && ids[order[m - 1]].Length <= t)
                    m--;
                if (m == 0)
                    break;
                for (int r = 0; r < m; r++)
                {
                    work.HPrev[r] = t == 0 ? h0 : hA + (long)r * h;
                    work.Init[r] = table + (long)ids[order[r]][t] * 4 * h;
                    work.C[r] = c + (long)r * h;
                    work.HOut[r] = hB + (long)r * h;
                }
                works[0].M = m;
                PackedLstm.Step(works);
                for (int r = 0; r < m; r++)
                {
                    int seq = order[r], w = wordAt[seq][t];
                    if (w >= 0)
                        new Span<float>(hB + (long)r * h, h).CopyTo(result[seq].AsSpan(w * h, h));
                }
                var swap = hA;
                hA = hB;
                hB = swap;
            }
        }
        GC.KeepAlive(pointers);
        GC.KeepAlive(buffers);
        return result;
    }

    private int Id(string unit) => _vocab.TryGetValue(unit, out var id) ? id : _unkId;
}

/// <summary>
/// Managed twin of <see cref="HighwayLstm"/>'s packed path: per layer, one GEMM computes both LSTM directions'
/// input projections and the gate and highway layers (they all read the same input), then the recurrence, then
/// h + σ(gate)·tanh(highway).
/// </summary>
internal sealed unsafe class ManagedHighwayLstm
{
    private readonly (PackedMatrix W, float[] Bias, PackedLstm Lstm)[] _layers;
    private readonly int _hidden;

    public ManagedHighwayLstm(Checkpoint ckpt, JsonNode stateDict, string name, int inputSize, int hidden, int layers)
    {
        _hidden = hidden;
        float[] T(string key) => ckpt.Tensor<float>(stateDict[key]);
        var hInit = T($"{name}_h_init");
        var cInit = T($"{name}_c_init");
        float[] Init(float[] all, int i) => all.AsSpan(i * hidden, hidden).ToArray();
        _layers = new (PackedMatrix, float[], PackedLstm)[layers];
        int inSize = inputSize;
        for (int l = 0; l < layers; l++)
        {
            string lstm = $"{name}.lstm.{l}.lstm.";
            var (w, bias) = PackedLstm.PackInput(hidden, inSize,
                [T(lstm + "weight_ih_l0"), T(lstm + "weight_ih_l0_reverse")],
                [T(lstm + "bias_ih_l0"), T(lstm + "bias_ih_l0_reverse")],
                [T(lstm + "bias_hh_l0"), T(lstm + "bias_hh_l0_reverse")],
                (T($"{name}.gate.{l}.weight"), T($"{name}.gate.{l}.bias")),
                (T($"{name}.highway.{l}.weight"), T($"{name}.highway.{l}.bias")));
            var recurrent = new PackedLstm(hidden, [T(lstm + "weight_hh_l0"), T(lstm + "weight_hh_l0_reverse")],
                [Init(hInit, 2 * l), Init(hInit, 2 * l + 1)], [Init(cInit, 2 * l), Init(cInit, 2 * l + 1)]);
            _layers[l] = (w, bias, recurrent);
            inSize = 2 * hidden;
        }
    }

    /// <param name="input">Packed rows [words, inputSize] (time-major, as <see cref="Rnn.Pack"/> takes them).</param>
    /// <returns>The last layer's packed rows [words, 2·hidden].</returns>
    public float[] Forward(float[] input, long[] lengths)
    {
        var batchSizes = PackedLstm.BatchSizes(lengths);
        int rows = batchSizes.Sum(), h2 = 2 * _hidden;
        var x = input;
        foreach (var (w, bias, lstm) in _layers)
        {
            int inSize = x.Length / rows, ldp = w.PaddedN;
            var p = GC.AllocateUninitializedArray<float>(rows * ldp);
            var output = new float[rows * h2];
            fixed (float* px = x, pb = bias, pp = p, po = output)
            {
                Gemm.Run(px, rows, inSize, w, pb, pp, ldp);
                lstm.Recur(pp, ldp, batchSizes, po, h2);
                // h += σ(gate)·tanh(highway); gate and highway follow the two directions' 8H gate columns.
                nint ip = (nint)pp, io = (nint)po;
                int gate = 4 * h2, chunk = 64;
                ManagedThreads.For((rows + chunk - 1) / chunk, task =>
                {
                    for (int n = task * chunk, end = Math.Min(rows, n + chunk); n < end; n++)
                    {
                        float* g = (float*)ip + (long)n * ldp + gate, hw = g + h2, o = (float*)io + (long)n * h2;
                        int j = 0;
                        for (; j <= h2 - 8; j += 8)
                            (Vector256.Load(o + j) + Act.Sigmoid(Vector256.Load(g + j)) * Act.Tanh(Vector256.Load(hw + j))).Store(o + j);
                        for (; j < h2; j++)
                            o[j] += 1f / (1f + MathF.Exp(-g[j])) * MathF.Tanh(hw[j]);
                    }
                });
            }
            x = output;
        }
        return x;
    }
}
