using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace StanzaSharp.Nn.Managed;

// Per-call buffers come from ArrayPool<T>.Shared: a call reuses what an earlier call returned, so the large
// [rows, width] buffers aren't garbage for the GC to find late. Each call rents its own, so concurrent calls never
// share one. Rented arrays hold stale data; every buffer below is written before it is read.

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
    private readonly float[] _table; // [vocab, 4H], gates in PackedLstm.GateOrder; pinned
    private readonly PackedMatrix _whh;
    private readonly float[] _h0, _c0; // _h0 pinned

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
        _h0 = GC.AllocateArray<float>(h, pinned: true);
        ckpt.Tensor<float>(state["charlstm_h_init"]).CopyTo(_h0, 0);
        _c0 = ckpt.Tensor<float>(state["charlstm_c_init"]);
    }

    public static ManagedCharLanguageModel Load(string basePath) => new(Checkpoint.Load(basePath));

    /// <summary>For each sentence, [words, HiddenDim] row-major: the LSTM state at the space after each word.</summary>
    public List<float[]> BuildCharRepresentation(IReadOnlyList<IReadOnlyList<string>> sentences, CancellationToken ct = default)
    {
        int h = HiddenDim, total = sentences.Sum(s => s.Count);
        var flat = new float[total * h];
        fixed (float* o = flat)
            BuildCharRepresentation(sentences, o, h, ct);
        var result = new List<float[]>(sentences.Count);
        int offset = 0;
        foreach (var s in sentences)
        {
            result.Add(flat.AsSpan(offset, s.Count * h).ToArray());
            offset += s.Count * h;
        }
        return result;
    }

    /// <inheritdoc cref="BuildCharRepresentation(IReadOnlyList{IReadOnlyList{string}}, float*, int, CancellationToken)"/>
    public void BuildCharRepresentation(IReadOnlyList<IReadOnlyList<string>> sentences, Span<float> output, int ldo, CancellationToken ct = default)
    {
        int words = sentences.Sum(s => s.Count);
        if (words > 0 && output.Length < (words - 1) * ldo + HiddenDim)
            throw new ArgumentException("output is too small");
        fixed (float* o = output)
            BuildCharRepresentation(sentences, o, ldo, ct);
    }

    /// <summary>
    /// Writes the LSTM state at the space after each word to <paramref name="output"/>: row n (stride
    /// <paramref name="ldo"/> ≥ HiddenDim) is the n-th word counting through the sentences in order.
    /// <paramref name="ct"/> is checked before each character step.
    /// </summary>
    public void BuildCharRepresentation(IReadOnlyList<IReadOnlyList<string>> sentences, float* output, int ldo, CancellationToken ct = default)
    {
        int n = sentences.Count, h = HiddenDim;
        var ids = new int[n][];
        var rowAt = new int[n][]; // position → output row, or -1
        for (int i = 0, firstRow = 0; i < n; i++)
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
            rowAt[i] = new int[chars.Count];
            Array.Fill(rowAt[i], -1);
            for (int w = 0; w < ends.Count; w++)
                rowAt[i][ends[w]] = firstRow + (IsForward ? w : ends.Count - 1 - w);
            firstRow += words.Count;
        }
        if (n == 0)
            return;

        // Packed: sequences sorted longest first, so the rows still running at step t are a prefix.
        var order = Enumerable.Range(0, n).OrderByDescending(i => ids[i].Length).ToArray();
        var pointers = ArrayPool<nint>.Shared.Rent(4 * n);
        var buffers = ArrayPool<float>.Shared.Rent(3 * n * h); // h ping-pong, c
        try
        {
            fixed (nint* pointerBase = pointers)
            fixed (float* bufferBase = buffers, table = _table, c0 = _c0)
            {
                float** ptr = (float**)pointerBase;
                float* hA = bufferBase, hB = hA + n * h, c = hB + n * h;
                float* h0 = (float*)Unsafe.AsPointer(ref _h0[0]);
                var works = new[] { new StepRows { Whh = _whh, HPrev = ptr, Init = ptr + n, C = ptr + 2 * n, HOut = ptr + 3 * n } };
                ref var work = ref works[0];
                for (int r = 0; r < n; r++)
                    new Span<float>(c0, h).CopyTo(new Span<float>(c + (long)r * h, h));
                int m = n;
                for (int t = 0; m > 0; t++)
                {
                    while (m > 0 && ids[order[m - 1]].Length <= t)
                        m--;
                    if (m == 0)
                        break;
                    ct.ThrowIfCancellationRequested();
                    for (int r = 0; r < m; r++)
                    {
                        work.HPrev[r] = t == 0 ? h0 : hA + (long)r * h;
                        work.Init[r] = table + (long)ids[order[r]][t] * 4 * h;
                        work.C[r] = c + (long)r * h;
                        work.HOut[r] = hB + (long)r * h;
                    }
                    work.M = m;
                    PackedLstm.Step(works);
                    for (int r = 0; r < m; r++)
                    {
                        int row = rowAt[order[r]][t];
                        if (row >= 0)
                            new Span<float>(hB + (long)r * h, h).CopyTo(new Span<float>(output + (long)row * ldo, h));
                    }
                    var swap = hA;
                    hA = hB;
                    hB = swap;
                }
            }
        }
        finally
        {
            ArrayPool<nint>.Shared.Return(pointers);
            ArrayPool<float>.Shared.Return(buffers);
        }
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

    /// <summary>Width of an output row: both directions' h.</summary>
    public int OutputSize => 2 * _hidden;

    /// <inheritdoc cref="Forward(ReadOnlySpan{float}, long[], Span{float}, CancellationToken)"/>
    public float[] Forward(float[] input, long[] lengths, CancellationToken ct = default)
    {
        var output = new float[lengths.Sum() * OutputSize];
        Forward(input, lengths, output, ct);
        return output;
    }

    /// <param name="input">Packed rows [words, inputSize] (time-major, as <see cref="Rnn.Pack"/> takes them).</param>
    /// <param name="output">Gets the last layer's packed rows [words, 2·hidden].</param>
    /// <param name="ct">Checked per GEMM block and before each time step.</param>
    public void Forward(ReadOnlySpan<float> input, long[] lengths, Span<float> output, CancellationToken ct = default)
    {
        var batchSizes = PackedLstm.BatchSizes(lengths);
        int rows = batchSizes.Sum(), h2 = OutputSize, ldp = _layers.Max(l => l.W.PaddedN);
        if (output.Length < rows * h2)
            throw new ArgumentException("output is too small");
        var p = ArrayPool<float>.Shared.Rent(rows * ldp);
        // Layer outputs ping-pong between two temporaries; the last layer writes to output.
        var temps = Enumerable.Range(0, Math.Min(_layers.Length - 1, 2)).Select(_ => ArrayPool<float>.Shared.Rent(rows * h2)).ToArray();
        try
        {
            fixed (float* pp = p)
            {
                for (int l = 0; l < _layers.Length; l++)
                {
                    var (w, bias, lstm) = _layers[l];
                    var src = l == 0 ? input : temps[(l - 1) % 2].AsSpan(0, rows * h2);
                    var dst = l == _layers.Length - 1 ? output : temps[l % 2].AsSpan(0, rows * h2);
                    int inSize = src.Length / rows;
                    fixed (float* px = src, pb = bias, po = dst)
                    {
                        Gemm.Run(px, rows, inSize, w, pb, pp, ldp, ct);
                        lstm.Recur(pp, ldp, batchSizes, po, h2, ct);
                        // h += σ(gate)·tanh(highway); gate and highway follow the two directions' 8H gate columns.
                        nint ip = (nint)pp, io = (nint)po;
                        int gate = 4 * h2, chunk = 64;
                        ManagedThreads.For((rows + chunk - 1) / chunk, task =>
                        {
                            for (int n = task * chunk, end = Math.Min(rows, n + chunk); n < end; n++)
                            {
                                float* g = (float*)ip + (long)n * ldp + gate;
                                Act.AddGatedTanh((float*)io + (long)n * h2, g, g + h2, h2);
                            }
                        });
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(p);
            foreach (var t in temps)
                ArrayPool<float>.Shared.Return(t);
        }
    }
}

/// <summary>
/// Managed twin of a PyTorch <c>nn.LSTM(batch_first: true)</c> (any layers, uni- or bidirectional, zero initial state)
/// run through <see cref="Rnn.RunPacked"/>: padded input in, padded output out, zero past each row's length.
/// </summary>
/// <remarks>
/// A hidden size that isn't a multiple of 4 (<see cref="PackedLstm"/>'s panel) is padded with zero units: zero weights
/// and biases keep their c and h exactly 0 (σ(0)·0 + σ(0)·tanh(0)), so the real units are unchanged.
/// </remarks>
internal sealed unsafe class ManagedLstm
{
    private readonly (PackedMatrix W, float[] Bias, PackedLstm Lstm)[] _layers;
    private readonly int _hidden, _padded, _dirs;

    public int InputSize { get; }
    /// <summary>Width of an output row: every direction's h.</summary>
    public int OutputSize => _dirs * _hidden;

    /// <param name="prefix">The LSTM's key prefix in <paramref name="stateDict"/>, e.g. <c>"rnn."</c>.</param>
    public ManagedLstm(Checkpoint ckpt, JsonNode stateDict, string prefix, int inputSize, int hidden, int layers, bool bidirectional)
    {
        (InputSize, _hidden, _dirs) = (inputSize, hidden, bidirectional ? 2 : 1);
        int h = hidden, hp = _padded = (hidden + 3) / 4 * 4;
        string[] suffixes = bidirectional ? ["", "_reverse"] : [""];
        _layers = new (PackedMatrix, float[], PackedLstm)[layers];
        for (int l = 0; l < layers; l++)
        {
            float[] T(string name, string suffix) => ckpt.Tensor<float>(stateDict[$"{prefix}{name}_l{l}{suffix}"]
                ?? throw new KeyNotFoundException($"Checkpoint has no weight '{prefix}{name}_l{l}{suffix}'"));
            // Layer > 0 reads the previous layer's padded output: direction d's unit u is column d·hp + u.
            int k = l == 0 ? inputSize : _dirs * h, kp = l == 0 ? inputSize : _dirs * hp;
            Func<int, int> col = l == 0 ? c => c : c => c / h * hp + c % h;
            var wih = suffixes.Select(s => PadGates(T("weight_ih", s), h, hp, k, kp, col)).ToArray();
            var bih = suffixes.Select(s => PadGates(T("bias_ih", s), h, hp, 1, 1, c => c)).ToArray();
            var bhh = suffixes.Select(s => PadGates(T("bias_hh", s), h, hp, 1, 1, c => c)).ToArray();
            var whh = suffixes.Select(s => PadGates(T("weight_hh", s), h, hp, h, hp, c => c)).ToArray();
            var (w, bias) = PackedLstm.PackInput(hp, kp, wih, bih, bhh);
            var zeros = suffixes.Select(_ => new float[hp]).ToArray();
            _layers[l] = (w, bias, new PackedLstm(hp, whh, zeros, zeros));
        }
    }

    /// <summary>A gate-stacked [4·h, k] matrix as [4·hp, kp]: row g·h + u goes to g·hp + u, column c to col(c), zeros elsewhere.</summary>
    private static float[] PadGates(float[] w, int h, int hp, int k, int kp, Func<int, int> col)
    {
        if (h == hp && k == kp)
            return w;
        var result = new float[4 * hp * kp];
        for (int g = 0; g < 4; g++)
            for (int u = 0; u < h; u++)
                for (int c = 0; c < k; c++)
                    result[(g * hp + u) * kp + col(c)] = w[(g * h + u) * k + c];
        return result;
    }

    /// <param name="input">[batch, width, InputSize], batch-first.</param>
    /// <param name="lengths">Each row's length (1 ≤ length ≤ width); rows run packed at it.</param>
    /// <param name="output">Gets [batch, width, OutputSize], zero past each row's length.</param>
    /// <param name="ct">Checked per GEMM block and before each time step.</param>
    public void ForwardPadded(ReadOnlySpan<float> input, int batch, int width, long[] lengths, Span<float> output, CancellationToken ct = default)
    {
        if (input.Length < batch * width * InputSize || output.Length < batch * width * OutputSize)
            throw new ArgumentException("input or output is too small");
        // pack_padded_sequence: rows sorted longest first; step t holds the batchSizes[t] rows longer than t.
        var order = Enumerable.Range(0, batch).OrderByDescending(i => lengths[i]).ToArray();
        var batchSizes = PackedLstm.BatchSizes(lengths);
        int rows = batchSizes.Sum(), ldo = _dirs * _padded, ldp = _layers.Max(l => l.W.PaddedN);
        var x = ArrayPool<float>.Shared.Rent(rows * Math.Max(InputSize, ldo));
        var y = ArrayPool<float>.Shared.Rent(rows * ldo);
        var p = ArrayPool<float>.Shared.Rent(rows * ldp);
        try
        {
            for (int t = 0, n = 0; t < batchSizes.Length; t++)
                for (int r = 0; r < batchSizes[t]; r++, n++)
                    input.Slice((order[r] * width + t) * InputSize, InputSize).CopyTo(x.AsSpan(n * InputSize));
            fixed (float* px = x, py = y, pp = p)
            {
                float* src = px, dst = py;
                int inSize = InputSize;
                foreach (var (w, bias, lstm) in _layers)
                {
                    fixed (float* pb = bias)
                        Gemm.Run(src, rows, inSize, w, pb, pp, ldp, ct);
                    lstm.Recur(pp, ldp, batchSizes, dst, ldo, ct);
                    var swap = src;
                    src = dst;
                    dst = swap;
                    inSize = ldo;
                }
                // src holds the last layer's packed output.
                output[..(batch * width * OutputSize)].Clear();
                for (int t = 0, n = 0; t < batchSizes.Length; t++)
                    for (int r = 0; r < batchSizes[t]; r++, n++)
                        for (int d = 0; d < _dirs; d++)
                            new ReadOnlySpan<float>(src + (long)n * ldo + d * _padded, _hidden)
                                .CopyTo(output.Slice((order[r] * width + t) * OutputSize + d * _hidden, _hidden));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(y);
            ArrayPool<float>.Shared.Return(p);
        }
    }
}
