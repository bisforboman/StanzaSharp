using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace StanzaSharp.Nn.Managed;

/// <summary>One direction's rows for one LSTM time step: row r reads HPrev[r] and Init[r] and updates C[r], HOut[r].</summary>
internal unsafe struct StepRows
{
    public PackedMatrix Whh;
    public int M;
    /// <summary>h_{t-1} rows (K = hidden floats each).</summary>
    public float** HPrev;
    /// <summary>x_t·W_ihᵀ + b_ih + b_hh rows, columns in <see cref="PackedLstm.GateOrder"/> (4·hidden floats each).</summary>
    public float** Init;
    /// <summary>Cell state rows, updated in place.</summary>
    public float** C;
    public float** HOut;
}

/// <summary>
/// A (bi)LSTM with PyTorch semantics (gates i, f, g, o; b_ih + b_hh; given initial states) over packed batches,
/// like pack_padded_sequence → nn.LSTM. The input projection x·W_ihᵀ is one big GEMM over all rows (done by the
/// caller, so it can be fused with other layers reading the same input); each time step is then
/// h_{t-1}·W_hhᵀ with the cell update fused into the GEMM's epilogue.
/// </summary>
/// <remarks>
/// W_hh's rows are permuted (<see cref="GateOrder"/>) so that each 16-column panel holds the i, f, g, o gates of
/// four hidden units: the thread computing a panel can finish those units' c and h without waiting for others.
/// </remarks>
internal sealed unsafe class PackedLstm
{
    private readonly PackedMatrix[] _whh;
    private readonly float[][] _h0, _c0;

    public int Hidden { get; }
    public int Directions => _whh.Length;

    /// <param name="whh">Per direction, W_hh [4·hidden, hidden].</param>
    /// <param name="h0">Per direction, the initial h [hidden] (the same for every sequence).</param>
    public PackedLstm(int hidden, float[][] whh, float[][] h0, float[][] c0)
    {
        if (hidden % 4 != 0)
            throw new NotSupportedException("hidden must be a multiple of 4");
        Hidden = hidden;
        var order = GateOrder(hidden);
        _whh = whh.Select(w => new PackedMatrix(w, 4 * hidden, hidden, order)).ToArray();
        _h0 = h0;
        _c0 = c0;
    }

    /// <summary>Column j of the permuted gates holds PyTorch gate row GateOrder[j]: panel p = units 4p..4p+3 as [i×4, f×4, g×4, o×4].</summary>
    public static int[] GateOrder(int hidden) =>
        Enumerable.Range(0, 4 * hidden).Select(j => (j % 16 / 4) * hidden + j / 16 * 4 + j % 4).ToArray();

    /// <summary>
    /// The input projections of all directions as one packed matrix [directions·4H + extra rows, inputSize], gates
    /// permuted, with the bias b_ih + b_hh (extra layers' biases appended). Extra layers (e.g. a highway's gate)
    /// read the same input, so they share the GEMM.
    /// </summary>
    public static (PackedMatrix W, float[] Bias) PackInput(int hidden, int inputSize, float[][] wih, float[][] bih, float[][] bhh,
        params (float[] W, float[] B)[] extra)
    {
        var order = GateOrder(hidden);
        var rows = new List<float[]>();
        var bias = new List<float>();
        for (int d = 0; d < wih.Length; d++)
            foreach (int j in order)
            {
                rows.Add(wih[d].AsSpan(j * inputSize, inputSize).ToArray());
                bias.Add(bih[d][j] + bhh[d][j]);
            }
        foreach (var (w, b) in extra)
        {
            for (int j = 0; j < b.Length; j++)
                rows.Add(w.AsSpan(j * inputSize, inputSize).ToArray());
            bias.AddRange(b);
        }
        var packed = new PackedMatrix(rows.SelectMany(r => r).ToArray(), rows.Count, inputSize);
        var paddedBias = new float[packed.PaddedN];
        bias.CopyTo(paddedBias);
        return (packed, paddedBias);
    }

    /// <summary>Batch sizes of a packed batch: at step t, the number of sequences longer than t.</summary>
    public static int[] BatchSizes(long[] lengths)
    {
        var sizes = new int[lengths.Max()];
        foreach (var len in lengths)
            for (int t = 0; t < len; t++)
                sizes[t]++;
        return sizes;
    }

    /// <summary>Array form of <see cref="Recur(float*, int, int[], float*, int)"/>: returns [rows, Directions·H].</summary>
    public float[] Recur(float[] p, int ldp, int[] batchSizes)
    {
        int ldo = Directions * Hidden;
        var output = new float[batchSizes.Sum() * ldo];
        fixed (float* pp = p, po = output)
            Recur(pp, ldp, batchSizes, po, ldo);
        return output;
    }

    /// <summary>
    /// Runs the recurrence over a packed batch (rows time-major, as pack_padded_sequence lays them out).
    /// </summary>
    /// <param name="p">Input projections, row n at p + n·ldp; direction d's gates start at column d·4H.</param>
    /// <param name="output">Row n gets direction d's h at column d·H (ldo ≥ Directions·H).</param>
    public void Recur(float* p, int ldp, int[] batchSizes, float* output, int ldo)
    {
        int steps = batchSizes.Length, batch = batchSizes[0], h = Hidden;
        var start = new int[steps];
        for (int t = 1; t < steps; t++)
            start[t] = start[t - 1] + batchSizes[t - 1];

        int dirs = Directions;
        var pointers = GC.AllocateArray<nint>(dirs * 4 * batch, pinned: true);
        var cells = GC.AllocateArray<float>(dirs * batch * h, pinned: true);
        var h0 = _h0.Select(x => GC.AllocateArray<float>(h, pinned: true)).ToArray();
        for (int d = 0; d < dirs; d++)
            _h0[d].CopyTo(h0[d], 0);
        var works = new StepRows[dirs];
        float** basePtr = (float**)Unsafe.AsPointer(ref pointers[0]);
        float* cBase = (float*)Unsafe.AsPointer(ref cells[0]);
        for (int d = 0; d < dirs; d++)
            works[d] = new StepRows
            {
                Whh = _whh[d],
                HPrev = basePtr + (d * 4 + 0) * batch,
                Init = basePtr + (d * 4 + 1) * batch,
                C = basePtr + (d * 4 + 2) * batch,
                HOut = basePtr + (d * 4 + 3) * batch,
            };

        for (int s = 0; s < steps; s++)
        {
            for (int d = 0; d < dirs; d++)
            {
                ref var w = ref works[d];
                bool forward = d == 0;
                int t = forward ? s : steps - 1 - s;
                w.M = batchSizes[t];
                float* h0d = (float*)Unsafe.AsPointer(ref h0[d][0]);
                for (int r = 0; r < w.M; r++)
                {
                    float* cRow = cBase + ((long)d * batch + r) * h;
                    // Forward: every row continues from the previous step's row r (or h0 at t = 0). Backward: a
                    // sequence starts at its own last element, i.e. where row r wasn't present at step t + 1.
                    bool fresh = forward ? t == 0 : t == steps - 1 || r >= batchSizes[t + 1];
                    int prev = forward ? t - 1 : t + 1;
                    w.HPrev[r] = fresh ? h0d : output + (long)(start[prev] + r) * ldo + d * h;
                    if (fresh)
                        _c0[d].CopyTo(new Span<float>(cRow, h));
                    w.C[r] = cRow;
                    w.Init[r] = p + (long)(start[t] + r) * ldp + d * 4 * h;
                    w.HOut[r] = output + (long)(start[t] + r) * ldo + d * h;
                }
            }
            Step(works);
        }
        GC.KeepAlive(pointers);
        GC.KeepAlive(cells);
        GC.KeepAlive(h0);
    }

    /// <summary>One time step for every direction in <paramref name="works"/>, split across threads by panel.</summary>
    public static void Step(StepRows[] works)
    {
        int total = 0;
        foreach (var w in works)
            total += w.Whh.Panels;
        int tasks = Math.Min(ManagedThreads.Count, total);
        ManagedThreads.For(tasks, task =>
        {
            int g0 = (int)((long)total * task / tasks), g1 = (int)((long)total * (task + 1) / tasks);
            var aRows = stackalloc float*[Gemm.MR];
            var init = stackalloc float*[Gemm.MR];
            var outRows = stackalloc float*[Gemm.MR];
            var scratch = stackalloc float[Gemm.MR * PackedMatrix.NR];
            for (int r = 0; r < Gemm.MR; r++)
                outRows[r] = scratch + r * PackedMatrix.NR;
            int offset = 0;
            foreach (var w in works)
            {
                int lo = Math.Max(g0 - offset, 0), hi = Math.Min(g1 - offset, w.Whh.Panels);
                offset += w.Whh.Panels;
                for (int p = lo; p < hi; p++)
                {
                    var panel = w.Whh.Data + (long)p * PackedMatrix.NR * w.Whh.K;
                    for (int r0 = 0; r0 < w.M; r0 += Gemm.MR)
                    {
                        int mr = Math.Min(Gemm.MR, w.M - r0);
                        for (int r = 0; r < Gemm.MR; r++)
                        {
                            int row = r0 + Math.Min(r, mr - 1);
                            aRows[r] = w.HPrev[row];
                            init[r] = w.Init[row] + p * PackedMatrix.NR;
                        }
                        Gemm.Kernel(mr, aRows, panel, w.Whh.K, init, outRows);
                        for (int r = 0; r < mr; r++)
                            Cell(scratch + r * PackedMatrix.NR, w.C[r0 + r] + p * 4, w.HOut[r0 + r] + p * 4);
                    }
                }
            }
        });
    }

    /// <summary>c = σ(f)·c + σ(i)·tanh(g); h = σ(o)·tanh(c), for four units whose gates are [i×4, f×4, g×4, o×4].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Cell(float* gates, float* c, float* h)
    {
        var ifg = Act.Sigmoid(Vector256.Load(gates));
        var go = Vector256.Load(gates + 8);
        var cNew = ifg.GetUpper() * Vector128.Load(c) + ifg.GetLower() * Act.Tanh(go.GetLower());
        cNew.Store(c);
        (Act.Sigmoid(go.GetUpper()) * Act.Tanh(cNew)).Store(h);
    }
}
