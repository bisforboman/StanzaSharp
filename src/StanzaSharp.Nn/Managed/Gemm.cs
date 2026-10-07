using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace StanzaSharp.Nn.Managed;

// Spike for issue #29 (Phase 0): a pure managed float32 backend. Nothing in the pipeline uses this code;
// docs/managed-backend-spike.md has the method and the numbers.

/// <summary>Thread count of the managed kernels (the managed twin of <c>torch.set_num_threads</c>).</summary>
/// <remarks>
/// A persistent team of threads meeting at a <see cref="Barrier"/> (which spins briefly before blocking): an LSTM
/// runs one parallel region per time step, thousands per call, so the per-region cost matters.
/// ponytail: one process-wide team behind a lock, so concurrent callers take turns; a real backend needs per-caller
/// teams or work stealing.
/// </remarks>
internal static class ManagedThreads
{
    private static readonly Lock Gate = new();
    private static int _count = Environment.ProcessorCount;
    private static Team? _team;

    public static int Count
    {
        get => _count;
        set => _count = Math.Max(1, value);
    }

    /// <summary>Runs <paramref name="body"/>(i) for i in [0, n) on up to <see cref="Count"/> threads.</summary>
    public static void For(int n, Action<int> body)
    {
        if (n == 1 || Count == 1)
        {
            for (int i = 0; i < n; i++)
                body(i);
            return;
        }
        lock (Gate)
        {
            if (_team?.Size != Count)
            {
                _team?.Dispose();
                _team = new Team(Count);
            }
            _team.Run(n, body);
        }
    }

    private sealed class Team : IDisposable
    {
        private readonly Barrier _barrier;
        private Action<int>? _body;
        private int _n, _next;
        private bool _stop;

        public int Size { get; }

        public Team(int size)
        {
            Size = size;
            _barrier = new Barrier(size);
            for (int i = 1; i < size; i++)
                new Thread(Work) { IsBackground = true, Name = $"managed-kernel-{i}" }.Start();
        }

        public void Run(int n, Action<int> body)
        {
            (_body, _n, _next) = (body, n, 0);
            _barrier.SignalAndWait();
            Drain();
            _barrier.SignalAndWait();
            _body = null;
        }

        private void Drain()
        {
            for (int i; (i = Interlocked.Increment(ref _next) - 1) < _n;)
                _body!(i);
        }

        private void Work()
        {
            while (true)
            {
                _barrier.SignalAndWait();
                if (_stop)
                    return;
                Drain();
                _barrier.SignalAndWait();
            }
        }

        public void Dispose()
        {
            _stop = true;
            _barrier.SignalAndWait();
        }
    }
}

/// <summary>
/// A weight matrix W [N, K] (PyTorch's Linear/LSTM layout, rows are outputs) packed once for <see cref="Gemm"/>:
/// panels of <see cref="NR"/> output columns, each stored as [K][NR], N padded with zero rows.
/// </summary>
internal sealed unsafe class PackedMatrix
{
    public const int NR = 16;

    private readonly float[] _data; // pinned (POH), so Data stays valid

    public int N { get; }
    public int K { get; }
    public int Panels { get; }
    /// <summary>N rounded up to a whole panel: the width a GEMM output must have.</summary>
    public int PaddedN => Panels * NR;
    public float* Data => (float*)Unsafe.AsPointer(ref _data[0]);

    /// <param name="w">[n, k] row-major.</param>
    /// <param name="rows">Optional: output column j is source row rows[j] (length n).</param>
    public PackedMatrix(ReadOnlySpan<float> w, int n, int k, int[]? rows = null)
    {
        if (w.Length != n * k)
            throw new ArgumentException($"Expected {n}x{k} floats, got {w.Length}");
        N = n;
        K = k;
        Panels = (n + NR - 1) / NR;
        _data = GC.AllocateArray<float>(Panels * NR * k, pinned: true);
        for (int j = 0; j < n; j++)
        {
            var src = w.Slice((rows?[j] ?? j) * k, k);
            int panel = j / NR, lane = j % NR;
            int dst = panel * NR * k + lane;
            for (int kk = 0; kk < k; kk++)
                _data[dst + kk * NR] = src[kk];
        }
    }
}

/// <summary>Blocked, SIMD (AVX2 + FMA) float32 GEMM: C = A·Wᵀ + init, with a 6×16 register-blocked micro-kernel.</summary>
internal static unsafe class Gemm
{
    public const int MR = 6;
    private const int KC = 256, MC = 96, NC = 8; // NC panels (128 columns) per task

    /// <summary>A [m, K] row-major → C [m, PaddedN] = A·Wᵀ + bias (bias null or PaddedN floats).</summary>
    public static float[] Run(float[] a, int m, PackedMatrix w, float[]? bias)
    {
        var c = new float[m * w.PaddedN];
        fixed (float* pa = a, pb = bias, pc = c)
            Run(pa, m, w.K, w, pb, pc, w.PaddedN);
        return c;
    }

    /// <summary>
    /// C[m, :] = A[m, :]·Wᵀ + bias for m in [0, M). C must have room for <see cref="PackedMatrix.PaddedN"/> columns
    /// per row (ldc ≥ PaddedN); bias (null for none) must have PaddedN floats.
    /// </summary>
    public static void Run(float* a, int m, int lda, PackedMatrix w, float* bias, float* c, int ldc)
    {
        if (m == 0)
            return;
        if (ldc < w.PaddedN)
            throw new ArgumentException("ldc must cover the padded width");
        int mBlocks = (m + MC - 1) / MC, nBlocks = (w.Panels + NC - 1) / NC;
        nint pa = (nint)a, pb = (nint)bias, pc = (nint)c;
        ManagedThreads.For(mBlocks * nBlocks, task =>
        {
            var a_ = (float*)pa;
            var bias_ = (float*)pb;
            var c_ = (float*)pc;
            int m0 = task / nBlocks * MC, m1 = Math.Min(m, m0 + MC);
            int p0 = task % nBlocks * NC, p1 = Math.Min(w.Panels, p0 + NC);
            var aRows = stackalloc float*[MR];
            var init = stackalloc float*[MR];
            var cRows = stackalloc float*[MR];
            var zero = stackalloc float[PackedMatrix.NR];
            var dummy = stackalloc float[PackedMatrix.NR];
            for (int i = 0; i < PackedMatrix.NR; i++)
                zero[i] = 0;
            for (int k0 = 0; k0 < w.K; k0 += KC)
            {
                int kc = Math.Min(KC, w.K - k0);
                for (int p = p0; p < p1; p++)
                {
                    var panel = w.Data + (long)p * PackedMatrix.NR * w.K + (long)k0 * PackedMatrix.NR;
                    for (int r0 = m0; r0 < m1; r0 += MR)
                    {
                        int mr = Math.Min(MR, m1 - r0);
                        for (int r = 0; r < MR; r++)
                        {
                            int row = r0 + Math.Min(r, mr - 1);
                            aRows[r] = a_ + (long)row * lda + k0;
                            var cRow = c_ + (long)row * ldc + p * PackedMatrix.NR;
                            init[r] = k0 > 0 ? cRow : bias_ != null ? bias_ + p * PackedMatrix.NR : zero;
                            cRows[r] = r < mr ? cRow : dummy;
                        }
                        Kernel(mr, aRows, panel, kc, init, cRows);
                    }
                }
            }
        });
    }

    /// <summary>
    /// out[r][0..16) = init[r][0..16) + Σ_k a[r][k]·panel[k][0..16) for six rows r. Rows the caller doesn't need
    /// may repeat another row's pointers and write to a scratch buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Kernel(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        var c00 = Vector256.Load(init[0]); var c01 = Vector256.Load(init[0] + 8);
        var c10 = Vector256.Load(init[1]); var c11 = Vector256.Load(init[1] + 8);
        var c20 = Vector256.Load(init[2]); var c21 = Vector256.Load(init[2] + 8);
        var c30 = Vector256.Load(init[3]); var c31 = Vector256.Load(init[3] + 8);
        var c40 = Vector256.Load(init[4]); var c41 = Vector256.Load(init[4] + 8);
        var c50 = Vector256.Load(init[5]); var c51 = Vector256.Load(init[5] + 8);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector256.Load(b);
            var b1 = Vector256.Load(b + 8);
            var x = Vector256.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01);
            x = Vector256.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11);
            x = Vector256.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21);
            x = Vector256.Create(a3[p]); c30 = Fma(x, b0, c30); c31 = Fma(x, b1, c31);
            x = Vector256.Create(a4[p]); c40 = Fma(x, b0, c40); c41 = Fma(x, b1, c41);
            x = Vector256.Create(a5[p]); c50 = Fma(x, b0, c50); c51 = Fma(x, b1, c51);
        }
        c00.Store(output[0]); c01.Store(output[0] + 8);
        c10.Store(output[1]); c11.Store(output[1] + 8);
        c20.Store(output[2]); c21.Store(output[2] + 8);
        c30.Store(output[3]); c31.Store(output[3] + 8);
        c40.Store(output[4]); c41.Store(output[4] + 8);
        c50.Store(output[5]); c51.Store(output[5] + 8);
    }

    /// <summary><see cref="Kernel"/> for mr rows, with smaller kernels for 1 to 3 rows (a padded 6-row block wastes up to 6× the work).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Kernel(int mr, float** a, float* b, int k, float** init, float** output)
    {
        if (mr == 1)
            Kernel1(a[0], b, k, init[0], output[0]);
        else if (mr <= 3)
            Kernel3(a, b, k, init, output);
        else
            Kernel(a, b, k, init, output);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        var c00 = Vector256.Load(init[0]); var c01 = Vector256.Load(init[0] + 8);
        var c10 = Vector256.Load(init[1]); var c11 = Vector256.Load(init[1] + 8);
        var c20 = Vector256.Load(init[2]); var c21 = Vector256.Load(init[2] + 8);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector256.Load(b);
            var b1 = Vector256.Load(b + 8);
            var x = Vector256.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01);
            x = Vector256.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11);
            x = Vector256.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21);
        }
        c00.Store(output[0]); c01.Store(output[0] + 8);
        c10.Store(output[1]); c11.Store(output[1] + 8);
        c20.Store(output[2]); c21.Store(output[2] + 8);
    }

    /// <summary>One row (a GEMV panel): four independent accumulator pairs over k, to hide the FMA latency.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel1(float* a, float* b, int k, float* init, float* output)
    {
        var c0 = Vector256.Load(init); var c1 = Vector256.Load(init + 8);
        Vector256<float> d0 = default, d1 = default, e0 = default, e1 = default, f0 = default, f1 = default;
        int p = 0;
        for (; p <= k - 4; p += 4, b += 4 * PackedMatrix.NR)
        {
            var x = Vector256.Create(a[p]); c0 = Fma(x, Vector256.Load(b), c0); c1 = Fma(x, Vector256.Load(b + 8), c1);
            x = Vector256.Create(a[p + 1]); d0 = Fma(x, Vector256.Load(b + 16), d0); d1 = Fma(x, Vector256.Load(b + 24), d1);
            x = Vector256.Create(a[p + 2]); e0 = Fma(x, Vector256.Load(b + 32), e0); e1 = Fma(x, Vector256.Load(b + 40), e1);
            x = Vector256.Create(a[p + 3]); f0 = Fma(x, Vector256.Load(b + 48), f0); f1 = Fma(x, Vector256.Load(b + 56), f1);
        }
        for (; p < k; p++, b += PackedMatrix.NR)
        {
            var x = Vector256.Create(a[p]); c0 = Fma(x, Vector256.Load(b), c0); c1 = Fma(x, Vector256.Load(b + 8), c1);
        }
        ((c0 + d0) + (e0 + f0)).Store(output);
        ((c1 + d1) + (e1 + f1)).Store(output + 8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Fma(Vector256<float> a, Vector256<float> b, Vector256<float> c) =>
        System.Runtime.Intrinsics.X86.Fma.IsSupported ? System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(a, b, c) : a * b + c;
}

/// <summary>Vectorized activations (PyTorch semantics; values agree to a few ulp).</summary>
internal static class Act
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sigmoid(Vector256<float> x) => Vector256<float>.One / (Vector256<float>.One + Vector256.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Sigmoid(Vector128<float> x) => Vector128<float>.One / (Vector128<float>.One + Vector128.Exp(-x));

    /// <summary>tanh(x) = 2σ(2x) − 1 (absolute error ≈ 1e-7 near 0).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Tanh(Vector128<float> x) => Sigmoid(x + x) * Vector128.Create(2f) - Vector128<float>.One;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Tanh(Vector256<float> x) => Sigmoid(x + x) * Vector256.Create(2f) - Vector256<float>.One;
}
