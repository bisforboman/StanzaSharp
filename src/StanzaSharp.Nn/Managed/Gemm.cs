using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace StanzaSharp.Nn.Managed;

// Issue #29: a pure managed float32 backend, used by the processors ported so far when the backend is
// Backend.Managed. docs/managed-backend-spike.md (kernels) and docs/backends.md (the seam) have the numbers.

/// <summary>The SIMD flavor of the managed kernels. All four compute the same thing; tests run each of them.</summary>
internal enum KernelPath
{
    /// <summary>
    /// AVX-512 (x64): one 16-float register per row of a panel, and two panels at a time in a GEMM. Each
    /// output gets the fused multiply-adds of <see cref="Vector256"/> in the same order, so the same bits.
    /// </summary>
    Vector512,

    /// <summary>AVX2 + FMA (x64): 6×16 micro-kernel in 12 registers.</summary>
    Vector256,

    /// <summary>NEON on Arm64 (6×16 in 24 of its 32 registers), or SSE on x64 without AVX2.</summary>
    Vector128,

    /// <summary>Plain scalar code, for platforms without hardware SIMD.</summary>
    Scalar,
}

/// <summary>
/// A weight matrix W [N, K] (PyTorch's Linear/LSTM layout, rows are outputs) packed once for <see cref="Gemm"/>:
/// panels of <see cref="NR"/> output columns, each stored as [K][NR], N padded with zero rows. Every
/// <see cref="KernelPath"/> reads the same layout.
/// </summary>
internal sealed unsafe class PackedMatrix
{
    public const int NR = 16;

    private readonly float[]? _data; // pinned (POH), so Data stays valid
    private readonly float* _mapped; // or a mapped cache view (PackCache, round 34 prototype)

    public int N { get; }
    public int K { get; }
    public int Panels { get; }

    /// <summary>N rounded up to a whole panel: the width a GEMM output must have.</summary>
    public int PaddedN => Panels * NR;

    public float* Data => _mapped != null ? _mapped + _offset : (float*)Unsafe.AsPointer(ref _data![_offset]);

    private readonly int _offset;

    /// <summary>
    /// Initializes a new instance of the <see cref="PackedMatrix"/> class. A view of <paramref name="panels"/> of <paramref name="source"/>'s panels from <paramref name="firstPanel"/> on
    /// (output columns [16·firstPanel, 16·(firstPanel + panels)), sharing its data). A GEMM column depends only on its
    /// panel, so those columns come out with the same bits as from the whole matrix.
    /// </summary>
    public PackedMatrix(PackedMatrix source, int firstPanel, int panels)
    {
        if (firstPanel < 0 || panels <= 0 || firstPanel + panels > source.Panels)
            throw new ArgumentOutOfRangeException(nameof(panels));
        _data = source._data;
        _mapped = source._mapped;
        _offset = source._offset + firstPanel * NR * source.K;
        K = source.K;
        Panels = panels;
        N = Math.Min(source.N - firstPanel * NR, panels * NR);
    }

    /// <param name="w">[n, k] row-major.</param>
    /// <param name="rows">Optional: output column j is source row rows[j] (length n).</param>
    public PackedMatrix(ReadOnlySpan<float> w, int n, int k, int[]? rows = null)
    {
        if (w.Length != n * k)
            throw new ArgumentException($"Expected {n}x{k} floats, got {w.Length}");
        N = n;
        K = k;
        Panels = (n + NR - 1) / NR;
        int length = Panels * NR * k;
        Interlocked.Add(ref LoadProfile.PackedFloats, length);
        var cache = PackCache.Current;
        if (cache != null && cache.TryHit(length, out _data, out _mapped))
            return;
        using var _ = LoadProfile.Measure(LoadProfile.Stage.Pack);
        if (Mode == "old")
        {
            _data = GC.AllocateArray<float>(length, pinned: true);
            for (int j = 0; j < n; j++)
            {
                var src = w.Slice((rows?[j] ?? j) * k, k);
                int panel = j / NR, lane = j % NR;
                int dst = panel * NR * k + lane;
                for (int kk = 0; kk < k; kk++)
                    _data[dst + kk * NR] = src[kk];
            }
        }
        else
        {
            _data = GC.AllocateUninitializedArray<float>(length, pinned: true);
            fixed (float* source = w)
            {
                nint src = (nint)source, dst = (nint)Data;
                if ((long)n * k >= ParallelPackFloats && Environment.ProcessorCount > 1 && Mode == "parallel")
                    Parallel.For(0, Panels, p => PackPanel((float*)src, (float*)dst, p, n, k, rows));
                else
                {
                    for (int p = 0; p < Panels; p++)
                        PackPanel((float*)src, (float*)dst, p, n, k, rows);
                }
            }
        }

        cache?.Record(_data);
    }

    /// <summary>Round 34 prototype: STANZASHARP_PACK = old (main's loop), fast (default here), parallel.</summary>
    internal static readonly string Mode = Environment.GetEnvironmentVariable("STANZASHARP_PACK") ?? "fast";

    private const long ParallelPackFloats = 1 << 20;

    /// <summary>Panel p of W [n, k]: its [k][NR] block, rows past n zero. A copy, so the bits don't depend on how it's done.</summary>
    private static void PackPanel(float* w, float* data, int p, int n, int k, int[]? rows)
    {
        var src = stackalloc nint[NR];
        int lanes = Math.Min(NR, n - p * NR);
        for (int lane = 0; lane < lanes; lane++)
        {
            int j = p * NR + lane, r = rows?[j] ?? j;
            if ((uint)r >= (uint)n)
                throw new ArgumentOutOfRangeException(nameof(rows));
            src[lane] = (nint)(w + (long)r * k);
        }

        float* dst = data + (long)p * NR * k;
        if (lanes < NR)
            new Span<float>(dst, NR * k).Clear();
        for (int lane = 0; lane < lanes; lane++)
        {
            float* s = (float*)src[lane];
            float* d = dst + lane;
            for (int kk = 0; kk < k; kk++)
                d[(long)kk * NR] = s[kk];
        }
    }
}

/// <summary>Blocked SIMD float32 GEMM: C = A·Wᵀ + init, with a 6×16 register-blocked micro-kernel per <see cref="KernelPath"/>.</summary>
internal static unsafe class Gemm
{
    public const int MR = 6;
    private const int KC = 256, MC = 96, NC = 8; // NC panels (128 columns) per task

    /// <summary>The kernels' SIMD flavor, detected at startup. Settable so tests can run every path on one machine.</summary>
    public static KernelPath Path { get; set; } = Detect();

    /// <remarks>
    /// Vector512 only where .NET accelerates it: CPUs with AVX-512 that don't throttle on it, and not when
    /// <c>DOTNET_PreferredVectorBitWidth=256</c> (or <c>DOTNET_EnableAVX512F=0</c>) is set, which selects Vector256.
    /// </remarks>
    public static KernelPath Detect() =>
        Vector512.IsHardwareAccelerated ? KernelPath.Vector512
        : Vector256.IsHardwareAccelerated ? KernelPath.Vector256
        : Vector128.IsHardwareAccelerated ? KernelPath.Vector128
        : KernelPath.Scalar;

    /// <summary>
    /// A [m, K] row-major → C [m, PaddedN] = A·Wᵀ + bias (bias null or PaddedN floats), into <paramref name="c"/>
    /// if given (at least m·PaddedN floats; a rented array is fine, every element of those rows is written).
    /// </summary>
    public static float[] Run(float[] a, int m, PackedMatrix w, float[]? bias, float[]? c = null)
    {
        c ??= new float[m * w.PaddedN];
        if (c.Length < m * w.PaddedN)
            throw new ArgumentException("c is too small");
        fixed (float* pa = a, pb = bias, pc = c)
            Run(pa, m, w.K, w, pb, pc, w.PaddedN);
        return c;
    }

    /// <summary>
    /// C[m, :] = A[m, :]·Wᵀ + bias for m in [0, M). C must have room for <see cref="PackedMatrix.PaddedN"/> columns
    /// per row (ldc ≥ PaddedN); bias (null for none) must have PaddedN floats. <paramref name="ct"/> is checked
    /// before each block of 96 rows × 128 columns.
    /// </summary>
    /// <param name="rowInvariant">Compute every row with the same arithmetic whatever <paramref name="m"/> is (no
    /// single-row kernel), so a row's result doesn't depend on the other rows in the call.</param>
    /// <param name="epilogue">Called by the task that computed them, once rows [m0, m1) × panels [p0, p1) of C are final:
    /// (m0, m1, p0, p1). Tasks are split the same way whatever the thread count.</param>
    public static void Run(float* a, int m, int lda, PackedMatrix w, float* bias, float* c, int ldc, CancellationToken ct = default,
        bool rowInvariant = false, Action<int, int, int, int>? epilogue = null)
    {
        if (m == 0)
            return;
        if (ldc < w.PaddedN)
            throw new ArgumentException("ldc must cover the padded width");
        int mBlocks = (m + MC - 1) / MC, nBlocks = (w.Panels + NC - 1) / NC;
        nint pa = (nint)a, pb = (nint)bias, pc = (nint)c;
        ManagedThreads.For(mBlocks * nBlocks, task =>
        {
            ct.ThrowIfCancellationRequested();
            var a_ = (float*)pa;
            var bias_ = (float*)pb;
            var c_ = (float*)pc;
            int m0 = task / nBlocks * MC, m1 = Math.Min(m, m0 + MC);
            int p0 = task % nBlocks * NC, p1 = Math.Min(w.Panels, p0 + NC);
            var aRows = stackalloc float*[MR];
            var init = stackalloc float*[MR];
            var cRows = stackalloc float*[MR];
            var zero = stackalloc float[2 * PackedMatrix.NR]; // two panels wide, for the Vector512 pairs
            var dummy = stackalloc float[2 * PackedMatrix.NR];
            for (int i = 0; i < 2 * PackedMatrix.NR; i++)
                zero[i] = 0;
            bool pairs = Path == KernelPath.Vector512;
            for (int k0 = 0; k0 < w.K; k0 += KC)
            {
                int kc = Math.Min(KC, w.K - k0);
                for (int p = p0; p < p1; p++)
                {
                    bool pair = pairs && p + 1 < p1;
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

                        if (pair)
                            KernelPair512(mr, aRows, panel, (long)PackedMatrix.NR * w.K, kc, init, cRows, rowInvariant);
                        else
                            Kernel(mr, aRows, panel, kc, init, cRows, rowInvariant);
                    }

                    if (pair)
                        p++;
                }
            }

            epilogue?.Invoke(m0, m1, p0, p1);
        });
    }

    /// <summary>
    /// Rows [<paramref name="r0"/>, <paramref name="r1"/>) of <c>Run(a, m, ...)</c>, with the bits that call gives them, so
    /// a big GEMM can be computed a few rows at a time. Every kernel but the single-row one gives a row the same
    /// arithmetic, and <c>Run</c> uses that one only for row m - 1 when m % <see cref="MR"/> is 1 (MC is a multiple of MR).
    /// </summary>
    /// <param name="a">Row r0 of A.</param>
    /// <param name="c">Row r0 of C.</param>
    public static void RunRows(float* a, int r0, int r1, int m, int lda, PackedMatrix w, float* bias, float* c, int ldc, CancellationToken ct = default)
    {
        int end = r1 == m && m % MR == 1 ? r1 - 1 : r1;
        Run(a, end - r0, lda, w, bias, c, ldc, ct, rowInvariant: true);
        if (end < r1)
            Run(a + (long)(end - r0) * lda, 1, lda, w, bias, c + (long)(end - r0) * ldc, ldc, ct);
    }

    /// <summary>
    /// out[r][0..16) = init[r][0..16) + Σ_k a[r][k]·panel[k][0..16) for the first mr ≤ 6 rows, on <see cref="Path"/>.
    /// Smaller kernels serve 1 to 3 rows (a padded 6-row block wastes up to 6× the work). Rows past mr must repeat
    /// another row's inputs; the SIMD kernels write them too, so their outputs must be scratch. With
    /// <paramref name="rowInvariant"/>, one row also goes through the 3-row kernel, whose per-row arithmetic is the
    /// 6-row kernel's (the 1-row kernel splits the sum over k into four).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Kernel(int mr, float** a, float* b, int k, float** init, float** output, bool rowInvariant = false)
    {
        switch (Path)
        {
            case KernelPath.Vector512:
                if (mr == 1 && !rowInvariant)
                    Kernel1x512(a[0], b, k, init[0], output[0]);
                else if (mr <= 3)
                    Kernel3x512(a, b, k, init, output);
                else
                    Kernel6x512(a, b, k, init, output);
                break;
            case KernelPath.Vector256:
                if (mr == 1 && !rowInvariant)
                    Kernel1x256(a[0], b, k, init[0], output[0]);
                else if (mr <= 3)
                    Kernel3x256(a, b, k, init, output);
                else
                    Kernel6x256(a, b, k, init, output);
                break;
            case KernelPath.Vector128:
                if (mr == 1 && !rowInvariant)
                {
                    Kernel1x128(a[0], b, k, init[0], output[0]);
                }
                else if (mr <= 3)
                {
                    if (AdvSimd.Arm64.IsSupported) Kernel3x128Neon(a, b, k, init, output); else Kernel3x128(a, b, k, init, output);
                }
                else if (AdvSimd.Arm64.IsSupported)
                {
                    Kernel6x128Neon(a, b, k, init, output);
                }
                else
                {
                    Kernel6x128(a, b, k, init, output);
                }

                break;
            default:
                KernelScalar(mr, a, b, k, init, output);
                break;
        }
    }

    /// <summary>
    /// <see cref="Kernel"/> on Vector512 for two adjacent panels (<paramref name="b"/> and <paramref name="b"/> +
    /// <paramref name="bStride"/>), whose init and output columns follow the first panel's (+16): 12 accumulators for
    /// 6 rows instead of 6, enough independent chains to cover the FMA latency. Each column's arithmetic is the
    /// single-panel kernel's.
    /// </summary>
    private static void KernelPair512(int mr, float** a, float* b, long bStride, int k, float** init, float** output, bool rowInvariant)
    {
        if (mr == 1 && !rowInvariant)
        {
            Kernel1x512(a[0], b, k, init[0], output[0]);
            Kernel1x512(a[0], b + bStride, k, init[0] + PackedMatrix.NR, output[0] + PackedMatrix.NR);
        }
        else if (mr <= 3)
        {
            Kernel3x512Pair(a, b, bStride, k, init, output);
        }
        else
        {
            Kernel6x512Pair(a, b, bStride, k, init, output);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x512(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        var c0 = Vector512.Load(init[0]); var c1 = Vector512.Load(init[1]); var c2 = Vector512.Load(init[2]);
        var c3 = Vector512.Load(init[3]); var c4 = Vector512.Load(init[4]); var c5 = Vector512.Load(init[5]);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector512.Load(b);
            c0 = Fma(Vector512.Create(a0[p]), b0, c0); c1 = Fma(Vector512.Create(a1[p]), b0, c1);
            c2 = Fma(Vector512.Create(a2[p]), b0, c2); c3 = Fma(Vector512.Create(a3[p]), b0, c3);
            c4 = Fma(Vector512.Create(a4[p]), b0, c4); c5 = Fma(Vector512.Create(a5[p]), b0, c5);
        }

        c0.Store(output[0]); c1.Store(output[1]); c2.Store(output[2]);
        c3.Store(output[3]); c4.Store(output[4]); c5.Store(output[5]);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x512(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        var c0 = Vector512.Load(init[0]); var c1 = Vector512.Load(init[1]); var c2 = Vector512.Load(init[2]);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector512.Load(b);
            c0 = Fma(Vector512.Create(a0[p]), b0, c0); c1 = Fma(Vector512.Create(a1[p]), b0, c1); c2 = Fma(Vector512.Create(a2[p]), b0, c2);
        }

        c0.Store(output[0]); c1.Store(output[1]); c2.Store(output[2]);
    }

    /// <summary>One row: four accumulators over k, summed like <see cref="Kernel1x256"/>'s, so the same bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel1x512(float* a, float* b, int k, float* init, float* output)
    {
        var c = Vector512.Load(init);
        Vector512<float> d = default, e = default, f = default;
        int p = 0;
        for (; p <= k - 4; p += 4, b += 4 * PackedMatrix.NR)
        {
            c = Fma(Vector512.Create(a[p]), Vector512.Load(b), c);
            d = Fma(Vector512.Create(a[p + 1]), Vector512.Load(b + 16), d);
            e = Fma(Vector512.Create(a[p + 2]), Vector512.Load(b + 32), e);
            f = Fma(Vector512.Create(a[p + 3]), Vector512.Load(b + 48), f);
        }

        for (; p < k; p++, b += PackedMatrix.NR)
            c = Fma(Vector512.Create(a[p]), Vector512.Load(b), c);

        ((c + d) + (e + f)).Store(output);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x512Pair(float** a, float* b, long bStride, int k, float** init, float** output)
    {
        const int nr = PackedMatrix.NR;
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        float* bb = b + bStride;
        var c00 = Vector512.Load(init[0]); var c01 = Vector512.Load(init[0] + nr);
        var c10 = Vector512.Load(init[1]); var c11 = Vector512.Load(init[1] + nr);
        var c20 = Vector512.Load(init[2]); var c21 = Vector512.Load(init[2] + nr);
        var c30 = Vector512.Load(init[3]); var c31 = Vector512.Load(init[3] + nr);
        var c40 = Vector512.Load(init[4]); var c41 = Vector512.Load(init[4] + nr);
        var c50 = Vector512.Load(init[5]); var c51 = Vector512.Load(init[5] + nr);
        for (int p = 0; p < k; p++, b += nr, bb += nr)
        {
            var b0 = Vector512.Load(b);
            var b1 = Vector512.Load(bb);
            var x = Vector512.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01);
            x = Vector512.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11);
            x = Vector512.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21);
            x = Vector512.Create(a3[p]); c30 = Fma(x, b0, c30); c31 = Fma(x, b1, c31);
            x = Vector512.Create(a4[p]); c40 = Fma(x, b0, c40); c41 = Fma(x, b1, c41);
            x = Vector512.Create(a5[p]); c50 = Fma(x, b0, c50); c51 = Fma(x, b1, c51);
        }

        c00.Store(output[0]); c01.Store(output[0] + nr);
        c10.Store(output[1]); c11.Store(output[1] + nr);
        c20.Store(output[2]); c21.Store(output[2] + nr);
        c30.Store(output[3]); c31.Store(output[3] + nr);
        c40.Store(output[4]); c41.Store(output[4] + nr);
        c50.Store(output[5]); c51.Store(output[5] + nr);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x512Pair(float** a, float* b, long bStride, int k, float** init, float** output)
    {
        const int nr = PackedMatrix.NR;
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        float* bb = b + bStride;
        var c00 = Vector512.Load(init[0]); var c01 = Vector512.Load(init[0] + nr);
        var c10 = Vector512.Load(init[1]); var c11 = Vector512.Load(init[1] + nr);
        var c20 = Vector512.Load(init[2]); var c21 = Vector512.Load(init[2] + nr);
        for (int p = 0; p < k; p++, b += nr, bb += nr)
        {
            var b0 = Vector512.Load(b);
            var b1 = Vector512.Load(bb);
            var x = Vector512.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01);
            x = Vector512.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11);
            x = Vector512.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21);
        }

        c00.Store(output[0]); c01.Store(output[0] + nr);
        c10.Store(output[1]); c11.Store(output[1] + nr);
        c20.Store(output[2]); c21.Store(output[2] + nr);
    }

    // Fma is MultiplyAddEstimate: one fused instruction where the CPU has one (x64 FMA3, every Arm64), else a·b + c.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x256(float** a, float* b, int k, float** init, float** output)
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x256(float** a, float* b, int k, float** init, float** output)
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
    private static void Kernel1x256(float* a, float* b, int k, float* init, float* output)
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

    /// <summary>6×16 in 24 accumulators + 4 panel registers + 1 broadcast: fits Arm64's 32 vector registers.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x128(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        float* i0 = init[0], i1 = init[1], i2 = init[2], i3 = init[3], i4 = init[4], i5 = init[5];
        var c00 = Vector128.Load(i0); var c01 = Vector128.Load(i0 + 4); var c02 = Vector128.Load(i0 + 8); var c03 = Vector128.Load(i0 + 12);
        var c10 = Vector128.Load(i1); var c11 = Vector128.Load(i1 + 4); var c12 = Vector128.Load(i1 + 8); var c13 = Vector128.Load(i1 + 12);
        var c20 = Vector128.Load(i2); var c21 = Vector128.Load(i2 + 4); var c22 = Vector128.Load(i2 + 8); var c23 = Vector128.Load(i2 + 12);
        var c30 = Vector128.Load(i3); var c31 = Vector128.Load(i3 + 4); var c32 = Vector128.Load(i3 + 8); var c33 = Vector128.Load(i3 + 12);
        var c40 = Vector128.Load(i4); var c41 = Vector128.Load(i4 + 4); var c42 = Vector128.Load(i4 + 8); var c43 = Vector128.Load(i4 + 12);
        var c50 = Vector128.Load(i5); var c51 = Vector128.Load(i5 + 4); var c52 = Vector128.Load(i5 + 8); var c53 = Vector128.Load(i5 + 12);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector128.Load(b); var b1 = Vector128.Load(b + 4); var b2 = Vector128.Load(b + 8); var b3 = Vector128.Load(b + 12);
            var x = Vector128.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01); c02 = Fma(x, b2, c02); c03 = Fma(x, b3, c03);
            x = Vector128.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11); c12 = Fma(x, b2, c12); c13 = Fma(x, b3, c13);
            x = Vector128.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21); c22 = Fma(x, b2, c22); c23 = Fma(x, b3, c23);
            x = Vector128.Create(a3[p]); c30 = Fma(x, b0, c30); c31 = Fma(x, b1, c31); c32 = Fma(x, b2, c32); c33 = Fma(x, b3, c33);
            x = Vector128.Create(a4[p]); c40 = Fma(x, b0, c40); c41 = Fma(x, b1, c41); c42 = Fma(x, b2, c42); c43 = Fma(x, b3, c43);
            x = Vector128.Create(a5[p]); c50 = Fma(x, b0, c50); c51 = Fma(x, b1, c51); c52 = Fma(x, b2, c52); c53 = Fma(x, b3, c53);
        }

        float* o = output[0]; c00.Store(o); c01.Store(o + 4); c02.Store(o + 8); c03.Store(o + 12);
        o = output[1]; c10.Store(o); c11.Store(o + 4); c12.Store(o + 8); c13.Store(o + 12);
        o = output[2]; c20.Store(o); c21.Store(o + 4); c22.Store(o + 8); c23.Store(o + 12);
        o = output[3]; c30.Store(o); c31.Store(o + 4); c32.Store(o + 8); c33.Store(o + 12);
        o = output[4]; c40.Store(o); c41.Store(o + 4); c42.Store(o + 8); c43.Store(o + 12);
        o = output[5]; c50.Store(o); c51.Store(o + 4); c52.Store(o + 8); c53.Store(o + 12);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x128(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        float* i0 = init[0], i1 = init[1], i2 = init[2];
        var c00 = Vector128.Load(i0); var c01 = Vector128.Load(i0 + 4); var c02 = Vector128.Load(i0 + 8); var c03 = Vector128.Load(i0 + 12);
        var c10 = Vector128.Load(i1); var c11 = Vector128.Load(i1 + 4); var c12 = Vector128.Load(i1 + 8); var c13 = Vector128.Load(i1 + 12);
        var c20 = Vector128.Load(i2); var c21 = Vector128.Load(i2 + 4); var c22 = Vector128.Load(i2 + 8); var c23 = Vector128.Load(i2 + 12);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector128.Load(b); var b1 = Vector128.Load(b + 4); var b2 = Vector128.Load(b + 8); var b3 = Vector128.Load(b + 12);
            var x = Vector128.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01); c02 = Fma(x, b2, c02); c03 = Fma(x, b3, c03);
            x = Vector128.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11); c12 = Fma(x, b2, c12); c13 = Fma(x, b3, c13);
            x = Vector128.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21); c22 = Fma(x, b2, c22); c23 = Fma(x, b3, c23);
        }

        float* o = output[0]; c00.Store(o); c01.Store(o + 4); c02.Store(o + 8); c03.Store(o + 12);
        o = output[1]; c10.Store(o); c11.Store(o + 4); c12.Store(o + 8); c13.Store(o + 12);
        o = output[2]; c20.Store(o); c21.Store(o + 4); c22.Store(o + 8); c23.Store(o + 12);
    }

    /// <summary>
    /// Arm64: <see cref="Kernel6x128"/> with each A value loaded into lane 0 and multiplied by element (`fmla v.4s, v.4s,
    /// v.s[0]`). The generic kernel's broadcast compiles to `ldr s` + `dup`, and the `dup`s take vector-pipe slots from the
    /// FMAs: 58–60% → 86–89% of peak on Neoverse N2. Same fused operations in the same order, so bitwise the same results.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x128Neon(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        float* i0 = init[0], i1 = init[1], i2 = init[2], i3 = init[3], i4 = init[4], i5 = init[5];
        var c00 = Vector128.Load(i0); var c01 = Vector128.Load(i0 + 4); var c02 = Vector128.Load(i0 + 8); var c03 = Vector128.Load(i0 + 12);
        var c10 = Vector128.Load(i1); var c11 = Vector128.Load(i1 + 4); var c12 = Vector128.Load(i1 + 8); var c13 = Vector128.Load(i1 + 12);
        var c20 = Vector128.Load(i2); var c21 = Vector128.Load(i2 + 4); var c22 = Vector128.Load(i2 + 8); var c23 = Vector128.Load(i2 + 12);
        var c30 = Vector128.Load(i3); var c31 = Vector128.Load(i3 + 4); var c32 = Vector128.Load(i3 + 8); var c33 = Vector128.Load(i3 + 12);
        var c40 = Vector128.Load(i4); var c41 = Vector128.Load(i4 + 4); var c42 = Vector128.Load(i4 + 8); var c43 = Vector128.Load(i4 + 12);
        var c50 = Vector128.Load(i5); var c51 = Vector128.Load(i5 + 4); var c52 = Vector128.Load(i5 + 8); var c53 = Vector128.Load(i5 + 12);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector128.Load(b); var b1 = Vector128.Load(b + 4); var b2 = Vector128.Load(b + 8); var b3 = Vector128.Load(b + 12);
            var x0 = Vector128.CreateScalarUnsafe(a0[p]);
            c00 = FmaLane0(x0, b0, c00); c01 = FmaLane0(x0, b1, c01); c02 = FmaLane0(x0, b2, c02); c03 = FmaLane0(x0, b3, c03);
            var x1 = Vector128.CreateScalarUnsafe(a1[p]);
            c10 = FmaLane0(x1, b0, c10); c11 = FmaLane0(x1, b1, c11); c12 = FmaLane0(x1, b2, c12); c13 = FmaLane0(x1, b3, c13);
            var x2 = Vector128.CreateScalarUnsafe(a2[p]);
            c20 = FmaLane0(x2, b0, c20); c21 = FmaLane0(x2, b1, c21); c22 = FmaLane0(x2, b2, c22); c23 = FmaLane0(x2, b3, c23);
            var x3 = Vector128.CreateScalarUnsafe(a3[p]);
            c30 = FmaLane0(x3, b0, c30); c31 = FmaLane0(x3, b1, c31); c32 = FmaLane0(x3, b2, c32); c33 = FmaLane0(x3, b3, c33);
            var x4 = Vector128.CreateScalarUnsafe(a4[p]);
            c40 = FmaLane0(x4, b0, c40); c41 = FmaLane0(x4, b1, c41); c42 = FmaLane0(x4, b2, c42); c43 = FmaLane0(x4, b3, c43);
            var x5 = Vector128.CreateScalarUnsafe(a5[p]);
            c50 = FmaLane0(x5, b0, c50); c51 = FmaLane0(x5, b1, c51); c52 = FmaLane0(x5, b2, c52); c53 = FmaLane0(x5, b3, c53);
        }

        float* o = output[0]; c00.Store(o); c01.Store(o + 4); c02.Store(o + 8); c03.Store(o + 12);
        o = output[1]; c10.Store(o); c11.Store(o + 4); c12.Store(o + 8); c13.Store(o + 12);
        o = output[2]; c20.Store(o); c21.Store(o + 4); c22.Store(o + 8); c23.Store(o + 12);
        o = output[3]; c30.Store(o); c31.Store(o + 4); c32.Store(o + 8); c33.Store(o + 12);
        o = output[4]; c40.Store(o); c41.Store(o + 4); c42.Store(o + 8); c43.Store(o + 12);
        o = output[5]; c50.Store(o); c51.Store(o + 4); c52.Store(o + 8); c53.Store(o + 12);
    }

    /// <summary>Arm64: <see cref="Kernel3x128"/> by element, like <see cref="Kernel6x128Neon"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x128Neon(float** a, float* b, int k, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        float* i0 = init[0], i1 = init[1], i2 = init[2];
        var c00 = Vector128.Load(i0); var c01 = Vector128.Load(i0 + 4); var c02 = Vector128.Load(i0 + 8); var c03 = Vector128.Load(i0 + 12);
        var c10 = Vector128.Load(i1); var c11 = Vector128.Load(i1 + 4); var c12 = Vector128.Load(i1 + 8); var c13 = Vector128.Load(i1 + 12);
        var c20 = Vector128.Load(i2); var c21 = Vector128.Load(i2 + 4); var c22 = Vector128.Load(i2 + 8); var c23 = Vector128.Load(i2 + 12);
        for (int p = 0; p < k; p++, b += PackedMatrix.NR)
        {
            var b0 = Vector128.Load(b); var b1 = Vector128.Load(b + 4); var b2 = Vector128.Load(b + 8); var b3 = Vector128.Load(b + 12);
            var x0 = Vector128.CreateScalarUnsafe(a0[p]);
            c00 = FmaLane0(x0, b0, c00); c01 = FmaLane0(x0, b1, c01); c02 = FmaLane0(x0, b2, c02); c03 = FmaLane0(x0, b3, c03);
            var x1 = Vector128.CreateScalarUnsafe(a1[p]);
            c10 = FmaLane0(x1, b0, c10); c11 = FmaLane0(x1, b1, c11); c12 = FmaLane0(x1, b2, c12); c13 = FmaLane0(x1, b3, c13);
            var x2 = Vector128.CreateScalarUnsafe(a2[p]);
            c20 = FmaLane0(x2, b0, c20); c21 = FmaLane0(x2, b1, c21); c22 = FmaLane0(x2, b2, c22); c23 = FmaLane0(x2, b3, c23);
        }

        float* o = output[0]; c00.Store(o); c01.Store(o + 4); c02.Store(o + 8); c03.Store(o + 12);
        o = output[1]; c10.Store(o); c11.Store(o + 4); c12.Store(o + 8); c13.Store(o + 12);
        o = output[2]; c20.Store(o); c21.Store(o + 4); c22.Store(o + 8); c23.Store(o + 12);
    }

    /// <summary>One row: two independent accumulator sets over k (8 registers), to hide the FMA latency.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel1x128(float* a, float* b, int k, float* init, float* output)
    {
        var c0 = Vector128.Load(init); var c1 = Vector128.Load(init + 4); var c2 = Vector128.Load(init + 8); var c3 = Vector128.Load(init + 12);
        Vector128<float> d0 = default, d1 = default, d2 = default, d3 = default;
        int p = 0;
        for (; p <= k - 2; p += 2, b += 2 * PackedMatrix.NR)
        {
            var x = Vector128.Create(a[p]);
            c0 = Fma(x, Vector128.Load(b), c0); c1 = Fma(x, Vector128.Load(b + 4), c1);
            c2 = Fma(x, Vector128.Load(b + 8), c2); c3 = Fma(x, Vector128.Load(b + 12), c3);
            x = Vector128.Create(a[p + 1]);
            d0 = Fma(x, Vector128.Load(b + 16), d0); d1 = Fma(x, Vector128.Load(b + 20), d1);
            d2 = Fma(x, Vector128.Load(b + 24), d2); d3 = Fma(x, Vector128.Load(b + 28), d3);
        }

        for (; p < k; p++, b += PackedMatrix.NR)
        {
            var x = Vector128.Create(a[p]);
            c0 = Fma(x, Vector128.Load(b), c0); c1 = Fma(x, Vector128.Load(b + 4), c1);
            c2 = Fma(x, Vector128.Load(b + 8), c2); c3 = Fma(x, Vector128.Load(b + 12), c3);
        }

        (c0 + d0).Store(output); (c1 + d1).Store(output + 4); (c2 + d2).Store(output + 8); (c3 + d3).Store(output + 12);
    }

    /// <summary>Any mr ≤ 6 rows without SIMD. Writes only the first mr outputs.</summary>
    private static void KernelScalar(int mr, float** a, float* b, int k, float** init, float** output)
    {
        const int nr = PackedMatrix.NR;
        var acc = stackalloc float[MR * nr];
        for (int r = 0; r < mr; r++)
            new Span<float>(init[r], nr).CopyTo(new Span<float>(acc + r * nr, nr));
        for (int p = 0; p < k; p++, b += nr)
        {
            for (int r = 0; r < mr; r++)
            {
                float x = a[r][p];
                float* row = acc + r * nr;
                for (int j = 0; j < nr; j++)
                    row[j] += x * b[j];
            }
        }

        for (int r = 0; r < mr; r++)
            new Span<float>(acc + r * nr, nr).CopyTo(new Span<float>(output[r], nr));
    }

    /// <summary>
    /// out[r][0..16) = (Σ over blocks of <paramref name="kb"/> k's, each summed from zero) + init[r][0..16), for the
    /// first mr ≤ 6 rows, on <see cref="Path"/>. Shorter chains than <see cref="Kernel"/>'s one sum from init: the LSTM
    /// recurrence uses it, where K is up to 1024 (the charlm), for about half the rounding error (docs/backends.md).
    /// Rows past mr as in <see cref="Kernel"/>; every row gets the same arithmetic whatever mr is.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void KernelBlocked(int mr, float** a, float* b, int k, int kb, float** init, float** output)
    {
        switch (Path)
        {
            case KernelPath.Vector512:
                if (mr <= 3)
                    Kernel3x512Blocked(a, b, k, kb, init, output);
                else
                    Kernel6x512Blocked(a, b, k, kb, init, output);
                break;
            case KernelPath.Vector256:
                if (mr <= 3)
                    Kernel3x256Blocked(a, b, k, kb, init, output);
                else
                    Kernel6x256Blocked(a, b, k, kb, init, output);
                break;
            case KernelPath.Vector128:
                if (mr <= 3)
                    Kernel3x128Blocked(a, b, k, kb, init, output);
                else
                    Kernel6x128Blocked(a, b, k, kb, init, output);
                break;
            default:
                KernelScalarBlocked(mr, a, b, k, kb, init, output);
                break;
        }
    }

    // The blocked kernels: per block the accumulators start at zero; after it they are added to the output (the sum of
    // the earlier blocks), and after the last block init is added.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x256Blocked(float** a, float* b, int k, int kb, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        float* o0 = output[0], o1 = output[1], o2 = output[2], o3 = output[3], o4 = output[4], o5 = output[5];
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector256<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default,
                c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += PackedMatrix.NR)
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

            if (k0 > 0)
            {
                c00 = Vector256.Load(o0) + c00; c01 = Vector256.Load(o0 + 8) + c01;
                c10 = Vector256.Load(o1) + c10; c11 = Vector256.Load(o1 + 8) + c11;
                c20 = Vector256.Load(o2) + c20; c21 = Vector256.Load(o2 + 8) + c21;
                c30 = Vector256.Load(o3) + c30; c31 = Vector256.Load(o3 + 8) + c31;
                c40 = Vector256.Load(o4) + c40; c41 = Vector256.Load(o4 + 8) + c41;
                c50 = Vector256.Load(o5) + c50; c51 = Vector256.Load(o5 + 8) + c51;
            }

            if (end == k)
            {
                c00 += Vector256.Load(init[0]); c01 += Vector256.Load(init[0] + 8);
                c10 += Vector256.Load(init[1]); c11 += Vector256.Load(init[1] + 8);
                c20 += Vector256.Load(init[2]); c21 += Vector256.Load(init[2] + 8);
                c30 += Vector256.Load(init[3]); c31 += Vector256.Load(init[3] + 8);
                c40 += Vector256.Load(init[4]); c41 += Vector256.Load(init[4] + 8);
                c50 += Vector256.Load(init[5]); c51 += Vector256.Load(init[5] + 8);
            }

            c00.Store(o0); c01.Store(o0 + 8);
            c10.Store(o1); c11.Store(o1 + 8);
            c20.Store(o2); c21.Store(o2 + 8);
            c30.Store(o3); c31.Store(o3 + 8);
            c40.Store(o4); c41.Store(o4 + 8);
            c50.Store(o5); c51.Store(o5 + 8);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x256Blocked(float** a, float* b, int k, int kb, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        float* o0 = output[0], o1 = output[1], o2 = output[2];
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector256<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += PackedMatrix.NR)
            {
                var b0 = Vector256.Load(b);
                var b1 = Vector256.Load(b + 8);
                var x = Vector256.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01);
                x = Vector256.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11);
                x = Vector256.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21);
            }

            if (k0 > 0)
            {
                c00 = Vector256.Load(o0) + c00; c01 = Vector256.Load(o0 + 8) + c01;
                c10 = Vector256.Load(o1) + c10; c11 = Vector256.Load(o1 + 8) + c11;
                c20 = Vector256.Load(o2) + c20; c21 = Vector256.Load(o2 + 8) + c21;
            }

            if (end == k)
            {
                c00 += Vector256.Load(init[0]); c01 += Vector256.Load(init[0] + 8);
                c10 += Vector256.Load(init[1]); c11 += Vector256.Load(init[1] + 8);
                c20 += Vector256.Load(init[2]); c21 += Vector256.Load(init[2] + 8);
            }

            c00.Store(o0); c01.Store(o0 + 8);
            c10.Store(o1); c11.Store(o1 + 8);
            c20.Store(o2); c21.Store(o2 + 8);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x512Blocked(float** a, float* b, int k, int kb, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector512<float> c0 = default, c1 = default, c2 = default, c3 = default, c4 = default, c5 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += PackedMatrix.NR)
            {
                var b0 = Vector512.Load(b);
                c0 = Fma(Vector512.Create(a0[p]), b0, c0); c1 = Fma(Vector512.Create(a1[p]), b0, c1);
                c2 = Fma(Vector512.Create(a2[p]), b0, c2); c3 = Fma(Vector512.Create(a3[p]), b0, c3);
                c4 = Fma(Vector512.Create(a4[p]), b0, c4); c5 = Fma(Vector512.Create(a5[p]), b0, c5);
            }

            bool first = k0 == 0, last = end == k;
            Flush512(output[0], init[0], first, last, c0); Flush512(output[1], init[1], first, last, c1);
            Flush512(output[2], init[2], first, last, c2); Flush512(output[3], init[3], first, last, c3);
            Flush512(output[4], init[4], first, last, c4); Flush512(output[5], init[5], first, last, c5);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x512Blocked(float** a, float* b, int k, int kb, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector512<float> c0 = default, c1 = default, c2 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += PackedMatrix.NR)
            {
                var b0 = Vector512.Load(b);
                c0 = Fma(Vector512.Create(a0[p]), b0, c0); c1 = Fma(Vector512.Create(a1[p]), b0, c1); c2 = Fma(Vector512.Create(a2[p]), b0, c2);
            }

            bool first = k0 == 0, last = end == k;
            Flush512(output[0], init[0], first, last, c0); Flush512(output[1], init[1], first, last, c1);
            Flush512(output[2], init[2], first, last, c2);
        }
    }

    /// <summary>
    /// <see cref="KernelBlocked"/> on Vector512 for two adjacent panels (<paramref name="b"/> and <paramref name="b"/> +
    /// <paramref name="bStride"/>), whose init and output columns follow the first panel's (+16), like
    /// <see cref="KernelPair512"/>: per k, 2 panel loads and mr broadcasts feed 2·mr FMAs instead of 1 and mr feeding
    /// mr, which held the single-panel kernel back on Intel (load ports; docs/performance/round-19-vector512-intel-two-panels.md). Each
    /// column's arithmetic is the single-panel kernel's.
    /// </summary>
    public static void KernelBlockedPair512(int mr, float** a, float* b, long bStride, int k, int kb, float** init, float** output)
    {
        if (mr <= 3)
            Kernel3x512BlockedPair(a, b, bStride, k, kb, init, output);
        else
            Kernel6x512BlockedPair(a, b, bStride, k, kb, init, output);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x512BlockedPair(float** a, float* b, long bStride, int k, int kb, float** init, float** output)
    {
        const int nr = PackedMatrix.NR;
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        float* bb = b + bStride;
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default,
                c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += nr, bb += nr)
            {
                var b0 = Vector512.Load(b);
                var b1 = Vector512.Load(bb);
                var x = Vector512.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01);
                x = Vector512.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11);
                x = Vector512.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21);
                x = Vector512.Create(a3[p]); c30 = Fma(x, b0, c30); c31 = Fma(x, b1, c31);
                x = Vector512.Create(a4[p]); c40 = Fma(x, b0, c40); c41 = Fma(x, b1, c41);
                x = Vector512.Create(a5[p]); c50 = Fma(x, b0, c50); c51 = Fma(x, b1, c51);
            }

            bool first = k0 == 0, last = end == k;
            Flush512(output[0], init[0], first, last, c00); Flush512(output[0] + nr, init[0] + nr, first, last, c01);
            Flush512(output[1], init[1], first, last, c10); Flush512(output[1] + nr, init[1] + nr, first, last, c11);
            Flush512(output[2], init[2], first, last, c20); Flush512(output[2] + nr, init[2] + nr, first, last, c21);
            Flush512(output[3], init[3], first, last, c30); Flush512(output[3] + nr, init[3] + nr, first, last, c31);
            Flush512(output[4], init[4], first, last, c40); Flush512(output[4] + nr, init[4] + nr, first, last, c41);
            Flush512(output[5], init[5], first, last, c50); Flush512(output[5] + nr, init[5] + nr, first, last, c51);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x512BlockedPair(float** a, float* b, long bStride, int k, int kb, float** init, float** output)
    {
        const int nr = PackedMatrix.NR;
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        float* bb = b + bStride;
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector512<float> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += nr, bb += nr)
            {
                var b0 = Vector512.Load(b);
                var b1 = Vector512.Load(bb);
                var x = Vector512.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01);
                x = Vector512.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11);
                x = Vector512.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21);
            }

            bool first = k0 == 0, last = end == k;
            Flush512(output[0], init[0], first, last, c00); Flush512(output[0] + nr, init[0] + nr, first, last, c01);
            Flush512(output[1], init[1], first, last, c10); Flush512(output[1] + nr, init[1] + nr, first, last, c11);
            Flush512(output[2], init[2], first, last, c20); Flush512(output[2] + nr, init[2] + nr, first, last, c21);
        }
    }

    /// <summary>A blocked kernel's row after a block, like <see cref="Flush128"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Flush512(float* o, float* init, bool first, bool last, Vector512<float> c)
    {
        if (!first)
            c = Vector512.Load(o) + c;
        if (last)
            c += Vector512.Load(init);
        c.Store(o);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x128Blocked(float** a, float* b, int k, int kb, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector128<float> c00 = default, c01 = default, c02 = default, c03 = default, c10 = default, c11 = default, c12 = default, c13 = default,
                c20 = default, c21 = default, c22 = default, c23 = default, c30 = default, c31 = default, c32 = default, c33 = default,
                c40 = default, c41 = default, c42 = default, c43 = default, c50 = default, c51 = default, c52 = default, c53 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += PackedMatrix.NR)
            {
                var b0 = Vector128.Load(b); var b1 = Vector128.Load(b + 4); var b2 = Vector128.Load(b + 8); var b3 = Vector128.Load(b + 12);
                var x = Vector128.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01); c02 = Fma(x, b2, c02); c03 = Fma(x, b3, c03);
                x = Vector128.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11); c12 = Fma(x, b2, c12); c13 = Fma(x, b3, c13);
                x = Vector128.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21); c22 = Fma(x, b2, c22); c23 = Fma(x, b3, c23);
                x = Vector128.Create(a3[p]); c30 = Fma(x, b0, c30); c31 = Fma(x, b1, c31); c32 = Fma(x, b2, c32); c33 = Fma(x, b3, c33);
                x = Vector128.Create(a4[p]); c40 = Fma(x, b0, c40); c41 = Fma(x, b1, c41); c42 = Fma(x, b2, c42); c43 = Fma(x, b3, c43);
                x = Vector128.Create(a5[p]); c50 = Fma(x, b0, c50); c51 = Fma(x, b1, c51); c52 = Fma(x, b2, c52); c53 = Fma(x, b3, c53);
            }

            bool first = k0 == 0, last = end == k;
            Flush128(output[0], init[0], first, last, c00, c01, c02, c03);
            Flush128(output[1], init[1], first, last, c10, c11, c12, c13);
            Flush128(output[2], init[2], first, last, c20, c21, c22, c23);
            Flush128(output[3], init[3], first, last, c30, c31, c32, c33);
            Flush128(output[4], init[4], first, last, c40, c41, c42, c43);
            Flush128(output[5], init[5], first, last, c50, c51, c52, c53);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel3x128Blocked(float** a, float* b, int k, int kb, float** init, float** output)
    {
        float* a0 = a[0], a1 = a[1], a2 = a[2];
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            Vector128<float> c00 = default, c01 = default, c02 = default, c03 = default, c10 = default, c11 = default, c12 = default, c13 = default,
                c20 = default, c21 = default, c22 = default, c23 = default;
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += PackedMatrix.NR)
            {
                var b0 = Vector128.Load(b); var b1 = Vector128.Load(b + 4); var b2 = Vector128.Load(b + 8); var b3 = Vector128.Load(b + 12);
                var x = Vector128.Create(a0[p]); c00 = Fma(x, b0, c00); c01 = Fma(x, b1, c01); c02 = Fma(x, b2, c02); c03 = Fma(x, b3, c03);
                x = Vector128.Create(a1[p]); c10 = Fma(x, b0, c10); c11 = Fma(x, b1, c11); c12 = Fma(x, b2, c12); c13 = Fma(x, b3, c13);
                x = Vector128.Create(a2[p]); c20 = Fma(x, b0, c20); c21 = Fma(x, b1, c21); c22 = Fma(x, b2, c22); c23 = Fma(x, b3, c23);
            }

            bool first = k0 == 0, last = end == k;
            Flush128(output[0], init[0], first, last, c00, c01, c02, c03);
            Flush128(output[1], init[1], first, last, c10, c11, c12, c13);
            Flush128(output[2], init[2], first, last, c20, c21, c22, c23);
        }
    }

    /// <summary>A blocked kernel's row after a block: o = (first ? 0 : o) + block sum, then + init after the last block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Flush128(float* o, float* init, bool first, bool last, Vector128<float> c0, Vector128<float> c1, Vector128<float> c2, Vector128<float> c3)
    {
        if (!first)
        {
            c0 = Vector128.Load(o) + c0; c1 = Vector128.Load(o + 4) + c1; c2 = Vector128.Load(o + 8) + c2; c3 = Vector128.Load(o + 12) + c3;
        }

        if (last)
        {
            c0 += Vector128.Load(init); c1 += Vector128.Load(init + 4); c2 += Vector128.Load(init + 8); c3 += Vector128.Load(init + 12);
        }

        c0.Store(o); c1.Store(o + 4); c2.Store(o + 8); c3.Store(o + 12);
    }

    /// <summary>Any mr ≤ 6 rows without SIMD, blocked. Writes only the first mr outputs.</summary>
    private static void KernelScalarBlocked(int mr, float** a, float* b, int k, int kb, float** init, float** output)
    {
        const int nr = PackedMatrix.NR;
        var acc = stackalloc float[MR * nr];
        for (int k0 = 0; k0 < k; k0 += kb)
        {
            new Span<float>(acc, mr * nr).Clear();
            int end = Math.Min(k, k0 + kb);
            for (int p = k0; p < end; p++, b += nr)
            {
                for (int r = 0; r < mr; r++)
                {
                    float x = a[r][p];
                    float* row = acc + r * nr;
                    for (int j = 0; j < nr; j++)
                        row[j] += x * b[j];
                }
            }

            for (int r = 0; r < mr; r++)
            {
                for (int j = 0; j < nr; j++)
                {
                    float sum = k0 > 0 ? output[r][j] + acc[r * nr + j] : acc[r * nr + j];
                    output[r][j] = end == k ? sum + init[r][j] : sum;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Fma(Vector512<float> a, Vector512<float> b, Vector512<float> c) => Vector512.MultiplyAddEstimate(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Fma(Vector256<float> a, Vector256<float> b, Vector256<float> c) => Vector256.MultiplyAddEstimate(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Fma(Vector128<float> a, Vector128<float> b, Vector128<float> c) => Vector128.MultiplyAddEstimate(a, b, c);

    /// <summary>c + x[0]·b in one fused Arm64 instruction (`fmla` by element); the same result as <c>Fma(broadcast(x[0]), b, c)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> FmaLane0(Vector128<float> x, Vector128<float> b, Vector128<float> c) => AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c, b, x, 0);
}

/// <summary>Activations and the fused element-wise steps, per <see cref="Gemm.Path"/> (PyTorch semantics; values agree to a few ulp).</summary>
internal static unsafe class Act
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Sigmoid(Vector512<float> x) => Vector512<float>.One / (Vector512<float>.One + VectorExp.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Sigmoid(Vector256<float> x) => Vector256<float>.One / (Vector256<float>.One + VectorExp.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Sigmoid(Vector128<float> x) => Vector128<float>.One / (Vector128<float>.One + VectorExp.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>torch.argmax: the first index of the maximum.</summary>
    public static int ArgMax(ReadOnlySpan<float> s)
    {
        int best = 0;
        for (int i = 1; i < s.Length; i++)
        {
            if (s[i] > s[best])
                best = i;
        }

        return best;
    }

    /// <summary>tanh(x) = 2σ(2x) − 1 (absolute error ≈ 1e-7 near 0).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Tanh(Vector128<float> x) => Sigmoid(x + x) * Vector128.Create(2f) - Vector128<float>.One;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<float> Tanh(Vector512<float> x) => Sigmoid(x + x) * Vector512.Create(2f) - Vector512<float>.One;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Tanh(Vector256<float> x) => Sigmoid(x + x) * Vector256.Create(2f) - Vector256<float>.One;

    /// <summary>
    /// LSTM cell for four units whose gates are [i×4, f×4, g×4, o×4] (<see cref="PackedLstm.GateOrder"/>):
    /// c = σ(f)·c + σ(i)·tanh(g); h = σ(o)·tanh(c).
    /// </summary>
    // AggressiveOptimization (also on AddGatedTanh): optimized from the first call, which a caller still at tier 0 makes
    // per four units (round 17); optimized callers still inline it.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void LstmCell(float* gates, float* c, float* h)
    {
        switch (Gemm.Path)
        {
            case KernelPath.Vector512:
                {
                    // One exp for all 16 lanes: σ(i), σ(f), σ(2g), σ(o); tanh(g) = 2σ(2g) − 1, as in Tanh.
                    var g = Vector128.Load(gates + 8);
                    var s = Sigmoid(Vector512.Create(Vector256.Load(gates), Vector256.Create(g + g, Vector128.Load(gates + 12))));
                    var lower = s.GetLower();
                    var upper = s.GetUpper();
                    var cNew = lower.GetUpper() * Vector128.Load(c) + lower.GetLower() * (upper.GetLower() * Vector128.Create(2f) - Vector128<float>.One);
                    cNew.Store(c);
                    (upper.GetUpper() * Tanh(cNew)).Store(h);
                    break;
                }
            case KernelPath.Vector256:
                {
                    var ifg = Sigmoid(Vector256.Load(gates));
                    var go = Vector256.Load(gates + 8);
                    var cNew = ifg.GetUpper() * Vector128.Load(c) + ifg.GetLower() * Tanh(go.GetLower());
                    cNew.Store(c);
                    (Sigmoid(go.GetUpper()) * Tanh(cNew)).Store(h);
                    break;
                }
            case KernelPath.Vector128:
                {
                    var cNew = Sigmoid(Vector128.Load(gates + 4)) * Vector128.Load(c) + Sigmoid(Vector128.Load(gates)) * Tanh(Vector128.Load(gates + 8));
                    cNew.Store(c);
                    (Sigmoid(Vector128.Load(gates + 12)) * Tanh(cNew)).Store(h);
                    break;
                }
            default:
                for (int u = 0; u < 4; u++)
                {
                    float cNew = Sigmoid(gates[4 + u]) * c[u] + Sigmoid(gates[u]) * MathF.Tanh(gates[8 + u]);
                    c[u] = cNew;
                    h[u] = Sigmoid(gates[12 + u]) * MathF.Tanh(cNew);
                }

                break;
        }
    }

    /// <summary>
    /// Σ a[q]·b[q] in double with four accumulators s0..s3 over q mod 4, then (s0 + s1) + (s2 + s3); the tail past the
    /// last multiple of 4 goes into s0. Every path gives the same bits: the SIMD lanes are the accumulators, a float
    /// product is exact in double, and each lane adds in the scalar order.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = b.Length, q = 0;
        if (a.Length < n)
            throw new ArgumentException("a is shorter than b", nameof(a));
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        fixed (float* pa = a, pb = b)
        {
            if (Gemm.Path is KernelPath.Vector512 or KernelPath.Vector256)
            {
                var s = Vector256<double>.Zero;
                for (; q <= n - 4; q += 4)
                    s += Vector256.WidenLower(Vector128.Load(pa + q).ToVector256Unsafe()) * Vector256.WidenLower(Vector128.Load(pb + q).ToVector256Unsafe());
                (s0, s1, s2, s3) = (s[0], s[1], s[2], s[3]);
            }
            else if (Gemm.Path == KernelPath.Vector128)
            {
                Vector128<double> s01 = Vector128<double>.Zero, s23 = Vector128<double>.Zero;
                for (; q <= n - 4; q += 4)
                {
                    var (x, y) = (Vector128.Load(pa + q), Vector128.Load(pb + q));
                    s01 += Vector128.WidenLower(x) * Vector128.WidenLower(y);
                    s23 += Vector128.WidenUpper(x) * Vector128.WidenUpper(y);
                }

                (s0, s1, s2, s3) = (s01[0], s01[1], s23[0], s23[1]);
            }
            else
            {
                for (; q <= n - 4; q += 4)
                {
                    s0 += (double)pa[q] * pb[q];
                    s1 += (double)pa[q + 1] * pb[q + 1];
                    s2 += (double)pa[q + 2] * pb[q + 2];
                    s3 += (double)pa[q + 3] * pb[q + 3];
                }
            }

            for (; q < n; q++)
                s0 += (double)pa[q] * pb[q];
        }

        return (s0 + s1) + (s2 + s3);
    }

    /// <summary>The highway step o[j] += σ(gate[j])·tanh(highway[j]) for j in [0, n).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddGatedTanh(float* o, float* gate, float* highway, int n)
    {
        int j = 0;
        if (Gemm.Path == KernelPath.Vector512)
        {
            for (; j <= n - 16; j += 16)
                (Vector512.Load(o + j) + Sigmoid(Vector512.Load(gate + j)) * Tanh(Vector512.Load(highway + j))).Store(o + j);
        }

        if (Gemm.Path is KernelPath.Vector512 or KernelPath.Vector256)
        {
            for (; j <= n - 8; j += 8)
                (Vector256.Load(o + j) + Sigmoid(Vector256.Load(gate + j)) * Tanh(Vector256.Load(highway + j))).Store(o + j);
        }

        if (Gemm.Path != KernelPath.Scalar)
        {
            for (; j <= n - 4; j += 4)
                (Vector128.Load(o + j) + Sigmoid(Vector128.Load(gate + j)) * Tanh(Vector128.Load(highway + j))).Store(o + j);
        }

        for (; j < n; j++)
            o[j] += Sigmoid(gate[j]) * MathF.Tanh(highway[j]);
    }
}
