using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace StanzaSharp.Nn.Managed;

// Issue #29: a pure managed float32 backend, used by the processors ported so far when the backend is
// Backend.Managed. docs/managed-backend-spike.md (kernels) and docs/backends.md (the seam) have the numbers.

/// <summary>The SIMD flavor of the managed kernels. All three compute the same thing; tests run each of them.</summary>
internal enum KernelPath
{
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

/// <summary>Blocked SIMD float32 GEMM: C = A·Wᵀ + init, with a 6×16 register-blocked micro-kernel per <see cref="KernelPath"/>.</summary>
internal static unsafe class Gemm
{
    public const int MR = 6;
    private const int KC = 256, MC = 96, NC = 8; // NC panels (128 columns) per task

    /// <summary>The kernels' SIMD flavor, detected at startup. Settable so tests can run every path on one machine.</summary>
    public static KernelPath Path { get; set; } = Detect();

    public static KernelPath Detect() =>
        Vector256.IsHardwareAccelerated ? KernelPath.Vector256
        : Vector128.IsHardwareAccelerated ? KernelPath.Vector128
        : KernelPath.Scalar;

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
    /// per row (ldc ≥ PaddedN); bias (null for none) must have PaddedN floats. <paramref name="ct"/> is checked
    /// before each block of 96 rows × 128 columns.
    /// </summary>
    /// <param name="rowInvariant">Compute every row with the same arithmetic whatever <paramref name="m"/> is (no
    /// single-row kernel), so a row's result doesn't depend on the other rows in the call.</param>
    public static void Run(float* a, int m, int lda, PackedMatrix w, float* bias, float* c, int ldc, CancellationToken ct = default,
        bool rowInvariant = false)
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
                        Kernel(mr, aRows, panel, kc, init, cRows, rowInvariant);
                    }
                }
            }
        });
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
                    Kernel1x128(a[0], b, k, init[0], output[0]);
                else if (mr <= 3)
                {
                    if (AdvSimd.Arm64.IsSupported) Kernel3x128Neon(a, b, k, init, output); else Kernel3x128(a, b, k, init, output);
                }
                else if (AdvSimd.Arm64.IsSupported)
                    Kernel6x128Neon(a, b, k, init, output);
                else
                    Kernel6x128(a, b, k, init, output);
                break;
            default:
                KernelScalar(mr, a, b, k, init, output);
                break;
        }
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
            for (int r = 0; r < mr; r++)
            {
                float x = a[r][p];
                float* row = acc + r * nr;
                for (int j = 0; j < nr; j++)
                    row[j] += x * b[j];
            }
        for (int r = 0; r < mr; r++)
            new Span<float>(acc + r * nr, nr).CopyTo(new Span<float>(output[r], nr));
    }

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
    public static Vector256<float> Sigmoid(Vector256<float> x) => Vector256<float>.One / (Vector256<float>.One + Vector256.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Sigmoid(Vector128<float> x) => Vector128<float>.One / (Vector128<float>.One + Vector128.Exp(-x));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>tanh(x) = 2σ(2x) − 1 (absolute error ≈ 1e-7 near 0).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<float> Tanh(Vector128<float> x) => Sigmoid(x + x) * Vector128.Create(2f) - Vector128<float>.One;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<float> Tanh(Vector256<float> x) => Sigmoid(x + x) * Vector256.Create(2f) - Vector256<float>.One;

    /// <summary>
    /// LSTM cell for four units whose gates are [i×4, f×4, g×4, o×4] (<see cref="PackedLstm.GateOrder"/>):
    /// c = σ(f)·c + σ(i)·tanh(g); h = σ(o)·tanh(c).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void LstmCell(float* gates, float* c, float* h)
    {
        switch (Gemm.Path)
        {
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

    /// <summary>The highway step o[j] += σ(gate[j])·tanh(highway[j]) for j in [0, n).</summary>
    public static void AddGatedTanh(float* o, float* gate, float* highway, int n)
    {
        int j = 0;
        if (Gemm.Path == KernelPath.Vector256)
            for (; j <= n - 8; j += 8)
                (Vector256.Load(o + j) + Sigmoid(Vector256.Load(gate + j)) * Tanh(Vector256.Load(highway + j))).Store(o + j);
        if (Gemm.Path != KernelPath.Scalar)
            for (; j <= n - 4; j += 4)
                (Vector128.Load(o + j) + Sigmoid(Vector128.Load(gate + j)) * Tanh(Vector128.Load(highway + j))).Store(o + j);
        for (; j < n; j++)
            o[j] += Sigmoid(gate[j]) * MathF.Tanh(highway[j]);
    }
}
