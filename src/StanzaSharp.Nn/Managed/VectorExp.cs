using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// <c>Vector128/256/512.Exp(float)</c> of .NET 10 (<c>VectorMath.ExpSingle</c>, from AMD's aocl-libm <c>vrs4_expf</c>),
/// copied operation for operation, so the bits are the same, but compiled with full optimization from the first call.
/// </summary>
/// <remarks>
/// The runtime's own <c>ExpSingle</c> is too large to inline and has no usable ReadyToRun code (it depends on the
/// machine's vector width), so every sigmoid and tanh of a fresh process ran it at JIT tier 0 until it tiered up,
/// even inside our fully optimized kernels (docs/performance.md, round 17). Each method below takes the same
/// branches as the runtime's (which double vector width, which conversion instructions), and
/// <c>ManagedBackendTests.VectorExp_GivesTheRuntimesBits</c> compares them over every float pattern in a stride.
/// </remarks>
internal static class VectorExp
{
    private const uint ArgMax = 0x42AE0000;
    private const float ExpfMin = -103.97208f, ExpfMax = +88.72284f;
    private const double Huge = 6755399441055744, Ln2Inv = 1.4426950408889634;
    private const double C1 = 1.0000000754895704, C2 = 0.6931472254087585, C3 = 0.2402210737432219;
    private const double C4 = 0.05550297297702539, C5 = 0.009676036358193323, C6 = 0.001341000536524434;

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Vector128<float> Exp(Vector128<float> x)
    {
        if (!Vector128.IsHardwareAccelerated)
            return Vector128.Exp(x);
        Vector128<float> result;
        if (Vector256.IsHardwareAccelerated)
        {
            var wide = Avx.IsSupported ? Avx.ConvertToVector256Double(x) : Vector256.Create(Vector128.WidenLower(x), Vector128.WidenUpper(x));
            var e = Core(wide);
            result = Avx.IsSupported ? Avx.ConvertToVector128Single(e) : Vector128.Narrow(e.GetLower(), e.GetUpper());
        }
        else
        {
            result = Vector128.Narrow(Core(Vector128.WidenLower(x)), Core(Vector128.WidenUpper(x)));
        }

        if (Vector128.GreaterThanAny(Vector128.Abs(x).AsUInt32(), Vector128.Create(ArgMax)))
        {
            result = Vector128.ConditionalSelect(Vector128.GreaterThan(x, Vector128.Create(ExpfMax)), Vector128.Create(float.PositiveInfinity), result);
            result = Vector128.AndNot(result, Vector128.LessThan(x, Vector128.Create(ExpfMin)));
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Vector256<float> Exp(Vector256<float> x)
    {
        if (!Vector256.IsHardwareAccelerated)
            return Vector256.Exp(x);
        Vector256<float> result;
        if (Vector512.IsHardwareAccelerated)
        {
            var wide = Avx512F.IsSupported ? Avx512F.ConvertToVector512Double(x) : Vector512.Create(Vector256.WidenLower(x), Vector256.WidenUpper(x));
            var e = Core(wide);
            result = Avx512F.IsSupported ? Avx512F.ConvertToVector256Single(e) : Vector256.Narrow(e.GetLower(), e.GetUpper());
        }
        else
        {
            result = Vector256.Narrow(Core(Vector256.WidenLower(x)), Core(Vector256.WidenUpper(x)));
        }

        if (Vector256.GreaterThanAny(Vector256.Abs(x).AsUInt32(), Vector256.Create(ArgMax)))
        {
            result = Vector256.ConditionalSelect(Vector256.GreaterThan(x, Vector256.Create(ExpfMax)), Vector256.Create(float.PositiveInfinity), result);
            result = Vector256.AndNot(result, Vector256.LessThan(x, Vector256.Create(ExpfMin)));
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Vector512<float> Exp(Vector512<float> x)
    {
        if (!Vector512.IsHardwareAccelerated)
            return Vector512.Exp(x);
        var result = Vector512.Narrow(Core(Vector512.WidenLower(x)), Core(Vector512.WidenUpper(x)));
        if (Vector512.GreaterThanAny(Vector512.Abs(x).AsUInt32(), Vector512.Create(ArgMax)))
        {
            result = Vector512.ConditionalSelect(Vector512.GreaterThan(x, Vector512.Create(ExpfMax)), Vector512.Create(float.PositiveInfinity), result);
            result = Vector512.AndNot(result, Vector512.LessThan(x, Vector512.Create(ExpfMin)));
        }

        return result;
    }

    // ExpSingle's CoreImpl: 2^(x/ln 2) as 2^n · p(r), n the nearest integer, r = z − n, p a degree-5 polynomial.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<double> Core(Vector128<double> x)
    {
        var z = x * Vector128.Create(Ln2Inv);
        var huge = Vector128.Create(Huge);
        var dn = z + huge;
        var n = dn.AsUInt64();
        var r = z - (dn - huge);
        var r2 = r * r;
        var r4 = r2 * r2;
        var poly = Vector128.MultiplyAddEstimate(
            r4,
            Vector128.MultiplyAddEstimate(Vector128.Create(C6), r, Vector128.Create(C5)),
            Vector128.MultiplyAddEstimate(
                r2,
                Vector128.MultiplyAddEstimate(Vector128.Create(C4), r, Vector128.Create(C3)),
                Vector128.MultiplyAddEstimate(Vector128.Create(C2), r, Vector128.Create(C1))));
        return (poly.AsUInt64() + (n << 52)).AsDouble();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> Core(Vector256<double> x)
    {
        var z = x * Vector256.Create(Ln2Inv);
        var huge = Vector256.Create(Huge);
        var dn = z + huge;
        var n = dn.AsUInt64();
        var r = z - (dn - huge);
        var r2 = r * r;
        var r4 = r2 * r2;
        var poly = Vector256.MultiplyAddEstimate(
            r4,
            Vector256.MultiplyAddEstimate(Vector256.Create(C6), r, Vector256.Create(C5)),
            Vector256.MultiplyAddEstimate(
                r2,
                Vector256.MultiplyAddEstimate(Vector256.Create(C4), r, Vector256.Create(C3)),
                Vector256.MultiplyAddEstimate(Vector256.Create(C2), r, Vector256.Create(C1))));
        return (poly.AsUInt64() + (n << 52)).AsDouble();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> Core(Vector512<double> x)
    {
        var z = x * Vector512.Create(Ln2Inv);
        var huge = Vector512.Create(Huge);
        var dn = z + huge;
        var n = dn.AsUInt64();
        var r = z - (dn - huge);
        var r2 = r * r;
        var r4 = r2 * r2;
        var poly = Vector512.MultiplyAddEstimate(
            r4,
            Vector512.MultiplyAddEstimate(Vector512.Create(C6), r, Vector512.Create(C5)),
            Vector512.MultiplyAddEstimate(
                r2,
                Vector512.MultiplyAddEstimate(Vector512.Create(C4), r, Vector512.Create(C3)),
                Vector512.MultiplyAddEstimate(Vector512.Create(C2), r, Vector512.Create(C1))));
        return (poly.AsUInt64() + (n << 52)).AsDouble();
    }
}
