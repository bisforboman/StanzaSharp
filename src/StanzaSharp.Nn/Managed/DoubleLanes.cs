using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// Several vectors of doubles that every operation runs on one after the other (round 51), for element-wise code whose
/// dependency chains are too long for the CPU to overlap one vector at a time (the encoder's GELU). Each lane gets the
/// scalar operations: a multiply and an add are never fused, so any width gives the scalar code's bits.
/// </summary>
/// <typeparam name="T">The lane group type.</typeparam>
internal interface IDoubleLanes<T> where T : struct, IDoubleLanes<T>
{
    static abstract T operator +(T x, T y);

    static abstract T operator +(T x, double y);

    static abstract T operator +(double x, T y);

    static abstract T operator -(T x, T y);

    static abstract T operator -(T x, double y);

    static abstract T operator -(double x, T y);

    static abstract T operator -(T x);

    static abstract T operator *(T x, T y);

    static abstract T operator *(T x, double y);

    static abstract T operator *(double x, T y);

    /// <summary>x / y per lane.</summary>
    static abstract T Divide(double x, T y);

    static abstract T Abs(T x);

    /// <summary>Math.Max(x, y) per lane.</summary>
    static abstract T Max(T x, double y);

    /// <summary>x ≥ 0 ? ifTrue : ifFalse per lane.</summary>
    static abstract T SelectNonNegative(T x, T ifTrue, T ifFalse);

    /// <summary>2^n per lane, for integral n in the normal exponent range.</summary>
    static abstract T TwoToThe(T n);
}

/// <summary>Four <see cref="Vector512{T}"/> of doubles (32 lanes), for <see cref="KernelPath.Vector512"/>.</summary>
internal readonly struct Lanes512(Vector512<double> a, Vector512<double> b, Vector512<double> c, Vector512<double> d) : IDoubleLanes<Lanes512>
{
    public readonly Vector512<double> A = a, B = b, C = c, D = d;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator +(Lanes512 x, Lanes512 y) => new(x.A + y.A, x.B + y.B, x.C + y.C, x.D + y.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator +(Lanes512 x, double y) => x + Of(y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator +(double x, Lanes512 y) => Of(x) + y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator -(Lanes512 x, Lanes512 y) => new(x.A - y.A, x.B - y.B, x.C - y.C, x.D - y.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator -(Lanes512 x, double y) => x - Of(y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator -(double x, Lanes512 y) => Of(x) - y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator -(Lanes512 x) => new(-x.A, -x.B, -x.C, -x.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator *(Lanes512 x, Lanes512 y) => new(x.A * y.A, x.B * y.B, x.C * y.C, x.D * y.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator *(Lanes512 x, double y) => x * Of(y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 operator *(double x, Lanes512 y) => Of(x) * y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 Divide(double x, Lanes512 y)
    {
        var v = Vector512.Create(x);
        return new(v / y.A, v / y.B, v / y.C, v / y.D);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 Abs(Lanes512 x) => new(Vector512.Abs(x.A), Vector512.Abs(x.B), Vector512.Abs(x.C), Vector512.Abs(x.D));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 Max(Lanes512 x, double y)
    {
        var v = Vector512.Create(y);
        return new(Vector512.Max(x.A, v), Vector512.Max(x.B, v), Vector512.Max(x.C, v), Vector512.Max(x.D, v));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 SelectNonNegative(Lanes512 x, Lanes512 ifTrue, Lanes512 ifFalse) =>
        new(Select(x.A, ifTrue.A, ifFalse.A), Select(x.B, ifTrue.B, ifFalse.B), Select(x.C, ifTrue.C, ifFalse.C), Select(x.D, ifTrue.D, ifFalse.D));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Lanes512 TwoToThe(Lanes512 n) => new(Pow2(n.A), Pow2(n.B), Pow2(n.C), Pow2(n.D));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Lanes512 Of(double x)
    {
        var v = Vector512.Create(x);
        return new(v, v, v, v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> Select(Vector512<double> x, Vector512<double> ifTrue, Vector512<double> ifFalse) =>
        Vector512.ConditionalSelect(Vector512.GreaterThanOrEqual(x, Vector512<double>.Zero), ifTrue, ifFalse);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> Pow2(Vector512<double> n) => Vector512.ShiftLeft(Vector512.ConvertToInt64(n) + Vector512.Create(1023L), 52).AsDouble();
}

/// <summary>Four <see cref="Vector{T}"/> of doubles, for the other paths.</summary>
internal readonly struct LanesVector(Vector<double> a, Vector<double> b, Vector<double> c, Vector<double> d) : IDoubleLanes<LanesVector>
{
    public readonly Vector<double> A = a, B = b, C = c, D = d;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator +(LanesVector x, LanesVector y) => new(x.A + y.A, x.B + y.B, x.C + y.C, x.D + y.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator +(LanesVector x, double y) => x + Of(y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator +(double x, LanesVector y) => Of(x) + y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator -(LanesVector x, LanesVector y) => new(x.A - y.A, x.B - y.B, x.C - y.C, x.D - y.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator -(LanesVector x, double y) => x - Of(y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator -(double x, LanesVector y) => Of(x) - y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator -(LanesVector x) => new(-x.A, -x.B, -x.C, -x.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator *(LanesVector x, LanesVector y) => new(x.A * y.A, x.B * y.B, x.C * y.C, x.D * y.D);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator *(LanesVector x, double y) => x * Of(y);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector operator *(double x, LanesVector y) => Of(x) * y;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector Divide(double x, LanesVector y)
    {
        var v = new Vector<double>(x);
        return new(v / y.A, v / y.B, v / y.C, v / y.D);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector Abs(LanesVector x) => new(Vector.Abs(x.A), Vector.Abs(x.B), Vector.Abs(x.C), Vector.Abs(x.D));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector Max(LanesVector x, double y)
    {
        var v = new Vector<double>(y);
        return new(Vector.Max(x.A, v), Vector.Max(x.B, v), Vector.Max(x.C, v), Vector.Max(x.D, v));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector SelectNonNegative(LanesVector x, LanesVector ifTrue, LanesVector ifFalse) =>
        new(Select(x.A, ifTrue.A, ifFalse.A), Select(x.B, ifTrue.B, ifFalse.B), Select(x.C, ifTrue.C, ifFalse.C), Select(x.D, ifTrue.D, ifFalse.D));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static LanesVector TwoToThe(LanesVector n) => new(Pow2(n.A), Pow2(n.B), Pow2(n.C), Pow2(n.D));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static LanesVector Of(double x)
    {
        var v = new Vector<double>(x);
        return new(v, v, v, v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> Select(Vector<double> x, Vector<double> ifTrue, Vector<double> ifFalse) =>
        Vector.ConditionalSelect(Vector.GreaterThanOrEqual(x, Vector<double>.Zero), ifTrue, ifFalse);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> Pow2(Vector<double> n) => Vector.AsVectorDouble(Vector.ShiftLeft(Vector.ConvertToInt64(n) + new Vector<long>(1023), 52));
}
