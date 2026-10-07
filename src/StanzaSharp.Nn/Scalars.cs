using System.Reflection;
using TorchSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// Scalars that live as long as the process, for every TorchSharp call that takes one.
/// </summary>
/// <remarks>
/// A TorchSharp <see cref="Scalar"/> frees its native value in a finalizer and is not tracked by any
/// DisposeScope. TorchSharp passes <c>scalar.Handle</c> to native code without keeping the Scalar alive, so a
/// temporary one (an implicit conversion such as <c>t + 1</c> or <c>masked_fill(m, 0)</c>, or the hidden
/// <c>alpha = 1</c> of <c>a + b</c> and <c>add(b)</c>) can be finalized by a garbage collection that starts
/// during the call, while libtorch still reads it. That crashed concurrent Process calls on Arm64. So:
/// no tensor <c>+</c>, no <c>add(Tensor)</c>, no numbers where TorchSharp expects a Scalar. Use these
/// (at::Scalar is immutable, so sharing them across threads is fine), or a <c>using var</c> local for a
/// value known only at run time.
/// </remarks>
internal static class Scalars
{
    public static readonly Scalar Zero = 0, One = 1, Two = 2, NegativeInfinity = float.NegativeInfinity;

    // F.softplus(x) converts its beta = 1.0 and threshold = 20.0 to temporary Scalars and has no Scalar
    // overload, so call TorchSharp's private Scalar version with kept ones. Should a later TorchSharp drop
    // it, fall back to F.softplus rather than fail.
    private static readonly Scalar SoftplusBeta = 1.0, SoftplusThreshold = 20.0;
    internal static readonly Func<Tensor, Scalar, Scalar, Tensor>? SoftplusWithScalars =
        typeof(Tensor).GetMethod("softplus1", BindingFlags.NonPublic | BindingFlags.Instance, [typeof(Scalar), typeof(Scalar)])
            ?.CreateDelegate<Func<Tensor, Scalar, Scalar, Tensor>>();

    /// <summary><c>F.softplus(x)</c> with beta 1 and threshold 20.</summary>
    public static Tensor Softplus(Tensor x) =>
        SoftplusWithScalars?.Invoke(x, SoftplusBeta, SoftplusThreshold) ?? nn.functional.softplus(x);
}
