using StanzaSharp.Nn;
using Xunit;

namespace StanzaSharp.Tests;

public class ScalarsTests
{
    // Scalars.Softplus falls back to the racy F.softplus if TorchSharp drops its private softplus1.
    [Fact]
    public void Softplus_UsesTorchSharpsScalarOverload() => Assert.NotNull(Scalars.SoftplusWithScalars);
}
