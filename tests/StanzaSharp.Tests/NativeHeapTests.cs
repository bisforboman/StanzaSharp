using Xunit;

namespace StanzaSharp.Tests;

[Trait("Backend", "Managed")]
public class NativeHeapTests
{
    // malloc_trim is found on Linux (CI's runners and the Docker check use glibc) and nowhere else; Trim never throws.
    [Fact]
    public void Trim_RunsOnlyOnLinux()
    {
        Assert.Equal(OperatingSystem.IsLinux(), NativeHeap.CanTrim);
        NativeHeap.Trim();
    }
}
