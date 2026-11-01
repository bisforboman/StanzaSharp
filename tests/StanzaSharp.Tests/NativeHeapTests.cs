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

    // The managed models live on the GC heap: a managed pipeline never trims (it would only take glibc's arena locks).
    [Fact]
    public void ManagedModels_DontTrimTheNativeHeap()
    {
        using var models = PipelineBackend.Managed.Models();
        Assert.False(models.UsesNativeHeap);
    }
}

[Trait("Backend", "TorchSharp")]
public class NativeHeapTorchSharpTests
{
    // libtorch allocates tensors with malloc, so a TorchSharp pipeline trims after large calls.
    [Fact]
    public void TorchSharpModels_TrimTheNativeHeap()
    {
        using var models = CudaBackend.Cpu.Models();
        Assert.True(models.UsesNativeHeap);
    }
}
