using StanzaSharp.Nn;

namespace StanzaSharp.Tests;

/// <summary>Paths into the repository, found by walking up from the test binaries to StanzaSharp.slnx.</summary>
public static class Repo
{
    public static readonly string Root = FindRoot();
    public static readonly string Golden = Path.Combine(Root, "tests", "golden");
    // STANZASHARP_MODELS overrides the location, e.g. for a git worktree without its own models/.
    public static readonly string Models =
        Environment.GetEnvironmentVariable("STANZASHARP_MODELS") is { Length: > 0 } dir ? dir : Path.Combine(Root, "models", "converted", "en");

    // The original Stanza download (<processor>/<name>.pt), found next to the converted models: models/stanza/en.
    public static readonly string StanzaModels = Path.GetFullPath(Path.Combine(Models, "..", "..", "stanza", "en"));

    public static string Model(string relativeBase) => Path.Combine(Models, relativeBase);

    /// <summary>The backend golden tests run on (issue #29): <see cref="Backend.Managed"/> or TorchSharp.</summary>
    internal static Backend Backend(bool managed) => managed ? Nn.Backend.Managed : Nn.Backend.TorchSharp;

    /// <summary>The shared word vectors in the backend's form: a plain array when managed, so no libtorch is needed
    /// (the linux-arm64 CI job runs the managed cases without one).</summary>
    internal static Pretrain LoadPretrain(bool managed) =>
        managed ? Pretrain.LoadManaged(Model("pretrain/conll17")) : Pretrain.Load(Model("pretrain/conll17"));

    /// <summary>The same as a pipeline option.</summary>
    internal static PipelineBackend PipelineBackendFor(bool managed) => managed ? PipelineBackend.Managed : CudaBackend.Cpu;

    private static string FindRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
            if (File.Exists(Path.Combine(dir.FullName, "StanzaSharp.slnx")))
                return dir.FullName;
        throw new InvalidOperationException("StanzaSharp.slnx not found above " + AppContext.BaseDirectory);
    }
}

/// <summary>A fact that is skipped when the converted models are missing.</summary>
public sealed class ModelFactAttribute : FactAttribute
{
    public ModelFactAttribute()
    {
        if (!Directory.Exists(Repo.Models))
            Skip = "models/converted/en not found; run setup.ps1 -Models";
    }
}

/// <summary>A theory that is skipped when the converted models are missing.</summary>
public sealed class ModelTheoryAttribute : TheoryAttribute
{
    public ModelTheoryAttribute()
    {
        if (!Directory.Exists(Repo.Models))
            Skip = "models/converted/en not found; run setup.ps1 -Models";
    }
}

/// <summary>A theory that is skipped unless both the original .pt models and the converted ones are present.</summary>
public sealed class PtModelTheoryAttribute : TheoryAttribute
{
    public PtModelTheoryAttribute()
    {
        if (!Directory.Exists(Repo.Models) || !Directory.Exists(Repo.StanzaModels))
            Skip = "models/stanza/en or models/converted/en not found; run setup.ps1 -Models";
    }
}

/// <summary>A fact that is skipped when the original .pt models are missing.</summary>
public sealed class PtModelFactAttribute : FactAttribute
{
    public PtModelFactAttribute()
    {
        if (!Directory.Exists(Repo.StanzaModels))
            Skip = "models/stanza/en not found; run setup.ps1 -Models";
    }
}
