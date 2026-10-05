namespace StanzaSharp.Tests;

/// <summary>Paths into the repository, found by walking up from the test binaries to StanzaSharp.slnx.</summary>
public static class Repo
{
    public static readonly string Root = FindRoot();
    public static readonly string Golden = Path.Combine(Root, "tests", "golden");
    // STANZASHARP_MODELS overrides the location, e.g. for a git worktree without its own models/.
    public static readonly string Models =
        Environment.GetEnvironmentVariable("STANZASHARP_MODELS") is { Length: > 0 } dir ? dir : Path.Combine(Root, "models", "converted", "en");

    public static string Model(string relativeBase) => Path.Combine(Models, relativeBase);

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
