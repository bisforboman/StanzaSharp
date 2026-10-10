using System.Diagnostics;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// The managed backend needs no native libtorch (issue #29, the 0.5 requirement). This test process has libtorch loaded,
/// so it runs tests/StanzaSharp.ManagedCheck in a process of its own: an app that references no TorchSharp-cpu or
/// libtorch-cpu-* package, so libtorch is not on disk next to it, and whose NuGet folder is empty, so TorchSharp's
/// fallback (copying libtorch from the NuGet cache into <c>cpu/</c>) finds nothing either. Any TorchSharp call that loads
/// native code then throws. The app loads and runs both packages, compares the golden CoNLL-U byte for byte, and checks
/// that no native torch module (LibTorchSharp, torch_cpu, c10) is loaded.
/// </summary>
public class ManagedCheckTests(ITestOutputHelper output)
{
    [ModelFact]
    [Trait("Backend", "Managed")]
    public void ManagedPipeline_RunsWithoutNativeLibtorch()
    {
        // tests/StanzaSharp.Tests/bin/<Configuration>/<tfm>/ → tests/StanzaSharp.ManagedCheck/bin/<Configuration>/<tfm>/
        var tfm = new DirectoryInfo(AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        var bin = Path.Combine(Repo.Root, "tests", "StanzaSharp.ManagedCheck", "bin", tfm.Parent!.Name, tfm.Name);
        var app = Path.Combine(bin, "StanzaSharp.ManagedCheck.dll");
        Assert.True(File.Exists(app), $"{app} not found; build StanzaSharp.slnx");

        // Left by a run of the app with libtorch in the NuGet cache (TorchSharp's fallback): remove it, then nothing
        // of libtorch may be on disk.
        if (Directory.Exists(Path.Combine(bin, "cpu")))
            Directory.Delete(Path.Combine(bin, "cpu"), recursive: true);
        string[] prefixes = ["torch_cpu", "torch.", "libtorch_cpu", "libtorch.", "c10.", "libc10."];
        var libtorch = Directory.GetFiles(bin, "*", SearchOption.AllDirectories)
            .Where(f => prefixes.Any(p => Path.GetFileName(f).StartsWith(p, StringComparison.OrdinalIgnoreCase)))
            .ToList();
        Assert.True(libtorch.Count == 0, "libtorch is next to the app: " + string.Join(", ", libtorch));

        var emptyNuget = Directory.CreateTempSubdirectory("stanzasharp-no-nuget-");
        try
        {
            var start = new ProcessStartInfo(Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") is { Length: > 0 } host ? host : "dotnet")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            foreach (var arg in new[] { app, Repo.Models, Repo.Golden })
                start.ArgumentList.Add(arg);
            start.Environment["NUGET_PACKAGES"] = emptyNuget.FullName;
            using var process = Process.Start(start)!;
            var stderr = process.StandardError.ReadToEndAsync();
            var stdout = process.StandardOutput.ReadToEnd();
            process.WaitForExit();
            output.WriteLine(stdout);
            output.WriteLine(stderr.Result);
            Assert.True(process.ExitCode == 0, $"exit code {process.ExitCode}\n{stdout}\n{stderr.Result}");
            Assert.Contains("Torch modules loaded: none", stdout);
            Assert.Contains("default: processed", stdout);
            Assert.Contains("default_fast: processed", stdout);
            Assert.False(Directory.Exists(Path.Combine(bin, "cpu")), "TorchSharp tried to consolidate libtorch next to the app");
        }
        finally
        {
            emptyNuget.Delete(recursive: true);
        }
    }
}
