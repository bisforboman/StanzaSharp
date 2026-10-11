using System.Globalization;
using StanzaSharp.Tool;
using static StanzaSharp.Tool.BenchmarkCommand;

namespace StanzaSharp.Tests;

/// <summary><c>stanzasharp benchmark</c>: the report, the CPU detection and a quick run.</summary>
public class BenchmarkCommandTests
{
    private static readonly Machine TestMachine = new("Test CPU 9000", 4, 8, 15.6, "Linux (X64)", ".NET 10.0.0", "Vector256", "1.2.3");

    private static Side TestSide(double scale) =>
        new(4, 1 * scale, new() { ["tokenize"] = 0.5 * scale, ["pos"] = 1.5 * scale }, 10 * scale, 20 * scale, 1000 * scale);

    [Fact]
    public void Format_WithPython_ShowsBothSidesAndTheRatio()
    {
        var saved = CultureInfo.CurrentCulture;
        CultureInfo.CurrentCulture = new CultureInfo("sv-SE"); // decimal comma: the block must not depend on it
        try
        {
            var text = Format(new Result(TestMachine, "default", "tokenize,pos", 2000, 100, 3, 50, TestSide(1), TestSide(2.5),
                "Stanza 1.15.0, torch 2.14.1, Python 3.12.0", "identical CoNLL-U on both sides"));
            Assert.Contains("| CPU | Test CPU 9000: 4 cores, 8 logical processors |", text);
            Assert.Contains("| RAM | 15.6 GB |", text);
            Assert.Contains("| .NET | .NET 10.0.0, SIMD Vector256 |", text);
            Assert.Contains("| StanzaSharp | 1.2.3, managed backend |", text);
            Assert.Contains("| Python Stanza | Stanza 1.15.0, torch 2.14.1, Python 3.12.0 |", text);
            Assert.Contains("2,000 words in 100 sentences, median of 3 timed runs after a warm-up, the two sides alternating", text);
            Assert.Contains("| Output | identical CoNLL-U on both sides |", text);
            Assert.Contains("| | StanzaSharp | Python Stanza | StanzaSharp's advantage |", text);
            Assert.Contains("| load | 1.00 s | 2.50 s | 2.50x |", text);
            Assert.Contains("| pos | 1.50 s | 3.75 s | 2.50x |", text);
            Assert.Contains("| **total** | 2.00 s | 5.00 s | 2.50x |", text);
            Assert.Contains("| words/s | 1,000 | 400 | 2.50x |", text);
            Assert.Contains("| per call, p90 | 20.0 ms | 50.0 ms | 2.50x |", text);
            Assert.Contains("| peak memory | 1,000 MB | 2,500 MB | 2.50x |", text);
        }
        finally
        {
            CultureInfo.CurrentCulture = saved;
        }
    }

    [Fact]
    public void Format_WithoutPython_HasOneColumn()
    {
        var text = Format(new Result(TestMachine with { Cores = null }, "default_fast", "tokenize,pos", 2000, 100, 1, 10, TestSide(1), null,
            "not run: could not start python", null));
        Assert.Contains("| CPU | Test CPU 9000: 8 logical processors |", text);
        Assert.Contains("| Python Stanza | not run: could not start python |", text);
        Assert.Contains("median of 1 timed run after a warm-up;", text);
        Assert.DoesNotContain("| Output |", text);
        Assert.Contains("| | StanzaSharp |\n", text.ReplaceLineEndings("\n"));
        Assert.Contains("| **total** | 2.00 s |\n", text.ReplaceLineEndings("\n"));
    }

    [Fact]
    public void ParseLscpu_X64()
    {
        const string lscpu = """
            Architecture:                x86_64
            CPU(s):                      4
            Vendor ID:                   AuthenticAMD
              BIOS Vendor ID:            Advanced Micro Devices, Inc.
              Model name:                AMD EPYC 7763 64-Core Processor
                BIOS Model name:         AMD EPYC 7763 64-Core Processor                 None CPU @ 2.4GHz
                Thread(s) per core:      2
                Core(s) per socket:      2
                Socket(s):               1
            """;
        Assert.Equal(("AMD EPYC 7763 64-Core Processor", 2), ParseLscpu(lscpu));
    }

    [Fact]
    public void ParseLscpu_ArmAndHybrid()
    {
        const string arm = "Architecture: aarch64\nVendor ID: ARM\n  Model name: Neoverse-N2\n    Thread(s) per core: 1\n    Core(s) per socket: 4\n    Socket(s): 1\n";
        Assert.Equal(("Neoverse-N2", 4), ParseLscpu(arm));
        const string hybrid = "Vendor ID: ARM\n  Model name: Cortex-A55\n    Core(s) per cluster: 4\n  Model name: Cortex-A76\n    Core(s) per cluster: 4\n";
        Assert.Equal(("Cortex-A55 + Cortex-A76", 4), ParseLscpu(hybrid));
        Assert.Equal((null, null), ParseLscpu(""));
    }

    [Fact]
    public void ParseCpuInfo_FirstModelName() =>
        Assert.Equal("Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz",
            ParseCpuInfo("processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz\n\nprocessor\t: 1\nmodel name\t: other\n"));

    /// <summary>This machine, through its OS's path (Windows: registry and GetLogicalProcessorInformation; Linux: lscpu; macOS: sysctl).</summary>
    [Fact]
    public void DescribeMachine_FindsThisCpu()
    {
        var m = DescribeMachine();
        Assert.DoesNotContain("model unknown", m.Cpu);
        Assert.Equal(Environment.ProcessorCount, m.LogicalProcessors);
        if (m.Cores is { } cores)
            Assert.InRange(cores, 1, m.LogicalProcessors);
        else
            Assert.True(OperatingSystem.IsLinux(), "only lscpu may lack the core count");
        Assert.True(m.RamGB > 0.5);
        Assert.Contains(m.Simd, new[] { "Vector256", "Vector128", "Scalar" });
    }

    [Fact]
    public void Paragraphs_CutsOrRepeatsTheGoldenTexts()
    {
        var shortText = Paragraphs(50);
        Assert.StartsWith(File.ReadAllText(Path.Combine(Repo.Golden, "validation.txt")).Split("\n\n")[0].Trim('\n'), shortText[0]);
        int Words(List<string> ps) => ps.Sum(p => p.Split([' ', '\t', '\r', '\n'], StringSplitOptions.RemoveEmptyEntries).Length);
        Assert.InRange(Words(shortText), 50, 400);
        var longText = Paragraphs(6000); // more than the ~2,400 of one copy
        Assert.InRange(Words(longText), 6000, 7000);
        Assert.True(longText.IndexOf(longText[0], 1) > 0, "the text repeats");
    }

    [Fact]
    public void Run_BadArgumentsExitWith2()
    {
        Assert.Equal(2, BenchmarkCommand.Run(["--nonsense"], "usage", TextWriter.Null));
        Assert.Equal(2, BenchmarkCommand.Run(["--threads", "0"], "usage", TextWriter.Null));
        Assert.Equal(2, BenchmarkCommand.Run(["--package", "nope"], "usage", TextWriter.Null));
        Assert.Equal(2, BenchmarkCommand.Run(["--models", "no-such-dir"], "usage", TextWriter.Null));
    }

    /// <summary>The smoke run without Python: StanzaSharp's column only, with the reason Stanza didn't run.</summary>
    [ModelFact]
    public void Run_Quick_WithoutPython()
    {
        var output = new StringWriter();
        Assert.Equal(0, BenchmarkCommand.Run(["--quick", "--models", Repo.Models, "--python", "no-such-python"], "usage", output));
        var text = output.ToString();
        Assert.Contains("| Python Stanza | not run: ", text);
        Assert.Contains("| **total** | ", text);
        Assert.Contains("| per call, median | ", text);
        Assert.DoesNotContain("| Output |", text);
    }

    /// <summary>The whole command with Python Stanza (tools/.venv): both columns, and both sides' output identical.</summary>
    [PythonStanzaFact]
    public void Run_PythonStanza_Quick()
    {
        var output = new StringWriter();
        Assert.Equal(0, BenchmarkCommand.Run(["--quick", "--models", Repo.StanzaModels, "--python", PythonStanzaFactAttribute.Python], "usage", output));
        var text = output.ToString();
        Assert.Contains("| Python Stanza | Stanza 1.15.0, torch ", text);
        Assert.Contains("| Output | identical CoNLL-U on both sides |", text);
        Assert.Contains("| | StanzaSharp | Python Stanza | StanzaSharp's advantage |", text);
    }
}
