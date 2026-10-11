using StanzaSharp.Tool;

namespace StanzaSharp.Tests;

/// <summary><c>stanzasharp compare</c>: the diff and offset logic, and one run against Python Stanza.</summary>
public class CompareCommandTests
{
    private const string Sentence0 = "# sent_id = 0\n# text = Hi.\n1\tHi\thi\tINTJ\tUH\t_\t0\troot\t_\tstart_char=0|end_char=2\n";
    private const string Sentence1 = "# sent_id = 1\n# text = Go.\n1\tGo\tgo\tVERB\tVB\t_\t0\troot\t_\tstart_char=4|end_char=6\n";

    [Fact]
    public void Find_IdenticalIsNull() =>
        Assert.Null(ConlluDiff.Find(Sentence0 + "\n" + Sentence1 + "\n", Sentence0 + "\n" + Sentence1 + "\n"));

    [Fact]
    public void Find_ReportsTheFirstDifferingLineAndItsSentence()
    {
        var changed = Sentence1.Replace("\tgo\t", "\tgone\t");
        var d = ConlluDiff.Find(Sentence0 + "\n" + Sentence1 + "\n", Sentence0 + "\n" + changed + "\n")!;
        Assert.Equal(7, d.Line);
        Assert.Equal("1", d.SentId);
        Assert.Equal("Go.", d.Text);
        Assert.Contains("\tgo\t", d.Stanza);
        Assert.Contains("\tgone\t", d.StanzaSharp);
        Assert.Equal((2, 1), (d.Sentences, d.DifferentSentences));
    }

    [Fact]
    public void Find_MissingSentence()
    {
        var d = ConlluDiff.Find(Sentence0 + "\n" + Sentence1 + "\n", Sentence0 + "\n")!;
        Assert.Equal(5, d.Line);
        Assert.Equal("1", d.SentId);
        Assert.Equal("# sent_id = 1", d.Stanza);
        Assert.Equal("", d.StanzaSharp);
        Assert.Equal((2, 1), (d.Sentences, d.DifferentSentences));
    }

    [Fact]
    public void ToCodePointOffsets_ConvertsOnlyMiscOffsets()
    {
        // "😀" is two UTF-16 code units but one code point, so "ok" starts at 3 in .NET and 2 in Python.
        const string text = "😀 ok";
        var conllu = "# text = 😀 ok\n1\t😀\t😀\tSYM\tNFP\t_\t0\troot\t_\tstart_char=0|end_char=2\n" +
            "2\tok\tstart_char=3\tINTJ\tUH\t_\t1\tdiscourse\t_\tSpaceAfter=No|start_char=3|end_char=5|ner=O\n\n";
        Assert.Equal("# text = 😀 ok\n1\t😀\t😀\tSYM\tNFP\t_\t0\troot\t_\tstart_char=0|end_char=1\n" +
            "2\tok\tstart_char=3\tINTJ\tUH\t_\t1\tdiscourse\t_\tSpaceAfter=No|start_char=2|end_char=4|ner=O\n\n",
            ConlluDiff.ToCodePointOffsets(conllu, text));
    }

    [Fact]
    public void Run_BadArgumentsExitWith2()
    {
        Assert.Equal(2, CompareCommand.Run([], "usage", TextWriter.Null));
        Assert.Equal(2, CompareCommand.Run(["missing.txt"], "usage", TextWriter.Null));
        Assert.Equal(2, CompareCommand.Run([Path.Combine(Repo.Golden, "corpus.txt"), "--nonsense"], "usage", TextWriter.Null));
    }

    /// <summary>The whole command, as a new user runs it: Python Stanza (tools/.venv) and StanzaSharp agree on corpus.txt.</summary>
    [PythonStanzaFact]
    public void Run_PythonStanza_CorpusIsIdentical()
    {
        var output = new StringWriter();
        int exit = CompareCommand.Run([Path.Combine(Repo.Golden, "corpus.txt"), "--models", Repo.StanzaModels, "--python", PythonStanzaFactAttribute.Python],
            "usage", output);
        Assert.True(exit == 0, output.ToString());
        Assert.Contains("Identical: the CoNLL-U output is the same, byte for byte.", output.ToString());
        Assert.Contains("10 sentences, 140 words", output.ToString());
    }
}

/// <summary>
/// A fact that is skipped without Python Stanza (tools/.venv, or STANZASHARP_PYTHON, e.g. for a git worktree) or Stanza's
/// .pt models (models/stanza/en). CI's no-skip checks allow this one skip: CI has no tools/.venv.
/// </summary>
public sealed class PythonStanzaFactAttribute : FactAttribute
{
    internal static readonly string Python = Environment.GetEnvironmentVariable("STANZASHARP_PYTHON") is { Length: > 0 } python ? python
        : Path.Combine(Repo.Root, "tools", ".venv", OperatingSystem.IsWindows() ? "Scripts/python.exe" : "bin/python");

    public PythonStanzaFactAttribute()
    {
        if (!File.Exists(Python))
            Skip = $"{Python} not found; see tools/requirements.txt";
        else if (!Directory.Exists(Repo.StanzaModels))
            Skip = $"{Repo.StanzaModels} not found; run setup.ps1 -Models";
    }
}
