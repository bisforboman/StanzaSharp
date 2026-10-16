using System.Text.Json;
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
        var conllu = "# text = 😀 ok\n1\t😀\t😀\tSYM\tNFP\t_\t0\troot\t_\tstart_char=0|end_char=2\n"
            + "2\tok\tstart_char=3\tINTJ\tUH\t_\t1\tdiscourse\t_\tSpaceAfter=No|start_char=3|end_char=5|ner=O\n\n";
        Assert.Equal("# text = 😀 ok\n1\t😀\t😀\tSYM\tNFP\t_\t0\troot\t_\tstart_char=0|end_char=1\n"
            + "2\tok\tstart_char=3\tINTJ\tUH\t_\t1\tdiscourse\t_\tSpaceAfter=No|start_char=2|end_char=4|ner=O\n\n",
            ConlluDiff.ToCodePointOffsets(conllu, text));
    }

    [Fact]
    public void Run_BadArgumentsExitWith2()
    {
        Assert.Equal(2, CompareCommand.Run([], "usage", TextWriter.Null));
        Assert.Equal(2, CompareCommand.Run(["missing.txt"], "usage", TextWriter.Null));
        Assert.Equal(2, CompareCommand.Run([Path.Combine(Repo.Golden, "corpus.txt"), "--nonsense"], "usage", TextWriter.Null));
        Assert.Equal(2, CompareCommand.Run([Path.Combine(Repo.Golden, "corpus.txt"), "--lang", "xx"], "usage", TextWriter.Null));
        Assert.Equal(2, CompareCommand.Run([Path.Combine(Repo.Golden, "corpus.txt"), "--lang", "de", "--package", "gsd", "--processors", "ner"], "usage",
            TextWriter.Null));
        Assert.Equal(2, CompareCommand.Run([Path.Combine(Repo.Golden, "corpus.txt"), "--lang", "de", "--models", "no-such-dir"], "usage", TextWriter.Null));
    }

    /// <summary>Each language's packages; a partial one (German's treebank packages) runs only its processors.</summary>
    [Fact]
    public void ProcessorList_PerLanguage()
    {
        Assert.Equal(Pipeline.AllProcessors, CompareCommand.ProcessorList("default", null, "de"));
        Assert.Equal("tokenize,mwt,pos,lemma,depparse", CompareCommand.ProcessorList("gsd", null, "de"));
        Assert.Equal("tokenize,mwt,pos,lemma,depparse", CompareCommand.ProcessorList("hdt", "depparse", "de"));
        Assert.Equal("tokenize,mwt,pos,lemma,depparse,sentiment,ner", CompareCommand.ProcessorList("default_fast", null, "de"));
        Assert.Throws<ArgumentException>(() => CompareCommand.ProcessorList("gsd", null, "en"));
        Assert.Throws<ArgumentException>(() => CompareCommand.ProcessorList("combined", "ner", "de"));
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

    /// <summary>German, all eight processors: Stanza's German pipeline and StanzaSharp agree on validation.txt.</summary>
    [PythonStanzaFact("de")]
    public void Run_PythonStanza_GermanIsIdentical()
    {
        var output = new StringWriter();
        int exit = CompareCommand.Run([Path.Combine(Repo.GoldenFor("de"), "validation.txt"), "--lang", "de", "--models", PythonStanzaFactAttribute.ModelsFor("de"),
            "--python", PythonStanzaFactAttribute.Python], "usage", output);
        Assert.True(exit == 0, output.ToString());
        Assert.Contains("language:    de", output.ToString());
        Assert.Contains("Identical: the CoNLL-U output is the same, byte for byte.", output.ToString());
    }

    /// <summary>A German treebank package: both sides run only its processors (tokenize to depparse) with its models.</summary>
    [PythonStanzaFact("de")]
    public void Run_PythonStanza_GermanTreebankPackage()
    {
        var output = new StringWriter();
        int exit = CompareCommand.Run([Path.Combine(Repo.GoldenFor("de"), "corpus.txt"), "--lang", "de", "--package", "gsd", "--models",
            PythonStanzaFactAttribute.ModelsFor("de"), "--python", PythonStanzaFactAttribute.Python], "usage", output);
        Assert.True(exit == 0, output.ToString());
        Assert.Contains("package:     gsd (tokenize,mwt,pos,lemma,depparse)", output.ToString());
        Assert.Contains("Identical: the CoNLL-U output is the same, byte for byte.", output.ToString());
    }

    /// <summary>A text without sentences: both CoNLL-U outputs are empty (compare.py wrote "\n" for Stanza's).</summary>
    [PythonStanzaFact]
    public void Run_PythonStanza_BlankTextIsIdentical()
    {
        var file = Path.GetTempFileName();
        try
        {
            File.WriteAllText(file, "  \n\n\t");
            var output = new StringWriter();
            int exit = CompareCommand.Run([file, "--processors", "tokenize,mwt", "--models", Repo.StanzaModels, "--python", PythonStanzaFactAttribute.Python],
                "usage", output);
            Assert.True(exit == 0, output.ToString());
        }
        finally
        {
            File.Delete(file);
        }
    }

    /// <summary>Stanza reads <c>&lt;dir&gt;/en/...</c>; compare.py links a folder of any other name as en.</summary>
    [PythonStanzaFact]
    public void Run_PythonStanza_ModelsFolderOfAnyName()
    {
        var models = Directory.CreateTempSubdirectory("stanzasharp-models-");
        try
        {
            foreach (var (path, _) in ModelDownloader.FilesFor("tokenize,mwt"))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(models.FullName, path))!);
                File.Copy(Path.Combine(Repo.StanzaModels, path), Path.Combine(models.FullName, path));
            }

            var output = new StringWriter();
            int exit = CompareCommand.Run([Path.Combine(Repo.Golden, "corpus.txt"), "--processors", "tokenize,mwt", "--models", models.FullName,
                "--python", PythonStanzaFactAttribute.Python], "usage", output);
            Assert.True(exit == 0, output.ToString());
            Assert.True(File.Exists(Path.Combine(models.FullName, "tokenize", "combined_nocharlm.pt")), "the link's target survives");
        }
        finally
        {
            models.Delete(recursive: true);
        }
    }

    /// <summary>
    /// compare.py's resources.json entry must be the Stanza release whose models ModelDownloader pins. After a Stanza
    /// version change, regenerate it as CLAUDE.md says (StanzaSharp.Tool, compare).
    /// </summary>
    [Theory]
    [InlineData("en")]
    [InlineData("de")]
    public void EmbeddedResources_MatchModelDownloader(string language)
    {
        using var resource = typeof(CompareCommand).Assembly.GetManifestResourceStream($"stanza_resources_{language}.json")!;
        using var json = JsonDocument.Parse(resource);
        Assert.Equal(ModelDownloader.StanzaVersion, json.RootElement.GetProperty("stanza_version").GetString());
        var resources = json.RootElement.GetProperty(language);
        foreach (var (path, md5) in ModelDownloader.Files[language])
        {
            var (processor, name) = (path.Split('/')[0], Path.GetFileNameWithoutExtension(path));
            var listed = resources.TryGetProperty(processor, out var models) && models.TryGetProperty(name, out var model) ? model.GetProperty("md5").GetString() : null;
            Assert.True(md5 == listed, $"{path}: stanza_resources_{language}.json lists MD5 {listed ?? "(none)"}, ModelDownloader.Files {md5}");
        }

        // Each package's models are Stanza's, and each model's shared files (the alternatives' too) are its resources.json dependencies.
        foreach (var (package, models) in Pipeline.Packages[language])
        {
            var stanza = resources.GetProperty("packages").GetProperty(package);
            Assert.Equal(stanza.EnumerateObject().Select(p => p.Name).Where(p => p != "optional").Order(), models.Keys.Order());
            foreach (var (processor, name) in models)
            {
                Assert.Equal(stanza.GetProperty(processor).GetString(), name);
                if (name != Pipeline.IdentityLemma) // Stanza's identity lemmatizer: no model in resources.json
                    AssertDependencies(processor, name);
            }
        }

        foreach (var (processor, names) in Pipeline.Alternatives[language])
        {
            foreach (var name in names)
                AssertDependencies(processor, name);
        }

        void AssertDependencies(string processor, string name)
        {
            var model = resources.GetProperty(processor).GetProperty(name);
            var dependencies = model.TryGetProperty("dependencies", out var d)
                ? d.EnumerateArray().Select(x => $"{x.GetProperty("model").GetString()}/{x.GetProperty("package").GetString()}").Order().ToList()
                : [];
            Assert.Equal(dependencies, Pipeline.SharedModels(new Dictionary<string, string> { [processor] = name }, language).Order());
        }
    }
}

/// <summary>
/// A fact that is skipped without Python Stanza (tools/.venv, or STANZASHARP_PYTHON, e.g. for a git worktree) or Stanza's
/// .pt models of the language (models/stanza/en, or models/stanza/de next to it). CI runs these in golden-shard 2
/// (STANZASHARP_PYTHON, both languages' models); the other jobs' no-skip checks allow the skip, as they have no Python Stanza.
/// </summary>
public sealed class PythonStanzaFactAttribute : FactAttribute
{
    internal static readonly string Python = Environment.GetEnvironmentVariable("STANZASHARP_PYTHON") is { Length: > 0 } python
        ? python
        : Path.Combine(Repo.Root, "tools", ".venv", OperatingSystem.IsWindows() ? "Scripts/python.exe" : "bin/python");

    public PythonStanzaFactAttribute(string lang = "en")
    {
        if (!File.Exists(Python))
            Skip = $"{Python} not found; see tools/requirements.txt";
        else if (!Directory.Exists(ModelsFor(lang)))
            Skip = $"{ModelsFor(lang)} not found; run stanzasharp download --lang {lang}";
    }

    /// <summary>Stanza's .pt files of a language: <see cref="Repo.StanzaModels"/> for English, its sibling folder for the others.</summary>
    internal static string ModelsFor(string lang) => lang == "en" ? Repo.StanzaModels : Path.GetFullPath(Path.Combine(Repo.StanzaModels, "..", lang));
}
