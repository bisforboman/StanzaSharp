namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="PipelineOptions.ModelPaths"/> (Stanza's <c>{processor}_model_path</c>): models of one's own by file. The
/// golden rows Stanza ran are <see cref="OptionTests"/>' <c>*_model_path</c> cases; these check what the paths change.
/// </summary>
public sealed class ModelPathTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("stanzasharp-model-paths-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void CopiedModels_UnderOtherNames_GiveThePackagesOutput(bool managed)
    {
        // The tagger and every shared file it reads, copied under names no table knows: the files come from the paths,
        // what they read from the ModelPath, and the output is the package's (the lemmatizer reads the package's tags).
        var pos = new ModelPath(Copy("pos/combined_charlm", "tagger"), Copy("pretrain/conll17", "vectors"),
            Copy("forward_charlm/1billion", "forward"), Copy("backward_charlm/1billion", "backward"));
        using var nlp = Pipeline.Load(Repo.StanzaModels, new PipelineOptions
        {
            Processors = "tokenize,mwt,pos,lemma",
            ModelPaths = new Dictionary<string, ModelPath> { [Processor.Pos] = pos },
            Backend = Repo.PipelineBackendFor(managed),
        });
        var expected = File.ReadAllText(Path.Combine(Repo.Golden, "lemma", "corpus.conllu"));
        Assert.Equal(expected, Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")))));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void CharlmTagger_InDefaultFast_ReadsTheCharlmsItsModelPathNames(bool managed)
    {
        // default_fast's tagger reads no charlms: a _charlm one in its place gets them from its ModelPath, and tags as the
        // default package does.
        var stanza = Repo.StanzaModels;
        using var nlp = Pipeline.Load(stanza, new PipelineOptions
        {
            Package = "default_fast",
            Processors = "tokenize,mwt,pos,lemma",
            ModelPaths = new Dictionary<string, ModelPath>
            {
                [Processor.Pos] = new(Path.Combine(stanza, "pos", "combined_charlm.pt"),
                    ForwardCharlm: Path.Combine(stanza, "forward_charlm", "1billion.pt"), BackwardCharlm: Path.Combine(stanza, "backward_charlm", "1billion.pt")),
            },
            Backend = Repo.PipelineBackendFor(managed),
        });
        var expected = File.ReadAllText(Path.Combine(Repo.Golden, "lemma", "corpus.conllu"));
        Assert.Equal(expected, Conllu.Write(nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")))));
    }

    [ModelFact]
    public void CharlmTagger_WithoutCharlms_ThrowsNamingItsModelPath()
    {
        var e = Assert.Throws<ArgumentException>(() => Pipeline.Load(Repo.StanzaModels, new PipelineOptions
        {
            Package = "default_fast",
            Processors = "tokenize,mwt,pos",
            ModelPaths = new Dictionary<string, ModelPath> { [Processor.Pos] = Path.Combine(Repo.StanzaModels, "pos", "combined_charlm.pt") },
        }));
        Assert.Contains("ModelPaths[\"pos\"]", e.Message);
        Assert.Contains("ForwardCharlm", e.Message);
    }

    [ModelFact]
    public void VerifyChecksums_LeavesOutTheCallersFiles()
    {
        // The copy has no published checksum; every package file still gets checked.
        using var nlp = Pipeline.Load(Repo.StanzaModels, new PipelineOptions
        {
            Processors = "tokenize,mwt",
            ModelPaths = new Dictionary<string, ModelPath> { [Processor.Mwt] = Copy("mwt/combined", "mwt") + ".pt" },
            VerifyChecksums = true,
        });
        Assert.Equal(2, nlp.Process("I don't know.").Sentences[0].Tokens[1].Words.Count);
    }

    [Fact]
    public void MissingFile_NamesItsModelPath()
    {
        var file = Path.Combine(_dir, "absent.pt");
        var e = Assert.Throws<FileNotFoundException>(() => Pipeline.Load(_dir, new PipelineOptions
        {
            ModelPaths = new Dictionary<string, ModelPath> { [Processor.Ner] = new(Path.Combine(_dir, "ner.pt"), Pretrain: file) },
        }));
        Assert.Contains("ModelPaths[\"ner\"].Model", e.Message); // the model is checked first
        File.WriteAllBytes(Path.Combine(_dir, "ner.pt"), []);
        e = Assert.Throws<FileNotFoundException>(() => Pipeline.Load(_dir, new PipelineOptions
        {
            ModelPaths = new Dictionary<string, ModelPath> { [Processor.Ner] = new(Path.Combine(_dir, "ner.pt"), Pretrain: file) },
        }));
        Assert.Contains($"{file} (ModelPaths[\"ner\"].Pretrain)", e.Message);
    }

    [Theory]
    [InlineData("tokenize,mwt,pos", "pos", "Both Models and ModelPaths name pos")]
    [InlineData("tokenize,mwt", "ner", "ModelPaths names ner, which Processors (tokenize,mwt) doesn't list")]
    [InlineData(null, "parser", "Unknown processor 'parser' in ModelPaths")]
    public void Conflicts_Throw(string? processors, string key, string message)
    {
        var e = Assert.Throws<ArgumentException>(() => Pipeline.Load(_dir, new PipelineOptions
        {
            Processors = processors,
            Models = key == "pos" ? new Dictionary<string, string> { ["pos"] = "combined_nocharlm" } : null,
            ModelPaths = new Dictionary<string, ModelPath> { [key] = "my.pt" },
        }));
        Assert.Contains(message, e.Message);
    }

    [Fact]
    public void FilesFor_LeavesOutWhatThePathsReplace()
    {
        // A tagger of one's own reading the package's shared files: they are still downloaded, its model isn't.
        var files = Files(new() { [Processor.Pos] = "my/pos.pt" });
        Assert.DoesNotContain("pos/combined_charlm.pt", files);
        Assert.Contains("pretrain/conll17.pt", files);
        Assert.Contains("forward_charlm/1billion.pt", files);

        // With its own shared files too, and no other reader of the package's, none of them is.
        files = Files(new() { [Processor.Pos] = new("my/pos.pt", "my/vectors.pt", "my/forward.pt", "my/backward.pt") });
        Assert.Equal(["mwt/combined.pt", "tokenize/combined_nocharlm.pt"], files.Order(StringComparer.Ordinal));

        // A processor default_fast lacks: added, reading only what its ModelPath names.
        var fast = ModelDownloader.FilesFor("tokenize,mwt,pos,constituency", "default_fast", modelPaths: new Dictionary<string, ModelPath>
        {
            [Processor.Constituency] = new("my/parser.pt", "my/vectors.pt", "my/forward.pt", "my/backward.pt"),
        }).Select(f => f.Path).ToList();
        Assert.DoesNotContain(fast, f => f.StartsWith("constituency/", StringComparison.Ordinal) || f.StartsWith("forward_charlm/", StringComparison.Ordinal)
            || f.StartsWith("backward_charlm/", StringComparison.Ordinal));

        static List<string> Files(Dictionary<string, ModelPath> paths) =>
            [.. ModelDownloader.FilesFor("tokenize,mwt,pos", modelPaths: paths).Select(f => f.Path)];
    }

    /// <summary>Copies <paramref name="file"/> (a path under the models, without .pt) to the test's folder as <paramref name="name"/>.pt; returns its base path.</summary>
    private string Copy(string file, string name)
    {
        var target = Path.Combine(_dir, name);
        File.Copy(Path.Combine(Repo.StanzaModels, file + ".pt"), target + ".pt");
        return target;
    }
}
