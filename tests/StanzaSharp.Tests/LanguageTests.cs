using System.Net;
using StanzaSharp.Tool;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="PipelineOptions.Language"/>: the per-language package, dependency and checksum tables, the downloader, and
/// German's golden data (tests/golden/de; the processors' test classes run German against it).
/// </summary>
[Trait("Backend", "Managed")]
public class LanguageTests
{
    private static readonly string GermanGolden = Path.Combine(Repo.Golden, "de");

    private const string GermanShared = "pretrain/conll17.pt forward_charlm/newswiki.pt backward_charlm/newswiki.pt";

    [Theory]
    [InlineData("default", null, "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt lemma/combined_nocharlm.pt constituency/spmrl_charlm.pt "
        + "depparse/combined_charlm.pt sentiment/sb10k_charlm.pt ner/germeval2014.pt pretrain/fasttextwiki.pt " + GermanShared)]

    // NER reads fastText's vectors and the charlms, not conll17.
    [InlineData("default", "tokenize,ner", "tokenize/combined_nocharlm.pt ner/germeval2014.pt pretrain/fasttextwiki.pt forward_charlm/newswiki.pt backward_charlm/newswiki.pt")]
    [InlineData("default", "tokenize,mwt", "tokenize/combined_nocharlm.pt mwt/combined.pt")]

    // ner and sentiment keep their charlm models, so the fast package still needs both pretrains and the charlms.
    [InlineData("default_fast", null, "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_nocharlm.pt lemma/combined_nocharlm.pt depparse/combined_nocharlm.pt "
        + "sentiment/sb10k_charlm.pt ner/germeval2014.pt pretrain/fasttextwiki.pt " + GermanShared)]
    [InlineData("default_fast", "depparse", "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_nocharlm.pt lemma/combined_nocharlm.pt depparse/combined_nocharlm.pt pretrain/conll17.pt")]
    [InlineData("combined", null, "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt lemma/combined_nocharlm.pt depparse/combined_charlm.pt " + GermanShared)]
    [InlineData("gsd", null, "tokenize/gsd_nocharlm.pt mwt/gsd.pt pos/gsd_charlm.pt lemma/gsd_nocharlm.pt depparse/gsd_charlm.pt " + GermanShared)]
    [InlineData("hdt", null, "tokenize/hdt_nocharlm.pt mwt/hdt.pt pos/hdt_charlm.pt lemma/hdt_nocharlm.pt depparse/hdt_charlm.pt " + GermanShared)]
    [InlineData("hdt", "tokenize,mwt", "tokenize/hdt_nocharlm.pt mwt/hdt.pt")]
    public void FilesFor_German(string package, string? processors, string expected) =>
        Assert.Equal(expected.Split(' ').Order(), ModelDownloader.FilesFor(processors, package, "options", "de").Select(f => f.Path).Order());

    [Fact]
    public void FileTable_IsEveryGermanPackagesFiles()
    {
        Assert.Equal(25, ModelDownloader.Files["de"].Length);
        Assert.Equal(ModelDownloader.Files["de"].Select(f => f.Path).Order(),
            Pipeline.Packages["de"].Keys.SelectMany(p => ModelDownloader.FilesFor(null, p, "options", "de")).Select(f => f.Path)
                .Concat(ModelDownloaderTests.AlternativesFiles("de")).Distinct().Order());
        Assert.Equal(Pipeline.Packages.Keys.Order(), ModelDownloader.Files.Keys.Order());
        Assert.Equal(Pipeline.Packages.Keys.Order(), Pipeline.Dependencies.Keys.Order());
    }

    [Fact]
    public void SharedModels_ArePretrainsThenCharlms()
    {
        var models = Pipeline.SelectModels("default", null, addRequired: false, "options", "de");
        Assert.Equal(["pretrain/conll17", "pretrain/fasttextwiki", "forward_charlm/newswiki", "backward_charlm/newswiki"], Pipeline.SharedModels(models, "de"));
        Assert.Equal(Pipeline.AllProcessors.Split(',').Order(), models.Keys.Order());
        Assert.Equal("spmrl_charlm", models["constituency"]);
        Assert.Empty(Pipeline.SharedModels(Pipeline.SelectModels("default", "tokenize,mwt", addRequired: false, "options", "de"), "de"));
    }

    [Fact]
    public void UnknownLanguagesAndPackagesThrowWithTheSupportedList()
    {
        var ex = Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(null, "default", "options", "fr"));
        Assert.Contains("en, de", ex.Message);
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(null, "default", "options", "DE"));
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(null, "default", "options", null!));

        ex = Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(null, "default_accurate", "options", "de"));
        Assert.Contains("transformer", ex.Message);
        Assert.Contains("default, default_fast, combined, gsd, hdt", ex.Message);

        // The treebank packages are German only, and have no ner; the error names the packages that do.
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(null, "gsd"));
        ex = Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor("ner", "gsd", "options", "de"));
        Assert.Contains("\"default\" or \"default_fast\"", ex.Message);
    }

    [Fact]
    public void Load_RejectsUnknownLanguagesAndPackages()
    {
        // Options are checked before anything loads, as for English.
        Assert.Throws<ArgumentException>(() => Pipeline.Load(Repo.Root, new PipelineOptions { Language = "de", Package = "default_accurate" }));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(Repo.Root, new PipelineOptions { Language = "fr" }));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(Repo.Root, new PipelineOptions { Language = "de", Package = "gsd", Processors = "tokenize,ner" }));
    }

    /// <summary>A German pipeline loads both pretrains (fastText for NER) and the newswiki charlms, each once.</summary>
    [GermanFact]
    public void Load_German_ReadsBothPretrains()
    {
        using var nlp = Pipeline.Load(Repo.ModelsFor("de"), new PipelineOptions { Language = "de", Processors = "tokenize,mwt,pos,ner" });
        var pretrains = (Nn.Pretrain[])typeof(Pipeline).GetField("_pretrains", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(nlp)!;
        Assert.Equal([100, 300], pretrains.Select(p => p.Dim).Order());
        var doc = nlp.Process("Angela Merkel wurde in Hamburg geboren.");
        Assert.Equal(["Angela Merkel/PER", "Hamburg/LOC"], doc.Entities.Select(e => $"{e.Text}/{e.Type}"));
    }

    [Fact]
    public async Task Download_German_UsesTheGermanRepository()
    {
        var server = new UrlRecorder();
        var dir = Directory.CreateTempSubdirectory("stanzasharp-dl-");
        try
        {
            await Assert.ThrowsAsync<HttpRequestException>(() => ModelDownloader.DownloadAsync(dir.FullName,
                ModelDownloader.FilesFor("tokenize", "default", "options", "de"), new HttpClient(server), null, default, "de"));
            Assert.Equal($"/stanfordnlp/stanza-de/resolve/v{ModelDownloader.StanzaVersion}/models/tokenize/combined_nocharlm.pt", Assert.Single(server.Requested));
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    [Fact]
    public void DownloadCommand_DefaultsToTheLanguagesFolder()
    {
        Assert.Equal(Path.Combine("models", "stanza", "en"), DownloadCommand.DefaultDir);
        Assert.Equal(Path.Combine("models", "stanza", "de"), DownloadCommand.DirFor("de"));
    }

    /// <summary>Every German golden CoNLL-U file parses and writes back byte for byte.</summary>
    [Fact]
    public void GermanGolden_ReadsAndRoundTrips()
    {
        var files = Directory.GetFiles(GermanGolden, "*.conllu", SearchOption.AllDirectories);
        Assert.True(files.Length > 50, $"{files.Length} files");
        foreach (var file in files)
        {
            var text = File.ReadAllText(file);
            var doc = Conllu.Read(text);
            Assert.NotEmpty(doc.Sentences);
            Assert.True(text == Conllu.Write(doc), Path.GetRelativePath(GermanGolden, file));
        }

        // The texts the golden data was made from, and its parts.
        string[] parts =
        [
            "corpus.txt", "pipeline.conllu", "intermediates.safetensors", "mwt.json", "lemma/words.json", "sentiment/reviews.txt",
            "bulk/default.json", "bulk/default_fast.json", "pretokenized/inputs.json", "long_token.txt", "tokenize_stress.txt",
        ];
        foreach (var name in parts)
        {
            Assert.True(File.Exists(Path.Combine(GermanGolden, name)), name);
        }
    }

    /// <summary>Answers every request with 404 and records its path.</summary>
    private sealed class UrlRecorder : HttpMessageHandler
    {
        public List<string> Requested { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requested.Add(request.RequestUri!.AbsolutePath);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
