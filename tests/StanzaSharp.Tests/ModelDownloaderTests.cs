using System.Net;
using System.Security.Cryptography;

namespace StanzaSharp.Tests;

[Trait("Backend", "Managed")]
public class ModelDownloaderTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "stanzasharp-dl-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    /// <summary>Serves fixed bytes per URL path and records which paths were requested.</summary>
    private sealed class FakeServer(Dictionary<string, byte[]> files) : HttpMessageHandler
    {
        public List<string> Requested { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri!.AbsolutePath;
            Requested.Add(path);
            var match = files.FirstOrDefault(f => path.EndsWith("/models/" + f.Key, StringComparison.Ordinal));
            return Task.FromResult(match.Value is { } bytes
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) }
                : new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }

    private static string Md5(byte[] data) => Convert.ToHexStringLower(MD5.HashData(data));

    [Fact]
    public async Task DownloadsVerifiesAndSkipsUpToDateFiles()
    {
        byte[] a = [1, 2, 3], b = [4, 5];
        var server = new FakeServer(new() { ["tokenize/a.pt"] = a, ["pos/b.pt"] = b });
        (string, string)[] files = [("tokenize/a.pt", Md5(a)), ("pos/b.pt", Md5(b))];

        await ModelDownloader.DownloadAsync(_dir, files, new HttpClient(server), null, default);
        Assert.Equal(a, File.ReadAllBytes(Path.Combine(_dir, "tokenize/a.pt")));
        Assert.Equal(b, File.ReadAllBytes(Path.Combine(_dir, "pos/b.pt")));
        Assert.Equal(2, server.Requested.Count);
        Assert.All(server.Requested, p => Assert.StartsWith($"/stanfordnlp/stanza-en/resolve/{ModelDownloader.Revisions["en"]}/models/", p));

        // A second run finds both files with the right checksum and downloads nothing.
        var progress = new List<string>();
        await ModelDownloader.DownloadAsync(_dir, files, new HttpClient(server), new SyncProgress(progress), default);
        Assert.Equal(2, server.Requested.Count);
        Assert.All(progress, line => Assert.EndsWith("up to date", line));
    }

    [Fact]
    public async Task RejectsAChecksumMismatchAndLeavesNoFile()
    {
        var server = new FakeServer(new() { ["mwt/c.pt"] = [9, 9, 9] });
        var ex = await Assert.ThrowsAsync<InvalidDataException>(() =>
            ModelDownloader.DownloadAsync(_dir, [("mwt/c.pt", Md5([1]))], new HttpClient(server), null, default));
        Assert.Contains("mwt/c.pt", ex.Message);
        Assert.Empty(Directory.GetFiles(_dir, "*", SearchOption.AllDirectories));
    }

    [Fact]
    public async Task ReplacesACorruptedFile()
    {
        byte[] good = [7, 7];
        var target = Path.Combine(_dir, "pretrain/d.pt");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        File.WriteAllBytes(target, [0]);
        var server = new FakeServer(new() { ["pretrain/d.pt"] = good });

        await ModelDownloader.DownloadAsync(_dir, [("pretrain/d.pt", Md5(good))], new HttpClient(server), null, default);
        Assert.Equal(good, File.ReadAllBytes(target));
        Assert.Single(Directory.GetFiles(_dir, "*", SearchOption.AllDirectories));
    }

    /// <summary>Serves fixed bytes for exact URL paths.</summary>
    private sealed class PathServer(Dictionary<string, byte[]> files) : HttpMessageHandler
    {
        public List<string> Requested { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var path = request.RequestUri!.AbsolutePath;
            Requested.Add(path);
            return Task.FromResult(files.TryGetValue(path, out var bytes)
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) }
                : new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }

    [Fact]
    public async Task DownloadsATransformerAtItsRevisionAndChecksItsSha256()
    {
        byte[] good = [1, 2, 3];
        var server = new PathServer(new()
        {
            ["/org/model/resolve/abc123/tokenizer.json"] = good,
            ["/org/model/resolve/abc123/weights.bin"] = [9],
        });
        var sha = Convert.ToHexStringLower(SHA256.HashData(good));
        await ModelDownloader.DownloadTransformerAsync(_dir, "org/model", "abc123", [("tokenizer.json", sha)], new HttpClient(server), null, default);
        var directory = ModelDownloader.TransformerDirectory(_dir, "org/model");
        Assert.Equal(Path.Combine(_dir, "transformer", "org", "model"), directory);
        Assert.Equal(good, File.ReadAllBytes(Path.Combine(directory, "tokenizer.json")));

        // Up to date: nothing is requested again. A wrong checksum leaves no file.
        await ModelDownloader.DownloadTransformerAsync(_dir, "org/model", "abc123", [("tokenizer.json", sha)], new HttpClient(server), null, default);
        Assert.Single(server.Requested);
        var ex = await Assert.ThrowsAsync<InvalidDataException>(() =>
            ModelDownloader.DownloadTransformerAsync(_dir, "org/model", "abc123", [("weights.bin", sha)], new HttpClient(server), null, default));
        Assert.Contains("org/model/weights.bin", ex.Message);
        Assert.False(File.Exists(Path.Combine(directory, "weights.bin")));
    }

    [Fact]
    public async Task UnknownTransformersAndFilesThrow()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => ModelDownloader.DownloadTransformerAsync(_dir, "org/nothing"));
        await Assert.ThrowsAsync<ArgumentException>(() => ModelDownloader.DownloadTransformerAsync(_dir, "google/electra-large-discriminator", ["vocab.txt"]));
        Assert.All(ModelDownloader.Transformers.Values, t =>
        {
            Assert.Matches("^[0-9a-f]{40}$", t.Revision);
            Assert.All(t.Files, f => Assert.Matches("^[0-9a-f]{64}$", f.Sha256));
        });
    }

    [Fact]
    public void FileTableCoversEveryModelThePipelineLoads()
    {
        // Pipeline.Load reads <processor>/<name>; the downloader must fetch exactly those files.
        string[] expected =
        [
            "tokenize/combined_nocharlm.pt", "mwt/combined.pt", "pos/combined_charlm.pt", "lemma/combined_nocharlm.pt", "constituency/ptb3-revised_charlm.pt",
            "pretrain/conll17.pt", "forward_charlm/1billion.pt", "backward_charlm/1billion.pt",
            "depparse/combined_charlm.pt", "ner/ontonotes-ww-multi_charlm.pt", "sentiment/sstplus_charlm.pt",
            "pos/combined_nocharlm.pt", "depparse/combined_nocharlm.pt", "ner/ontonotes-ww-multi_nocharlm.pt",
        ];
        Assert.Equal(expected.Order(), new[] { "default", "default_fast" }.SelectMany(p => ModelDownloader.FilesFor(null, p)).Select(f => f.Path).Distinct().Order());

        // Every package's files and Pipeline.Alternatives', together, are the table (the treebank packages' too).
        Assert.Equal(ModelDownloader.Files["en"].Select(f => f.Path).Order(),
            Pipeline.Packages["en"].Keys.SelectMany(p => ModelDownloader.FilesFor(null, p)).Select(f => f.Path).Concat(AlternativesFiles("en")).Distinct().Order());
    }

    /// <summary>The files of each of <see cref="Pipeline.Alternatives"/>' models of <paramref name="lang"/>: its own and its shared ones.</summary>
    internal static IEnumerable<string> AlternativesFiles(string lang) =>
        Pipeline.Alternatives[lang].SelectMany(kv => kv.Value.SelectMany(name => Pipeline.ModelFiles(new Dictionary<string, string> { [kv.Key] = name }, lang)))
            .Select(p => p + ".pt");

    [Theory]

    // Without mwt (pos requires it only where the package has it) and with the identity lemmatizer, which has no file.
    [InlineData("eslspok", null, "tokenize/eslspok_nocharlm.pt pos/eslspok_charlm.pt depparse/eslspok_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("eslspok", "tokenize,lemma", "tokenize/eslspok_nocharlm.pt")]
    [InlineData("ewt", "lemma", "tokenize/ewt_nocharlm.pt mwt/ewt.pt pos/ewt_charlm.pt lemma/ewt_nocharlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("craft", null, "tokenize/craft_nocharlm.pt pos/craft_nocharlm.pt lemma/craft_nocharlm.pt depparse/craft_nocharlm.pt pretrain/biomed.pt")]
    [InlineData("mimic", "pos", "tokenize/mimic_nocharlm.pt pos/mimic_charlm.pt pretrain/mimic.pt forward_charlm/mimic.pt backward_charlm/mimic.pt")]
    [InlineData("combined", null, "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt lemma/combined_nocharlm.pt depparse/combined_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    public void FilesFor_SelectsTheTreebankPackagesFiles(string package, string? processors, string expected) =>
        Assert.Equal(expected.Split(' ').Order(), ModelDownloader.FilesFor(processors, package).Select(f => f.Path).Order());

    [Theory]

    // No constituency, nocharlm pos/depparse/ner; sentiment still needs the charlms.
    [InlineData(null, "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_nocharlm.pt lemma/combined_nocharlm.pt depparse/combined_nocharlm.pt ner/ontonotes-ww-multi_nocharlm.pt sentiment/sstplus_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]

    // Without sentiment, no charlm at all.
    [InlineData("depparse,ner", "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_nocharlm.pt lemma/combined_nocharlm.pt depparse/combined_nocharlm.pt ner/ontonotes-ww-multi_nocharlm.pt pretrain/conll17.pt")]
    [InlineData("tokenize,mwt", "tokenize/combined_nocharlm.pt mwt/combined.pt")]
    public void FilesFor_SelectsTheFastPackagesFiles(string? processors, string expected) =>
        Assert.Equal(expected.Split(' ').Order(), ModelDownloader.FilesFor(processors, "default_fast").Select(f => f.Path).Order());

    [Theory]
    [InlineData("tokenize", "tokenize/combined_nocharlm.pt mwt/combined.pt")] // mwt comes along, as in Stanza
    [InlineData("tokenize,mwt", "tokenize/combined_nocharlm.pt mwt/combined.pt")]

    // pos pulls in what it requires (tokenize, mwt) and the shared pretrain and charlms.
    [InlineData("pos", "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("tokenize,lemma", "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt lemma/combined_nocharlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("tokenize,sentiment", "tokenize/combined_nocharlm.pt mwt/combined.pt sentiment/sstplus_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData(" NER , tokenize ", "tokenize/combined_nocharlm.pt mwt/combined.pt ner/ontonotes-ww-multi_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    public void FilesFor_SelectsWhatTheProcessorsNeed(string processors, string expected) =>
        Assert.Equal(expected.Split(' ').Order(), ModelDownloader.FilesFor(processors).Select(f => f.Path).Order());

    [Fact]
    public void FilesFor_AllProcessorsIsTheDefaultPackageAndUnknownNamesThrow()
    {
        var all = ModelDownloader.FilesFor(Pipeline.AllProcessors).Order().ToList();
        Assert.Equal(11, all.Count);
        Assert.Equal(all, ModelDownloader.FilesFor(null).Order());
        Assert.DoesNotContain(all, f => f.Path.Contains("_nocharlm") && !f.Path.StartsWith("tokenize/") && !f.Path.StartsWith("lemma/"));
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor("tokenize,morphseg"));

        // coref is optional: never among a package's processors, its model only when named.
        Assert.DoesNotContain(all, f => f.Path.StartsWith("coref/", StringComparison.Ordinal));
        Assert.Contains(ModelDownloader.FilesFor("tokenize,coref"), f => f.Path == "coref/udcoref_xlm-roberta-lora.pt");
        Assert.Contains(ModelDownloader.FilesFor("tokenize,coref", "default_fast"), f => f.Path == "coref/udcoref_xlm-roberta-lora.pt");
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor("tokenize,coref", "ewt"));
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor(""));
    }

    [PtModelTheory]
    [InlineData("tokenize,mwt", "default")]
    [InlineData("tokenize,mwt,pos,lemma", "default")]
    [InlineData("tokenize,sentiment", "default")]
    [InlineData(null, "default_fast")]
    [InlineData("tokenize,mwt,pos,lemma,depparse,ner", "default_fast")]
    public void FilesFor_AreEnoughToLoadThePipeline(string? processors, string package)
    {
        // Only the selected .pt files, copied from the full download: Pipeline.Load must find all it reads.
        var options = new PipelineOptions { Processors = processors, Package = package };
        foreach (var (path, _) in ModelDownloader.FilesFor(options.Processors, options.Package))
        {
            var target = Path.Combine(_dir, path);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(Path.Combine(Repo.StanzaModels, path), target);
        }

        using var nlp = Pipeline.Load(_dir, options);
        Assert.NotEmpty(nlp.Process("It works.").Sentences);
    }

    /// <summary>Progress&lt;T&gt; reports on the thread pool; this one records synchronously.</summary>
    private sealed class SyncProgress(List<string> lines) : IProgress<string>
    {
        public void Report(string value) => lines.Add(value);
    }
}
