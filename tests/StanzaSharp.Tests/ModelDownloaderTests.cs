using System.Net;
using System.Security.Cryptography;

namespace StanzaSharp.Tests;

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
        Assert.All(server.Requested, p => Assert.StartsWith($"/stanfordnlp/stanza-en/resolve/v{ModelDownloader.StanzaVersion}/models/", p));

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
        Assert.Equal(expected.Order(), ModelDownloader.Files.Select(f => f.Path).Order());
        // Every package's files, together, are the table.
        Assert.Equal(expected.Order(), Pipeline.Packages.Keys.SelectMany(p => ModelDownloader.FilesFor(null, p)).Select(f => f.Path).Distinct().Order());
    }

    [Theory]
    // No constituency, nocharlm pos/depparse/ner; sentiment still needs the charlms.
    [InlineData(null, "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_nocharlm.pt lemma/combined_nocharlm.pt depparse/combined_nocharlm.pt ner/ontonotes-ww-multi_nocharlm.pt sentiment/sstplus_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    // Without sentiment, no charlm at all.
    [InlineData("depparse,ner", "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_nocharlm.pt lemma/combined_nocharlm.pt depparse/combined_nocharlm.pt ner/ontonotes-ww-multi_nocharlm.pt pretrain/conll17.pt")]
    [InlineData("tokenize,mwt", "tokenize/combined_nocharlm.pt mwt/combined.pt")]
    public void FilesFor_SelectsTheFastPackagesFiles(string? processors, string expected) =>
        Assert.Equal(expected.Split(' ').Order(), ModelDownloader.FilesFor(processors, "default_fast").Select(f => f.Path).Order());

    [Theory]
    [InlineData("tokenize", "tokenize/combined_nocharlm.pt")]
    [InlineData("tokenize,mwt", "tokenize/combined_nocharlm.pt mwt/combined.pt")]
    // pos pulls in what it requires (tokenize, mwt) and the shared pretrain and charlms.
    [InlineData("pos", "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("tokenize,lemma", "tokenize/combined_nocharlm.pt mwt/combined.pt pos/combined_charlm.pt lemma/combined_nocharlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData("tokenize,sentiment", "tokenize/combined_nocharlm.pt sentiment/sstplus_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    [InlineData(" NER , tokenize ", "tokenize/combined_nocharlm.pt ner/ontonotes-ww-multi_charlm.pt pretrain/conll17.pt forward_charlm/1billion.pt backward_charlm/1billion.pt")]
    public void FilesFor_SelectsWhatTheProcessorsNeed(string processors, string expected) =>
        Assert.Equal(expected.Split(' ').Order(), ModelDownloader.FilesFor(processors).Select(f => f.Path).Order());

    [Fact]
    public void FilesFor_AllProcessorsIsTheDefaultPackageAndUnknownNamesThrow()
    {
        var all = ModelDownloader.FilesFor(Pipeline.AllProcessors).Order().ToList();
        Assert.Equal(11, all.Count);
        Assert.Equal(all, ModelDownloader.FilesFor(null).Order());
        Assert.DoesNotContain(all, f => f.Path.Contains("_nocharlm") && !f.Path.StartsWith("tokenize/") && !f.Path.StartsWith("lemma/"));
        Assert.Throws<ArgumentException>(() => ModelDownloader.FilesFor("tokenize,coref"));
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
