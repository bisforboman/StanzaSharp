using Microsoft.Extensions.Logging;
using TorchSharp;

namespace StanzaSharp.Tests;

/// <summary>PipelineOptions.Threads, VerifyChecksums and Logger, and the Processor constants.</summary>
[Collection(TorchSettingsCollection.Name)]
public class OptionsTests
{
    [ModelFact]
    public void Threads_SetsTorchsThreadCount_AndNullCapsItAtProcessorCount()
    {
        int old = torch.get_num_threads();
        try
        {
            using (Pipeline.Load(Repo.Models, new PipelineOptions { Processors = Processor.Tokenize, Threads = 2 }))
                Assert.Equal(2, torch.get_num_threads());
            // Null keeps a lower count set by the caller.
            using (Pipeline.Load(Repo.Models, new PipelineOptions { Processors = Processor.Tokenize }))
                Assert.Equal(2, torch.get_num_threads());
            torch.set_num_threads(Environment.ProcessorCount + 4);
            using (Pipeline.Load(Repo.Models, new PipelineOptions { Processors = Processor.Tokenize }))
                Assert.Equal(Environment.ProcessorCount, torch.get_num_threads());
            Assert.Throws<ArgumentOutOfRangeException>(() => Pipeline.Load(Repo.Models, new PipelineOptions { Threads = 0 }));
        }
        finally
        {
            torch.set_num_threads(old);
        }
    }

    [PtModelFact]
    public void VerifyChecksums_AcceptsStanzasFiles_AndNamesACorruptedOne()
    {
        var options = new PipelineOptions { Processors = "tokenize,mwt", VerifyChecksums = true };
        using (Pipeline.Load(Repo.StanzaModels, options)) { }

        // A copy of the two files with one byte of mwt changed.
        var dir = Directory.CreateTempSubdirectory("stanzasharp-md5-").FullName;
        try
        {
            foreach (var file in new[] { "tokenize/combined_nocharlm.pt", "mwt/combined.pt" })
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(dir, file))!);
                File.Copy(Path.Combine(Repo.StanzaModels, file), Path.Combine(dir, file));
            }
            var bytes = File.ReadAllBytes(Path.Combine(dir, "mwt/combined.pt"));
            bytes[^1] ^= 1;
            File.WriteAllBytes(Path.Combine(dir, "mwt/combined.pt"), bytes);
            var e = Assert.Throws<InvalidDataException>(() => Pipeline.Load(dir, options));
            Assert.Contains("combined.pt", e.Message);
            Assert.Contains("dc569dbb26cc72b71847f9d0b8775d43", e.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void VerifyChecksums_RefusesConvertedModels()
    {
        // Only the converted file's presence matters: the check runs before anything is read.
        var dir = Directory.CreateTempSubdirectory("stanzasharp-converted-").FullName;
        try
        {
            Directory.CreateDirectory(Path.Combine(dir, "tokenize"));
            File.WriteAllText(Path.Combine(dir, "tokenize", "combined_nocharlm.json"), "{}");
            var e = Assert.Throws<InvalidOperationException>(() =>
                Pipeline.Load(dir, new PipelineOptions { Processors = Processor.Tokenize, VerifyChecksums = true }));
            Assert.Contains("combined_nocharlm.json", e.Message);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [ModelFact]
    public void Logger_ReceivesLoadAndProcessTimings()
    {
        var logger = new ListLogger();
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,pos", Logger = logger });
        nlp.Process("Hello world.");
        var lines = logger.Lines;
        Assert.Contains(lines, l => l.StartsWith("Information: Loaded tokenize/combined_nocharlm in "));
        Assert.Contains(lines, l => l.StartsWith("Information: Loaded pretrain/conll17 in "));
        Assert.Contains(lines, l => l.StartsWith("Information: Using ") && l.Contains("torch intra-op threads"));
        Assert.Contains(lines, l => l.StartsWith("Information: Loaded the default pipeline (tokenize,mwt,pos) in "));
        foreach (var p in new[] { "tokenize", "mwt", "pos" })
            Assert.Contains(lines, l => l.StartsWith($"Debug: {p} took "));
    }

    [Fact]
    public void ProcessorConstants_SpellAllProcessors()
    {
        Assert.Equal(Pipeline.AllProcessors, string.Join(",",
            Processor.Tokenize, Processor.Mwt, Processor.Pos, Processor.Lemma, Processor.Constituency, Processor.Depparse, Processor.Sentiment, Processor.Ner));
    }

    private sealed class ListLogger : ILogger
    {
        public List<string> Lines { get; } = [];
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            lock (Lines)
                Lines.Add($"{logLevel}: {formatter(state, exception)}");
        }
    }
}
