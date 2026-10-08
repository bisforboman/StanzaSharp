using System.Diagnostics;
using TorchSharp;

namespace StanzaSharp.Tests;

/// <summary>Concurrent <see cref="Pipeline.Process(string)"/> calls on one pipeline, and cancellation.</summary>
/// <remarks>
/// Runs alone (after the parallel collections): the cancellation test times each processor once and then cancels at
/// half that time, so heavy tests running beside it (the both-backend golden theories) made the timed run several
/// times slower than the canceled ones, and every attempt finished before its cancel ("lemma: every call finished
/// before it was canceled", in 3 of 3 full runs).
/// </remarks>
[Collection(Name)]
public class ConcurrencyTests(Xunit.Abstractions.ITestOutputHelper output)
{
    public const string Name = "Concurrency and cancellation";

    private static readonly string[] Texts =
        [.. new[] { "corpus.txt" }.Concat(Directory.GetFiles(Repo.Golden, "validation*.txt").Select(Path.GetFileName).Order())
            .Select(f => File.ReadAllText(Path.Combine(Repo.Golden, f!)))];

    /// <summary>One call of each kind: plain text, pretokenized text and bulk.</summary>
    private static string Run(Pipeline nlp, int i, CancellationToken ct = default) => (i % 4) switch
    {
        // Pretokenized: the text's own tokens, as the tokenizer splits them.
        3 => Conllu.Write(nlp.Process(nlp.Process(Texts[i % Texts.Length], ct).Sentences.Select(s => s.Tokens.Select(t => t.Text)), ct)),
        2 => string.Join("\n", nlp.Process([Texts[i % Texts.Length], Texts[(i + 1) % Texts.Length]], ct).Select(Conllu.Write)),
        _ => Conllu.Write(nlp.Process(Texts[i % Texts.Length], ct)),
    };

    [ModelTheory]
    [InlineData("default", false)]
    [InlineData("default_fast", false)]
    [InlineData("default", true)]
    [InlineData("default_fast", true)]
    public void ConcurrentCalls_EqualSequentialOutput(string package, bool managed)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Package = package, Backend = Repo.Backend(managed) });
        int jobs = Texts.Length * 2;
        var expected = Enumerable.Range(0, jobs).Select(i => Run(nlp, i)).ToArray();

        // 8 threads start together; each takes every 8th job, from its own offset, so different texts and processors overlap.
        const int threads = 8;
        var actual = new string[threads][];
        using var start = new Barrier(threads);
        var workers = Enumerable.Range(0, threads).Select(t => new Thread(() =>
        {
            start.SignalAndWait();
            actual[t] = Enumerable.Range(0, jobs).Where(i => i % threads == t).Select(i => Run(nlp, i)).ToArray();
        })).ToList();
        workers.ForEach(w => w.Start());
        workers.ForEach(w => w.Join());

        for (int t = 0; t < threads; t++)
        {
            var mine = Enumerable.Range(0, jobs).Where(i => i % threads == t).ToList();
            for (int k = 0; k < mine.Count; k++)
                Assert.True(expected[mine[k]] == actual[t][k], $"job {mine[k]} on thread {t} differs from the sequential run");
        }
    }

    /// <summary>
    /// Cancels a long text in the middle of each processor in turn (timed from an uncanceled run), so the checks inside
    /// every processor's batch loop are reached. TorchSharp's live tensor count on this thread must not grow by more than
    /// an uncanceled call's: TorchSharp counts the two undefined index tensors of every
    /// <c>pack_padded_sequence(enforce_sorted: true)</c> as live forever (they hold no memory), which the lemmatizer's
    /// encoder uses like Stanza's. So a call stopped before the lemmatizer must leave the count unchanged, one stopped
    /// after it must add exactly what an uncanceled call adds, and one stopped inside it is not counted.
    /// </summary>
    [ModelTheory]
    [InlineData(null, false)]
    [InlineData("tokenize,mwt,pos,constituency,sentiment,ner", false)]
    [InlineData(null, true)]
    public void Canceled_InsideEachProcessor_ThrowsPromptly_AndThePipelineStillWorks(string? processors, bool managed)
    {
        var timings = new TimingLogger();
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = processors, Logger = timings, Backend = Repo.Backend(managed) });
        var corpus = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"));
        var stats = DisposeScopeManager.Statistics; // per thread, and Process runs on this one
        long live = stats.ThreadTotalLiveCount;
        var before = Conllu.Write(nlp.Process(corpus));
        long corpusLeak = stats.ThreadTotalLiveCount - live; // 0 without the lemmatizer, as is lemmaLeak below
        if (processors == null)
            Assert.Equal(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")), before);

        // Already canceled: nothing runs.
        Assert.Throws<OperationCanceledException>(() => nlp.Process(corpus, new CancellationToken(true)));
        Assert.Throws<OperationCanceledException>(() => nlp.Process([corpus], new CancellationToken(true)));
        Assert.Throws<OperationCanceledException>(() => nlp.Process([["A", "."]], new CancellationToken(true)));

        var big = string.Join("\n\n", Enumerable.Repeat(string.Join("\n\n", Texts), 2));
        live = stats.ThreadTotalLiveCount;
        timings.Steps.Clear();
        nlp.Process(big);
        var steps = timings.Steps.ToList();
        long lemmaLeak = stats.ThreadTotalLiveCount - live;
        live = stats.ThreadTotalLiveCount;
        for (int i = 0; i < steps.Count; i++)
        {
            var (processor, ms) = steps[i];
            OperationCanceledException? e = null;
            Stopwatch latency = new();
            // Halfway through the processor: timed from the end of the one before it (its log message). A faster run
            // can finish before the cancel arrives (CI runners vary a lot); then try again, cancelling sooner.
            for (int attempt = 0; e == null; attempt++)
            {
                Assert.True(attempt < 4, $"{processor}: every call finished before it was canceled");
                using var cts = new CancellationTokenSource();
                latency = new Stopwatch();
                cts.Token.Register(latency.Start);
                var delay = TimeSpan.FromMilliseconds(ms / (2 << attempt));
                if (i == 0)
                    cts.CancelAfter(delay);
                else
                    timings.OnStep = p => { if (p == steps[i - 1].Processor) cts.CancelAfter(delay); };
                timings.Steps.Clear();
                try
                {
                    nlp.Process(big, cts.Token);
                    live = stats.ThreadTotalLiveCount; // a finished call leaves the lemmatizer's tensors, like any call
                }
                catch (OperationCanceledException canceled)
                {
                    e = canceled;
                }
                timings.OnStep = null;
            }
            output.WriteLine($"{processor} ({ms:F0} ms): canceled after {latency.Elapsed.TotalMilliseconds:F0} ms, " +
                $"in {e.StackTrace!.Split('\n').Select(l => l.Trim()).FirstOrDefault(l => l.StartsWith("at StanzaSharp.") && !l.Contains("Pipeline.Step"))}");
            Assert.True(latency.Elapsed < MaxLatency, $"canceled in {processor}, the call returned {latency.Elapsed.TotalMilliseconds:F0} ms later");
            // Where the call stopped decides the count (a loaded machine can move it to the next processor).
            if (timings.Steps.Any(s => s.Processor == "lemma"))
                Assert.Equal(lemmaLeak, stats.ThreadTotalLiveCount - live);
            else if (!e.StackTrace!.Contains("StanzaSharp.Lemma."))
                Assert.Equal(0, stats.ThreadTotalLiveCount - live);
            live = stats.ThreadTotalLiveCount;
        }
        Assert.Equal(before, Conllu.Write(nlp.Process(corpus, CancellationToken.None)));
        Assert.Equal(corpusLeak, stats.ThreadTotalLiveCount - live);
    }

    /// <summary>
    /// Every batched processor that takes seconds per batch (pos, depparse, sentiment) also checks inside its batches, so the
    /// longest gap between two checks is one charlm pass, LSTM layer or scorer over a 5000-word batch. On an 8-core desktop
    /// the worst latency measured (3 runs of both cases) was 0.56 s, 0.82 s with the machine 2x loaded; before, a whole
    /// POS or sentiment batch took up to 0.8 and 1.2 s. CI runners are 3-4x slower than the desktop (macOS and Windows
    /// Arm64 the slowest), so about 3 s there under load; 5 s leaves room for a noisy runner.
    /// </summary>
    private static readonly TimeSpan MaxLatency = TimeSpan.FromSeconds(5);

    /// <summary>Collects the pipeline's "{processor} took {ms} ms" debug messages.</summary>
    private sealed class TimingLogger : Microsoft.Extensions.Logging.ILogger
    {
        public List<(string Processor, double Milliseconds)> Steps { get; } = [];
        public Action<string>? OnStep { get; set; }
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
        public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, Microsoft.Extensions.Logging.EventId eventId, TState state,
            Exception? exception, Func<TState, Exception?, string> formatter)
        {
            if (state is IReadOnlyList<KeyValuePair<string, object?>> values && values.Any(v => v.Key == "Processor"))
            {
                var processor = (string)values.First(v => v.Key == "Processor").Value!;
                Steps.Add((processor, (double)values.First(v => v.Key == "Milliseconds").Value!));
                OnStep?.Invoke(processor);
            }
        }
    }
}

/// <summary><see cref="ConcurrencyTests"/> runs alone; see its remarks.</summary>
[CollectionDefinition(ConcurrencyTests.Name, DisableParallelization = true)]
public sealed class ConcurrencyCollection;
