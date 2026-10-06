using System.Diagnostics;
using TorchSharp;

namespace StanzaSharp.Tests;

/// <summary>Concurrent <see cref="Pipeline.Process(string)"/> calls on one pipeline, and cancellation.</summary>
public class ConcurrencyTests(Xunit.Abstractions.ITestOutputHelper output)
{
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
    [InlineData("default")]
    [InlineData("default_fast")]
    public void ConcurrentCalls_EqualSequentialOutput(string package)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Package = package });
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
    /// every processor's batch loop are reached. With <paramref name="countTensors"/>, TorchSharp's live tensor count on
    /// this thread must be unchanged afterwards. The lemmatizer is left out of that count: TorchSharp counts the two
    /// undefined index tensors of every <c>pack_padded_sequence(enforce_sorted: true)</c> as live forever (they hold no
    /// memory), which the lemmatizer's encoder uses like Stanza's.
    /// </summary>
    [ModelTheory]
    [InlineData(null, false)]
    [InlineData("tokenize,mwt,pos,constituency,sentiment,ner", true)]
    public void Canceled_InsideEachProcessor_ThrowsPromptly_AndThePipelineStillWorks(string? processors, bool countTensors)
    {
        var timings = new TimingLogger();
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = processors, Logger = timings });
        var corpus = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"));
        var before = Conllu.Write(nlp.Process(corpus));
        if (processors == null)
            Assert.Equal(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")), before);

        // Already canceled: nothing runs.
        Assert.Throws<OperationCanceledException>(() => nlp.Process(corpus, new CancellationToken(true)));
        Assert.Throws<OperationCanceledException>(() => nlp.Process([corpus], new CancellationToken(true)));
        Assert.Throws<OperationCanceledException>(() => nlp.Process([["A", "."]], new CancellationToken(true)));

        var big = string.Join("\n\n", Enumerable.Repeat(string.Join("\n\n", Texts), 2));
        timings.Steps.Clear();
        nlp.Process(big);
        var steps = timings.Steps.ToList();

        var stats = DisposeScopeManager.Statistics; // per thread, and Process runs on this one
        long live = stats.ThreadTotalLiveCount;
        for (int i = 0; i < steps.Count; i++)
        {
            var (processor, ms) = steps[i];
            using var cts = new CancellationTokenSource();
            var latency = new Stopwatch();
            cts.Token.Register(latency.Start);
            // Halfway through the processor: timed from the end of the one before it (its log message).
            var delay = TimeSpan.FromMilliseconds(ms / 2);
            if (i == 0)
                cts.CancelAfter(delay);
            else
                timings.OnStep = p => { if (p == steps[i - 1].Processor) cts.CancelAfter(delay); };
            var e = Assert.ThrowsAny<OperationCanceledException>(() => nlp.Process(big, cts.Token));
            timings.OnStep = null;
            output.WriteLine($"{processor} ({ms:F0} ms): canceled after {latency.Elapsed.TotalMilliseconds:F0} ms, " +
                $"in {e.StackTrace!.Split('\n').Select(l => l.Trim()).FirstOrDefault(l => l.StartsWith("at StanzaSharp.") && !l.Contains("Pipeline.Step"))}");
            Assert.True(latency.Elapsed < TimeSpan.FromSeconds(5), $"canceled in {processor}, the call returned {latency.Elapsed.TotalMilliseconds:F0} ms later");
            if (countTensors)
                Assert.Equal(live, stats.ThreadTotalLiveCount);
        }
        Assert.Equal(before, Conllu.Write(nlp.Process(corpus, CancellationToken.None)));
        if (countTensors)
            Assert.Equal(live, stats.ThreadTotalLiveCount);
    }

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
