using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using Xunit.Abstractions;
using static TorchSharp.torch;

namespace StanzaSharp.Tests;

/// <summary>
/// The managed backend's kernels (issue #29, docs/managed-backend-spike.md) against TorchSharp and the golden data, on
/// every <see cref="KernelPath"/> this machine runs natively. <see cref="Gemm.Path"/> and <see cref="ManagedThreads.Count"/>
/// are process-wide, and only this class sets them; xUnit runs a class's tests one at a time.
/// </summary>
public class ManagedBackendTests(ITestOutputHelper output)
{
    /// <summary>The paths to test: Vector256 only where it is hardware (on Arm64 it would be emulated, never chosen).</summary>
    private static readonly KernelPath[] NativePaths =
        [.. Enum.GetValues<KernelPath>().Where(p => p != KernelPath.Vector256 || Vector256.IsHardwareAccelerated)];

    // Theory data are path names: KernelPath is internal, and a public test method can't take it.
    public static TheoryData<string> Paths() => [.. NativePaths.Select(p => p.ToString())];

    public static TheoryData<string, int, int, int, int> GemmCases()
    {
        var data = new TheoryData<string, int, int, int, int>();
        foreach (var path in NativePaths.Select(p => p.ToString()))
        {
            data.Add(path, 1, 7, 16, 1);
            data.Add(path, 37, 300, 125, 4); // ragged M, K over one K block, N not a whole panel
            data.Add(path, 200, 1024, 256, 8);
        }
        return data;
    }

    private static T With<T>(string path, int threads, Func<T> body) => With(Enum.Parse<KernelPath>(path), threads, body);

    /// <summary>Runs <paramref name="body"/> with the given path and thread count, then restores both.</summary>
    private static T With<T>(KernelPath path, int threads, Func<T> body)
    {
        var (beforePath, beforeThreads) = (Gemm.Path, ManagedThreads.Count);
        (Gemm.Path, ManagedThreads.Count) = (path, threads);
        try
        {
            return body();
        }
        finally
        {
            (Gemm.Path, ManagedThreads.Count) = (beforePath, beforeThreads);
        }
    }

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
    }

    private static float[] Unpad(float[] padded, int rows, int width, int paddedWidth) =>
        Enumerable.Range(0, rows).SelectMany(r => padded.AsSpan(r * paddedWidth, width).ToArray()).ToArray();

    [Theory]
    [MemberData(nameof(GemmCases))]
    public void Gemm_MatchesLinear(string path, int m, int k, int n, int threads)
    {
        var a = Random(m * k, 1);
        var w = Random(n * k, 2);
        var bias = Random(n, 3);
        var packed = new PackedMatrix(w, n, k);
        var paddedBias = new float[packed.PaddedN];
        bias.CopyTo(paddedBias, 0);
        var actual = Unpad(With(path, threads, () => Gemm.Run(a, m, packed, paddedBias)), m, n, packed.PaddedN);
        using var expected = nn.functional.linear(torch.tensor(a, [m, k]), torch.tensor(w, [n, k]), torch.tensor(bias));
        TokenizerTests.AssertClose(expected.data<float>().ToArray(), actual, 1e-4f, "gemm");
    }

    /// <summary>A random bidirectional LSTM with its packed input, both as nn.LSTM and as <see cref="PackedLstm"/>.</summary>
    private sealed class RandomLstm
    {
        public readonly PackedLstm Managed;
        public readonly PackedMatrix W;
        public readonly float[] Bias, Input;
        public readonly long[] Lengths;
        public readonly int Rows;
        public readonly float[] Expected;

        public RandomLstm(int input, int hidden, long[] lengths, int seed)
        {
            using var _ = torch.no_grad();
            using var scope = NewDisposeScope();
            Lengths = lengths;
            torch.manual_seed(seed);
            var lstm = nn.LSTM(input, hidden, batchFirst: true, bidirectional: true);
            float[] P(string name) => lstm.get_parameter(name)!.data<float>().ToArray();
            var h0 = Random(2 * hidden, seed + 1);
            var c0 = Random(2 * hidden, seed + 2);
            var x = torch.randn(lengths.Length, lengths.Max(), input);
            var packed = nn.utils.rnn.pack_padded_sequence(x, torch.tensor(lengths), batch_first: true, enforce_sorted: false);
            var state = (torch.tensor(h0, [2, 1, hidden]).expand(2, lengths.Length, hidden).contiguous(),
                torch.tensor(c0, [2, 1, hidden]).expand(2, lengths.Length, hidden).contiguous());
            var (expected, _, _) = lstm.call(packed, state);
            Expected = expected.data.data<float>().ToArray();
            (W, Bias) = PackedLstm.PackInput(hidden, input, [P("weight_ih_l0"), P("weight_ih_l0_reverse")],
                [P("bias_ih_l0"), P("bias_ih_l0_reverse")], [P("bias_hh_l0"), P("bias_hh_l0_reverse")]);
            Managed = new PackedLstm(hidden, [P("weight_hh_l0"), P("weight_hh_l0_reverse")],
                [h0[..hidden], h0[hidden..]], [c0[..hidden], c0[hidden..]]);
            Input = packed.data.data<float>().ToArray();
            Rows = Input.Length / input;
        }

        public float[] Run(CancellationToken ct = default) =>
            Managed.Recur(Gemm.Run(Input, Rows, W, Bias), W.PaddedN, PackedLstm.BatchSizes(Lengths), ct);
    }

    [Theory]
    [MemberData(nameof(Paths))]
    public void PackedBiLstm_MatchesTorchLstm(string path)
    {
        // Unsorted lengths with ties: packing must be pack_padded_sequence's.
        var lstm = new RandomLstm(10, 12, [5, 3, 5, 1, 2, 4], 7);
        TokenizerTests.AssertClose(lstm.Expected, With(path, 4, () => lstm.Run()), 1e-5f, "bilstm");
    }

    [Fact]
    public void ConcurrentCalls_EqualSequentialOutput()
    {
        // Batches of 1 to 8 sequences hit the 1-, 3- and 6-row kernels; 64 units are 16 panels per direction.
        var lstms = Enumerable.Range(0, 8).Select(i => new RandomLstm(48, 64,
            Enumerable.Range(0, i + 1).Select(j => (long)(5 + (i * 7 + j * 13) % 40)).ToArray(), 100 + i)).ToArray();
        foreach (var path in NativePaths)
            With(path, 4, () =>
            {
                var sequential = lstms.Select(l => l.Run()).ToArray();
                for (int i = 0; i < lstms.Length; i++)
                    TokenizerTests.AssertClose(lstms[i].Expected, sequential[i], 1e-5f, $"{path} lstm {i}");
                // 8 callers start together, each running every LSTM several times from its own offset.
                const int callers = 8, rounds = 5;
                var mismatches = 0;
                using var start = new Barrier(callers);
                var threads = Enumerable.Range(0, callers).Select(c => new Thread(() =>
                {
                    start.SignalAndWait();
                    for (int r = 0; r < rounds * lstms.Length; r++)
                    {
                        int i = (c + r) % lstms.Length;
                        if (!lstms[i].Run().AsSpan().SequenceEqual(sequential[i]))
                            Interlocked.Increment(ref mismatches);
                    }
                })).ToList();
                threads.ForEach(t => t.Start());
                threads.ForEach(t => t.Join());
                Assert.True(mismatches == 0, $"{path}: {mismatches} concurrent results differ from the sequential ones");
                return 0;
            });
    }

    [Fact]
    public void Cancellation_StopsBetweenSteps_AndThePoolStaysUsable()
    {
        var lstm = new RandomLstm(32, 256, [3000, 2500, 2000], 11);
        With(Gemm.Detect(), 4, () =>
        {
            using (var canceled = new CancellationTokenSource())
            {
                canceled.Cancel();
                Assert.ThrowsAny<OperationCanceledException>(() => lstm.Run(canceled.Token));
            }
            var watch = Stopwatch.StartNew();
            using (var later = new CancellationTokenSource(TimeSpan.FromMilliseconds(30)))
                Assert.ThrowsAny<OperationCanceledException>(() => lstm.Run(later.Token));
            output.WriteLine($"canceled after {watch.ElapsedMilliseconds} ms");
            TokenizerTests.AssertClose(lstm.Expected, lstm.Run(), 1e-4f, "after cancel");
            return 0;
        });
    }

    [Fact]
    public void ParallelFor_RethrowsAWorkersException()
    {
        var error = With(Gemm.Path, 4, () => Assert.Throws<InvalidOperationException>(() =>
            ManagedThreads.For(16, i => { if (i == 9) throw new InvalidOperationException("item 9"); })));
        Assert.Equal("item 9", error.Message);
    }

    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void CharLanguageModels_MatchGoldenRepresentations(string path)
    {
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e => (IReadOnlyList<string>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList()).ToList();
        foreach (var direction in new[] { "forward", "backward" })
        {
            var charlm = ManagedCharLanguageModel.Load(Repo.Model($"{direction}_charlm/1billion"));
            var reps = With(path, Environment.ProcessorCount, () => charlm.BuildCharRepresentation(sentences));
            for (int i = 0; i < sentences.Count; i++)
                TokenizerTests.AssertClose(golden.Read<float>($"s{i}.charlm_{direction}"), reps[i], 1e-4f, $"{path} {direction} s{i}");
        }
    }

    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void HighwayLstm_MatchesTorchSharp(string path)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var ckpt = Checkpoint.Load(Repo.Model("pos/combined_charlm"));
        var model = ckpt.Root["model"]!;
        using var reference = new HighwayLstm(ckpt, model, "taggerlstm", 2248, 200, 2);
        var managed = new ManagedHighwayLstm(ckpt, model, "taggerlstm", 2248, 200, 2);

        long[] lengths = [7, 30, 1, 12, 12];
        var input = Random((int)lengths.Sum() * 2248, 9);
        var expected = reference.Forward(Rnn.Pack(torch.tensor(input, [lengths.Sum(), 2248]), lengths)).data.data<float>().ToArray();
        TokenizerTests.AssertClose(expected, With(path, Environment.ProcessorCount, () => managed.Forward(input, lengths)), 1e-4f, $"{path} highway");
    }

    /// <summary>
    /// Times managed vs TorchSharp for both charlms and the tagger's highway biLSTM on a modest input, on every path
    /// this machine runs natively, so CI logs show each OS and architecture (Arm64 included). Never fails on speed.
    /// </summary>
    [ModelFact]
    public void Speed_IsReported()
    {
        using var _ = torch.no_grad();
        int threads = torch.get_num_threads();
        // 16 pseudo-sentences of 20 words from corpus.txt (no tokenizer needed).
        var words = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var sentences = Enumerable.Range(0, 16)
            .Select(i => (IReadOnlyList<string>)Enumerable.Range(i * 20, 20).Select(j => words[j % words.Length]).ToList()).ToList();
        long[] lengths = [.. sentences.Select(s => (long)s.Count)];
        var input = Random((int)lengths.Sum() * 2248, 5);

        var lms = new[] { "forward", "backward" }.Select(d => Repo.Model($"{d}_charlm/1billion"))
            .Select(p => (Torch: CharLanguageModel.Load(p), Managed: ManagedCharLanguageModel.Load(p))).ToArray();
        var ckpt = Checkpoint.Load(Repo.Model("pos/combined_charlm"));
        using var highway = new HighwayLstm(ckpt, ckpt.Root["model"]!, "taggerlstm", 2248, 200, 2);
        var managedHighway = new ManagedHighwayLstm(ckpt, ckpt.Root["model"]!, "taggerlstm", 2248, 200, 2);

        static double Median(Action run)
        {
            run(); // warm-up
            var times = new List<double>();
            for (int i = 0; i < 3; i++)
            {
                var watch = Stopwatch.StartNew();
                run();
                times.Add(watch.Elapsed.TotalMilliseconds);
            }
            return times.Order().ElementAt(1);
        }
        void Report(string line)
        {
            output.WriteLine(line);
            Console.WriteLine(line);
        }

        double torchCharlm = Median(() =>
        {
            foreach (var (lm, _) in lms)
                foreach (var t in lm.BuildCharRepresentation(sentences))
                    t.Dispose();
        });
        double torchHighway = Median(() =>
        {
            using var scope = NewDisposeScope();
            highway.Forward(Rnn.Pack(torch.tensor(input, [lengths.Sum(), 2248]), lengths), disposeInput: true);
        });
        Report($"managed-backend speed: {RuntimeInformation.OSDescription}, {RuntimeInformation.ProcessArchitecture}, " +
               $"{Environment.ProcessorCount} logical CPUs, torch threads {threads}, detected path {Gemm.Detect()}, " +
               $"{sentences.Count} sentences / {lengths.Sum()} words");
        Report($"  TorchSharp: charlm {torchCharlm,8:F1} ms, highway {torchHighway,8:F1} ms");
        foreach (var path in NativePaths)
        {
            var (charlm, hw) = With(path, threads, () => (
                Median(() =>
                {
                    foreach (var (_, lm) in lms)
                        lm.BuildCharRepresentation(sentences);
                }),
                Median(() => managedHighway.Forward(input, lengths))));
            Report($"  {path,-10}: charlm {charlm,8:F1} ms ({charlm / torchCharlm:F2}x), highway {hw,8:F1} ms ({hw / torchHighway:F2}x)");
        }
    }
}
