using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using Xunit.Abstractions;
using StanzaSharp.Mwt;
using StanzaSharp.Tokenize;
using static TorchSharp.torch;

namespace StanzaSharp.Tests;

/// <summary>
/// The managed backend's kernels (issue #29, docs/managed-backend-spike.md) against TorchSharp and the golden data, on
/// every <see cref="KernelPath"/> this machine runs natively. <see cref="Gemm.Path"/> and <see cref="ManagedThreads.Count"/>
/// are process-wide, and only this class sets them; xUnit runs a class's tests one at a time, and the collection runs
/// alone, so tests of the managed backend elsewhere never see another path.
/// </summary>
[Collection(ManagedKernelsCollection.Name)]
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
    /// The seam's LSTM block against nn.LSTM through <see cref="Rnn.RunPacked"/>, on real checkpoints: the tokenizer's
    /// (hidden 64) and MWT's 2-layer encoder (hidden 50, padded to 52 units inside), with ragged lengths and a row at
    /// full width.
    /// </summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void ManagedLstm_MatchesRunPacked(string path)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        foreach (var (model, prefix, input, hidden, layers) in new[] { ("tokenize/combined_nocharlm", "rnn.", 41, 64, 1), ("mwt/combined", "encoder.", 50, 50, 2) })
        {
            var ckpt = Checkpoint.Load(Repo.Model(model));
            var state = ckpt.Root["model"]!;
            using var reference = nn.LSTM(input, hidden, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, state, prefix);
            var managed = new ManagedLstm(ckpt, state, prefix, input, hidden, layers, bidirectional: true);
            long[] lengths = [9, 3, 17, 1, 17, 6, 11];
            int width = 17;
            var x = Random(lengths.Length * width * input, 21);
            var expected = Rnn.RunPacked(reference, torch.tensor(x, [lengths.Length, width, input]), lengths).data<float>().ToArray();
            var actual = new float[expected.Length];
            With(path, 4, () => { managed.ForwardPadded(x, lengths.Length, width, lengths, actual); return 0; });
            TokenizerTests.AssertClose(expected, actual, 1e-5f, $"{path} {model}");
        }
    }

    /// <summary>The tokenizer's network, both backends on the same batch (one row at full width, as later windows run).</summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void TokenizerNet_ManagedMatchesTorchSharp(string path)
    {
        var ckpt = Checkpoint.Load(Repo.Model("tokenize/combined_nocharlm"));
        using var reference = new TokenizerNet(ckpt);
        var managed = new ManagedTokenizerNet(ckpt);
        long[] lengths = [40, 12, 1, 25];
        int width = 40, vocab = (int)ckpt.Shape(ckpt.Root["model"]!["embeddings.weight"])[0];
        var rng = new Random(5);
        var ids = Enumerable.Range(0, lengths.Length * width).Select(_ => (long)rng.Next(vocab)).ToArray();
        var feats = Enumerable.Range(0, lengths.Length * width * 9).Select(_ => rng.Next(4) == 0 ? 1f : 0f).ToArray();
        var expected = reference.Forward(ids, feats, lengths.Length, width, lengths, default);
        var actual = With(path, 4, () => managed.Forward(ids, feats, lengths.Length, width, lengths, default));
        output.WriteLine($"{path}: max |diff| {TokenizerTests.AssertClose(expected, actual, 1e-4f, path):E2}");
    }

    /// <summary>MWT's classifier network, both backends on the same batch of ragged rows.</summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void MwtNet_ManagedMatchesTorchSharp(string path)
    {
        var ckpt = Checkpoint.Load(Repo.Model("mwt/combined"));
        var config = ckpt.Root["config"]!;
        using var reference = new MwtNet(ckpt, config, 0);
        var managed = new ManagedMwtNet(ckpt, config);
        long[] lengths = [14, 3, 9, 14, 6];
        int width = 14, vocab = config["vocab_size"]!.GetValue<int>();
        var rng = new Random(8);
        var ids = Enumerable.Range(0, lengths.Length * width).Select(_ => (long)rng.Next(vocab)).ToArray();
        var expected = reference.Forward(ids, lengths.Length, width, lengths);
        var actual = With(path, 4, () => managed.Forward(ids, lengths.Length, width, lengths));
        output.WriteLine($"{path}: max |diff| {TokenizerTests.AssertClose(expected, actual, 1e-4f, path):E2}");
    }

    /// <summary>
    /// The character model, both variants: NER's (bidirectional, final states) and the tagger's (unidirectional,
    /// attention pooling), with words of 1 to 20 characters including unknown ones.
    /// </summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void CharacterModel_ManagedMatchesTorchSharp(string path)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var words = "a The NER-tagger's charmodel reads every character: 𝔘nicode, ü, and supercalifragilistic !".Split(' ');
        foreach (var (name, bidirectional, attention) in new[] { ("ner/ontonotes-ww-multi_nocharlm", true, false), ("pos/combined_nocharlm", false, true) })
        {
            var ckpt = Checkpoint.Load(Repo.Model(name));
            var (model, config, vocab) = (ckpt.Root["model"]!, ckpt.Root["config"]!, ckpt.Root["vocab"]!["char"]!);
            using var reference = new CharacterModel(ckpt, model, config, vocab, "charmodel.", bidirectional, attention);
            var managed = new ManagedCharacterModel(ckpt, model, config, vocab, "charmodel.", bidirectional, attention);
            var expected = reference.ForwardWords([words.Select(reference.CharIds).ToList()]).data<float>().ToArray();
            var actual = With(path, 4, () => managed.Forward(words.Select(managed.CharIds).ToList()));
            output.WriteLine($"{path} {name}: max |diff| {TokenizerTests.AssertClose(expected, actual, 1e-5f, $"{path} {name}"):E2}");
        }
    }

    /// <summary>NER's network, both backends and both checkpoints, on one padded batch of real sentences (one alone at full width).</summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void NerNet_ManagedMatchesTorchSharp(string path)
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        using var backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var managedForward = ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        var managedBackward = ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var sentences = new[] { 12, 3, 30, 1, 17 }.Select((n, i) => (IReadOnlyList<string>)Enumerable.Range(i * 31, n).Select(k => text[k % text.Length]).ToList()).ToList();
        foreach (var name in new[] { "ner/ontonotes-ww-multi_charlm", "ner/ontonotes-ww-multi_nocharlm" })
        {
            bool charlm = name.EndsWith("_charlm");
            using var reference = Ner.NerTagger.Load(Repo.Model(name), pretrain, charlm ? forward : null, charlm ? backward : null);
            using var managed = Ner.NerTagger.LoadManaged(Repo.Model(name), pretrain, charlm ? managedForward : null, charlm ? managedBackward : null);
            var expectedTags = reference.Predict(sentences, out var expected);
            var actualTags = With(path, 4, () => managed.Predict(sentences, out var e) is var t ? (t, e) : default);
            float diff = 0;
            for (int i = 0; i < sentences.Count; i++)
            {
                diff = Math.Max(diff, TokenizerTests.AssertClose(expected[i], actualTags.e[i], 1e-4f, $"{path} {name} s{i}"));
                Assert.Equal(expectedTags[i], actualTags.t[i]);
            }
            output.WriteLine($"{path} {name}: max |diff| {diff:E2}");
        }
    }

    /// <summary>
    /// The tagger's network, both backends and both checkpoints, on one batch of real sentences (one with a word
    /// simplify_punct changes). With the charlms, the managed tagger's cache entries (arrays) match the TorchSharp one's.
    /// </summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void PosNet_ManagedMatchesTorchSharp(string path)
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        using var backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var managedForward = ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        var managedBackward = ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var sentences = new[] { 12, 3, 30, 1, 17 }.Select((n, i) => (IReadOnlyList<string>)Enumerable.Range(i * 31, n).Select(k => text[k % text.Length]).ToList()).ToList();
        sentences.Add(["Really", "?!"]);
        var keys = sentences.Select(_ => (Sentence?)new Sentence()).ToList();
        foreach (var name in new[] { "pos/combined_charlm", "pos/combined_nocharlm" })
        {
            bool charlm = name.EndsWith("_charlm");
            using var reference = Pos.PosTagger.Load(Repo.Model(name), pretrain, charlm ? forward : null, charlm ? backward : null);
            using var managed = Pos.PosTagger.LoadManaged(Repo.Model(name), pretrain, charlm ? managedForward : null, charlm ? managedBackward : null);
            using var referenceCache = new CharlmCache();
            using var managedCache = new CharlmCache();
            var expectedTags = reference.Predict(sentences, out var expected, referenceCache, keys);
            var actualTags = With(path, 4, () => managed.Predict(sentences, out var e, managedCache, keys) is var t ? (t, e) : default);
            float diff = 0, cacheDiff = 0;
            for (int i = 0; i < sentences.Count; i++)
            {
                diff = Math.Max(diff, TokenizerTests.AssertClose(expected[i], actualTags.e[i], 1e-4f, $"{path} {name} s{i}"));
                Assert.Equal(expectedTags[i], actualTags.t[i]);
                bool kept = referenceCache.TryGetArrays(keys[i]!, out var r);
                Assert.Equal(kept, managedCache.TryGetArrays(keys[i]!, out var m));
                Assert.Equal(charlm && i < sentences.Count - 1, kept); // not the one simplify_punct changed
                if (kept)
                    cacheDiff = Math.Max(cacheDiff, Math.Max(TokenizerTests.AssertClose(r.Forward, m.Forward, 1e-4f, $"{path} s{i} forward"),
                        TokenizerTests.AssertClose(r.Backward, m.Backward, 1e-4f, $"{path} s{i} backward")));
            }
            output.WriteLine($"{path} {name}: UPOS scores max |diff| {diff:E2}, cached charlm max |diff| {cacheDiff:E2}");
        }
    }

    /// <summary>The managed tagger's UPOS logits against Python Stanza's (both packages' golden intermediates) on every path, at TorchSharp's 1e-4.</summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void PosUposLogits_MatchGoldenOnEveryPath(string path)
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        var managedForward = ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        var managedBackward = ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        foreach (var (name, golden) in new[] { ("pos/combined_charlm", Repo.Golden), ("pos/combined_nocharlm", Path.Combine(Repo.Golden, "fast")) })
        {
            bool charlm = name.EndsWith("_charlm");
            using var tagger = Pos.PosTagger.LoadManaged(Repo.Model(name), pretrain, charlm ? managedForward : null, charlm ? managedBackward : null);
            var tensors = SafeTensorFile.Load(Path.Combine(golden, "intermediates.safetensors"));
            var index = JsonNode.Parse(File.ReadAllText(Path.Combine(golden, "intermediates.json")))!["sentences"]!.AsArray();
            var sentences = index.Select(e => (IReadOnlyList<string>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList()).ToList();
            var (tags, logits) = With(path, 4, () => (tagger.Predict(sentences, out var l), l));
            float diff = 0;
            for (int i = 0; i < sentences.Count; i++)
            {
                diff = Math.Max(diff, TokenizerTests.AssertClose(tensors.Read<float>($"s{i}.pos.upos_logits"), logits[i], 1e-4f, $"{path} {name} s{i}"));
                Assert.Equal(index[i]!["xpos"]!.AsArray().Select(x => x!.GetValue<string>()), tags[i].Select(t => t.Xpos));
            }
            output.WriteLine($"{path} {name}: UPOS logits max |diff| vs Python {diff:E2}");
        }
    }

    /// <summary>
    /// The parser's network, both backends and both checkpoints, on one padded batch of golden sentences (Stanza's tags and
    /// lemmas; lengths 4 to 40, plus one with a word simplify_punct changes): arc log-probs of every real row against
    /// every column (padding included) and label scores of every real pair at 1e-4, identical parses.
    /// </summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void DepparseNet_ManagedMatchesTorchSharp(string path)
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        using var backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var managedForward = ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        var managedBackward = ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "depparse", "corpus.conllu")));
        var batch = doc.Sentences.Select(s => (IReadOnlyList<Word>)s.Words.ToList()).Where(s => s.Count <= 40)
            .DistinctBy(s => s.Count / 4).OrderByDescending(s => s.Count).ToList();
        batch.Add([new Word { Text = "Really", Upos = "ADV", Xpos = "RB", Lemma = "really" }, new Word { Text = "?!", Upos = "PUNCT", Xpos = ".", Lemma = "?!" }]);
        foreach (var name in new[] { "depparse/combined_charlm", "depparse/combined_nocharlm" })
        {
            bool charlm = name.EndsWith("_charlm");
            using var reference = Depparse.DependencyParser.Load(Repo.Model(name), pretrain, charlm ? forward : null, charlm ? backward : null);
            using var managed = Depparse.DependencyParser.LoadManaged(Repo.Model(name), pretrain, charlm ? managedForward : null, charlm ? managedBackward : null);
            var expected = reference.Scores(batch, labelScores: true);
            var (actual, parses) = With(path, 4, () => (managed.Scores(batch, labelScores: true), managed.Parse(batch)));
            int width = expected.Width, relations = expected.Relations;
            Assert.Equal(width, actual.Width);
            float arcs = 0, labels = 0;
            for (int b = 0; b < batch.Count; b++)
            {
                int n = batch[b].Count + 1;
                var rows = Enumerable.Range((b * width) * width, n * width).ToArray();
                Assert.Equal(rows.Select(k => float.IsNegativeInfinity(expected.ArcLogProbs[k])), rows.Select(k => float.IsNegativeInfinity(actual.ArcLogProbs[k])));
                float[] Finite(float[] a) => rows.Select(k => float.IsNegativeInfinity(a[k]) ? 0 : a[k]).ToArray();
                arcs = Math.Max(arcs, TokenizerTests.AssertClose(Finite(expected.ArcLogProbs), Finite(actual.ArcLogProbs), 1e-4f, $"{path} {name} s{b} arcs"));
                var pairs = Enumerable.Range(0, n).SelectMany(i => Enumerable.Range(((b * width + i) * width) * relations, n * relations)).ToArray();
                labels = Math.Max(labels, TokenizerTests.AssertClose(pairs.Select(k => expected.LabelScores![k]).ToArray(),
                    pairs.Select(k => actual.LabelScores![k]).ToArray(), 1e-4f, $"{path} {name} s{b} labels"));
            }
            var expectedParses = reference.Parse(batch);
            for (int b = 0; b < batch.Count; b++)
                Assert.Equal(expectedParses[b], parses[b]);
            output.WriteLine($"{path} {name}: {batch.Count} sentences, width {width}: arc log-probs max |diff| {arcs:E2}, label scores {labels:E2}");
        }
    }

    /// <summary>The managed parser's arc and label log-probs against Python Stanza's (both packages' golden intermediates) on every path, at TorchSharp's 1e-4.</summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void DepparseScores_MatchGoldenOnEveryPath(string path)
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        var managedForward = ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        var managedBackward = ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        foreach (var (name, golden, prefix) in new[] { ("depparse/combined_charlm", Path.Combine(Repo.Golden, "depparse"), ""), ("depparse/combined_nocharlm", Path.Combine(Repo.Golden, "fast"), "depparse.") })
        {
            bool charlm = name.EndsWith("_charlm");
            using var parser = Depparse.DependencyParser.LoadManaged(Repo.Model(name), pretrain, charlm ? managedForward : null, charlm ? managedBackward : null);
            var tensors = SafeTensorFile.Load(Path.Combine(golden, "intermediates.safetensors"));
            var doc = Conllu.Read(File.ReadAllText(Path.Combine(golden, "corpus.conllu")));
            float arcs = 0, labels = 0;
            for (int i = 0; i < 3; i++)
            {
                var scores = With(path, 4, () => parser.Scores([doc.Sentences[i].Words.ToList()], labelScores: true));
                var (a, l) = DepparseTests.AssertScoresMatch(scores, tensors.Read<float>($"s{i}.{prefix}unlabeled"), tensors.Read<float>($"s{i}.{prefix}deprel"), $"{path} {name} s{i}");
                (arcs, labels) = (Math.Max(arcs, a), Math.Max(labels, l));
            }
            output.WriteLine($"{path} {name}: arc log-probs max |diff| vs Python {arcs:E2}, label log-probs {labels:E2}");
        }
    }

    /// <summary>
    /// The sentiment classifier's network, both backends, on one padded batch of real words (lengths 1 to 30, so the
    /// LSTM and convolutions run over padding): logits at 1e-4, identical labels. Then the managed one with a cache
    /// holding both entry forms (TorchSharp tensors and managed arrays), against itself without the cache.
    /// </summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void SentimentNet_ManagedMatchesTorchSharp(string path)
    {
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"));
        using var forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        using var backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var managedForward = ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
        var managedBackward = ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
        var text = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var sentences = new[] { 12, 3, 30, 1, 17, 8 }.Select((n, i) => (IReadOnlyList<string>)Enumerable.Range(i * 31, n).Select(k => text[k % text.Length]).ToList()).ToList();
        using var reference = Sentiment.SentimentClassifier.Load(Repo.Model("sentiment/sstplus_charlm"), pretrain, forward, backward);
        using var managed = Sentiment.SentimentClassifier.LoadManaged(Repo.Model("sentiment/sstplus_charlm"), pretrain, managedForward, managedBackward);
        var expectedLabels = reference.Classify(sentences, out var expected);
        var (labels, logits) = With(path, 4, () => (managed.Classify(sentences, out var l), l));
        Assert.Equal(expectedLabels, labels);
        float diff = 0;
        for (int i = 0; i < sentences.Count; i++)
            diff = Math.Max(diff, TokenizerTests.AssertClose(expected[i], logits[i], 1e-4f, $"{path} s{i}"));

        using var cache = new CharlmCache();
        var keys = sentences.Select(_ => (Sentence?)new Sentence()).ToList();
        var torchReps = (forward.BuildCharRepresentation(sentences), backward.BuildCharRepresentation(sentences));
        var managedReps = (managedForward.BuildCharRepresentation(sentences), managedBackward.BuildCharRepresentation(sentences));
        for (int i = 0; i < sentences.Count; i++)
            Assert.True(i % 2 == 0 ? cache.TryAdd(keys[i]!, torchReps.Item1[i], torchReps.Item2[i])
                : cache.TryAdd(keys[i]!, managedReps.Item1[i], managedReps.Item2[i], sentences[i].Count));
        var (cachedLabels, cachedLogits) = With(path, 4, () => (managed.Classify(sentences, out var l, cache, keys), l));
        Assert.Equal(labels, cachedLabels);
        float cacheDiff = 0;
        for (int i = 0; i < sentences.Count; i++)
            cacheDiff = Math.Max(cacheDiff, TokenizerTests.AssertClose(logits[i], cachedLogits[i], 1e-4f, $"{path} cached s{i}"));
        output.WriteLine($"{path}: logits max |diff| {diff:E2}; with cached tensors and arrays {cacheDiff:E2}");
    }

    /// <summary>
    /// The lemmatizer's seq2seq network, both backends, on every word the golden lemma files send to the model (868, each
    /// file's misses as one Predict call, as the pipeline batches them) plus words.json: the same number of decoder steps,
    /// identical decoding and edits. Reports each step's log-prob drift (all entries, and those within 10 of the row's
    /// maximum) and the smallest top-2 margin of a row still decoding, on either backend: the room the greedy argmax has.
    /// </summary>
    /// <remarks>
    /// The log-probs are reported, not asserted: no tolerance exists for them yet (TorchSharp's lemmatizer is tested on
    /// its discrete output only), and 1e-4 is out of reach for both backends. On words.json, against Stanza run in float64,
    /// Stanza's own float32 is 2.9e-4 off, TorchSharp 2.7e-4 and the managed net 3.8e-4 (near the top: 1.6e-4, 1.4e-4,
    /// 1.3e-4); the decoder recurrence amplifies float noise. The owner decides (docs/backends.md, lemma).
    /// </remarks>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void LemmaNet_ManagedMatchesTorchSharp(string path)
    {
        using var reference = Lemma.Lemmatizer.Load(Repo.Model("lemma/combined_nocharlm"));
        using var managed = Lemma.Lemmatizer.Load(Repo.Model("lemma/combined_nocharlm"), backend: Backend.Managed);
        var calls = Directory.GetFiles(Path.Combine(Repo.Golden, "lemma"), "*.conllu").Order().Select(file =>
            Conllu.Read(File.ReadAllText(file)).Sentences.SelectMany(s => s.Words)
                .Where(w => reference.Lookup(w.Text, w.Upos) == null)
                .Select(w => new Word { Text = w.Text, Upos = w.Upos }).ToList()).ToList();
        calls.Add(JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "lemma", "words.json")))!.AsArray()
            .Select(g => new Word { Text = (string)g!["word"]!, Upos = (string)g["upos"]! }).ToList());
        Assert.Equal(868, calls.SkipLast(1).Sum(c => c.Count));

        double diff = 0, nearTop = 0, margin = double.PositiveInfinity;
        int steps = 0;
        static double Margin(ReadOnlySpan<float> row)
        {
            float first = float.NegativeInfinity, second = float.NegativeInfinity;
            foreach (var x in row)
                if (x > first)
                    (first, second) = (x, first);
                else if (x > second)
                    second = x;
            return first - second;
        }
        foreach (var words in calls)
        {
            var expectedSteps = new List<(float[] LogProbs, int Columns)>();
            var expected = reference.Predict(words, onStep: (l, c, _) => expectedSteps.Add(((float[])l.Clone(), c)));
            int k = 0;
            var actual = With(path, 4, () => managed.Predict(words, onStep: (l, c, done) =>
            {
                var (e, columns) = expectedSteps[k++];
                Assert.Equal(columns, c);
                for (int r = 0; r < done.Length; r++)
                {
                    var er = e.AsSpan(r * c, c);
                    var ar = l.AsSpan(r * c, c);
                    float top = float.NegativeInfinity;
                    foreach (var x in er)
                        top = Math.Max(top, x);
                    for (int v = 0; v < c; v++)
                    {
                        double d = Math.Abs(er[v] - ar[v]);
                        diff = Math.Max(diff, d);
                        if (er[v] > top - 10)
                            nearTop = Math.Max(nearTop, d);
                    }
                    if (!done[r])
                        margin = Math.Min(margin, Math.Min(Margin(er), Margin(ar)));
                }
            }));
            Assert.Equal(expectedSteps.Count, k);
            steps += k;
            Assert.Equal(expected.Decoded, actual.Decoded);
            Assert.Equal(expected.Edits, actual.Edits);
        }
        output.WriteLine($"{path}: {calls.Sum(c => c.Count)} words, {steps} decoder steps: log-probs max |diff| {diff:E2} " +
            $"(within 10 of the top {nearTop:E2}), smallest top-2 margin {margin:E2}");
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

/// <summary>
/// Tests that set <see cref="Gemm.Path"/> or <see cref="ManagedThreads.Count"/> run here, alone (after the parallel
/// collections), so the managed backend's golden tests elsewhere always run on the detected path.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ManagedKernelsCollection
{
    public const string Name = "Managed kernels";
}
