using System.Text.Json;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>The transformer's nets, loaded once per backend for the classes that use them.</summary>
public sealed class TransformerModels : IDisposable
{
    private readonly Lazy<ITransformerNet> _managed = new(() => ManagedTransformer.Load(TransformerFactAttribute.Directory));
    private readonly Lazy<ITransformerNet> _torch = new(() => TransformerNet.Load(TransformerFactAttribute.Directory));
    private readonly Lazy<WordPiece> _tokenizer = new(() => WordPiece.Load(Path.Combine(TransformerFactAttribute.Directory, "tokenizer.json")));

    internal TransformerEmbedder Embedder(bool managed) =>
        new(managed ? _managed.Value : _torch.Value, _tokenizer.Value, TransformerEmbedder.ReadMaxLength(TransformerFactAttribute.Directory), "google/electra-large-discriminator");

    public void Dispose()
    {
        if (_managed.IsValueCreated)
            _managed.Value.Dispose();
        if (_torch.IsValueCreated)
            _torch.Value.Dispose();
    }
}

/// <summary>
/// The encoder and <see cref="TransformerEmbedder"/> against tests/golden/transformer/hidden.safetensors, Stanza's
/// extract_bert_embeddings in one batch (with a sentence over 512 pieces): every value within 1e-4 of Stanza's float32
/// or float64 one, the badly conditioned sentences within their listed bound (owner's decision, 2026-10-17;
/// docs/transformer-plan.md).
/// </summary>
public class TransformerEncoderTests(TransformerModels models, ITestOutputHelper output) : IClassFixture<TransformerModels>
{
    internal static readonly string[] Files = ["config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin"];

    internal sealed record Golden(string[][] Sentences, double?[] Bounds, string[] Long, int LongStride, float[] MixWeight, SafeTensorFile Tensors);

    internal static Golden ReadGolden()
    {
        using var json = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Repo.Golden, "transformer", "hidden.json")));
        var root = json.RootElement;
        var sentences = root.GetProperty("sentences").EnumerateArray().ToArray();
        return new Golden(
            [.. sentences.Select(s => s.GetProperty("words").EnumerateArray().Select(w => w.GetString()!).ToArray())],
            [.. sentences.Select(s => s.GetProperty("bound").ValueKind == JsonValueKind.Null ? (double?)null : s.GetProperty("bound").GetDouble())],
            [.. root.GetProperty("long").EnumerateArray().Select(w => w.GetString()!)], root.GetProperty("long_stride").GetInt32(),
            [.. root.GetProperty("mix_weight").EnumerateArray().Select(w => (float)w.GetDouble())],
            SafeTensorFile.Load(Path.Combine(Repo.Golden, "transformer", "hidden.safetensors")));
    }

    /// <summary>Each value within <paramref name="bound"/> (null: 1e-4, relative above |100|) of the float32 or float64 reference.</summary>
    /// <returns>The largest distance to the nearer reference.</returns>
    internal static double AssertNear(float[] actual, float[] f32, float[] f64, double? bound, string what)
    {
        Assert.Equal(f32.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double d = Math.Min(Math.Abs(actual[i] - f32[i]), Math.Abs(actual[i] - f64[i]));
            double allowed = bound ?? Math.Max(TokenizerTests.Allowed(1e-4f, f32[i]), TokenizerTests.Allowed(1e-4f, f64[i]));
            if (!(d <= allowed))
                Assert.Fail($"{what}: value {i} (row {i / 1024}) is {actual[i]}, float32 {f32[i]}, float64 {f64[i]}: {d:E2} over {allowed:E2}");
            worst = Math.Max(worst, d);
        }

        return worst;
    }

    [TransformerTheory("config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin")]
    [InlineData(true)]
    [InlineData(false)]
    public void Embeddings_MatchStanza(bool managed)
    {
        var golden = ReadGolden();
        var embedder = models.Embedder(managed);
        string[][] batch = [.. golden.Sentences, golden.Long];
        var legacy = embedder.Embed(batch, keepEndpoints: false, layerMix: null);
        var mixed = embedder.Embed(batch, keepEndpoints: true, layerMix: golden.MixWeight);
        for (int s = 0; s < golden.Sentences.Length; s++)
        {
            string text = string.Join(' ', golden.Sentences[s]);
            double l = AssertNear(legacy[s], golden.Tensors.Read<float>($"s{s}.legacy"), golden.Tensors.Read<float>($"s{s}.legacy.f64"), golden.Bounds[s], $"{text} legacy");
            double m = AssertNear(mixed[s], golden.Tensors.Read<float>($"s{s}.mix"), golden.Tensors.Read<float>($"s{s}.mix.f64"), golden.Bounds[s], $"{text} mix");
            output.WriteLine($"{text}: legacy {l:E2}, mix {m:E2} (bound {golden.Bounds[s]?.ToString() ?? "1e-4"})");
        }

        int h = embedder.Net.Config.Hidden;
        var longRows = legacy[^1].Chunk(h).Where((_, i) => i % golden.LongStride == 0).SelectMany(r => r).ToArray();
        double longWorst = AssertNear(longRows, golden.Tensors.Read<float>("long.legacy"), golden.Tensors.Read<float>("long.legacy.f64"), null, "long sentence");
        output.WriteLine($"long sentence ({golden.Long.Length} words, windows of 512): legacy {longWorst:E2}");
    }
}

/// <summary>The managed encoder's invariances, bit for bit, and its drift on every SIMD path (they set Gemm.Path and the threads).</summary>
[Collection(ManagedKernelsCollection.Name)]
[Trait("Backend", "Managed")]
public class TransformerKernelTests(TransformerModels models, ITestOutputHelper output) : IClassFixture<TransformerModels>
{
    private static readonly KernelPath[] NativePaths =
        [.. Enum.GetValues<KernelPath>().Where(p => p switch
        {
            KernelPath.Vector512 => System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated,
            KernelPath.Vector256 => System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated,
            _ => true,
        })];

    public static TheoryData<string> Paths() => [.. NativePaths.Select(p => p.ToString())];

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

    private int[][] Ids(string[][] sentences)
    {
        var tokenizer = models.Embedder(managed: true).Tokenizer;
        return [.. sentences.Select(s => TransformerInput.Build(tokenizer, s).Ids)];
    }

    // The managed encoder rents its outputs, so an array can be longer than its sentence's states: compare those only.
    private static float[][] Encode(ITransformerNet net, IReadOnlyList<int[]> ids, int lastLayers) =>
        [.. net.Encode(ids, lastLayers).Select((a, i) => a[..(lastLayers * ids[i].Length * net.Config.Hidden)])];

    [TransformerFact("config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin")]
    public void ASentence_HasTheSameBitsAloneInABatchAndOnAnyThreadCount()
    {
        var net = models.Embedder(managed: true).Net;
        var ids = Ids(TransformerEncoderTests.ReadGolden().Sentences);
        var batched = With(Gemm.Path, 4, () => Encode(net, ids, 4));
        for (int s = 0; s < ids.Length; s++)
        {
            Assert.True(Encode(net, [ids[s]], 4)[0].AsSpan().SequenceEqual(batched[s]), $"sentence {s} alone differs from the batch");
            Assert.True(With(Gemm.Path, 1, () => Encode(net, [ids[s]], 4))[0].AsSpan().SequenceEqual(batched[s]), $"sentence {s} on 1 thread differs");
        }
    }

    /// <summary>Sentiment's checkpoint, relative to the transformer directory (TransformerFact's files).</summary>
    private const string SentimentModel = "../../../sentiment/sstplus_electra-large.pt";

    private ITransformerNet LoraNet()
    {
        var ckpt = Checkpoint.Load(Path.Combine(Repo.StanzaModels, "sentiment", "sstplus_electra-large"));
        var p = ckpt.Root["params"]!;
        return models.Embedder(managed: true).Net.WithLora(LoraWeights.Read(ckpt, p["bert_lora"]!, p["config"]!));
    }

    [TransformerTheory("config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin", SentimentModel)]
    [MemberData(nameof(Paths))]
    public void WithLora_ASentence_HasTheSameBitsAloneInABatchAndOnAnyThreadCount(string path)
    {
        var kernel = Enum.Parse<KernelPath>(path);
        using var net = LoraNet();
        var ids = Ids(TransformerEncoderTests.ReadGolden().Sentences[..3]);
        var batched = With(kernel, 4, () => Encode(net, ids, 4));
        var frozen = With(kernel, 4, () => Encode(models.Embedder(managed: true).Net, ids, 4));
        for (int s = 0; s < ids.Length; s++)
        {
            Assert.True(With(kernel, 4, () => Encode(net, [ids[s]], 4))[0].AsSpan().SequenceEqual(batched[s]), $"{path}: sentence {s} alone differs from the batch");
            Assert.True(With(kernel, 1, () => Encode(net, [ids[s]], 4))[0].AsSpan().SequenceEqual(batched[s]), $"{path}: sentence {s} on 1 thread differs");
            Assert.False(frozen[s].AsSpan().SequenceEqual(batched[s]), $"{path}: the adapters changed nothing in sentence {s}");
        }

        // Vector512 must keep Vector256's bits, so a runner's CPU can't change a result.
        if (kernel == KernelPath.Vector512)
            Assert.True(With(KernelPath.Vector256, 4, () => Encode(net, ids, 4)).Zip(batched).All(x => x.First.AsSpan().SequenceEqual(x.Second)));
    }

    [TransformerFact("config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin")]
    public void Vector512_GivesVector256sBits()
    {
        if (!NativePaths.Contains(KernelPath.Vector512))
            return; // nothing to compare on this CPU; ManagedBackendTests' kernel tests cover what it runs
        var net = models.Embedder(managed: true).Net;
        var ids = Ids(TransformerEncoderTests.ReadGolden().Sentences[..3]);
        var wide = With(KernelPath.Vector512, 4, () => Encode(net, ids, 4));
        var narrow = With(KernelPath.Vector256, 4, () => Encode(net, ids, 4));
        Assert.True(wide.Zip(narrow).All(x => x.First.AsSpan().SequenceEqual(x.Second)));
    }

    // Values GELU and LayerNorm see, and the extremes: zeros, tiny, huge, past erf's saturation.
    private static float[] Values(int n, int seed)
    {
        var random = new Random(seed);
        var values = new float[n];
        for (int i = 0; i < n; i++)
            values[i] = (float)(random.NextDouble() * 12 - 6) * (i % 7 == 0 ? 8 : 1);
        float[] extremes = [0, -0f, 1e-30f, -1e-30f, 5, -5, 10, -10, 40, -40, 1e30f, -1e30f, float.MaxValue, float.MinValue, float.Epsilon];
        extremes.CopyTo(values, 0);
        return values;
    }

    [Fact]
    public unsafe void Gelu_EveryVectorWidthGivesTheScalarsBits_AndStaysWithinAFloatUlpOfTheReference()
    {
        var values = Values(4096 + 7, 1);
        var blocked = (float[])values.Clone();
        fixed (float* p = blocked)
            ManagedTransformer.GeluBlock(p, blocked.Length);
        int differ = 0;
        for (int i = 0; i < values.Length; i++)
        {
            float scalar = (float)ManagedTransformer.Gelu((double)values[i]);
            Assert.True(BitConverter.SingleToInt32Bits(scalar) == BitConverter.SingleToInt32Bits(blocked[i]), $"GELU({values[i]}): vector {blocked[i]}, scalar {scalar}");

            // The reference: the same erfc polynomial with Math.Exp (the encoder's GELU before round 45).
            double x = values[i], z = Math.Abs(x * 0.70710678118654752), t = 1 / (1 + (0.5 * z));
            double poly = -0.82215223 + (t * 0.17087277);
            foreach (var c in new[] { 1.48851587, -1.13520398, 0.27886807, -0.18628806, 0.09678418, 0.37409196, 1.00002368 })
                poly = c + (t * poly);
            double erfc = t * Math.Exp((-z * z) - 1.26551223 + (t * poly));
            float reference = (float)(0.5 * x * (1 + (x >= 0 ? 1 - erfc : erfc - 1)));
            if (scalar != reference)
            {
                differ++;
                Assert.True(Math.Abs(BitConverter.SingleToInt32Bits(scalar) - BitConverter.SingleToInt32Bits(reference)) <= 1, $"GELU({x}): {scalar}, reference {reference}");
            }
        }

        output.WriteLine($"{differ} of {values.Length} values one float ulp from the reference");
    }

    [Fact]
    public void Exp_IsWithin1e15OfMathExp()
    {
        for (double x = -708; x <= 0; x += 0.0137)
        {
            double e = ManagedTransformer.Exp(x), reference = Math.Exp(x);
            Assert.True(Math.Abs((e / reference) - 1) <= 1e-15, $"exp({x}): {e}, Math.Exp {reference}");
        }
    }

    [Theory]
    [MemberData(nameof(Paths))]
    public unsafe void LayerNormRow_EveryPathGivesTheScalarsBits(string path)
    {
        foreach (int n in new[] { 1024, 1027 })
        {
            var row = Values(n, 2);
            var add = Values(n, 3);
            var gamma = Values(n, 4);
            var beta = Values(n, 5);
            for (int i = 0; i < n; i++)
                (row[i], add[i], gamma[i], beta[i]) = (row[i] / 64, add[i] / 64, gamma[i] / 64, beta[i] / 64); // finite sums
            float[] Run(KernelPath p, bool residual)
            {
                var copy = (float[])row.Clone();
                With(p, 1, () =>
                {
                    fixed (float* r = copy, a = add, g = gamma, b = beta)
                        ManagedTransformer.LayerNormRow(r, residual ? a : null, n, g, b, 1e-12);
                    return 0;
                });
                return copy;
            }

            foreach (bool residual in new[] { false, true })
                Assert.True(Run(Enum.Parse<KernelPath>(path), residual).AsSpan().SequenceEqual(Run(KernelPath.Scalar, residual)), $"{path}, n {n}, residual {residual}");
        }
    }

    [Fact]
    public unsafe void WeightedAdd_GivesTheScalarLoopsBits()
    {
        var v = Values(67, 6);
        double[] acc = [.. Values(67, 7).Select(x => (double)x)], expected = (double[])acc.Clone();
        for (int e = 0; e < expected.Length; e++)
            expected[e] += 0.37 * v[e];
        fixed (float* pv = v)
            ManagedTransformer.WeightedAdd(acc, 0.37, pv);
        Assert.Equal(expected, acc);
    }

    [TransformerTheory("config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin")]
    [MemberData(nameof(Paths))]
    public void EveryPath_MeetsTheTolerance(string path)
    {
        var golden = TransformerEncoderTests.ReadGolden();
        var embedder = models.Embedder(managed: true);
        var first = golden.Sentences[..3]; // the ordinary ones; Scalar is slow
        var legacy = With(Enum.Parse<KernelPath>(path), 4, () => embedder.Embed(first, keepEndpoints: false, layerMix: null));
        for (int s = 0; s < first.Length; s++)
        {
            double worst = TransformerEncoderTests.AssertNear(legacy[s], golden.Tensors.Read<float>($"s{s}.legacy"), golden.Tensors.Read<float>($"s{s}.legacy.f64"),
                null, $"{path} sentence {s}");
            output.WriteLine($"{path} sentence {s}: {worst:E2}");
        }
    }
}
