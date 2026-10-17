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

    [TransformerFact("config.json", "tokenizer.json", "tokenizer_config.json", "pytorch_model.bin")]
    public void ASentence_HasTheSameBitsAloneInABatchAndOnAnyThreadCount()
    {
        var net = models.Embedder(managed: true).Net;
        var ids = Ids(TransformerEncoderTests.ReadGolden().Sentences);
        var batched = With(Gemm.Path, 4, () => net.Encode(ids, 4));
        for (int s = 0; s < ids.Length; s++)
        {
            Assert.True(net.Encode([ids[s]], 4)[0].AsSpan().SequenceEqual(batched[s]), $"sentence {s} alone differs from the batch");
            Assert.True(With(Gemm.Path, 1, () => net.Encode([ids[s]], 4))[0].AsSpan().SequenceEqual(batched[s]), $"sentence {s} on 1 thread differs");
        }
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
