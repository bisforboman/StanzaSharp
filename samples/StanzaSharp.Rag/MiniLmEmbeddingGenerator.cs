using System.Buffers;
using System.Security.Cryptography;
using Microsoft.Extensions.AI;
using Microsoft.ML.Tokenizers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Rag;

/// <summary>
/// Sentence embeddings from sentence-transformers' all-MiniLM-L6-v2 (Apache 2.0) on StanzaSharp's managed transformer
/// encoder, offline and with no native library: the model's weights and its WordPiece vocabulary, mean pooling over the
/// tokens, then L2 normalization, as sentence-transformers runs it. 384 dimensions; texts longer than
/// <see cref="MaxTokens"/> are truncated, as the model was trained.
/// </summary>
/// <remarks>
/// The encoder (<see cref="ManagedTransformer"/>) is internal to StanzaSharp and not in the package's public API (#268):
/// this sample reaches it through the repository's InternalsVisibleTo. Load a <see cref="Pipeline"/> first: it sets the
/// managed backend's threads and kernel path.
/// </remarks>
internal sealed class MiniLmEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    public const int Dimensions = 384;
    public const int MaxTokens = 256; // sentence-transformers' max_seq_length for this model

    // Pinned like StanzaSharp's own transformer downloads: a commit, and each file's sha256.
    private const string Repository = "sentence-transformers/all-MiniLM-L6-v2";
    private const string Revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
    private static readonly (string File, string Sha256)[] Files =
    [
        ("config.json", "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"),
        ("pytorch_model.bin", "c3a85f238711653950f6a79ece63eb0ea93d76f6a6284be04019c53733baf256"),
        ("vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    ];

    private readonly ManagedTransformer _encoder;

    private MiniLmEmbeddingGenerator(string directory)
    {
        Tokenizer = BertTokenizer.Create(Path.Combine(directory, "vocab.txt"), new BertOptions { LowerCaseBeforeTokenization = true });

        // The checkpoint is a bare BertModel's state dict: no prefix before its tensors' names.
        var config = TransformerConfig.Load(Path.Combine(directory, "config.json")) with { Prefix = "" };
        _encoder = ManagedTransformer.Load(config, Checkpoint.Load(Path.Combine(directory, "pytorch_model.bin")));
    }

    /// <summary>The model's own tokenizer, also the right one to size chunks with.</summary>
    public BertTokenizer Tokenizer { get; }

    /// <summary>Downloads the model into <paramref name="directory"/> (files that match are kept) and loads it.</summary>
    public static async Task<MiniLmEmbeddingGenerator> LoadAsync(string directory, CancellationToken cancellationToken = default)
    {
        using var http = new HttpClient();
        foreach (var (file, sha256) in Files)
        {
            var path = Path.Combine(directory, file);
            if (File.Exists(path) && await Sha256(path, cancellationToken) == sha256)
                continue;
            Console.Error.WriteLine($"Downloading {Repository}/{file}");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var url = $"https://huggingface.co/{Repository}/resolve/{Revision}/{file}";
            await using (var source = await http.GetStreamAsync(url, cancellationToken))
            await using (var target = File.Create(path + ".part"))
                await source.CopyToAsync(target, cancellationToken);
            if (await Sha256(path + ".part", cancellationToken) != sha256)
                throw new InvalidDataException($"{url} doesn't have the expected sha256 {sha256}.");
            File.Move(path + ".part", path, overwrite: true);
        }

        return new MiniLmEmbeddingGenerator(directory);
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        var texts = values.ToList();
        var embeddings = new GeneratedEmbeddings<Embedding<float>>(texts.Count);
        if (texts.Count == 0)
            return Task.FromResult(embeddings);

        // [CLS] pieces [SEP], cut to MaxTokens. No padding: the encoder runs each text at its own length.
        var ids = texts.Select(Encode).ToList();
        var hidden = _encoder.Encode(ids, 1, cancellationToken); // per text: [pieces, Dimensions], the last layer
        try
        {
            for (var b = 0; b < ids.Count; b++)
                embeddings.Add(new Embedding<float>(Pool(hidden[b].AsSpan(0, ids[b].Length * Dimensions), ids[b].Length)));
        }
        finally
        {
            if (_encoder.RentsOutputs)
            {
                foreach (var array in hidden)
                    ArrayPool<float>.Shared.Return(array);
            }
        }

        return Task.FromResult(embeddings);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey == null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() => _encoder.Dispose();

    private int[] Encode(string text)
    {
        var ids = Tokenizer.EncodeToIds(text, addSpecialTokens: true, considerPreTokenization: true, considerNormalization: true);
        return ids.Count > MaxTokens ? [.. ids.Take(MaxTokens - 1), Tokenizer.SeparatorTokenId] : [.. ids];
    }

    // The mean of the pieces' states, then x / max(‖x‖, 1e-12), both in double.
    private static float[] Pool(ReadOnlySpan<float> states, int pieces)
    {
        var sum = new double[Dimensions];
        for (var t = 0; t < pieces; t++)
        {
            var row = states.Slice(t * Dimensions, Dimensions);
            for (var d = 0; d < Dimensions; d++)
                sum[d] += row[d];
        }

        var norm = 0.0;
        for (var d = 0; d < Dimensions; d++)
        {
            sum[d] /= pieces;
            norm += sum[d] * sum[d];
        }

        var scale = 1 / Math.Max(Math.Sqrt(norm), 1e-12);
        var vector = new float[Dimensions];
        for (var d = 0; d < Dimensions; d++)
            vector[d] = (float)(sum[d] * scale);
        return vector;
    }

    private static async Task<string> Sha256(string path, CancellationToken cancellationToken)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken));
    }
}
