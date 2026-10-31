using System.Security.Cryptography;
using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;

namespace StanzaSharp.Rag;

/// <summary>
/// Sentence embeddings from sentence-transformers' all-MiniLM-L6-v2 (Apache 2.0) on ONNX Runtime, offline: the model's
/// ONNX export and its WordPiece vocabulary, mean pooling over the tokens, then L2 normalization, as sentence-transformers
/// runs it. 384 dimensions; texts longer than <see cref="MaxTokens"/> are truncated, as the model was trained.
/// </summary>
internal sealed class MiniLmEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    public const int Dimensions = 384;
    public const int MaxTokens = 256; // sentence-transformers' max_seq_length for this model

    // Pinned like StanzaSharp's own transformer downloads: a commit, and each file's sha256.
    private const string Repository = "sentence-transformers/all-MiniLM-L6-v2";
    private const string Revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";
    private static readonly (string File, string Sha256)[] Files =
    [
        ("onnx/model.onnx", "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"),
        ("vocab.txt", "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"),
    ];

    private readonly InferenceSession _session;

    private MiniLmEmbeddingGenerator(string directory)
    {
        Tokenizer = BertTokenizer.Create(Path.Combine(directory, "vocab.txt"), new BertOptions { LowerCaseBeforeTokenization = true });
        _session = new InferenceSession(Path.Combine(directory, "onnx", "model.onnx"));
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

        // One padded batch: [CLS] pieces [SEP], cut to MaxTokens.
        var ids = texts.Select(Encode).ToList();
        int batch = ids.Count, width = ids.Max(i => i.Count);
        var inputIds = new long[batch * width];
        var mask = new long[batch * width];
        for (var b = 0; b < batch; b++)
        {
            for (var t = 0; t < ids[b].Count; t++)
            {
                inputIds[(b * width) + t] = ids[b][t];
                mask[(b * width) + t] = 1;
            }
        }

        long[] shape = [batch, width];
        using var idsValue = OrtValue.CreateTensorValueFromMemory(inputIds, shape);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, shape);
        using var typesValue = OrtValue.CreateTensorValueFromMemory(new long[batch * width], shape);
        using var runOptions = new RunOptions();
        using var outputs = _session.Run(runOptions,
            new Dictionary<string, OrtValue> { ["input_ids"] = idsValue, ["attention_mask"] = maskValue, ["token_type_ids"] = typesValue },
            ["last_hidden_state"]);
        var hidden = outputs[0].GetTensorDataAsSpan<float>(); // [batch, width, Dimensions]

        for (var b = 0; b < batch; b++)
        {
            var vector = new float[Dimensions];
            for (var t = 0; t < ids[b].Count; t++)
            {
                var row = hidden.Slice(((b * width) + t) * Dimensions, Dimensions);
                for (var d = 0; d < Dimensions; d++)
                    vector[d] += row[d];
            }

            var norm = 0.0;
            for (var d = 0; d < Dimensions; d++)
            {
                vector[d] /= ids[b].Count;
                norm += vector[d] * vector[d];
            }

            var scale = (float)(1 / Math.Max(Math.Sqrt(norm), 1e-12));
            for (var d = 0; d < Dimensions; d++)
                vector[d] *= scale;
            embeddings.Add(new Embedding<float>(vector));
        }

        return Task.FromResult(embeddings);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey == null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() => _session.Dispose();

    private List<int> Encode(string text)
    {
        var ids = Tokenizer.EncodeToIds(text, addSpecialTokens: true, considerPreTokenization: true, considerNormalization: true).ToList();
        if (ids.Count > MaxTokens)
            ids = [.. ids.Take(MaxTokens - 1), Tokenizer.SeparatorTokenId];
        return ids;
    }

    private static async Task<string> Sha256(string path, CancellationToken cancellationToken)
    {
        await using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(stream, cancellationToken));
    }
}
