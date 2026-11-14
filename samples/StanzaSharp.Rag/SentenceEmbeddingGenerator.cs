using System.Buffers;
using System.Security.Cryptography;
using Microsoft.Extensions.AI;
using Microsoft.ML.Tokenizers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Rag;

/// <summary>
/// A sentence-transformers model the sample can run (<c>--model</c>): where its files are, pinned like StanzaSharp's own
/// transformer downloads (a commit, and each file's sha256), and how sentence-transformers runs it.
/// </summary>
/// <param name="Name">The <c>--model</c> value.</param>
/// <param name="Repository">The Hugging Face repository; its last part names the model's folder.</param>
/// <param name="Revision">The pinned commit.</param>
/// <param name="MaxTokens">sentence-transformers' <c>max_seq_length</c>: longer texts are truncated.</param>
/// <param name="Normalize">Whether the model ends in sentence-transformers' <c>Normalize</c> module (L2).</param>
/// <param name="QueryPrefix">What the model expects before a query (E5: <c>query: </c>).</param>
/// <param name="PassagePrefix">What the model expects before a stored text (E5: <c>passage: </c>).</param>
/// <param name="Megabytes">The download's size.</param>
/// <param name="Config">sha256 of <c>config.json</c>.</param>
/// <param name="Weights">sha256 of <c>pytorch_model.bin</c>.</param>
/// <param name="TokenizerJson">sha256 of <c>tokenizer.json</c>.</param>
internal sealed record EmbeddingModel(string Name, string Repository, string Revision, int MaxTokens, bool Normalize, string QueryPrefix, string PassagePrefix,
    int Megabytes, string Config, string Weights, string TokenizerJson)
{
    /// <summary>The models measured in docs/embedding-spike.md: each gives sentence-transformers' vectors.</summary>
    public static readonly EmbeddingModel[] All =
    [
        // English, 384 dimensions, Apache 2.0
        new("minilm", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", 256, true, "", "", 91,
            "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41", "c3a85f238711653950f6a79ece63eb0ea93d76f6a6284be04019c53733baf256",
            "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"),

        // About 100 languages, 384 dimensions, MIT
        new("e5-small", "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3", 512, true, "query: ", "passage: ", 488,
            "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959", "f0ccb64a2edc689fce7b1895f25146740e4075c3da28c8d0e672a8ba49c51167",
            "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),

        // About 100 languages, 768 dimensions, MIT
        new("e5-base", "intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", 512, true, "query: ", "passage: ", 1129,
            "9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4", "f061cb7641880f52895cbacab7c4ab39b0844e2e6b73794f2798de460d9fa418",
            "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"),

        // Swedish and English, 768 dimensions, Apache 2.0; its vectors are not normalized
        new("swedish", "KBLab/sentence-bert-swedish-cased", "6b5e83cd29c03729cfdc33d13b1423399b0efb5c", 384, false, "", "", 500,
            "f52dc6d1445b08f2b02910b100592932cef5696c34784eb7bb6386dacb39b9e1", "6bea01d07d85dc7870dfd9907ca78bac21ee8b1ef81c838f25a88b6494c79b32",
            "0001cd9fd9ef17918f38ad084d7abd372ff0fa587d8b12298bf659154ad6f045"),
    ];

    /// <summary>The model's folder under the models directory.</summary>
    public string Folder => Repository[(Repository.IndexOf('/') + 1)..];

    public (string File, string Sha256)[] Files => [("config.json", Config), ("pytorch_model.bin", Weights), ("tokenizer.json", TokenizerJson)];
}

/// <summary>
/// Sentence embeddings from a sentence-transformers model (<see cref="EmbeddingModel.All"/>) on StanzaSharp's managed
/// transformer encoder, offline and with no native library: the model's weights and its own tokenizer (WordPiece, or
/// XLM-RoBERTa's SentencePiece Unigram for multilingual E5), mean pooling over the tokens, then L2 normalization where the
/// model has it, as sentence-transformers runs it. Texts longer than the model's limit are truncated, as it was trained.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="GenerateAsync"/> embeds texts to store (with the model's passage prefix); <see cref="EmbedQueryAsync"/>
/// embeds a question (with its query prefix). Only E5 tells the two apart.
/// </para>
/// <para>
/// The encoder (<see cref="ManagedTransformer"/>) and the tokenizers are internal to StanzaSharp and not in the package's
/// public API (#268): this sample reaches them through the repository's InternalsVisibleTo. Load a
/// <see cref="Pipeline"/> first: it sets the managed backend's threads and kernel path.
/// </para>
/// </remarks>
internal sealed class SentenceEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly ManagedTransformer _encoder;
    private readonly ITransformerTokenizer _pieces;

    private SentenceEmbeddingGenerator(EmbeddingModel model, string directory)
    {
        Model = model;
        var tokenizerJson = Path.Combine(directory, "tokenizer.json");
        _pieces = Unigram.IsUnigram(tokenizerJson) ? Unigram.Load(tokenizerJson) : WordPiece.Load(tokenizerJson);
        Tokenizer = new PieceCounter(_pieces);

        // The checkpoint is a bare BertModel's or XLMRobertaModel's state dict: no prefix before its tensors' names.
        var config = TransformerConfig.Load(Path.Combine(directory, "config.json")) with { Prefix = "" };
        _encoder = ManagedTransformer.Load(config, Checkpoint.Load(Path.Combine(directory, "pytorch_model.bin")));
        Dimensions = config.Hidden;
    }

    public EmbeddingModel Model { get; }

    /// <summary>The vectors' length (the encoder's hidden size).</summary>
    public int Dimensions { get; }

    /// <summary>
    /// The model's own tokenizer as DataIngestion's chunkers take one: it counts a text's pieces, with the two special
    /// tokens around them, exactly as <see cref="GenerateAsync"/> encodes it. The right one to size chunks with.
    /// </summary>
    public Tokenizer Tokenizer { get; }

    /// <summary>Downloads the model into <paramref name="directory"/> (files that match are kept) and loads it.</summary>
    public static async Task<SentenceEmbeddingGenerator> LoadAsync(EmbeddingModel model, string directory, CancellationToken cancellationToken = default)
    {
        using var http = new HttpClient();
        foreach (var (file, sha256) in model.Files)
        {
            var path = Path.Combine(directory, file);
            if (File.Exists(path) && await Sha256(path, cancellationToken) == sha256)
                continue;
            Console.Error.WriteLine($"Downloading {model.Repository}/{file}");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var url = $"https://huggingface.co/{model.Repository}/resolve/{model.Revision}/{file}";
            await using (var source = await http.GetStreamAsync(url, cancellationToken))
            await using (var target = File.Create(path + ".part"))
                await source.CopyToAsync(target, cancellationToken);
            if (await Sha256(path + ".part", cancellationToken) != sha256)
                throw new InvalidDataException($"{url} doesn't have the expected sha256 {sha256}.");
            File.Move(path + ".part", path, overwrite: true);
        }

        return new SentenceEmbeddingGenerator(model, directory);
    }

    /// <summary>Embeds texts to store: each with the model's passage prefix.</summary>
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default) =>
        Task.FromResult(new GeneratedEmbeddings<Embedding<float>>(Embed([.. values.Select(v => Model.PassagePrefix + v)], cancellationToken)));

    /// <summary>Embeds a question to search with: with the model's query prefix.</summary>
    public Task<ReadOnlyMemory<float>> EmbedQueryAsync(string query, CancellationToken cancellationToken = default) =>
        Task.FromResult(Embed([Model.QueryPrefix + query], cancellationToken)[0].Vector);

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey == null && serviceType.IsInstanceOfType(this) ? this : null;

    public void Dispose() => _encoder.Dispose();

    private List<Embedding<float>> Embed(List<string> texts, CancellationToken cancellationToken)
    {
        var embeddings = new List<Embedding<float>>(texts.Count);
        if (texts.Count == 0)
            return embeddings;

        // The first special token, the pieces, the last, cut to MaxTokens. No padding: the encoder runs each text at its own length.
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

        return embeddings;
    }

    // The whole text is one "word" to StanzaSharp's tokenizers: they split it at white space (and punctuation) themselves.
    // sentence-transformers' truncation: the last pieces go, the closing special token stays.
    private int[] Encode(string text)
    {
        var ids = _pieces.Encode([text]).Ids;
        return ids.Length > Model.MaxTokens ? [.. ids.AsSpan(0, Model.MaxTokens - 1), ids[^1]] : ids;
    }

    // The mean of the pieces' states, then x / max(‖x‖, 1e-12) where the model normalizes, both in double.
    private float[] Pool(ReadOnlySpan<float> states, int pieces)
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

        var scale = Model.Normalize ? 1 / Math.Max(Math.Sqrt(norm), 1e-12) : 1;
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

    /// <summary>
    /// A StanzaSharp transformer tokenizer behind Microsoft.ML.Tokenizers' <see cref="Microsoft.ML.Tokenizers.Tokenizer"/>, for counting only:
    /// the tokens carry their ids but no text or offsets, and nothing can be decoded.
    /// </summary>
    private sealed class PieceCounter(ITransformerTokenizer pieces) : Tokenizer
    {
        public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten) =>
            throw new NotSupportedException("This tokenizer only counts tokens.");

        protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
        {
            text ??= textSpan.ToString();
            var ids = pieces.Encode([text]).Ids;
            var count = Math.Min(ids.Length, settings.MaxTokenCount);
            return new EncodeResults<EncodedToken>
            {
                Tokens = [.. ids.Take(count).Select(id => new EncodedToken(id, "", default))],
                CharsConsumed = text.Length,
            };
        }
    }
}
