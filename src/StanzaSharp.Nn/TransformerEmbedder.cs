using System.Buffers;

namespace StanzaSharp.Nn;

/// <summary>
/// Stanza's extract_bert_embeddings for a BERT-style transformer (stanza/models/common/bert_embedding.py,
/// extract_base_embeddings and build_cloned_features): each sentence's words through <see cref="TransformerInput"/>, the
/// encoder, a mix of the last layers, then one vector per word (its last piece), optionally with the [CLS] and [SEP]
/// positions around them.
/// </summary>
/// <param name="maxLength">The tokenizer's <c>model_max_length</c> (tokenizer_config.json, 512 for ELECTRA): longer inputs
/// run in windows, as Stanza's build_cloned_features does.</param>
/// <param name="name">The Hugging Face repository, e.g. <c>google/electra-large-discriminator</c>: what a model's
/// <c>bert_model</c> names.</param>
internal sealed class TransformerEmbedder(ITransformerNet net, ITransformerTokenizer tokenizer, int maxLength, string name)
{
    public string Name { get; } = name;

    /// <summary>
    /// Loads the tokenizer and the encoder <paramref name="net"/> makes from a transformer directory
    /// (<c>config.json</c>, <c>tokenizer.json</c> or <c>vocab.txt</c>, <c>tokenizer_config.json</c>, <c>pytorch_model.bin</c>): a
    /// <see cref="Unigram"/> when tokenizer.json holds a Unigram model (XLM-RoBERTa), a <see cref="ByteLevelBpe"/> when a BPE one
    /// (RoBERTa), else a <see cref="WordPiece"/>.
    /// </summary>
    public static TransformerEmbedder Load(string directory, string name, Func<TransformerConfig, Checkpoint, ITransformerNet> net)
    {
        var config = TransformerConfig.Load(Path.Combine(directory, "config.json"));
        var json = Path.Combine(directory, "tokenizer.json");
        ITransformerTokenizer tokenizer = !File.Exists(json) ? WordPiece.LoadDirectory(directory)
            : Unigram.IsUnigram(json) ? Unigram.Load(json)
            : ByteLevelBpe.IsBpe(json) ? ByteLevelBpe.Load(json)
            : WordPiece.LoadDirectory(directory);
        int maxLength = ReadMaxLength(directory);
        var encoder = net(config, Checkpoint.Load(Path.Combine(directory, "pytorch_model.bin")));
        return new TransformerEmbedder(encoder, tokenizer, maxLength, name);
    }

    /// <summary>
    /// How a frozen-transformer model of <paramref name="processor"/> reads <paramref name="transformer"/>: its
    /// <c>bert_layer_mix.weight</c> when its config sets <c>bert_hidden_layers</c>, else null (the legacy mix). A model without
    /// <c>bert_model</c> gives null and needs no transformer.
    /// </summary>
    /// <exception cref="ArgumentException">The model reads a transformer other than <paramref name="transformer"/> (or none was given).</exception>
    /// <exception cref="NotSupportedException">A fine-tuned transformer (or a LoRA one, which only the overload with
    /// <c>lora</c> allows), which these readers don't port.</exception>
    public static float[]? LayerMix(Checkpoint ckpt, TransformerEmbedder? transformer, string processor) =>
        LayerMix(ckpt, ckpt.Root["config"]!, ckpt.Root["model"]!, transformer, processor);

    /// <summary>
    /// <see cref="LayerMix(Checkpoint, TransformerEmbedder?, string)"/> for a checkpoint whose config and weights are
    /// elsewhere (the constituency parser's and the classifier's <see langword="params"/>).
    /// </summary>
    /// <param name="lora">Whether the model's LoRA adapters (<c>bert_lora</c>) are applied: then <c>use_peft</c> and
    /// <c>bert_finetune</c> are allowed, as long as the transformer's own weights aren't in the checkpoint.</param>
    public static float[]? LayerMix(Checkpoint ckpt, System.Text.Json.Nodes.JsonNode config, System.Text.Json.Nodes.JsonNode model,
        TransformerEmbedder? transformer, string processor, bool lora = false)
    {
        var name = config["bert_model"]?.GetValue<string>();
        if (name == null)
            return null;
        if (transformer?.Name != name)
            throw new ArgumentException($"This {processor} model reads the transformer {name}, but {(transformer == null ? "none was loaded" : $"{transformer.Name} was")}");
        bool tuned = config["bert_finetune"]?.GetValue<bool>() == true || config["use_peft"]?.GetValue<bool>() == true;
        bool ownWeights = model.AsObject().Any(kv => kv.Key.StartsWith("bert_model.", StringComparison.Ordinal));
        if (ownWeights || (tuned && !lora))
            throw new NotSupportedException($"The {processor} checkpoint uses a fine-tuned transformer, which is not ported");
        int layers = config["bert_hidden_layers"]?.GetValue<int>() ?? 0;
        if (layers == 0)
            return null;
        var mix = ckpt.Tensor<float>(model["bert_layer_mix.weight"] ?? throw new InvalidDataException("No bert_layer_mix.weight"));
        return mix.Length == layers ? mix : throw new InvalidDataException($"bert_layer_mix has {mix.Length} weights for {layers} layers");
    }

    /// <summary>Sentences per encoder call, Stanza's 128 (the managed results don't depend on it).</summary>
    public const int BatchSize = 128;

    /// <summary>
    /// Pieces per encoder call at most (a sentence longer than this runs alone): the managed net's buffers grow with the
    /// pieces (the intermediate layer alone is pieces × 4096 floats), and 128 sentences of 512 pieces would rent 1.8 GB.
    /// </summary>
    public const int BatchPieces = 8192;

    public ITransformerNet Net { get; } = net;

    public ITransformerTokenizer Tokenizer { get; } = tokenizer;

    public int MaxLength { get; } = maxLength <= net.Config.MaxPositions
        ? maxLength
        : throw new ArgumentException($"model_max_length {maxLength} is over the model's {net.Config.MaxPositions} positions", nameof(maxLength));

    /// <summary>
    /// Reads <c>model_max_length</c> from a transformer directory's <c>tokenizer_config.json</c>, or the older <c>max_len</c>
    /// (german-nlp-group's ELECTRA), which transformers reads the same way.
    /// </summary>
    public static int ReadMaxLength(string directory)
    {
        var path = Path.Combine(directory, "tokenizer_config.json");
        var config = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path));
        var node = config?["model_max_length"] ?? config?["max_len"];
        return node?.GetValue<int>() ?? throw new InvalidDataException($"{path}: no model_max_length");
    }

    /// <summary>
    /// Per sentence, [words + 2 (with <paramref name="keepEndpoints"/>) or words, Hidden]: row w (+1) is word w's last
    /// piece; with endpoints, row 0 is [CLS] and the last row the piece after the last word ([SEP]).
    /// </summary>
    /// <param name="layerMix">A model's <c>bert_layer_mix.weight</c> (N weights): the mean of the last N layers plus their
    /// weighted sum. Null: Stanza's legacy mix, (h[-4] + h[-3] + h[-2]) / 4.</param>
    /// <param name="cache">If given, a sentence whose ids it holds isn't run again, and the others are kept in it (when they
    /// fit), for the next reader of the same call; only for a mix of <see cref="TransformerCache.Layers"/> layers.</param>
    public float[][] Embed(IReadOnlyList<IReadOnlyList<string>> sentences, bool keepEndpoints, float[]? layerMix, TransformerCache? cache = null,
        CancellationToken ct = default)
    {
        int h = Net.Config.Hidden, lastLayers = layerMix?.Length ?? 4;
        if (lastLayers != TransformerCache.Layers || cache?.Net != Net)
            cache = null;
        var inputs = sentences.Select(s => TransformerInput.Build(Tokenizer, s)).ToArray();
        var states = new float[inputs.Length][]; // per sentence [lastLayers, pieces, h]
        var owned = new bool[inputs.Length]; // rented by the encoder and not kept by the cache: returned once mixed
        bool rents = Net.RentsOutputs;
        for (int s = 0; s < inputs.Length && cache != null; s++)
            states[s] = cache.TryGet(inputs[s].Ids, out var cached) ? cached : null!;
        var short_ = Enumerable.Range(0, inputs.Length).Where(s => states[s] == null && inputs[s].Ids.Length <= MaxLength).ToList();
        for (int b = 0; b < short_.Count;)
        {
            int count = 0, pieces = 0;
            while (b + count < short_.Count && count < BatchSize && (count == 0 || pieces + inputs[short_[b + count]].Ids.Length <= BatchPieces))
                pieces += inputs[short_[b + count++]].Ids.Length;
            var batch = short_.GetRange(b, count);
            var encoded = Net.Encode([.. batch.Select(s => inputs[s].Ids)], lastLayers, ct);
            for (int i = 0; i < batch.Count; i++)
            {
                states[batch[i]] = encoded[i];
                owned[batch[i]] = cache?.TryAdd(inputs[batch[i]].Ids, encoded[i]) != true && rents;
            }

            b += count;
        }

        for (int s = 0; s < inputs.Length; s++)
        {
            if (states[s] != null)
                continue;
            states[s] = Windowed(inputs[s].Ids, lastLayers, ct);
            owned[s] = cache?.TryAdd(inputs[s].Ids, states[s]) != true && rents;
        }

        var pool = ArrayPool<float>.Shared;
        var result = new float[inputs.Length][];
        for (int s = 0; s < inputs.Length; s++)
        {
            var positions = inputs[s].Positions;
            int pieces = inputs[s].Ids.Length, first = keepEndpoints ? 0 : 1, rows = positions.Length - (keepEndpoints ? 0 : 2);
            var mixed = Mix(states[s], pieces, h, layerMix);
            if (owned[s])
                pool.Return(states[s]);
            result[s] = new float[rows * h];
            for (int r = 0; r < rows; r++)
                Array.Copy(mixed, positions[first + r] * h, result[s], r * h, h);
            pool.Return(mixed);
        }

        return result;
    }

    /// <summary>
    /// Encodes every reader's sentences of a call in one sweep over the layers (<see cref="ITransformerNet.EncodeJoint"/>,
    /// #169) and keeps the states in each reader's net's cache, so each reader's <see cref="Embed"/> then finds them: on a
    /// short call the transformer's weights are read once, not once per net (the frozen one and each LoRA one). Only when the
    /// net can, at least two readers read a 4-layer mix, and every distinct sentence fits one encoder call
    /// (<see cref="BatchPieces"/>), so the caches hold at most that much; otherwise nothing is done and each reader runs its
    /// own passes as before. The states are the ones each net gives alone, so nothing a reader sees changes.
    /// </summary>
    /// <param name="cacheOf">The call's cache of a net, made on first use.</param>
    /// <returns>Whether the sentences were encoded.</returns>
    public static bool Prefill(IReadOnlyList<TransformerReader> readers, Func<ITransformerNet, TransformerCache> cacheOf, CancellationToken ct = default)
    {
        readers = [.. readers.Where(r => (r.LayerMix?.Length ?? 4) == TransformerCache.Layers)];
        if (readers.Count < 2 || !readers[0].Embedder.Net.EncodesJointly)
            return false;

        // Per net, its readers' distinct id sequences that fit one window.
        var groups = new List<(ITransformerNet Net, List<int[]> Ids)>();
        var seen = new Dictionary<ITransformerNet, HashSet<int[]>>();
        int pieces = 0;
        foreach (var reader in readers)
        {
            var embedder = reader.Embedder;
            if (!seen.TryGetValue(embedder.Net, out var ids))
            {
                seen[embedder.Net] = ids = new HashSet<int[]>(TransformerCache.IdsComparer.Instance);
                groups.Add((embedder.Net, []));
            }

            var group = groups.First(g => g.Net == embedder.Net).Ids;
            foreach (var sentence in reader.Sentences)
            {
                var input = TransformerInput.Build(embedder.Tokenizer, sentence).Ids;
                if (input.Length > embedder.MaxLength || !ids.Add(input))
                    continue;
                group.Add(input);
                pieces += input.Length;
                if (pieces > BatchPieces)
                    return false;
            }
        }

        var encoded = readers[0].Embedder.Net.EncodeJoint([.. groups.Select(g => (g.Net, (IReadOnlyList<int[]>)g.Ids))], TransformerCache.Layers, ct);
        if (encoded == null)
            return false;
        for (int g = 0; g < groups.Count; g++)
        {
            var cache = cacheOf(groups[g].Net);
            for (int s = 0; s < encoded[g].Length; s++)
            {
                if (!cache.TryAdd(groups[g].Ids[s], encoded[g][s]) && groups[g].Net.RentsOutputs)
                    ArrayPool<float>.Shared.Return(encoded[g][s]);
            }
        }

        return true;
    }

    // build_cloned_features over MaxLength: windows of MaxLength pieces starting every sliceLen pieces (each with
    // positions from 0 and attention within the window), each later window's rows from prefixLen on.
    private float[] Windowed(int[] ids, int lastLayers, CancellationToken ct)
    {
        int max = MaxLength, h = Net.Config.Hidden, n = ids.Length;
        int sliceLen = Math.Max(max - 20, max / 2), prefixLen = max - sliceLen;
        var windows = new List<int[]>();
        for (int start = 0; ; start += sliceLen)
        {
            windows.Add(ids[start..Math.Min(n, start + max)]);
            if (n - start <= max)
                break;
        }

        var encoded = Net.Encode(windows, lastLayers, ct);
        var states = Net.RentsOutputs ? ArrayPool<float>.Shared.Rent(lastLayers * n * h) : new float[lastLayers * n * h];
        for (int k = 0; k < lastLayers; k++)
        {
            int row = 0;
            for (int w = 0; w < windows.Count; w++)
            {
                int len = windows[w].Length, skip = w == 0 ? 0 : prefixLen;
                Array.Copy(encoded[w], (((long)k * len) + skip) * h, states, (((long)k * n) + row) * h, (long)(len - skip) * h);
                row += len - skip;
            }
        }

        if (Net.RentsOutputs)
        {
            foreach (var window in encoded)
                ArrayPool<float>.Shared.Return(window);
        }

        return states;
    }

    // cloned_feature, elementwise in float32 in Python's order: sum(...) starts from 0 and adds left to right.
    // Rented: the caller returns it.
    private static float[] Mix(float[] states, int pieces, int h, float[]? weights)
    {
        int size = pieces * h, layers = weights?.Length ?? 4;
        var mixed = ArrayPool<float>.Shared.Rent(size);
        for (int i = 0; i < size; i++)
        {
            if (weights == null)
            {
                mixed[i] = (states[i] + states[size + i] + states[(2 * size) + i]) / 4;
                continue;
            }

            float uniform = 0, weighted = 0;
            for (int k = 0; k < layers; k++)
            {
                uniform += states[(k * size) + i];
                weighted += weights[k] * states[(k * size) + i];
            }

            mixed[i] = (uniform / layers) + weighted;
        }

        return mixed;
    }
}

/// <summary>
/// What a processor will send <see cref="TransformerEmbedder.Embed"/> in a call: its embedder (the pipeline's frozen one, or
/// its own with LoRA adapters), its layer mix and every sentence's words, as it will pass them
/// (<see cref="TransformerEmbedder.Prefill"/>).
/// </summary>
internal sealed record TransformerReader(TransformerEmbedder Embedder, float[]? LayerMix, IReadOnlyList<IReadOnlyList<string>> Sentences);
