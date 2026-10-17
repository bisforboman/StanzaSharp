namespace StanzaSharp.Nn;

/// <summary>
/// Stanza's extract_bert_embeddings for a BERT-style transformer (stanza/models/common/bert_embedding.py,
/// extract_base_embeddings and build_cloned_features): each sentence's words through <see cref="TransformerInput"/>, the
/// encoder, a mix of the last layers, then one vector per word (its last piece), optionally with the [CLS] and [SEP]
/// positions around them.
/// </summary>
/// <param name="maxLength">The tokenizer's <c>model_max_length</c> (tokenizer_config.json, 512 for ELECTRA): longer inputs
/// run in windows, as Stanza's build_cloned_features does.</param>
internal sealed class TransformerEmbedder(ITransformerNet net, WordPiece tokenizer, int maxLength)
{
    /// <summary>Sentences per encoder call, Stanza's 128 (the managed results don't depend on it).</summary>
    public const int BatchSize = 128;

    /// <summary>
    /// Pieces per encoder call at most (a sentence longer than this runs alone): the managed net's buffers grow with the
    /// pieces (the intermediate layer alone is pieces × 4096 floats), and 128 sentences of 512 pieces would rent 1.8 GB.
    /// </summary>
    public const int BatchPieces = 8192;

    public ITransformerNet Net { get; } = net;

    public WordPiece Tokenizer { get; } = tokenizer;

    public int MaxLength { get; } = maxLength <= net.Config.MaxPositions
        ? maxLength
        : throw new ArgumentException($"model_max_length {maxLength} is over the model's {net.Config.MaxPositions} positions", nameof(maxLength));

    /// <summary>Reads <c>model_max_length</c> from a transformer directory's <c>tokenizer_config.json</c>.</summary>
    public static int ReadMaxLength(string directory)
    {
        var path = Path.Combine(directory, "tokenizer_config.json");
        var node = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path))?["model_max_length"];
        return node?.GetValue<int>() ?? throw new InvalidDataException($"{path}: no model_max_length");
    }

    /// <summary>
    /// Per sentence, [words + 2 (with <paramref name="keepEndpoints"/>) or words, Hidden]: row w (+1) is word w's last
    /// piece; with endpoints, row 0 is [CLS] and the last row the piece after the last word ([SEP]).
    /// </summary>
    /// <param name="layerMix">A model's <c>bert_layer_mix.weight</c> (N weights): the mean of the last N layers plus their
    /// weighted sum. Null: Stanza's legacy mix, (h[-4] + h[-3] + h[-2]) / 4.</param>
    public float[][] Embed(IReadOnlyList<IReadOnlyList<string>> sentences, bool keepEndpoints, float[]? layerMix, CancellationToken ct = default)
    {
        int h = Net.Config.Hidden, lastLayers = layerMix?.Length ?? 4;
        var inputs = sentences.Select(s => TransformerInput.Build(Tokenizer, s)).ToArray();
        var states = new float[inputs.Length][]; // per sentence [lastLayers, pieces, h]
        var short_ = Enumerable.Range(0, inputs.Length).Where(s => inputs[s].Ids.Length <= MaxLength).ToList();
        for (int b = 0; b < short_.Count;)
        {
            int count = 0, pieces = 0;
            while (b + count < short_.Count && count < BatchSize && (count == 0 || pieces + inputs[short_[b + count]].Ids.Length <= BatchPieces))
                pieces += inputs[short_[b + count++]].Ids.Length;
            var batch = short_.GetRange(b, count);
            var encoded = Net.Encode([.. batch.Select(s => inputs[s].Ids)], lastLayers, ct);
            for (int i = 0; i < batch.Count; i++)
                states[batch[i]] = encoded[i];
            b += count;
        }

        for (int s = 0; s < inputs.Length; s++)
            states[s] ??= Windowed(inputs[s].Ids, lastLayers, ct);

        var result = new float[inputs.Length][];
        for (int s = 0; s < inputs.Length; s++)
        {
            var positions = inputs[s].Positions;
            int pieces = inputs[s].Ids.Length, first = keepEndpoints ? 0 : 1, rows = positions.Length - (keepEndpoints ? 0 : 2);
            var mixed = Mix(states[s], pieces, h, layerMix);
            result[s] = new float[rows * h];
            for (int r = 0; r < rows; r++)
                Array.Copy(mixed, positions[first + r] * h, result[s], r * h, h);
        }

        return result;
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
        var states = new float[lastLayers * n * h];
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

        return states;
    }

    // cloned_feature, elementwise in float32 in Python's order: sum(...) starts from 0 and adds left to right.
    private static float[] Mix(float[] states, int pieces, int h, float[]? weights)
    {
        int size = pieces * h, layers = weights?.Length ?? 4;
        var mixed = new float[size];
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
