using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Nn;

/// <summary>
/// <see cref="ITransformerNet"/> on TorchSharp: transformers' ElectraModel forward (embeddings, then per layer
/// self-attention with an additive padding mask, output dense + LayerNorm, intermediate GELU, output dense +
/// LayerNorm). Sentences are padded to the longest of each call, like Stanza's batches; padding rows are dropped.
/// </summary>
internal sealed class TransformerNet : ITransformerNet
{
    private sealed record Layer(Tensor Query, Tensor QueryBias, Tensor Key, Tensor KeyBias, Tensor Value, Tensor ValueBias, Tensor Out, Tensor OutBias,
        Tensor Ln1G, Tensor Ln1B, Tensor Up, Tensor UpBias, Tensor Down, Tensor DownBias, Tensor Ln2G, Tensor Ln2B);

    private readonly Tensor _words, _positions, _type0, _embG, _embB;
    private readonly Layer[] _layers;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public TransformerConfig Config { get; }

    public TransformerNet(TransformerConfig config, Checkpoint ckpt, string prefix = "electra.")
    {
        Config = config;
        var root = ckpt.Root;
        Tensor T(string name) => ckpt.ToTensor(root[prefix + name] ?? throw new InvalidDataException($"No tensor {prefix}{name}"));
        _words = T("embeddings.word_embeddings.weight");
        _positions = T("embeddings.position_embeddings.weight");
        using (var types = T("embeddings.token_type_embeddings.weight"))
            _type0 = types[0].clone();
        _embG = T("embeddings.LayerNorm.weight");
        _embB = T("embeddings.LayerNorm.bias");
        _layers = [.. Enumerable.Range(0, config.Layers).Select(l =>
        {
            string p = $"encoder.layer.{l}.";
            return new Layer(T(p + "attention.self.query.weight"), T(p + "attention.self.query.bias"), T(p + "attention.self.key.weight"),
                T(p + "attention.self.key.bias"), T(p + "attention.self.value.weight"), T(p + "attention.self.value.bias"),
                T(p + "attention.output.dense.weight"), T(p + "attention.output.dense.bias"), T(p + "attention.output.LayerNorm.weight"),
                T(p + "attention.output.LayerNorm.bias"), T(p + "intermediate.dense.weight"), T(p + "intermediate.dense.bias"),
                T(p + "output.dense.weight"), T(p + "output.dense.bias"), T(p + "output.LayerNorm.weight"), T(p + "output.LayerNorm.bias"));
        })];
    }

    /// <summary>Loads a transformer directory (<c>config.json</c>, <c>pytorch_model.bin</c>) on <see cref="Weights.Device"/>.</summary>
    public static TransformerNet Load(string directory) =>
        new(TransformerConfig.Load(Path.Combine(directory, "config.json")), Checkpoint.Load(Path.Combine(directory, "pytorch_model.bin")));

    public float[][] Encode(IReadOnlyList<int[]> sentences, int lastLayers, CancellationToken ct = default)
    {
        if (lastLayers < 1 || lastLayers > _layers.Length + 1)
            throw new ArgumentOutOfRangeException(nameof(lastLayers));
        int h = Config.Hidden, heads = Config.Heads, d = h / heads, batch = sentences.Count;
        var result = new float[batch][];
        int width = sentences.Count == 0 ? 0 : sentences.Max(s => s.Length);
        if (width > Config.MaxPositions)
            throw new ArgumentException($"A sentence has {width} pieces, over {Config.MaxPositions}");
        if (width == 0)
        {
            for (int s = 0; s < batch; s++)
                result[s] = [];
            return result;
        }

        var ids = new long[batch * width];
        var mask = new float[batch * width]; // transformers' additive mask: 0 for real pieces, the dtype's minimum for padding
        for (int s = 0; s < batch; s++)
        {
            for (int i = 0; i < width; i++)
            {
                ids[(s * width) + i] = i < sentences[s].Length ? sentences[s][i] : 0;
                mask[(s * width) + i] = i < sentences[s].Length ? 0 : float.MinValue;
            }
        }

        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var input = torch.tensor(ids, [batch, width], device: _device);
        var additive = torch.tensor(mask, [batch, 1, 1, width], device: _device);
        var x = _words[input].add(_positions[..width].unsqueeze(0), Scalars.One).add(_type0, Scalars.One);
        x = F.layer_norm(x, [h], _embG, _embB, Config.LayerNormEps);
        var kept = new List<Tensor>();
        if (_layers.Length + 1 - lastLayers <= 0)
            kept.Add(x);
        for (int l = 0; l < _layers.Length; l++)
        {
            ct.ThrowIfCancellationRequested();
            var layer = _layers[l];
            var previous = x;

            // A scope per layer: a batch's intermediates are hundreds of MB (the FFN's up projection alone is 4096 floats a
            // piece), so only the layer's output outlives it, and the previous one only if it is kept.
            using (NewDisposeScope())
            {
                Tensor Heads(Tensor t) => t.view(batch, width, heads, d).transpose(1, 2);
                var q = Heads(F.linear(x, layer.Query, layer.QueryBias));
                var k = Heads(F.linear(x, layer.Key, layer.KeyBias));
                var v = Heads(F.linear(x, layer.Value, layer.ValueBias));
                var attention = F.scaled_dot_product_attention(q, k, v, additive).transpose(1, 2).reshape(batch, width, h);
                var mid = F.layer_norm(F.linear(attention, layer.Out, layer.OutBias).add(x, Scalars.One), [h], layer.Ln1G, layer.Ln1B, Config.LayerNormEps);
                var up = F.gelu(F.linear(mid, layer.Up, layer.UpBias));
                x = F.layer_norm(F.linear(up, layer.Down, layer.DownBias).add(mid, Scalars.One), [h], layer.Ln2G, layer.Ln2B, Config.LayerNormEps)
                    .MoveToOuterDisposeScope();
            }

            if (!kept.Contains(previous))
                previous.Dispose();
            if (l + 1 >= _layers.Length + 1 - lastLayers)
                kept.Add(x);
        }

        var states = torch.stack(kept).cpu().ToArray<float>(); // [lastLayers, batch, width, h]
        for (int s = 0; s < batch; s++)
        {
            int n = sentences[s].Length;
            result[s] = new float[lastLayers * n * h];
            for (int k = 0; k < lastLayers; k++)
                Array.Copy(states, ((((long)k * batch) + s) * width) * h, result[s], (long)k * n * h, (long)n * h);
        }

        return result;
    }

    public void Dispose()
    {
        foreach (var t in new[] { _words, _positions, _type0, _embG, _embB })
            t.Dispose();
        foreach (var layer in _layers)
        {
            foreach (var t in new[] { layer.Query, layer.QueryBias, layer.Key, layer.KeyBias, layer.Value, layer.ValueBias, layer.Out, layer.OutBias,
                         layer.Ln1G, layer.Ln1B, layer.Up, layer.UpBias, layer.Down, layer.DownBias, layer.Ln2G, layer.Ln2B, })
            {
                t.Dispose();
            }
        }
    }
}
