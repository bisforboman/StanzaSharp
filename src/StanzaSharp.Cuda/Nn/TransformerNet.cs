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

    // A LoRA adapter: A [rank, in], B [out, rank] (LoraWeights).
    private sealed record Adapter(Tensor A, Tensor B);

    // Per layer, the adapters on query, value, the attention output, the intermediate and the output dense (null: none).
    private sealed record LayerAdapters(Adapter? Query, Adapter? Value, Adapter? Out, Adapter? Up, Adapter? Down);

    private readonly Tensor _words, _positions, _type0, _embG, _embB;
    private readonly Layer[] _layers;
    private readonly LayerAdapters[]? _adapters;
    private readonly Scalar? _scaling;
    private readonly bool _sharesWeights; // a WithLora net: the weights are the base net's
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

    // WithLora: the base's weights, shared (not disposed by this net), plus adapters.
    private TransformerNet(TransformerNet base_, LayerAdapters[] adapters, float scaling)
    {
        Config = base_.Config;
        (_words, _positions, _type0, _embG, _embB, _layers, _device) = (base_._words, base_._positions, base_._type0, base_._embG, base_._embB, base_._layers, base_._device);
        (_adapters, _scaling, _sharesWeights) = (adapters, scaling.ToScalar(), true);
    }

    public ITransformerNet WithLora(LoraWeights lora)
    {
        int h = Config.Hidden, inter = Config.Intermediate;
        Adapter? Make(int layer, string module, int input, int output)
        {
            if (lora.Of(layer, module) is not { } w)
                return null;
            if (w.A.Length != w.Rank * input || w.B.Length != output * w.Rank)
                throw new InvalidDataException($"LoRA adapter on layer {layer} {module} has the wrong shape");
            return new Adapter(torch.tensor(w.A, [w.Rank, input], device: _device), torch.tensor(w.B, [output, w.Rank], device: _device));
        }

        var adapters = Enumerable.Range(0, _layers.Length).Select(l => new LayerAdapters(Make(l, "attention.self.query", h, h),
            Make(l, "attention.self.value", h, h), Make(l, "attention.output.dense", h, h), Make(l, "intermediate.dense", h, inter),
            Make(l, "output.dense", inter, h))).ToArray();
        return new TransformerNet(this, adapters, lora.Scaling);
    }

    // peft's LoRA Linear in eval mode: base + B(A(x))·scaling.
    private Tensor Linear(Tensor x, Tensor weight, Tensor bias, Adapter? adapter)
    {
        var y = F.linear(x, weight, bias);
        return adapter == null ? y : y.add(F.linear(F.linear(x, adapter.A), adapter.B).mul(_scaling!), Scalars.One);
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
                var adapters = _adapters?[l];
                var q = Heads(Linear(x, layer.Query, layer.QueryBias, adapters?.Query));
                var k = Heads(F.linear(x, layer.Key, layer.KeyBias));
                var v = Heads(Linear(x, layer.Value, layer.ValueBias, adapters?.Value));
                var attention = F.scaled_dot_product_attention(q, k, v, additive).transpose(1, 2).reshape(batch, width, h);
                var mid = F.layer_norm(Linear(attention, layer.Out, layer.OutBias, adapters?.Out).add(x, Scalars.One), [h], layer.Ln1G, layer.Ln1B,
                    Config.LayerNormEps);
                var up = F.gelu(Linear(mid, layer.Up, layer.UpBias, adapters?.Up));
                x = F.layer_norm(Linear(up, layer.Down, layer.DownBias, adapters?.Down).add(mid, Scalars.One), [h], layer.Ln2G, layer.Ln2B, Config.LayerNormEps)
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
        foreach (var a in _adapters ?? [])
        {
            foreach (var adapter in new[] { a.Query, a.Value, a.Out, a.Up, a.Down }.OfType<Adapter>())
            {
                adapter.A.Dispose();
                adapter.B.Dispose();
            }
        }

        _scaling?.Dispose();
        if (_sharesWeights)
            return;
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
