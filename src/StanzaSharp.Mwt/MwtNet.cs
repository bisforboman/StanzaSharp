using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Mwt;

/// <summary>MWT's character classifier network, per <see cref="Backend"/>: <see cref="MwtNet"/> or <see cref="ManagedMwtNet"/>.</summary>
internal interface IMwtNet : IDisposable
{
    /// <param name="ids">[rows, width] character ids (<c>&lt;SOS&gt;</c> chars <c>&lt;EOS&gt;</c>, padded).</param>
    /// <param name="lengths">Each row's length; the encoder is packed at it.</param>
    /// <returns>[rows, width, 2] logits (keep, cut before this character), row-major.</returns>
    float[] Forward(long[] ids, int rows, int width, long[] lengths);
}

/// <summary>stanza/models/mwt/character_classifier.py on TorchSharp: embedding, biLSTM encoder, Linear → ReLU → Linear.</summary>
internal sealed class MwtNet : IMwtNet
{
    private readonly Embedding _embedding;
    private readonly LSTM _encoder;
    private readonly Linear _hidden, _output;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public MwtNet(Checkpoint ckpt, JsonNode config, int padId)
    {
        var model = ckpt.Root["model"]!;
        int emb = config["emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["num_layers"]!.GetValue<int>();
        _embedding = nn.Embedding(config["vocab_size"]!.GetValue<int>(), emb, padding_idx: padId).LoadFrom(ckpt, model, "embedding.");
        _encoder = nn.LSTM(emb, hidden / 2, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "encoder.");
        _hidden = nn.Linear(hidden, hidden).LoadFrom(ckpt, model, "output_layer.0.");
        _output = nn.Linear(hidden, 2).LoadFrom(ckpt, model, "output_layer.2.");
    }

    public float[] Forward(long[] ids, int rows, int width, long[] lengths)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var src = torch.tensor(ids, [rows, width], device: _device);
        var encoded = Rnn.RunPacked(_encoder, _embedding.forward(src), lengths);
        return _output.forward(nn.functional.relu(_hidden.forward(encoded))).ToArray<float>();
    }

    public void Dispose()
    {
        _embedding.Dispose();
        _encoder.Dispose();
        _hidden.Dispose();
        _output.Dispose();
    }
}

/// <summary>Managed twin of <see cref="MwtNet"/> (<see cref="Backend.Managed"/>).</summary>
internal sealed class ManagedMwtNet : IMwtNet
{
    private readonly float[] _embeddings; // [vocab, emb]
    private readonly int _emb;
    private readonly ManagedLstm _encoder;
    private readonly PackedMatrix _hidden, _output;
    private readonly float[] _hiddenBias, _outputBias;

    public ManagedMwtNet(Checkpoint ckpt, JsonNode config)
    {
        var model = ckpt.Root["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        _emb = config["emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        _embeddings = T("embedding.weight");
        _encoder = new ManagedLstm(ckpt, model, "encoder.", _emb, hidden / 2, config["num_layers"]!.GetValue<int>(), bidirectional: true);
        (_hidden, _hiddenBias) = Linear(T("output_layer.0.weight"), T("output_layer.0.bias"), hidden, hidden);
        (_output, _outputBias) = Linear(T("output_layer.2.weight"), T("output_layer.2.bias"), 2, hidden);
    }

    private static (PackedMatrix, float[]) Linear(float[] w, float[] b, int n, int k)
    {
        var packed = new PackedMatrix(w, n, k);
        var bias = new float[packed.PaddedN];
        b.CopyTo(bias, 0);
        return (packed, bias);
    }

    public float[] Forward(long[] ids, int rows, int width, long[] lengths)
    {
        int n = rows * width, h = _encoder.OutputSize;
        var x = ArrayPool<float>.Shared.Rent(n * _emb);
        var encoded = ArrayPool<float>.Shared.Rent(n * h);
        try
        {
            for (int i = 0; i < n; i++)
                _embeddings.AsSpan((int)ids[i] * _emb, _emb).CopyTo(x.AsSpan(i * _emb));
            _encoder.ForwardPadded(x, rows, width, lengths, encoded);
            var hidden = Gemm.Run(encoded, n, _hidden, _hiddenBias); // [n, PaddedN]
            int k = _hidden.N;
            var relu = new float[n * k];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < k; j++)
                    relu[i * k + j] = Math.Max(hidden[i * _hidden.PaddedN + j], 0f);
            var output = Gemm.Run(relu, n, _output, _outputBias);
            var logits = new float[n * 2];
            for (int i = 0; i < n; i++)
                output.AsSpan(i * _output.PaddedN, 2).CopyTo(logits.AsSpan(i * 2));
            return logits;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(encoded);
        }
    }

    public void Dispose()
    {
    }
}
