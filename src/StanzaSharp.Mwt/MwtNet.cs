using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Mwt;

/// <summary>MWT's character classifier network, per <see cref="Backend"/>: <c>MwtNet</c> or <see cref="ManagedMwtNet"/>.</summary>
internal interface IMwtNet : IDisposable
{
    /// <param name="ids">[rows, width] character ids (<c>&lt;SOS&gt;</c> chars <c>&lt;EOS&gt;</c>, padded).</param>
    /// <param name="lengths">Each row's length; the encoder is packed at it.</param>
    /// <returns>[rows, width, 2] logits (keep, cut before this character), row-major.</returns>
    float[] Forward(long[] ids, int rows, int width, long[] lengths);
}

/// <summary>Managed twin of <c>MwtNet</c> (<see cref="Backend.Managed"/>).</summary>
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
