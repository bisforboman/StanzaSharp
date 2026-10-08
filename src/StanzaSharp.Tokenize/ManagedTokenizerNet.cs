using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Tokenize;

/// <summary>
/// Managed twin of <see cref="TokenizerNet"/> (<see cref="Backend.Managed"/>): the same layers on
/// <see cref="ManagedLstm"/> and <see cref="Gemm"/>, with the three heads of each level in one small GEMM.
/// </summary>
internal sealed class ManagedTokenizerNet : ITokenizerNet
{
    private const int Classes = 5;

    private readonly float[] _embeddings; // [vocab, emb]
    private readonly int _emb;
    private readonly ManagedLstm _rnn, _rnn2;
    private readonly PackedMatrix _clf, _clf2; // rows tok, sent, mwt
    private readonly float[] _clfBias;
    private readonly float _hierInvTemp;

    public ManagedTokenizerNet(Checkpoint ckpt)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        // TokenizerNet checks the options; this twin is only built for checkpoints it accepts.
        int hidden = config["hidden_dim"]!.GetValue<int>();
        _emb = config["emb_dim"]!.GetValue<int>();
        _hierInvTemp = (float)config["hier_invtemp"]!.GetValue<double>();
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));

        _embeddings = T("embeddings.weight");
        _rnn = new ManagedLstm(ckpt, model, "rnn.", _emb + config["feat_dim"]!.GetValue<int>(), hidden,
            config["rnn_layers"]!.GetValue<int>(), bidirectional: true);
        _rnn2 = new ManagedLstm(ckpt, model, "rnn2.", 2 * hidden, hidden, 1, bidirectional: true);
        string[] heads = ["tok_clf", "sent_clf", "mwt_clf"];
        _clf = new PackedMatrix(heads.SelectMany(h => T(h + ".weight")).ToArray(), 3, 2 * hidden);
        _clfBias = new float[_clf.PaddedN];
        heads.SelectMany(h => T(h + ".bias")).ToArray().CopyTo(_clfBias, 0);
        _clf2 = new PackedMatrix(heads.SelectMany(h => T(h + "2.weight")).ToArray(), 3, 2 * hidden);
    }

    public float[] Forward(long[] ids, float[] feats, int rows, int width, long[] lengths, CancellationToken ct)
    {
        int n = rows * width, featDim = feats.Length / n, inDim = _emb + featDim, h2 = _rnn.OutputSize;
        var x = ArrayPool<float>.Shared.Rent(n * inDim);
        var inp = ArrayPool<float>.Shared.Rent(n * h2);
        var inp2 = ArrayPool<float>.Shared.Rent(n * h2);
        var hid2 = ArrayPool<float>.Shared.Rent(n * h2);
        try
        {
            for (int i = 0; i < n; i++)
            {
                _embeddings.AsSpan((int)ids[i] * _emb, _emb).CopyTo(x.AsSpan(i * inDim));
                feats.AsSpan(i * featDim, featDim).CopyTo(x.AsSpan(i * inDim + _emb));
            }
            _rnn.ForwardPadded(x, rows, width, lengths, inp, ct);
            var level1 = Gemm.Run(inp, n, _clf, _clfBias); // [n, 16]: tok0, sent0, mwt0

            // inp * (1 - sigmoid(-tok0 * hier_invtemp)), in TokenizerNet's order of operations.
            for (int i = 0; i < n; i++)
            {
                float keep = 1f - Sigmoid(-level1[i * _clf.PaddedN] * _hierInvTemp);
                for (int j = 0; j < h2; j++)
                    inp2[i * h2 + j] = inp[i * h2 + j] * keep;
            }
            _rnn2.ForwardPadded(inp2, rows, width, lengths, hid2, ct);
            var level2 = Gemm.Run(hid2, n, _clf2, null);

            var pred = new float[n * Classes];
            for (int i = 0; i < n; i++)
            {
                int a = i * _clf.PaddedN;
                float tok0 = level1[a] + level2[a], sent0 = level1[a + 1] + level2[a + 1], mwt0 = level1[a + 2] + level2[a + 2];
                float tok = LogSigmoid(tok0), sent = LogSigmoid(sent0), nonsent = LogSigmoid(-sent0);
                float mwt = LogSigmoid(mwt0), nonmwt = LogSigmoid(-mwt0);
                var o = pred.AsSpan(i * Classes, Classes);
                o[0] = LogSigmoid(-tok0);
                o[1] = tok + nonsent + nonmwt;
                o[2] = tok + sent + nonmwt;
                o[3] = tok + nonsent + mwt;
                o[4] = tok + sent + mwt;
            }
            return pred;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(inp);
            ArrayPool<float>.Shared.Return(inp2);
            ArrayPool<float>.Shared.Return(hid2);
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>log σ(x) = min(x, 0) − log1p(e^−|x|), stable for large |x| (as torch's log_sigmoid).</summary>
    private static float LogSigmoid(float x) => (float)(Math.Min(x, 0) - Math.Log(1 + Math.Exp(-Math.Abs(x))));

    public void Dispose()
    {
    }
}
