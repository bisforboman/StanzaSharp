using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Pos;

/// <summary>A batch's tagger outputs per word, sentences in order: UPOS scores [words, upos] and each column's ids.</summary>
internal sealed record PosOutput(float[] UposScores, long[] Upos, long[] Xpos, long[][] Feats);

/// <summary>The tagger's network, per <see cref="Backend"/>: <c>PosNet</c> or <see cref="ManagedPosNet"/>.</summary>
internal interface IPosNet : IDisposable
{
    /// <summary>Embeddings, character features, the highway biLSTM and the three heads (XPOS and feats on the argmax UPOS).</summary>
    /// <param name="sentences">The batch's words after simplify_punct.</param>
    /// <param name="wordIds">Word vocab id of each word (lowercased), sentences in order.</param>
    /// <param name="pretrainIds">Pretrain id of each word (lowercased), sentences in order.</param>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: receives sentence i's charlm outputs under its key if not null.</param>
    /// <param name="ct">Checked after each charlm pass (or the character model), between LSTM layers and between the heads.</param>
    PosOutput Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] pretrainIds,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct);
}

/// <summary>
/// Managed twin of <c>PosNet</c> (<see cref="Backend.Managed"/>). It works on the packed rows only, like
/// <c>PosNet</c>. The XPOS and feats biaffine scorers read the UPOS embedding, which takes one of a few values,
/// so they are contracted with every UPOS embedding at load: each becomes a linear layer per UPOS, all UPOS stacked.
/// </summary>
internal sealed unsafe class ManagedPosNet : IPosNet
{
    private readonly Pretrain _pretrain;
    private readonly ManagedCharLanguageModel? _charlmForward, _charlmBackward;
    private readonly ManagedCharacterModel? _charModel;
    private readonly PackedMatrix? _transChar;
    private readonly float[] _wordEmb; // [words, wordDim]
    private readonly int _wordDim, _transformed, _inputSize, _biaff, _upos, _xpos;
    private readonly int[] _feats, _featOffsets;
    private readonly PackedMatrix _transPretrained, _hid, _xposClf, _featsClf;
    private readonly float[] _hidBias, _xposBias, _featsBias;
    private readonly float[] _uposClf, _uposBias; // [upos, biaff], [upos]: dot products in double (see Heads)
    private readonly ManagedHighwayLstm _lstm;

    public ManagedPosNet(Checkpoint ckpt, int upos, int xpos, int[] feats, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        (_upos, _xpos, _feats) = (upos, xpos, feats);
        _featOffsets = new int[feats.Length];
        for (int f = 1; f < feats.Length; f++)
            _featOffsets[f] = _featOffsets[f - 1] + feats[f - 1];
        int hidden = config["hidden_dim"]!.GetValue<int>();
        _biaff = config["deep_biaff_hidden_dim"]!.GetValue<int>();
        int compositeBiaff = config["composite_deep_biaff_hidden_dim"]!.GetValue<int>();
        int tagEmb = config["tag_emb_dim"]!.GetValue<int>();
        _transformed = config["transformed_dim"]!.GetValue<int>();
        _wordDim = config["word_emb_dim"]!.GetValue<int>();
        _inputSize = _wordDim + _transformed;
        if (_charlmForward != null)
            _inputSize += _charlmForward.HiddenDim + _charlmBackward!.HiddenDim;
        else
        {
            _charModel = new ManagedCharacterModel(ckpt, model, config, ckpt.Root["vocab"]!["char"]!, "charmodel.",
                bidirectional: config["char_bidirectional"]?.GetValue<bool>() == true, attention: true);
            _transChar = new PackedMatrix(T("trans_char.weight"), _transformed, _charModel.OutputDim);
            _inputSize += _transformed;
        }

        _wordEmb = T("word_emb.weight");
        _transPretrained = new PackedMatrix(T("trans_pretrained.weight"), _transformed, pretrain.Dim);
        _lstm = new ManagedHighwayLstm(ckpt, model, "taggerlstm", _inputSize, hidden, config["num_layers"]!.GetValue<int>());

        // upos_hid, tag_hid.xpos and tag_hid.feats all read the LSTM output: one GEMM.
        (_hid, _hidBias) = Linear([.. T("upos_hid.weight"), .. T("tag_hid.xpos.weight"), .. T("tag_hid.feats.weight")],
            [.. T("upos_hid.bias"), .. T("tag_hid.xpos.bias"), .. T("tag_hid.feats.bias")], 2 * _biaff + compositeBiaff, 2 * hidden);
        (_uposClf, _uposBias) = (T("upos_clf.weight"), T("upos_clf.bias"));
        var uposEmb = T("upos_emb.weight");
        (_xposClf, _xposBias) = BiaffinePerUpos([(T("tag_clf.xpos.W_bilin.weight"), T("tag_clf.xpos.W_bilin.bias"))], [xpos], uposEmb, upos, _biaff, tagEmb);
        (_featsClf, _featsBias) = BiaffinePerUpos(feats.Select((_, f) => (T($"tag_clf.feats.{f}.W_bilin.weight"), T($"tag_clf.feats.{f}.W_bilin.bias"))).ToArray(),
            feats, uposEmb, upos, compositeBiaff, tagEmb);
    }

    private static (PackedMatrix, float[]) Linear(float[] w, float[] b, int n, int k)
    {
        var packed = new PackedMatrix(w, n, k);
        var bias = new float[packed.PaddedN];
        b.CopyTo(bias, 0);
        return (packed, bias);
    }

    /// <summary>
    /// Biaffine scorers (<c>Bilinear(in1 + 1, in2 + 1, out)</c> over [x, 1] and [UPOS embedding, 1]) as one linear layer
    /// over x: output column u·Σout + offset(head) + k is head's score k given UPOS u. The constant 1 of x goes to the bias.
    /// </summary>
    private static (PackedMatrix, float[]) BiaffinePerUpos((float[] W, float[] B)[] heads, int[] outs, float[] uposEmb, int upos, int in1, int in2)
    {
        int total = outs.Sum();
        var w = new float[upos * total * in1];
        var bias = new float[upos * total];
        for (int u = 0; u < upos; u++)
            for (int h = 0, offset = 0; h < heads.Length; offset += outs[h], h++)
                for (int k = 0; k < outs[h]; k++)
                {
                    int row = u * total + offset + k;
                    for (int i = 0; i <= in1; i++)
                    {
                        // W[k, i, :] · [emb[u], 1]
                        int at = (k * (in1 + 1) + i) * (in2 + 1);
                        double s = heads[h].W[at + in2];
                        for (int j = 0; j < in2; j++)
                            s += (double)heads[h].W[at + j] * uposEmb[u * in2 + j];
                        if (i < in1)
                            w[row * in1 + i] = (float)s;
                        else
                            bias[row] = (float)(s + heads[h].B[k]);
                    }
                }
        return Linear(w, bias, upos * total, in1);
    }

    public PosOutput Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] pretrainIds,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct)
    {
        int batch = sentences.Count, inSize = _inputSize;
        var lengths = sentences.Select(s => (long)s.Count).ToArray();
        // pack_padded_sequence: sentences longest first; word (i, j) is packed row start[j] + rank[i], and word k of the
        // batch (sentences in order) is row packedRow[k].
        var order = Enumerable.Range(0, batch).OrderByDescending(i => lengths[i]).ToArray();
        var rank = new int[batch];
        for (int r = 0; r < batch; r++)
            rank[order[r]] = r;
        var batchSizes = PackedLstm.BatchSizes(lengths);
        var start = new int[batchSizes.Length];
        for (int t = 1; t < start.Length; t++)
            start[t] = start[t - 1] + batchSizes[t - 1];
        int rows = batchSizes.Sum();
        var packedRow = new int[rows];
        for (int i = 0, k = 0; i < batch; i++)
            for (int j = 0; j < lengths[i]; j++, k++)
                packedRow[k] = start[j] + rank[i];

        int ldHid = _hid.PaddedN, ldx = _xposClf.PaddedN, ldf = _featsClf.PaddedN;
        var x = ArrayPool<float>.Shared.Rent(rows * inSize);
        var scratch = ArrayPool<float>.Shared.Rent(rows * Math.Max(Math.Max(ldHid, ldx), ldf));
        // The LSTM output; before that, scratch for the pretrain vectors and the character model's output.
        var hidden = ArrayPool<float>.Shared.Rent(rows * Math.Max(_lstm.OutputSize, Math.Max(_pretrain.Dim, _charModel?.OutputDim ?? 0)));
        try
        {
            // Word embedding and trans_pretrained(pretrain vector): columns [0, wordDim) and [wordDim, wordDim + transformed).
            var vectors = _pretrain.CpuVectors();
            int dim = _pretrain.Dim, ldt = _transPretrained.PaddedN;
            for (int k = 0; k < rows; k++)
            {
                int r = packedRow[k];
                _wordEmb.AsSpan((int)wordIds[k] * _wordDim, _wordDim).CopyTo(x.AsSpan(r * inSize));
                vectors.Slice((int)pretrainIds[k] * dim, dim).CopyTo(hidden.AsSpan(r * dim)); // hidden as scratch
            }
            fixed (float* ph = hidden, ps = scratch)
                Gemm.Run(ph, rows, dim, _transPretrained, null, ps, ldt, ct);
            for (int r = 0; r < rows; r++)
                scratch.AsSpan(r * ldt, _transformed).CopyTo(x.AsSpan(r * inSize + _wordDim));

            int charCol = _wordDim + _transformed;
            if (_charModel != null)
            {
                // The character model's vector per word (packed row order), then trans_char.
                var words = new int[rows][];
                for (int i = 0, k = 0; i < batch; i++)
                    for (int j = 0; j < lengths[i]; j++, k++)
                        words[packedRow[k]] = _charModel.CharIds(sentences[i][j]);
                int ldc = _transChar!.PaddedN;
                fixed (float* ph = hidden, ps = scratch)
                {
                    _charModel.Forward(words, ph, _charModel.OutputDim, ct);
                    Gemm.Run(ph, rows, _charModel.OutputDim, _transChar, null, ps, ldc, ct);
                }
                for (int r = 0; r < rows; r++)
                    scratch.AsSpan(r * ldc, _transformed).CopyTo(x.AsSpan(r * inSize + charCol));
            }
            else
                CharlmFeatures(sentences, packedRow, x, charCol, charlms, cacheKeys, ct);
            ct.ThrowIfCancellationRequested();

            _lstm.Forward(x.AsSpan(0, rows * inSize), lengths, hidden.AsSpan(0, rows * _lstm.OutputSize), ct);
            ct.ThrowIfCancellationRequested();
            return Heads(hidden, rows, packedRow, scratch, ct);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(scratch);
            ArrayPool<float>.Shared.Return(hidden);
        }
    }

    /// <summary>Both charlms over the batch into each word's input row; sentences with a cache key are also kept in the cache.</summary>
    private void CharlmFeatures(IReadOnlyList<IReadOnlyList<string>> sentences, int[] packedRow, float[] x, int col,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct)
    {
        int dim = _charlmForward!.HiddenDim, words = packedRow.Length;
        var forward = ArrayPool<float>.Shared.Rent(words * dim);
        var backward = ArrayPool<float>.Shared.Rent(words * dim);
        try
        {
            _charlmForward.BuildCharRepresentation(sentences, forward, dim, ct);
            ct.ThrowIfCancellationRequested();
            _charlmBackward!.BuildCharRepresentation(sentences, backward, dim, ct);
            for (int k = 0; k < words; k++)
            {
                int r = packedRow[k] * _inputSize + col;
                forward.AsSpan(k * dim, dim).CopyTo(x.AsSpan(r));
                backward.AsSpan(k * dim, dim).CopyTo(x.AsSpan(r + dim));
            }
            if (cacheKeys != null)
                for (int i = 0, k = 0; i < sentences.Count; k += sentences[i].Count, i++)
                {
                    int n = sentences[i].Count;
                    if (cacheKeys[i] is { } key && charlms!.HasRoom(n))
                        charlms.TryAdd(key, forward.AsSpan(k * dim, n * dim).ToArray(), backward.AsSpan(k * dim, n * dim).ToArray(), n);
                }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(forward);
            ArrayPool<float>.Shared.Return(backward);
        }
    }

    /// <summary>The UPOS MLP, then the XPOS and feats scorers given the argmax UPOS; outputs in sentence order.</summary>
    private PosOutput Heads(float[] hidden, int rows, int[] packedRow, float[] scratch, CancellationToken ct)
    {
        int ldHid = _hid.PaddedN;
        var hid = ArrayPool<float>.Shared.Rent(rows * ldHid);
        var uposScores = new float[rows * _upos];
        var upos = new long[rows];
        var xpos = new long[rows];
        var feats = _feats.Select(_ => new long[rows]).ToArray();
        try
        {
            fixed (float* ph = hidden, pd = hid, ps = scratch, pb = _hidBias, pxb = _xposBias, pfb = _featsBias)
            {
                Gemm.Run(ph, rows, _lstm.OutputSize, _hid, pb, pd, ldHid, ct);
                for (long n = 0, end = (long)rows * ldHid; n < end; n++)
                    pd[n] = Math.Max(pd[n], 0f);

                // upos_clf accumulates in double: its 400-term float sums of large scores (|logit| up to ~150) drifted up
                // to 1.07e-4 from Stanza's (Scalar path), over the 1e-4 tolerance; in double, 3.1e-5 to 6.9e-5. It is
                // 21 outputs per word, so this costs next to nothing.
                nint hidBase = (nint)pd;
                ManagedThreads.For((rows + 63) / 64, task =>
                {
                    var h = (float*)hidBase;
                    for (int k = task * 64, end = Math.Min(rows, k + 64); k < end; k++)
                    {
                        var x = new ReadOnlySpan<float>(h + (long)packedRow[k] * ldHid, _biaff);
                        var s = uposScores.AsSpan(k * _upos, _upos);
                        for (int c = 0; c < _upos; c++)
                        {
                            var w = _uposClf.AsSpan(c * _biaff, _biaff);
                            double sum = _uposBias[c];
                            for (int a = 0; a < w.Length; a++)
                                sum += (double)w[a] * x[a];
                            s[c] = (float)sum;
                        }
                        upos[k] = ArgMax(s);
                    }
                });

                int ldx = _xposClf.PaddedN;
                Gemm.Run(pd + _biaff, rows, ldHid, _xposClf, pxb, ps, ldx, ct);
                for (int k = 0; k < rows; k++)
                    xpos[k] = ArgMax(scratch.AsSpan(packedRow[k] * ldx + (int)upos[k] * _xpos, _xpos));
                ct.ThrowIfCancellationRequested();

                int ldf = _featsClf.PaddedN, total = _feats.Sum();
                Gemm.Run(pd + 2 * _biaff, rows, ldHid, _featsClf, pfb, ps, ldf, ct);
                for (int k = 0; k < rows; k++)
                    for (int f = 0; f < _feats.Length; f++)
                        feats[f][k] = ArgMax(scratch.AsSpan(packedRow[k] * ldf + (int)upos[k] * total + _featOffsets[f], _feats[f]));
            }
            return new(uposScores, upos, xpos, feats);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hid);
        }
    }

    /// <summary>torch.argmax: the first index of the maximum.</summary>
    private static int ArgMax(ReadOnlySpan<float> s)
    {
        int best = 0;
        for (int i = 1; i < s.Length; i++)
            if (s[i] > s[best])
                best = i;
        return best;
    }

    public void Dispose()
    {
    }
}
