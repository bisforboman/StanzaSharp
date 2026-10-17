using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Ner;

/// <summary>NER's network, per <see cref="Backend"/>: <c>NerNet</c> or <see cref="ManagedNerNet"/>.</summary>
internal interface INerNet : IDisposable
{
    /// <summary>Embeddings (pretrain + delta, if the model has one), character features, input_transform, biLSTM and the tag set's linear layer.</summary>
    /// <param name="sentences">The batch's token texts.</param>
    /// <param name="wordIds">[batch, width] pretrain ids (lowercased tokens), padded.</param>
    /// <param name="deltaIds">[batch, width] delta ids, padded; unused without a delta embedding.</param>
    /// <param name="logits">Gets the [batch, width, tags] emission scores, row-major; rows past a sentence's end are unspecified.</param>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: charlm outputs to reuse; row i's key, if not null, is looked up.</param>
    /// <param name="transformer">A <c>_electra-large</c> model's transformer vectors, per sentence [tokens, hidden]: the input
    /// columns after the word vector, before the character features.</param>
    void Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] deltaIds, int width, Span<float> logits,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct, float[][]? transformer = null);
}

/// <summary>
/// Managed twin of <c>NerNet</c> (<see cref="Backend.Managed"/>). It works on the packed rows only: each token's
/// input row is built at its packed position, so input_transform, the biLSTM and the tag layer never see padding.
/// </summary>
internal sealed unsafe class ManagedNerNet : INerNet
{
    private readonly Pretrain _pretrain;
    private readonly ManagedCharLanguageModel? _charlmForward, _charlmBackward;
    private readonly ManagedCharacterModel? _charModel;
    private readonly float[]? _deltaEmb; // [delta vocab, wordDim], if any
    private readonly int _wordDim, _transformerDim, _charCol, _inputSize, _tags;
    private readonly PackedMatrix _inputTransform, _tagClf;
    private readonly float[] _inputBias, _tagBias;
    private readonly ManagedLstm _lstm;

    /// <param name="transformerDim">The transformer's hidden size for a <c>_electra-large</c> model, else 0.</param>
    public ManagedNerNet(Checkpoint ckpt, int tags, int tagset, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward,
        int transformerDim = 0)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        if (charlmForward == null)
            _charModel = new ManagedCharacterModel(ckpt, model, config, ckpt.Root["vocab"]!["char"]!, "charmodel.", bidirectional: true, attention: false);

        _wordDim = config["word_emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["num_layers"]!.GetValue<int>();

        // model.py: the word vector, the transformer's (if any), then the character features.
        (_transformerDim, _charCol) = (transformerDim, _wordDim + transformerDim);
        _inputSize = _charCol + (_charModel?.OutputDim ?? _charlmForward!.HiddenDim + _charlmBackward!.HiddenDim);
        _tags = tags;
        _deltaEmb = model["delta_emb.weight"] != null ? T("delta_emb.weight") : null;
        (_inputTransform, _inputBias) = Linear(T("input_transform.weight"), T("input_transform.bias"), _inputSize, _inputSize);
        _lstm = new ManagedLstm(ckpt, model, "taggerlstm.lstm.", _inputSize, hidden, layers, bidirectional: true, T("taggerlstm_h_init"), T("taggerlstm_c_init"));
        var head = NerTagger.HeadPrefix(ckpt, "tag_clf", tagset);
        (_tagClf, _tagBias) = Linear(T(head + "weight"), T(head + "bias"), tags, 2 * hidden);
    }

    private static (PackedMatrix, float[]) Linear(float[] w, float[] b, int n, int k)
    {
        var packed = new PackedMatrix(w, n, k);
        var bias = new float[packed.PaddedN];
        b.CopyTo(bias, 0);
        return (packed, bias);
    }

    public void Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] deltaIds, int width, Span<float> logits,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct, float[][]? transformer = null)
    {
        int batch = sentences.Count, inSize = _inputSize, ldt = _inputTransform.PaddedN, ldc = _tagClf.PaddedN;
        var lengths = sentences.Select(s => (long)s.Count).ToArray();

        // pack_padded_sequence: sentences longest first; token (i, j) is packed row start[j] + rank[i].
        var order = Enumerable.Range(0, batch).OrderByDescending(i => lengths[i]).ToArray();
        var rank = new int[batch];
        for (int r = 0; r < batch; r++)
            rank[order[r]] = r;
        var batchSizes = PackedLstm.BatchSizes(lengths);
        var start = new int[batchSizes.Length];
        for (int t = 1; t < start.Length; t++)
            start[t] = start[t - 1] + batchSizes[t - 1];
        int rows = batchSizes.Sum();
        int Row(int i, int j) => start[j] + rank[i];

        var x = ArrayPool<float>.Shared.Rent(rows * inSize);
        var transformed = ArrayPool<float>.Shared.Rent(rows * ldt);
        var hidden = ArrayPool<float>.Shared.Rent(rows * _lstm.OutputSize);
        var scores = ArrayPool<float>.Shared.Rent(rows * ldc);
        try
        {
            // Word embeddings: pretrain + delta (if any).
            var vectors = _pretrain.CpuVectors();
            for (int i = 0; i < batch; i++)
            {
                for (int j = 0; j < lengths[i]; j++)
                {
                    var dst = x.AsSpan(Row(i, j) * inSize, _wordDim);
                    var pre = vectors.Slice((int)wordIds[i * width + j] * _wordDim, _wordDim);
                    if (_deltaEmb == null)
                    {
                        pre.CopyTo(dst);
                        continue;
                    }

                    var delta = _deltaEmb.AsSpan((int)deltaIds[i * width + j] * _wordDim, _wordDim);
                    for (int k = 0; k < _wordDim; k++)
                        dst[k] = pre[k] + delta[k];
                }
            }

            for (int i = 0; i < batch && transformer != null; i++)
            {
                for (int j = 0; j < lengths[i]; j++)
                    transformer[i].AsSpan(j * _transformerDim, _transformerDim).CopyTo(x.AsSpan(Row(i, j) * inSize + _wordDim));
            }

            fixed (float* px = x)
            {
                if (_charModel != null)
                {
                    // Words listed in packed row order, so each vector lands in its row.
                    var words = new int[rows][];
                    for (int i = 0; i < batch; i++)
                    {
                        for (int j = 0; j < lengths[i]; j++)
                            words[Row(i, j)] = _charModel.CharIds(sentences[i][j]);
                    }

                    _charModel.Forward(words, px + _charCol, inSize, ct);
                }
                else
                {
                    CharlmFeatures(sentences, charlms, cacheKeys, x, Row, ct);
                }

                fixed (float* pt = transformed, pib = _inputBias)
                    Gemm.Run(px, rows, inSize, _inputTransform, pib, pt, ldt, ct);
            }

            _lstm.ForwardPacked(transformed, ldt, batchSizes, hidden, ct);
            fixed (float* ph = hidden, ps = scores, ptb = _tagBias)
                Gemm.Run(ph, rows, _lstm.OutputSize, _tagClf, ptb, ps, ldc, ct);

            for (int i = 0; i < batch; i++)
            {
                for (int j = 0; j < lengths[i]; j++)
                    scores.AsSpan(Row(i, j) * ldc, _tags).CopyTo(logits[((i * width + j) * _tags)..]);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(transformed);
            ArrayPool<float>.Shared.Return(hidden);
            ArrayPool<float>.Shared.Return(scores);
        }
    }

    /// <summary>The forward and backward charlm columns of each token's input row: from the cache where it has them, else computed.</summary>
    private void CharlmFeatures(IReadOnlyList<IReadOnlyList<string>> sentences, CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys,
        float[] x, Func<int, int, int> row, CancellationToken ct)
    {
        int dim = _charlmForward!.HiddenDim, forwardCol = _charCol, backwardCol = _charCol + dim;
        void Put(int i, ReadOnlySpan<float> forward, ReadOnlySpan<float> backward)
        {
            for (int j = 0; j < sentences[i].Count; j++)
            {
                int r = row(i, j) * _inputSize;
                forward.Slice(j * dim, dim).CopyTo(x.AsSpan(r + forwardCol, dim));
                backward.Slice(j * dim, dim).CopyTo(x.AsSpan(r + backwardCol, dim));
            }
        }

        var missing = new List<int>();
        for (int i = 0; i < sentences.Count; i++)
        {
            if (cacheKeys?[i] is { } key && charlms!.TryGetArrays(key, out var reps))
                Put(i, reps.Forward, reps.Backward);
            else
                missing.Add(i);
        }

        if (missing.Count == 0)
            return;

        var texts = missing.Select(i => sentences[i]).ToList();
        int words = texts.Sum(s => s.Count);
        var forwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        var backwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        try
        {
            // A single sentence the sentiment classifier ran alone (multi-word tokens) is taken from there.
            ManagedCharLanguageModel.BuildBoth(_charlmForward, _charlmBackward!, texts, forwardReps, backwardReps, charlms, ct);
            int offset = 0;
            foreach (int i in missing)
            {
                int n = sentences[i].Count * dim;
                Put(i, forwardReps.AsSpan(offset, n), backwardReps.AsSpan(offset, n));
                offset += n;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(forwardReps);
            ArrayPool<float>.Shared.Return(backwardReps);
        }
    }

    public void Dispose()
    {
    }
}
