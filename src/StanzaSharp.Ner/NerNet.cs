using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Ner;

/// <summary>NER's network, per <see cref="Backend"/>: <see cref="NerNet"/> or <see cref="ManagedNerNet"/>.</summary>
internal interface INerNet : IDisposable
{
    /// <summary>
    /// Embeddings (pretrain + delta), character features, input_transform, biLSTM and the tag set's linear layer.
    /// </summary>
    /// <param name="sentences">The batch's token texts.</param>
    /// <param name="wordIds">[batch, width] pretrain ids (lowercased tokens), padded.</param>
    /// <param name="deltaIds">[batch, width] delta ids, padded.</param>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: charlm outputs to reuse; row i's key, if not null, is looked up.</param>
    /// <returns>[batch, width, tags] emission scores, row-major; rows past a sentence's end are unspecified.</returns>
    float[] Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] deltaIds, int width,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct);
}

/// <summary>stanza/models/ner/model.py <c>NERTagger.forward</c> on TorchSharp (today's code).</summary>
internal sealed class NerNet : INerNet
{
    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel? _charlmForward, _charlmBackward;
    private readonly CharacterModel? _charModel;
    private readonly Embedding _deltaEmb;
    private readonly Linear _inputTransform, _tagClf;
    private readonly LSTM _lstm;
    private readonly Tensor _hInit, _cInit;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public NerNet(Checkpoint ckpt, int deltaCount, int tags, int tagset, Pretrain pretrain, CharLanguageModel? charlmForward, CharLanguageModel? charlmBackward)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        if (charlmForward == null)
            _charModel = new CharacterModel(ckpt, model, config, ckpt.Root["vocab"]!["char"]!, "charmodel.", bidirectional: true, attention: false);

        int wordDim = config["word_emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["num_layers"]!.GetValue<int>();
        int inputSize = wordDim + (_charModel?.OutputDim ?? _charlmForward!.HiddenDim + _charlmBackward!.HiddenDim);

        _deltaEmb = nn.Embedding(deltaCount, wordDim, padding_idx: NerTagger.PadId).LoadFrom(ckpt, model, "delta_emb.");
        _inputTransform = nn.Linear(inputSize, inputSize).LoadFrom(ckpt, model, "input_transform.");
        _lstm = nn.LSTM(inputSize, hidden, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "taggerlstm.lstm.");
        _hInit = ckpt.ToTensor(model["taggerlstm_h_init"]);
        _cInit = ckpt.ToTensor(model["taggerlstm_c_init"]);
        _tagClf = nn.Linear(hidden * 2, tags).LoadFrom(ckpt, model, $"tag_clfs.{tagset}.");
    }

    public float[] Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] deltaIds, int width,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        int batch = sentences.Count;
        var lengths = sentences.Select(s => (long)s.Count).ToArray();
        var words = _pretrain.Embeddings[torch.tensor(wordIds, [batch, width], device: _device)]
            .add(_deltaEmb.forward(torch.tensor(deltaIds, [batch, width], device: _device)), Scalars.One);

        Tensor[] chars;
        if (_charModel != null)
            chars = [_charModel.Forward(sentences.Select(s => (IReadOnlyList<long[]>)s.Select(_charModel.CharIds).ToList()).ToList())];
        else
        {
            var cached = Enumerable.Range(0, batch).Select(i =>
                cacheKeys?[i] is { } key && charlms!.TryGet(key, out var reps) ? reps : ((Tensor, Tensor)?)null).ToList();
            var missing = Enumerable.Range(0, batch).Where(i => cached[i] == null).ToList();
            var forward = _charlmForward!.BuildCharRepresentation(missing.Select(i => sentences[i]).ToList());
            var backward = _charlmBackward!.BuildCharRepresentation(missing.Select(i => sentences[i]).ToList());
            for (int k = 0; k < missing.Count; k++)
                cached[missing[k]] = (forward[k], backward[k]);
            chars = [Rnn.PadSequence(cached.Select(c => c!.Value.Item1).ToList()), Rnn.PadSequence(cached.Select(c => c!.Value.Item2).ToList())];
        }

        var input = _inputTransform.forward(cat([words, .. chars], 2));
        var h0 = _hInit.expand(_hInit.shape[0], batch, _hInit.shape[2]).contiguous();
        var c0 = _cInit.expand(_cInit.shape[0], batch, _cInit.shape[2]).contiguous();
        var output = Rnn.RunPacked(_lstm, input, lengths, (h0, c0));
        return _tagClf.forward(output).ToArray<float>();
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _deltaEmb, _inputTransform, _lstm, _tagClf })
            m.Dispose();
        _charModel?.Dispose();
        _hInit.Dispose();
        _cInit.Dispose();
    }
}

/// <summary>
/// Managed twin of <see cref="NerNet"/> (<see cref="Backend.Managed"/>). It works on the packed rows only: each token's
/// input row is built at its packed position, so input_transform, the biLSTM and the tag layer never see padding.
/// </summary>
internal sealed unsafe class ManagedNerNet : INerNet
{
    private readonly Pretrain _pretrain;
    private readonly ManagedCharLanguageModel? _charlmForward, _charlmBackward;
    private readonly ManagedCharacterModel? _charModel;
    private readonly float[] _deltaEmb; // [delta vocab, wordDim]
    private readonly int _wordDim, _inputSize, _tags;
    private readonly PackedMatrix _inputTransform, _tagClf;
    private readonly float[] _inputBias, _tagBias;
    private readonly ManagedLstm _lstm;

    public ManagedNerNet(Checkpoint ckpt, int tags, int tagset, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward)
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
        _inputSize = _wordDim + (_charModel?.OutputDim ?? _charlmForward!.HiddenDim + _charlmBackward!.HiddenDim);
        _tags = tags;
        _deltaEmb = T("delta_emb.weight");
        (_inputTransform, _inputBias) = Linear(T("input_transform.weight"), T("input_transform.bias"), _inputSize, _inputSize);
        _lstm = new ManagedLstm(ckpt, model, "taggerlstm.lstm.", _inputSize, hidden, layers, bidirectional: true, T("taggerlstm_h_init"), T("taggerlstm_c_init"));
        (_tagClf, _tagBias) = Linear(T($"tag_clfs.{tagset}.weight"), T($"tag_clfs.{tagset}.bias"), tags, 2 * hidden);
    }

    private static (PackedMatrix, float[]) Linear(float[] w, float[] b, int n, int k)
    {
        var packed = new PackedMatrix(w, n, k);
        var bias = new float[packed.PaddedN];
        b.CopyTo(bias, 0);
        return (packed, bias);
    }

    public float[] Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] deltaIds, int width,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct)
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
            // Word embeddings: pretrain + delta.
            var vectors = _pretrain.CpuVectors();
            for (int i = 0; i < batch; i++)
                for (int j = 0; j < lengths[i]; j++)
                {
                    var dst = x.AsSpan(Row(i, j) * inSize, _wordDim);
                    var pre = vectors.Slice((int)wordIds[i * width + j] * _wordDim, _wordDim);
                    var delta = _deltaEmb.AsSpan((int)deltaIds[i * width + j] * _wordDim, _wordDim);
                    for (int k = 0; k < _wordDim; k++)
                        dst[k] = pre[k] + delta[k];
                }

            fixed (float* px = x)
            {
                if (_charModel != null)
                {
                    // Words listed in packed row order, so each vector lands in its row.
                    var words = new int[rows][];
                    for (int i = 0; i < batch; i++)
                        for (int j = 0; j < lengths[i]; j++)
                            words[Row(i, j)] = _charModel.CharIds(sentences[i][j]);
                    _charModel.Forward(words, px + _wordDim, inSize, ct);
                }
                else
                    CharlmFeatures(sentences, charlms, cacheKeys, x, Row, ct);

                fixed (float* pt = transformed, pib = _inputBias)
                    Gemm.Run(px, rows, inSize, _inputTransform, pib, pt, ldt, ct);
            }
            _lstm.ForwardPacked(transformed, ldt, batchSizes, hidden, ct);
            fixed (float* ph = hidden, ps = scores, ptb = _tagBias)
                Gemm.Run(ph, rows, _lstm.OutputSize, _tagClf, ptb, ps, ldc, ct);

            var logits = new float[batch * width * _tags];
            for (int i = 0; i < batch; i++)
                for (int j = 0; j < lengths[i]; j++)
                    scores.AsSpan(Row(i, j) * ldc, _tags).CopyTo(logits.AsSpan((i * width + j) * _tags));
            return logits;
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
        int dim = _charlmForward!.HiddenDim, forwardCol = _wordDim, backwardCol = _wordDim + dim;
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
            if (cacheKeys?[i] is { } key && charlms!.TryGetArrays(key, out var reps))
                Put(i, reps.Forward, reps.Backward);
            else
                missing.Add(i);
        if (missing.Count == 0)
            return;

        var texts = missing.Select(i => sentences[i]).ToList();
        int words = texts.Sum(s => s.Count);
        var forwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        var backwardReps = ArrayPool<float>.Shared.Rent(words * dim);
        try
        {
            _charlmForward.BuildCharRepresentation(texts, forwardReps, dim, ct);
            _charlmBackward!.BuildCharRepresentation(texts, backwardReps, dim, ct);
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
