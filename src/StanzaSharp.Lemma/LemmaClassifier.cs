using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Lemma;

/// <summary>
/// One of a lemmatizer checkpoint's <c>contextual</c> models (English <c>combined_charlm</c>: <c>'s</c> AUX → be/have,
/// <c>her</c> PRON → she/her): a <c>LemmaClassifierLSTM</c> (stanza/models/lemma_classifier/lstm_model.py) that picks the
/// lemma of a target word from its sentence. Port of trainer.predict_contextual for one model: every target word of the
/// document (lowercased text in <c>target_words</c>, UPOS in <c>target_upos</c>) is one row of a single batch, with its
/// whole sentence, and the label with the highest logit replaces the lemma. Only the paths those checkpoints use: the
/// biLSTM (<c>num_heads</c> 0), UPOS embedding, both charlms.
/// </summary>
internal sealed class LemmaClassifier : IDisposable
{
    private readonly string[] _labels;
    private readonly Dictionary<string, int> _uposToId, _known = [];
    private readonly HashSet<string> _targetWords, _targetUpos;
    private readonly Pretrain _pretrain;
    private readonly ILemmaClassifierNet _net;

    /// <param name="node">The checkpoint's <c>contextual[i]</c> (get_save_dict).</param>
    /// <param name="net">Builds the network from the checkpoint and the node's state dict (a backend's).</param>
    public LemmaClassifier(Checkpoint ckpt, JsonNode node, Pretrain pretrain, Func<Checkpoint, JsonNode, ILemmaClassifierNet> net)
    {
        var args = node["args"]!;
        if (node["model_type"]?.GetValue<string>() != "LSTM" || args["num_heads"]?.GetValue<int>() != 0
            || args["use_charlm"]?.GetValue<bool>() != true || !(args["upos_emb_dim"]?.GetValue<int>() > 0))
        {
            throw new NotSupportedException("Contextual lemma classifiers other than the charlm biLSTM with UPOS input are not ported");
        }

        // label_decoder {label: id}; predict maps the argmax back through its inverse.
        var decoder = node["label_decoder"]!.AsObject();
        _labels = new string[decoder.Count];
        foreach (var (label, id) in decoder)
            _labels[id!.GetValue<int>()] = label;
        _uposToId = node["upos_to_id"]!.AsObject().ToDictionary(kv => kv.Key, kv => kv.Value!.GetValue<int>());

        // known_word_map = {word: idx for idx, word in enumerate(known_words)}: 0-based (the first word shares row 0, the
        // padding row, with unknown words) and the last of a repeated word wins, as in Stanza.
        var known = Checkpoint.Strings(node["known_words"]);
        for (int i = 0; i < known.Length; i++)
            _known[known[i]] = i;
        _targetWords = [.. Checkpoint.Strings(node["target_words"])];
        _targetUpos = [.. Checkpoint.Strings(node["target_upos"])];
        _pretrain = pretrain;
        _net = net(ckpt, node["params"]!);
    }

    /// <summary>
    /// Replaces the lemma of every target word: <paramref name="lemmas"/> holds the words of <paramref name="sentences"/>
    /// in order.
    /// </summary>
    /// <param name="onLogits">For tests: the batch's labels and [rows, labels] logits.</param>
    public void Apply(IReadOnlyList<IReadOnlyList<Word>> sentences, IList<string> lemmas, CancellationToken ct, Action<string[], float[]>? onLogits = null)
    {
        var rows = new List<(int Sentence, int Position, int Lemma)>();
        for (int s = 0, offset = 0; s < sentences.Count; offset += sentences[s].Count, s++)
        {
            for (int i = 0; i < sentences[s].Count; i++)
            {
                var word = sentences[s][i];
                if (word.Upos != null && _targetUpos.Contains(word.Upos) && _targetWords.Contains(PyString.Lower(word.Text)))
                    rows.Add((s, i, offset + i));
            }
        }

        if (rows.Count == 0)
            return;

        // pad_sequence: every row padded with id 0 to the longest sentence of the batch.
        int n = rows.Count, width = rows.Max(r => sentences[r.Sentence].Count);
        var ids = new long[n * width];
        var deltaIds = new long[n * width];
        var uposIds = new long[n * width];
        var texts = new IReadOnlyList<string>[n];
        for (int r = 0; r < n; r++)
        {
            var words = sentences[rows[r].Sentence];
            texts[r] = [.. words.Select(w => w.Text)];
            for (int t = 0; t < words.Count; t++)
            {
                var lower = PyString.Lower(words[t].Text);
                ids[r * width + t] = _pretrain.UnitToId(lower);
                deltaIds[r * width + t] = _known.GetValueOrDefault(lower);
                uposIds[r * width + t] = words[t].Upos is { } upos && _uposToId.TryGetValue(upos, out var id)
                    ? id
                    : throw new InvalidOperationException($"The contextual lemma classifier has no UPOS '{words[t].Upos}' (word '{words[t].Text}')");
            }
        }

        var logits = _net.Forward(texts, ids, deltaIds, uposIds, width, [.. rows.Select(r => r.Position)], ct);
        onLogits?.Invoke(_labels, logits);
        int classes = _labels.Length;
        for (int r = 0; r < n; r++)
        {
            int best = 0; // torch.argmax: the first maximum
            for (int k = 1; k < classes; k++)
            {
                if (logits[r * classes + k] > logits[r * classes + best])
                    best = k;
            }

            lemmas[rows[r].Lemma] = _labels[best];
        }
    }

    public void Dispose() => _net.Dispose();
}

/// <summary>A contextual lemma classifier's network, per <see cref="Backend"/>: <c>LemmaClassifierNet</c> or <see cref="ManagedLemmaClassifierNet"/>.</summary>
internal interface ILemmaClassifierNet : IDisposable
{
    /// <summary>
    /// LemmaClassifierLSTM.forward on one batch: [pretrain + delta embedding | UPOS embedding | forward charlm | backward
    /// charlm] per word, the biLSTM over the whole padded width (Stanza packs every row at the padded length, so padding
    /// reaches the backward direction), the output at each row's target position, then the MLP.
    /// </summary>
    /// <param name="sentences">Each row's word texts (a sentence with two targets is two rows).</param>
    /// <param name="ids">[batch, width] pretrain ids of the lowercased words, padded with 0.</param>
    /// <param name="deltaIds">[batch, width] delta embedding ids, padded with 0.</param>
    /// <param name="uposIds">[batch, width] UPOS ids, padded with 0.</param>
    /// <param name="positions">Each row's target word.</param>
    /// <param name="ct">Checked after each charlm pass and inside the LSTM.</param>
    /// <returns>[batch, labels] logits, row-major.</returns>
    float[] Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] ids, long[] deltaIds, long[] uposIds, int width, int[] positions, CancellationToken ct);
}

/// <summary>Managed twin of <c>LemmaClassifierNet</c> (<see cref="Backend.Managed"/>), built from the managed blocks sentiment uses.</summary>
internal sealed unsafe class ManagedLemmaClassifierNet : ILemmaClassifierNet
{
    private readonly Pretrain _pretrain;
    private readonly ManagedCharLanguageModel _charlmForward, _charlmBackward;
    private readonly float[] _delta, _upos;
    private readonly int _dim, _uposDim, _inSize;
    private readonly ManagedLstm _lstm;
    private readonly (PackedMatrix W, float[] Bias)[] _mlp;

    public ManagedLemmaClassifierNet(Checkpoint ckpt, JsonNode model, Pretrain pretrain, ManagedCharLanguageModel charlmForward, ManagedCharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        _dim = pretrain.Dim;
        _delta = T("delta_embedding.weight");
        _upos = T("upos_emb.weight");
        _uposDim = (int)ckpt.Shape(model["upos_emb.weight"])[1];
        _inSize = _dim + _uposDim + charlmForward.HiddenDim + charlmBackward.HiddenDim;
        var hh = ckpt.Shape(model["lstm.weight_hh_l0"]);
        _lstm = new ManagedLstm(ckpt, model, "lstm.", _inSize, (int)hh[1], 1, bidirectional: true);
        _mlp = [.. new[] { "mlp.0.", "mlp.2." }.Select(prefix =>
        {
            var shape = ckpt.Shape(model[prefix + "weight"]);
            var w = new PackedMatrix(T(prefix + "weight"), (int)shape[0], (int)shape[1]);
            var bias = new float[w.PaddedN];
            T(prefix + "bias").CopyTo(bias, 0);
            return (w, bias);
        })];
    }

    public float[] Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] ids, long[] deltaIds, long[] uposIds, int width, int[] positions, CancellationToken ct)
    {
        int n = sentences.Count, dim = _charlmForward.HiddenDim, words = sentences.Sum(s => s.Count);
        var forward = new float[words * dim];
        var backward = new float[words * dim];
        _charlmForward.BuildCharRepresentation(sentences, forward, dim, ct);
        ct.ThrowIfCancellationRequested();
        _charlmBackward.BuildCharRepresentation(sentences, backward, dim, ct);
        ct.ThrowIfCancellationRequested();
        var first = new int[n]; // each row's first word in the charlm outputs
        for (int i = 1; i < n; i++)
            first[i] = first[i - 1] + sentences[i - 1].Count;

        // The input rows, time-major (word (i, t) is row t·n + i): padding gets the pretrain's and the delta's row 0, UPOS
        // row 0 and zero charlms, as pad_sequence pads the ids with 0 and the charlm outputs with zeros.
        void Fill(int r0, int r1, float* x)
        {
            var vectors = _pretrain.CpuVectors();
            for (int r = r0; r < r1; r++)
            {
                int i = r % n, t = r / n, k = i * width + t;
                var dst = new Span<float>(x + (long)(r - r0) * _inSize, _inSize);
                var pre = vectors.Slice((int)ids[k] * _dim, _dim);
                var delta = _delta.AsSpan((int)deltaIds[k] * _dim, _dim);
                for (int d = 0; d < _dim; d++)
                    dst[d] = pre[d] + delta[d];
                _upos.AsSpan((int)uposIds[k] * _uposDim, _uposDim).CopyTo(dst[_dim..]);
                var charlms = dst[(_dim + _uposDim)..];
                if (t < sentences[i].Count)
                {
                    forward.AsSpan((first[i] + t) * dim, dim).CopyTo(charlms);
                    backward.AsSpan((first[i] + t) * dim, dim).CopyTo(charlms[dim..]);
                }
                else
                {
                    charlms.Clear();
                }
            }
        }

        int h2 = _lstm.OutputSize;
        var z = new float[n * width * h2]; // batch-major: row i·width + t
        _lstm.ForwardUnpacked(n, width, Fill, z, ct, lengths: [.. sentences.Select(s => s.Count)]);

        // The target word's output, then Linear, ReLU, Linear.
        var hidden = new float[n * h2];
        for (int i = 0; i < n; i++)
            z.AsSpan((i * width + positions[i]) * h2, h2).CopyTo(hidden.AsSpan(i * h2));
        var (w0, b0) = _mlp[0];
        var mid = Gemm.Run(hidden, n, w0, b0);
        for (int k = 0; k < mid.Length; k++)
            mid[k] = Math.Max(mid[k], 0);
        var (w1, b1) = _mlp[1];
        var output = new float[n * w1.PaddedN];
        fixed (float* a = mid, pb = b1, c = output)
            Gemm.Run(a, n, w0.PaddedN, w1, pb, c, w1.PaddedN, ct);
        var logits = new float[n * w1.N];
        for (int i = 0; i < n; i++)
            output.AsSpan(i * w1.PaddedN, w1.N).CopyTo(logits.AsSpan(i * w1.N));
        return logits;
    }

    public void Dispose()
    {
    }
}
