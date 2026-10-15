using System.Buffers;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Depparse;

/// <summary>
/// Managed twin of <c>DepparseNet</c> (<see cref="Backend.Managed"/>). The input and the highway biLSTM work on
/// the packed rows only. Each deep biaffine scorer runs in two steps: its <c>W1</c>/<c>W2</c> layers (one GEMM per
/// scorer, one scorer's buffers at a time), then T = in1·W_bilin per dependent (a GEMM) and T·in2 per word pair. Padding columns, which count in the
/// arc log-softmax, all see the same in2 (the LSTM output there is 0, so in2 = ReLU(W2's bias)), so a row's T·in2 for them
/// is computed once; padding rows are not scored, since nothing reads them.
/// </summary>
internal sealed unsafe class ManagedDepparseNet : IDepparseNet
{
    // The label scorer's T is [rows, (hidden + 1)·relations]: 79 KB per word. Whole sentences go through in chunks of
    // at most this many floats (32 MB), or one sentence if it is longer.
    private const int ChunkFloats = 8 << 20;

    private readonly Pretrain _pretrain;
    private readonly ManagedCharLanguageModel? _charlmForward, _charlmBackward;
    private readonly ManagedCharacterModel? _charModel;
    private readonly PackedMatrix? _transChar;
    private readonly float[] _wordEmb, _lemmaEmb, _uposEmb;
    private readonly float[][] _xposEmb; // one table per part of a composite XPOS vocabulary
    private readonly int _wordDim, _tagDim, _transformed, _inputSize, _biaff;
    private readonly PackedMatrix _transPretrained;
    private readonly ManagedHighwayLstm _lstm;
    private readonly Scorer _unlabeled, _deprel;
    private readonly Scorer? _linearization, _distance;

    /// <summary>
    /// A DeepBiaffineScorer: in1 = ReLU(W1·h) and in2 = ReLU(W2·h) are one GEMM (<see cref="Hid"/>, in1 at columns
    /// [0, hidden), in2 at [hidden, 2·hidden)); T[i, q·Out + o] = Σ_p in1[i, p]·W[p, q, o] + W[hidden, q, o] (in1's
    /// appended 1), so that score[i, j, o] = Σ_q T[i, q·Out + o]·in2[j, q] + T[i, hidden·Out + o] + Bias[o].
    /// </summary>
    /// <remarks>
    /// Until 1.0.0 all scorers' W1/W2 were one GEMM [rows, 3200]; each scorer's 800 columns were whole 16-column panels
    /// of it, and a GEMM column depends only on its panel, so per scorer the bits are the same at a quarter of the buffer.
    /// </remarks>
    private sealed class Scorer
    {
        public required int Out;             // relations
        public required PackedMatrix Hid;    // [2·hidden, LSTM output]: W1 then W2
        public required float[] HidBias;     // [Hid.PaddedN]
        public required PackedMatrix W;      // [(hidden + 1)·Out, hidden]
        public required float[] WBias;       // [W.PaddedN]
        public required float[] Bias;        // [Out]
        public required float[] Pad2;        // in2 of a padding word: ReLU(W2's bias)
    }

    public ManagedDepparseNet(Checkpoint ckpt, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        int hidden = config["hidden_dim"]!.GetValue<int>();
        _biaff = config["deep_biaff_hidden_dim"]!.GetValue<int>();
        _wordDim = config["word_emb_dim"]!.GetValue<int>();
        _tagDim = config["tag_emb_dim"]!.GetValue<int>();
        _transformed = config["transformed_dim"]!.GetValue<int>();

        // trans_pretrained, word, lemma, then the UPOS+XPOS embedding twice (see DepparseNet), then the characters.
        _inputSize = _transformed + 2 * _wordDim + 2 * _tagDim;
        if (_charlmForward != null)
        {
            _inputSize += _charlmForward.HiddenDim + _charlmBackward!.HiddenDim;
        }
        else
        {
            _charModel = new ManagedCharacterModel(ckpt, model, config, ckpt.Root["vocab"]!["char"]!, "charmodel.", bidirectional: false, attention: true);
            _transChar = new PackedMatrix(T("trans_char.weight"), _transformed, _charModel.OutputDim);
            _inputSize += _transformed;
        }

        (_wordEmb, _lemmaEmb, _uposEmb) = (T("word_emb.weight"), T("lemma_emb.weight"), T("upos_emb.weight"));
        var xpos = XposVocab.Read(ckpt.Root["vocab"]!);
        _xposEmb = xpos.IsComposite ? [.. xpos.Sizes.Select((_, i) => T($"xpos_emb.{i}.weight"))] : [T("xpos_emb.weight")];
        _transPretrained = new PackedMatrix(T("trans_pretrained.weight"), _transformed, pretrain.Dim);
        _lstm = new ManagedHighwayLstm(ckpt, model, "parserlstm", _inputSize, hidden, config["num_layers"]!.GetValue<int>());

        var names = new List<string> { "unlabeled", "deprel" };
        if (config["linearization"]!.GetValue<bool>())
            names.Add("linearization");
        if (config["distance"]!.GetValue<bool>())
            names.Add("distance");
        var scorers = names.Select(n => NewScorer(ckpt, model, n, 2 * hidden)).ToArray();
        (_unlabeled, _deprel) = (scorers[0], scorers[1]);
        _linearization = names.IndexOf("linearization") is > 0 and var l ? scorers[l] : null;
        _distance = names.IndexOf("distance") is > 0 and var d ? scorers[d] : null;
    }

    private Scorer NewScorer(Checkpoint ckpt, System.Text.Json.Nodes.JsonNode model, string name, int input)
    {
        int b = _biaff, b1 = b + 1;
        float[] T(string key) => ckpt.Tensor<float>(model[name + key]);
        var hid = new PackedMatrix([.. T(".W1.weight"), .. T(".W2.weight")], 2 * b, input);
        var hidBias = new float[hid.PaddedN];
        T(".W1.bias").CopyTo(hidBias, 0);
        T(".W2.bias").CopyTo(hidBias, b);
        var shape = ckpt.Shape(model[name + ".scorer.W_bilin.weight"]);
        if (!shape.SequenceEqual([b1, b1, shape[2]]))
            throw new InvalidOperationException($"{name}.scorer.W_bilin.weight has shape [{string.Join(", ", shape)}]");
        int outs = (int)shape[2];
        var wb = ckpt.Tensor<float>(model[name + ".scorer.W_bilin.weight"]); // [p, q, o]
        var w = new float[b1 * outs * b];
        var wBias = new float[(b1 * outs + PackedMatrix.NR - 1) / PackedMatrix.NR * PackedMatrix.NR];
        for (int p = 0; p < b1; p++)
        {
            for (int q = 0; q < b1; q++)
            {
                for (int o = 0; o < outs; o++)
                {
                    float v = wb[(p * b1 + q) * outs + o];
                    if (p < b)
                        w[(q * outs + o) * b + p] = v;
                    else
                        wBias[q * outs + o] = v;
                }
            }
        }

        return new Scorer
        {
            Out = outs,
            Hid = hid,
            HidBias = hidBias,
            W = new PackedMatrix(w, b1 * outs, b),
            WBias = wBias,
            Bias = ckpt.Tensor<float>(model[name + ".scorer.W_bilin.bias"]),
            Pad2 = hidBias.AsSpan(b, b).ToArray().Select(x => Math.Max(x, 0f)).ToArray(),
        };
    }

    public DepparseScores Forward(DepparseBatch batch, bool labelScores, CancellationToken ct)
    {
        int size = batch.Texts.Count, width = batch.Width;
        var lengths = batch.Lengths;

        // Rows in sentence order: word t of sentence b (0 is ROOT) is row offset[b] + t.
        var offset = new int[size + 1];
        for (int b = 0; b < size; b++)
            offset[b + 1] = offset[b] + (int)lengths[b];
        int rows = offset[size];

        // pack_padded_sequence: sentences longest first; row k is packed row packedRow[k].
        var order = Enumerable.Range(0, size).OrderByDescending(b => lengths[b]).ToArray();
        var rank = new int[size];
        for (int r = 0; r < size; r++)
            rank[order[r]] = r;
        var batchSizes = PackedLstm.BatchSizes(lengths);
        var start = new int[batchSizes.Length];
        for (int t = 1; t < start.Length; t++)
            start[t] = start[t - 1] + batchSizes[t - 1];
        var packedRow = new int[rows];
        for (int b = 0; b < size; b++)
        {
            for (int t = 0; t < lengths[b]; t++)
                packedRow[offset[b] + t] = start[t] + rank[b];
        }

        var h = ArrayPool<float>.Shared.Rent(rows * _lstm.OutputSize);
        try
        {
            Encode(batch, offset, packedRow, h, ct);
            ct.ThrowIfCancellationRequested();
            var arcs = Arcs(h, size, width, lengths, offset, batch.RentScores, ct);
            var (labels, scores) = Labels(h, size, width, lengths, offset, labelScores, batch.RentScores, ct);
            return new(width, _deprel.Out, arcs, labels, scores, batch.RentScores);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(h);
        }
    }

    /// <summary>A scorer's in1/in2 for every row: ReLU(h·[W1; W2]ᵀ + b), [rows, Hid.PaddedN], rented.</summary>
    private float[] Hidden(Scorer scorer, float[] h, int rows, CancellationToken ct)
    {
        int ld = scorer.Hid.PaddedN;
        var hid = ArrayPool<float>.Shared.Rent(rows * ld);
        fixed (float* ph = h, pd = hid, pb = scorer.HidBias)
            Gemm.Run(ph, rows, _lstm.OutputSize, scorer.Hid, pb, pd, ld, ct);
        for (long n = 0, end = (long)rows * ld; n < end; n++)
            hid[n] = Math.Max(hid[n], 0f);
        return hid;
    }

    /// <summary>The input rows and the highway biLSTM: <paramref name="output"/> gets [rows, 2·hidden] in sentence order.</summary>
    private void Encode(DepparseBatch batch, int[] offset, int[] packedRow, float[] output, CancellationToken ct)
    {
        int size = batch.Texts.Count, width = batch.Width, rows = packedRow.Length, inSize = _inputSize;
        int dim = _pretrain.Dim, ldt = _transPretrained.PaddedN, charDim = _charModel?.OutputDim ?? 0;
        int posCol = _transformed + 2 * _wordDim, charCol = posCol + 2 * _tagDim;
        var x = ArrayPool<float>.Shared.Rent(rows * inSize);
        var a = ArrayPool<float>.Shared.Rent(rows * Math.Max(Math.Max(dim, charDim), _lstm.OutputSize));
        var c = ArrayPool<float>.Shared.Rent(rows * Math.Max(ldt, _transChar?.PaddedN ?? 0));
        try
        {
            // Embeddings at each word's packed row; the pretrain vectors (packed order) go through trans_pretrained.
            var vectors = _pretrain.CpuVectors();
            for (int b = 0; b < size; b++)
            {
                for (int t = 0, k = offset[b]; k < offset[b + 1]; t++, k++)
                {
                    int id = b * width + t, r = packedRow[k];
                    var row = x.AsSpan(r * inSize, inSize);
                    _wordEmb.AsSpan((int)batch.Word[id] * _wordDim, _wordDim).CopyTo(row[_transformed..]);
                    _lemmaEmb.AsSpan((int)batch.Lemma[id] * _wordDim, _wordDim).CopyTo(row[(_transformed + _wordDim)..]);

                    // upos_emb + each XPOS part's embedding, added in order as Stanza's pos_emb += does.
                    var pos = row.Slice(posCol, _tagDim);
                    _uposEmb.AsSpan((int)batch.Upos[id] * _tagDim, _tagDim).CopyTo(pos);
                    for (int p = 0; p < _xposEmb.Length; p++)
                    {
                        var xpos = _xposEmb[p].AsSpan((int)batch.Xpos[id * _xposEmb.Length + p] * _tagDim, _tagDim);
                        for (int e = 0; e < _tagDim; e++)
                            pos[e] += xpos[e];
                    }

                    pos.CopyTo(row[(posCol + _tagDim)..]);
                    vectors.Slice((int)batch.Pretrained[id] * dim, dim).CopyTo(a.AsSpan(r * dim));
                }
            }

            fixed (float* pa = a, pc = c)
                Gemm.Run(pa, rows, dim, _transPretrained, null, pc, ldt, ct);
            for (int r = 0; r < rows; r++)
                c.AsSpan(r * ldt, _transformed).CopyTo(x.AsSpan(r * inSize));

            if (_charModel != null)
            {
                // ROOT is a word of the single character id ROOT_ID.
                var words = new int[rows][];
                for (int b = 0; b < size; b++)
                {
                    for (int t = 0, k = offset[b]; k < offset[b + 1]; t++, k++)
                        words[packedRow[k]] = t == 0 ? [DependencyParser.RootId] : _charModel.CharIds(batch.Texts[b][t - 1]);
                }

                int ldc = _transChar!.PaddedN;
                fixed (float* pa = a, pc = c)
                {
                    _charModel.Forward(words, pa, charDim, ct);
                    Gemm.Run(pa, rows, charDim, _transChar, null, pc, ldc, ct);
                }

                for (int r = 0; r < rows; r++)
                    c.AsSpan(r * ldc, _transformed).CopyTo(x.AsSpan(r * inSize + charCol));
            }
            else
            {
                CharlmFeatures(batch, packedRow, x, charCol, ct);
            }

            ct.ThrowIfCancellationRequested();

            _lstm.Forward(x.AsSpan(0, rows * inSize), batch.Lengths, a.AsSpan(0, rows * _lstm.OutputSize), ct);
            int h2 = _lstm.OutputSize;
            for (int k = 0; k < rows; k++)
                a.AsSpan(packedRow[k] * h2, h2).CopyTo(output.AsSpan(k * h2));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(x);
            ArrayPool<float>.Shared.Return(a);
            ArrayPool<float>.Shared.Return(c);
        }
    }

    /// <summary>Both charlms over "\n" (ROOT) + each sentence's words, into each word's input row.</summary>
    private void CharlmFeatures(DepparseBatch batch, int[] packedRow, float[] x, int col, CancellationToken ct)
    {
        int dim = _charlmForward!.HiddenDim, rows = packedRow.Length;
        var sentences = batch.Texts.Select(t => (IReadOnlyList<string>)t.Prepend("\n").ToList()).ToList();
        var reps = ArrayPool<float>.Shared.Rent(rows * dim);
        try
        {
            _charlmForward.BuildCharRepresentation(sentences, reps, dim, ct);
            for (int k = 0; k < rows; k++)
                reps.AsSpan(k * dim, dim).CopyTo(x.AsSpan(packedRow[k] * _inputSize + col));
            ct.ThrowIfCancellationRequested();
            BackwardCharlm(batch, sentences, reps, ct);
            for (int k = 0; k < rows; k++)
                reps.AsSpan(k * dim, dim).CopyTo(x.AsSpan(packedRow[k] * _inputSize + col + dim));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(reps);
        }
    }

    /// <summary>The backward charlm's rows for <see cref="CharlmFeatures"/>, ROOT then the words of each sentence in order.</summary>
    /// <remarks>
    /// Backward, ROOT comes last: "\n" + the words reversed is the tagger's sequence, then ROOT. So a sentence whose backward
    /// outputs the tagger kept (<see cref="CharlmCache.TryGetBackward"/>) takes its words' rows from there and runs on from
    /// the tagger's final state over ROOT only; the others run in full. A charlm step gives a row the same arithmetic in
    /// any batch (<see cref="Gemm.KernelBlocked"/>), so these are the bits a full pass over every sentence gives.
    /// </remarks>
    private void BackwardCharlm(DepparseBatch batch, List<IReadOnlyList<string>> sentences, float[] reps, CancellationToken ct)
    {
        int dim = _charlmBackward!.HiddenDim, n = sentences.Count;
        var first = new int[n]; // each sentence's first row (ROOT)
        for (int i = 1; i < n; i++)
            first[i] = first[i - 1] + sentences[i - 1].Count;
        var tagger = new (float[] Backward, float[] FinalC)[n];
        var reused = new List<int>();
        var rest = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (batch.Charlms != null && batch.Charlms.TryGetBackward(batch.Keys?[i], batch.Texts[i], out var backward, out var finalC))
            {
                tagger[i] = (backward, finalC);
                reused.Add(i);
            }
            else
            {
                rest.Add(i);
            }
        }

        if (reused.Count == 0)
        {
            _charlmBackward.BuildCharRepresentation(sentences, reps, dim, ct);
            return;
        }

        int m = reused.Count, restRows = rest.Sum(i => sentences[i].Count);
        var scratch = ArrayPool<float>.Shared.Rent(Math.Max(restRows, 3 * m) * dim);
        try
        {
            if (rest.Count > 0)
            {
                _charlmBackward.BuildCharRepresentation(rest.Select(i => sentences[i]).ToList(), scratch, dim, ct);
                for (int j = 0, k = 0; j < rest.Count; k += sentences[rest[j]].Count, j++)
                    scratch.AsSpan(k * dim, sentences[rest[j]].Count * dim).CopyTo(reps.AsSpan(first[rest[j]] * dim));
            }

            // Start states [m, dim] each: h is the first word's row (the tagger's pass ends there), c the final cell state.
            for (int j = 0; j < m; j++)
            {
                var (backward, finalC) = tagger[reused[j]];
                backward.AsSpan(0, dim).CopyTo(scratch.AsSpan(j * dim));
                finalC.CopyTo(scratch.AsSpan((m + j) * dim));
                backward.CopyTo(reps.AsSpan((first[reused[j]] + 1) * dim));
            }

            fixed (float* ps = scratch)
            {
                _charlmBackward.Continue(ps, ps + (long)m * dim, Enumerable.Repeat<IReadOnlyList<string>>(["\n"], m).ToList(),
                    ps + 2L * m * dim, dim, ct);
            }

            for (int j = 0; j < m; j++)
                scratch.AsSpan((2 * m + j) * dim, dim).CopyTo(reps.AsSpan(first[reused[j]] * dim));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(scratch);
        }
    }

    /// <summary>
    /// The unlabeled, linearization and distance scores of each real row against every column, combined as in
    /// GraphParser.forward_scores, then the log-softmax over the padded width. Pair sums and the combination are in double.
    /// One scorer at a time (its in1/in2 and T, then its term of every pair), so only one scorer's buffers are live.
    /// </summary>
    private float[] Arcs(float[] h, int size, int width, long[] lengths, int[] offset, bool rent, CancellationToken ct)
    {
        int rows = offset[size], b = _biaff;

        // Rented: rows past a sentence's length stay stale.
        var result = rent ? ArrayPool<float>.Shared.Rent(size * width * width) : new float[size * width * width];
        var sentenceOf = new int[rows];
        for (int s = 0; s < size; s++)
            Array.Fill(sentenceOf, s, offset[s], offset[s + 1] - offset[s]);
        var v = ArrayPool<double>.Shared.Rent(rows * width);
        try
        {
            foreach (var (scorer, term) in new[] { (_unlabeled, 0), (_linearization, 1), (_distance, 2) })
            {
                if (scorer == null)
                    continue;
                int ldH = scorer.Hid.PaddedN, ldT = scorer.W.PaddedN;
                var hid = Hidden(scorer, h, rows, ct);
                var t = ArrayPool<float>.Shared.Rent(rows * ldT);
                try
                {
                    fixed (float* pd = hid, pt = t, pb = scorer.WBias)
                    {
                        Gemm.Run(pd, rows, ldH, scorer.W, pb, pt, ldT, ct);
                        nint hidBase = (nint)pd, tBase = (nint)pt;
                        ManagedThreads.For(rows, k =>
                        {
                            ct.ThrowIfCancellationRequested();
                            int sent = sentenceOf[k], i = k - offset[sent], n = (int)lengths[sent];
                            var tRow = new ReadOnlySpan<float>((float*)tBase + (long)k * ldT, b + 1);
                            var vRow = v.AsSpan(k * width, width);
                            double constant = tRow[b] + (double)scorer.Bias[0];

                            // Every padding column has in2 = Pad2, so the same score.
                            double padScore = n < width ? Act.Dot(tRow, scorer.Pad2) + constant : 0;
                            for (int j = 0; j < width; j++)
                            {
                                double score = j < n
                                    ? Act.Dot(tRow, new ReadOnlySpan<float>((float*)hidBase + (long)(offset[sent] + j) * ldH + b, b)) + constant
                                    : padScore;
                                int offsetJ = j - i;
                                vRow[j] = term switch
                                {
                                    0 => score,
                                    1 => vRow[j] + LogSigmoid(score * Math.Sign(offsetJ)), // linearization
                                    _ => vRow[j] - Math.Log(Square(Math.Abs(offsetJ) - (1 + Softplus(score))) / 2 + 1), // distance
                                };
                            }
                        });
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(hid);
                    ArrayPool<float>.Shared.Return(t);
                }
            }

            ManagedThreads.For(rows, k =>
            {
                int sent = sentenceOf[k], i = k - offset[sent];
                var vRow = v.AsSpan(k * width, width);
                vRow[i] = double.NegativeInfinity;
                double max = double.NegativeInfinity, sum = 0;
                foreach (var x in vRow)
                    max = Math.Max(max, x);
                foreach (var x in vRow)
                    sum += Math.Exp(x - max);
                double logSum = max + Math.Log(sum);
                var o = result.AsSpan((sent * width + i) * width, width);
                for (int j = 0; j < width; j++)
                    o[j] = (float)(vRow[j] - logSum);
            });
        }
        finally
        {
            ArrayPool<double>.Shared.Return(v);
        }

        return result;
    }

    private static double Square(double x) => x * x;

    /// <summary>torch's softplus with beta 1 and threshold 20.</summary>
    private static double Softplus(double x) => x > 20 ? x : Math.Log(1 + Math.Exp(x));

    private static double LogSigmoid(double x) => Math.Min(x, 0) - Math.Log(1 + Math.Exp(-Math.Abs(x)));

    /// <summary>
    /// The label scorer for every pair of real words, a chunk of whole sentences at a time: T by GEMM, then for each
    /// dependent i, its T row repacked as a [hidden, relations] weight panel and every head j's in2 through the GEMM
    /// micro-kernel. Returns the argmax relations and, if asked for, the scores.
    /// </summary>
    private (int[] Labels, float[]? Scores) Labels(float[] h, int size, int width, long[] lengths, int[] offset, bool keepScores, bool rent,
        CancellationToken ct)
    {
        var scorer = _deprel;
        int outs = scorer.Out, b = _biaff, ldT = scorer.W.PaddedN, ldH = scorer.Hid.PaddedN;
        int panels = (outs + PackedMatrix.NR - 1) / PackedMatrix.NR, padded = panels * PackedMatrix.NR;
        var labels = rent ? ArrayPool<int>.Shared.Rent(size * width * width) : new int[size * width * width]; // rented: stale past the real pairs
        var scores = keepScores ? new float[size * width * width * outs] : null;
        int maxRows = Math.Max(ChunkFloats / ldT, (int)lengths.Max());
        var hid = Hidden(scorer, h, offset[size], ct);
        var t = ArrayPool<float>.Shared.Rent(Math.Min(maxRows, offset[size]) * ldT);
        try
        {
            fixed (float* pd = hid, pt = t, pb = scorer.WBias)
            {
                nint hidBase = (nint)pd, tBase = (nint)pt;
                for (int s0 = 0, s1; s0 < size; s0 = s1)
                {
                    ct.ThrowIfCancellationRequested();
                    for (s1 = s0 + 1; s1 < size && offset[s1 + 1] - offset[s0] <= maxRows; s1++)
                    {
                    }

                    int k0 = offset[s0], chunkRows = offset[s1] - k0, first = s0;
                    Gemm.Run(pd + (long)k0 * ldH, chunkRows, ldH, scorer.W, pb, pt, ldT, ct);
                    ManagedThreads.For(chunkRows, task =>
                    {
                        ct.ThrowIfCancellationRequested();
                        int k = k0 + task, sent = first;
                        while (offset[sent + 1] <= k)
                            sent++;
                        int i = k - offset[sent], n = (int)lengths[sent];
                        var tRow = (float*)tBase + (long)task * ldT;
                        var panel = ArrayPool<float>.Shared.Rent(padded * b + padded + Gemm.MR * padded);
                        try
                        {
                            fixed (float* pp = panel)
                            {
                                // Panel layout as PackedMatrix: [panel][q][lane], relation o = panel·16 + lane.
                                float* init = pp + padded * b, outRows = init + padded;
                                for (int p = 0; p < panels; p++)
                                {
                                    for (int q = 0; q < b; q++)
                                    {
                                        for (int lane = 0; lane < PackedMatrix.NR; lane++)
                                        {
                                            int o = p * PackedMatrix.NR + lane;
                                            pp[((long)p * b + q) * PackedMatrix.NR + lane] = o < outs ? tRow[q * outs + o] : 0;
                                        }
                                    }
                                }

                                for (int o = 0; o < padded; o++)
                                    init[o] = o < outs ? tRow[b * outs + o] + scorer.Bias[o] : 0;
                                var aRows = stackalloc float*[Gemm.MR];
                                var inits = stackalloc float*[Gemm.MR];
                                var outs_ = stackalloc float*[Gemm.MR];
                                for (int j0 = 0; j0 < n; j0 += Gemm.MR)
                                {
                                    int mr = Math.Min(Gemm.MR, n - j0);
                                    for (int r = 0; r < Gemm.MR; r++)
                                        aRows[r] = (float*)hidBase + (long)(offset[sent] + j0 + Math.Min(r, mr - 1)) * ldH + b;
                                    for (int p = 0; p < panels; p++)
                                    {
                                        for (int r = 0; r < Gemm.MR; r++)
                                        {
                                            inits[r] = init + p * PackedMatrix.NR;
                                            outs_[r] = outRows + r * padded + p * PackedMatrix.NR;
                                        }

                                        Gemm.Kernel(mr, aRows, pp + (long)p * b * PackedMatrix.NR, b, inits, outs_);
                                    }

                                    for (int r = 0; r < mr; r++)
                                    {
                                        var row = new ReadOnlySpan<float>(outRows + r * padded, outs);
                                        long at = ((long)sent * width + i) * width + j0 + r;
                                        labels[at] = Act.ArgMax(row);
                                        if (scores != null)
                                            row.CopyTo(scores.AsSpan((int)(at * outs), outs));
                                    }
                                }
                            }
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(panel);
                        }
                    });
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hid);
            ArrayPool<float>.Shared.Return(t);
        }

        return (labels, scores);
    }

    public void Dispose()
    {
    }
}
