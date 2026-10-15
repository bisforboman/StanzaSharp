using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Nn;

/// <summary>
/// seq2seq_model.py <c>Seq2SeqModel</c>, the lemmatizer's and German MWT's character model, per <see cref="Backend"/>:
/// <c>Seq2SeqNet</c> or <see cref="ManagedSeq2SeqNet"/>.
/// </summary>
internal interface ISeq2SeqNet : IDisposable
{
    /// <summary>Seq2SeqModel.encode on one batch, rows sorted longest first; the decoder then runs one step per call.</summary>
    /// <param name="ids">[batch, width] source ids (<c>&lt;SOS&gt;</c> chars <c>&lt;EOS&gt;</c>, padded with 0); ids past the
    /// vocabulary are DeltaVocab characters, embedded as <c>&lt;UNK&gt;</c> but copyable.</param>
    /// <param name="posIds">[batch] UPOS ids, the encoder's first step, with <see cref="Seq2SeqConfig.UsesPos"/>; else null.</param>
    /// <param name="lengths">Each row's encoder length: its source length, + 1 for the UPOS step.</param>
    /// <param name="beam">The decoder's rows per source row, for <see cref="Seq2SeqConfig.Beam"/>: row k · batch + b is beam k of
    /// source b, all starting from b's encoding (Seq2SeqModel.predict's repeat). As in Stanza, only the first batch rows (beam
    /// 0) see the copy distribution: decode scatters the unrepeated source into the repeated rows.</param>
    ISeq2SeqDecoder Encode(long[] ids, int batch, int width, long[]? posIds, long[] lengths, CancellationToken ct, int beam = 1);
}

/// <summary>One batch's decoder state (the per-call state of <see cref="ISeq2SeqNet"/>).</summary>
internal interface ISeq2SeqDecoder : IDisposable
{
    /// <summary>[batch, 3] edit classifier logits, with <see cref="Seq2SeqConfig.Edit"/>; else empty.</summary>
    float[] EditLogits { get; }

    /// <summary>Width of a <see cref="Step"/> row: the vocabulary, widened to the batch's largest source id + 1 with <see cref="Seq2SeqConfig.Copy"/>.</summary>
    int Columns { get; }

    /// <summary>
    /// Seq2SeqModel.decode for one step: feeds each row's previous character (<c>&lt;SOS&gt;</c> first) and returns
    /// [rows, <see cref="Columns"/>] log-probs, the vocabulary mixed with the copy distribution (with
    /// <see cref="Seq2SeqConfig.Copy"/>). The array may be reused by the next call.
    /// </summary>
    float[] Step(long[] previous, CancellationToken ct);

    /// <summary>Beam search's update_state: row r's decoder state (h and c) becomes row <paramref name="rows"/>[r]'s.</summary>
    void Reorder(int[] rows);
}

/// <summary>The network's sizes and options, from the checkpoint config.</summary>
/// <param name="PosDim">The UPOS embedding's size; 0 without POS input (MWT).</param>
/// <param name="Edit">Whether the model has the lemmatizer's 3-way edit classifier.</param>
/// <param name="Copy">Whether the model has the copy gate (<c>copy</c>; older lemmatizers, English craft, genia and mimic, don't).</param>
internal readonly record struct Seq2SeqConfig(int Vocab, int PosVocab, int Emb, int PosDim, int Hidden, bool Edit, bool Copy)
{
    public const int PadId = 0, UnkId = 1, SosId = 2, EosId = 3; // seq2seq_constant.py

    /// <summary>Whether a UPOS embedding is the encoder's first step (the lemmatizer).</summary>
    public bool UsesPos => PosDim > 0;

    public static Seq2SeqConfig From(JsonNode config)
    {
        bool pos = config["pos"]?.GetValue<bool>() == true && config["pos_dim"]?.GetValue<int>() > 0;
        return new(config["vocab_size"]!.GetValue<int>(), pos ? config["pos_vocab_size"]!.GetValue<int>() : 0, config["emb_dim"]!.GetValue<int>(),
            pos ? config["pos_dim"]!.GetValue<int>() : 0, config["hidden_dim"]!.GetValue<int>(), config["edit"]?.GetValue<bool>() == true,
            config["copy"]?.GetValue<bool>() == true);
    }

    /// <summary>
    /// Seq2SeqModel.predict_greedy's loop: one character per step for the whole batch, the argmax (torch's max: the first
    /// maximum) fed back, until every row has decoded <c>&lt;EOS&gt;</c> or <paramref name="maxDecLen"/> steps ran.
    /// </summary>
    /// <param name="neverDecodeUnk">MWT's <c>never_decode_unk</c>: <c>&lt;UNK&gt;</c> is never the argmax.</param>
    /// <param name="onStep">For tests: called after each step with its log-probs, the row width and which rows were done before it.</param>
    /// <returns>Each row's ids before its <c>&lt;EOS&gt;</c>.</returns>
    public static List<int>[] Greedy(ISeq2SeqDecoder decoder, int batch, int maxDecLen, bool neverDecodeUnk, CancellationToken ct,
        Action<float[], int, bool[]>? onStep = null)
    {
        var output = Enumerable.Range(0, batch).Select(_ => new List<int>()).ToArray();
        var previous = new long[batch];
        Array.Fill(previous, SosId);
        var done = new bool[batch];
        int totalDone = 0, columns = decoder.Columns;
        for (int step = 0; totalDone < batch && step < maxDecLen; step++)
        {
            ct.ThrowIfCancellationRequested();
            var logProbs = decoder.Step(previous, ct);
            onStep?.Invoke(logProbs, columns, done);
            for (int i = 0; i < batch; i++)
            {
                var row = logProbs.AsSpan(i * columns, columns);
                int best = 0;
                for (int v = 1; v < columns; v++)
                {
                    if ((!neverDecodeUnk || v != UnkId) && row[v] > row[best])
                        best = v;
                }

                previous[i] = best;
                if (done[i])
                    continue;
                if (best == EosId)
                {
                    done[i] = true;
                    totalDone++;
                }
                else
                {
                    output[i].Add(best);
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Seq2SeqModel.predict with <paramref name="beamSize"/> &gt; 1 (common/beam.py <c>Beam</c>): each source keeps the
    /// <paramref name="beamSize"/> best hypotheses by summed float32 log-prob (torch.topk; ties to the lower index), the first
    /// step expanding beam 0 alone; a source is done when its best hypothesis ends in <c>&lt;EOS&gt;</c>. The decoder must
    /// have been encoded with <c>beam: beamSize</c>. Stanza's MWT trainer runs this with the default 5 (it passes its
    /// <c>beam_size</c> 1 where predict takes <c>pos</c>).
    /// </summary>
    /// <param name="onStep">For tests: as <see cref="Greedy"/>'s, per decoder row (the done flags repeated over the beams).</param>
    /// <returns>Each source's best hypothesis before its first <c>&lt;EOS&gt;</c>.</returns>
    public static List<int>[] Beam(ISeq2SeqDecoder decoder, int batch, int beamSize, int maxDecLen, bool neverDecodeUnk, CancellationToken ct,
        Action<float[], int, bool[]>? onStep = null)
    {
        int k = beamSize, columns = decoder.Columns, rows = batch * k;
        var scores = new float[batch][];
        var prevKs = new List<int[]>[batch];
        var nextYs = new List<int[]>[batch];
        for (int b = 0; b < batch; b++)
        {
            scores[b] = new float[k];
            prevKs[b] = [];
            var first = new int[k]; // <SOS> in beam 0, <PAD> in the others
            first[0] = SosId;
            nextYs[b] = [first];
        }

        var done = new bool[batch];
        var rowDone = new bool[rows];
        var previous = new long[rows];
        var perm = new int[rows];
        var flat = new float[k * columns];
        for (int step = 0; step < maxDecLen; step++)
        {
            ct.ThrowIfCancellationRequested();
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < k; j++)
                    previous[j * batch + b] = nextYs[b][^1][j];
            }

            var logProbs = decoder.Step(previous, ct);
            if (neverDecodeUnk)
            {
                for (int r = 0; r < rows; r++)
                    logProbs[r * columns + UnkId] = float.NegativeInfinity;
            }

            onStep?.Invoke(logProbs, columns, rowDone);
            for (int b = 0; b < batch; b++)
            {
                if (done[b])
                    continue;

                // Beam.advance: the first step expands beam 0 only; later ones every beam's row plus its score.
                int n = prevKs[b].Count == 0 ? columns : k * columns;
                for (int j = 0; j < n / columns; j++)
                {
                    var row = logProbs.AsSpan((j * batch + b) * columns, columns);
                    for (int w = 0; w < columns; w++)
                        flat[j * columns + w] = prevKs[b].Count == 0 ? row[w] : row[w] + scores[b][j];
                }

                var best = TopK(flat.AsSpan(0, n), k);
                var prevK = new int[k];
                var nextY = new int[k];
                for (int j = 0; j < k; j++)
                {
                    scores[b][j] = flat[best[j]];
                    prevK[j] = best[j] / columns;
                    nextY[j] = best[j] - prevK[j] * columns;
                }

                prevKs[b].Add(prevK);
                nextYs[b].Add(nextY);
                done[b] = nextY[0] == EosId;
            }

            if (done.All(d => d))
                break;

            // update_state: each source's beams take their origins' states.
            for (int b = 0; b < batch; b++)
            {
                for (int j = 0; j < k; j++)
                {
                    perm[j * batch + b] = prevKs[b][^1][j] * batch + b;
                    rowDone[j * batch + b] = done[b];
                }
            }

            decoder.Reorder(perm);
        }

        var output = new List<int>[batch];
        for (int b = 0; b < batch; b++)
        {
            // sort_best's first, then get_hyp's walk back over the back pointers, pruned at <EOS> (prune_hyp).
            int at = 0;
            for (int j = 1; j < k; j++)
            {
                if (scores[b][j] > scores[b][at])
                    at = j;
            }

            var hyp = new List<int>();
            for (int t = prevKs[b].Count - 1; t >= 0; t--)
            {
                hyp.Add(nextYs[b][t + 1][at]);
                at = prevKs[b][t][at];
            }

            hyp.Reverse();
            int eos = hyp.IndexOf(EosId);
            output[b] = eos < 0 ? hyp : hyp[..eos];
        }

        return output;
    }

    /// <summary>The indices of the <paramref name="k"/> largest values, largest first; equal values by lower index first.</summary>
    private static int[] TopK(ReadOnlySpan<float> values, int k)
    {
        var best = new int[k];
        int count = 0;
        for (int i = 0; i < values.Length; i++)
        {
            float v = values[i];
            if (count == k && !(v > values[best[k - 1]]))
                continue;
            int at = Math.Min(count, k - 1);
            while (at > 0 && v > values[best[at - 1]])
            {
                best[at] = best[at - 1];
                at--;
            }

            best[at] = i;
            count = Math.Min(count + 1, k);
        }

        return best;
    }
}

/// <summary>
/// Managed twin of <c>Seq2SeqNet</c> (<see cref="Backend.Managed"/>). The encoder is a packed <see cref="ManagedLstm"/>;
/// each decoder step is three GEMMs over the batch (the LSTM cell with x and h in one K, linear_in; linear_out) and one for
/// dec2vocab with the copy gate as an extra output, with the attention and the copy mix per row in between. Sums over
/// attention positions and the vocabulary run in double.
/// </summary>
internal sealed unsafe class ManagedSeq2SeqNet : ISeq2SeqNet
{
    private readonly Seq2SeqConfig _c;
    private readonly float[] _emb;
    private readonly float[]? _posEmb;
    private readonly ManagedLstm _encoder;
    private readonly PackedMatrix _cell, _attnIn, _attnOut, _out;
    private readonly float[] _cellBias, _outBias;
    private readonly float[]? _edit0, _edit0Bias, _edit2, _edit2Bias;

    /// <summary>
    /// Study only (tools/lemma_divergence.py, the benchmark's <c>lemma-divergence --double-gates</c>): the decoder's
    /// LSTMCell gate sums in double (plain loops over the pool) instead of the GEMM. Read when a net is constructed. Off: the shipped path.
    /// </summary>
    internal static bool DoubleGates;
    private readonly float[]? _cellRows;  // with DoubleGates: [4·hidden, emb + hidden], rows in GateOrder
    private readonly double[]? _cellBias64;

    public ManagedSeq2SeqNet(Checkpoint ckpt, Seq2SeqConfig c)
    {
        _c = c;
        var model = ckpt.Root["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        int h = c.Hidden;
        if (h % 4 != 0 || (c.UsesPos && c.PosDim != c.Emb))
            throw new NotSupportedException("The managed seq2seq model needs hidden_dim % 4 == 0 and pos_dim == emb_dim");
        _emb = T("embedding.weight");
        _posEmb = c.UsesPos ? T("pos_embedding.weight") : null;
        _encoder = new ManagedLstm(ckpt, model, "encoder.", c.Emb, h / 2, 1, bidirectional: true);

        // LSTMCell: gates = [x | h]·[W_ih | W_hh]ᵀ + b_ih + b_hh, rows in PackedLstm.GateOrder for Act.LstmCell.
        var wih = T("decoder.lstm_cell.weight_ih");
        var whh = T("decoder.lstm_cell.weight_hh");
        var (bih, bhh) = (T("decoder.lstm_cell.bias_ih"), T("decoder.lstm_cell.bias_hh"));
        int k = c.Emb + h;
        var w = new float[4 * h * k];
        for (int j = 0; j < 4 * h; j++)
        {
            wih.AsSpan(j * c.Emb, c.Emb).CopyTo(w.AsSpan(j * k));
            whh.AsSpan(j * h, h).CopyTo(w.AsSpan(j * k + c.Emb));
        }

        var order = PackedLstm.GateOrder(h);
        _cell = new PackedMatrix(w, 4 * h, k, order);
        _cellBias = new float[_cell.PaddedN];
        for (int j = 0; j < order.Length; j++)
            _cellBias[j] = bih[order[j]] + bhh[order[j]];
        if (DoubleGates)
        {
            _cellRows = new float[order.Length * k];
            _cellBias64 = new double[order.Length];
            for (int j = 0; j < order.Length; j++)
            {
                w.AsSpan(order[j] * k, k).CopyTo(_cellRows.AsSpan(j * k));
                _cellBias64[j] = (double)bih[order[j]] + bhh[order[j]];
            }
        }

        _attnIn = new PackedMatrix(T("decoder.attention_layer.linear_in.weight"), h, h);
        _attnOut = new PackedMatrix(T("decoder.attention_layer.linear_out.weight"), h, 2 * h);

        // dec2vocab and the copy gate (if any) read the same h_tilde: one GEMM, the gate is column Vocab.
        _out = new PackedMatrix(c.Copy ? [.. T("dec2vocab.weight"), .. T("copy_gate.weight")] : T("dec2vocab.weight"), c.Vocab + (c.Copy ? 1 : 0), h);
        _outBias = new float[_out.PaddedN];
        T("dec2vocab.bias").CopyTo(_outBias, 0);
        if (c.Copy)
            _outBias[c.Vocab] = T("copy_gate.bias")[0];

        if (c.Edit)
            (_edit0, _edit0Bias, _edit2, _edit2Bias) = (T("edit_clf.0.weight"), T("edit_clf.0.bias"), T("edit_clf.2.weight"), T("edit_clf.2.bias"));
    }

    public ISeq2SeqDecoder Encode(long[] ids, int batch, int width, long[]? posIds, long[] lengths, CancellationToken ct, int beam = 1) =>
        new Decoder(this, ids, batch, width, posIds, lengths, ct, beam);

    /// <summary>
    /// One batch: the encoder output and the decoder's buffers, rented for the batch and returned on Dispose. The decoder has
    /// rows = batch · beam: row r decodes source r % batch, and only rows below batch mix in the copy distribution.
    /// </summary>
    private sealed class Decoder : ISeq2SeqDecoder
    {
        private readonly ManagedSeq2SeqNet _net;
        private readonly long[] _ids, _lengths;
        private readonly int _batch, _rows, _width, _steps, _first, _h, _ldTarget, _ldOut;
        private readonly float[] _ctx;    // [batch, steps, hidden]: the encoder output, real positions only
        private readonly float[] _xh;     // [rows, emb + hidden]: the cell's input and h
        private readonly float[] _cell;   // [rows, hidden]
        private readonly float[] _gates;  // [rows, 4·hidden]
        private readonly float[] _target; // [rows, ldTarget]: linear_in(h), then h_tilde
        private readonly float[] _wh;     // [rows, 2·hidden]: the attention-weighted context and h
        private readonly float[] _logits; // [rows, ldOut]: dec2vocab and the copy gate
        private readonly float[] _result;

        public float[] EditLogits { get; }
        public int Columns { get; }

        public Decoder(ManagedSeq2SeqNet net, long[] ids, int batch, int width, long[]? posIds, long[] lengths, CancellationToken ct, int beam)
        {
            (_net, _ids, _lengths, _batch, _rows, _width) = (net, ids, lengths, batch, batch * beam, width);
            var c = net._c;

            // The encoder's steps: the UPOS step (if any), then the characters. Source position t is encoder step t + _first.
            _first = c.UsesPos ? 1 : 0;
            int h = _h = c.Hidden, half = h / 2, steps = _steps = width + _first, emb = c.Emb;
            _ldTarget = net._attnIn.PaddedN;
            _ldOut = net._out.PaddedN;
            Columns = c.Copy ? (int)Math.Max(c.Vocab, ids.Max() + 1) : c.Vocab;
            var pool = ArrayPool<float>.Shared;
            _ctx = pool.Rent(batch * steps * h);
            _xh = pool.Rent(_rows * (emb + h));
            _cell = pool.Rent(_rows * h);
            _gates = pool.Rent(_rows * net._cell.PaddedN);
            _target = pool.Rent(_rows * _ldTarget);
            _wh = pool.Rent(_rows * 2 * h);
            _logits = pool.Rent(_rows * _ldOut);
            _result = new float[_rows * Columns];
            EditLogits = new float[c.Edit ? batch * 3 : 0];

            // The packed encoder input: the UPOS embedding first (if any), then the characters (<UNK>'s embedding past the vocabulary).
            var batchSizes = PackedLstm.BatchSizes(lengths);
            int rows = batchSizes.Sum();
            var start = new int[batchSizes.Length];
            for (int t = 1; t < start.Length; t++)
                start[t] = start[t - 1] + batchSizes[t - 1];
            var x = pool.Rent(rows * emb);
            var y = pool.Rent(rows * h);
            var finalC = pool.Rent(2 * batch * half);
            try
            {
                for (int t = 0; t < batchSizes.Length; t++)
                {
                    for (int r = 0; r < batchSizes[t]; r++)
                    {
                        long id = t < _first ? posIds![r] : ids[r * width + t - _first];
                        var table = t < _first ? net._posEmb! : net._emb;
                        if (t >= _first && id >= c.Vocab)
                            id = Seq2SeqConfig.UnkId;
                        table.AsSpan((int)id * emb, emb).CopyTo(x.AsSpan((start[t] + r) * emb));
                    }
                }

                net._encoder.ForwardPacked(x, emb, batchSizes, y, ct, finalC);
                for (int r = 0; r < batch; r++)
                {
                    for (int t = 0; t < lengths[r]; t++)
                        y.AsSpan((start[t] + r) * h, h).CopyTo(_ctx.AsSpan((r * steps + t) * h));

                    // h0 = cat(hn[-1], hn[-2]): the backward state (at t = 0), then the forward one (at the last step).
                    var h0 = _xh.AsSpan(r * (emb + h) + emb, h);
                    y.AsSpan(start[0] * h + r * h + half, half).CopyTo(h0);
                    y.AsSpan((start[lengths[r] - 1] + r) * h, half).CopyTo(h0[half..]);
                    finalC.AsSpan((batch + r) * half, half).CopyTo(_cell.AsSpan(r * h));
                    finalC.AsSpan(r * half, half).CopyTo(_cell.AsSpan(r * h + half));
                    if (c.Edit)
                        Edit(h0, EditLogits.AsSpan(r * 3, 3));
                }

                // The other beams start from their source's state (hn.repeat, cn.repeat).
                for (int r = batch; r < _rows; r++)
                {
                    _xh.AsSpan((r % batch) * (emb + h) + emb, h).CopyTo(_xh.AsSpan(r * (emb + h) + emb));
                    _cell.AsSpan((r % batch) * h, h).CopyTo(_cell.AsSpan(r * h));
                }
            }
            catch
            {
                Dispose();
                throw;
            }
            finally
            {
                pool.Return(x);
                pool.Return(y);
                pool.Return(finalC);
            }
        }

        /// <summary>edit_clf: Linear, ReLU, Linear, summed in double.</summary>
        private void Edit(ReadOnlySpan<float> h0, Span<float> logits)
        {
            var n = _net;
            int inner = n._edit0Bias!.Length;
            Span<double> hidden = stackalloc double[inner];
            for (int j = 0; j < inner; j++)
            {
                double s = n._edit0Bias[j];
                var w = n._edit0.AsSpan(j * _h, _h);
                for (int i = 0; i < _h; i++)
                    s += (double)w[i] * h0[i];
                hidden[j] = Math.Max(s, 0);
            }

            for (int k = 0; k < logits.Length; k++)
            {
                double s = n._edit2Bias![k];
                var w = n._edit2.AsSpan(k * inner, inner);
                for (int j = 0; j < inner; j++)
                    s += w[j] * hidden[j];
                logits[k] = (float)s;
            }
        }

        public float[] Step(long[] previous, CancellationToken ct)
        {
            var n = _net;
            var c = n._c;
            int b = _rows, h = _h, emb = c.Emb, ldxh = emb + h;
            for (int r = 0; r < b; r++)
            {
                long id = previous[r] >= c.Vocab ? Seq2SeqConfig.UnkId : previous[r];
                n._emb.AsSpan((int)id * emb, emb).CopyTo(_xh.AsSpan(r * ldxh));
            }

            fixed (float* xh = _xh, gates = _gates, cell = _cell, cellBias = n._cellBias, target = _target, wh = _wh,
                logits = _logits, outBias = n._outBias)
            {
                // LSTMCell: one GEMM, then the cell update per four units (a 16-column panel).
                int ldg = n._cell.PaddedN;
                if (n._cellRows is { } rows)
                {
                    // Gate rows in chunks of 16 across the pool; each sum in double, 4 rows of x at a time.
                    nint pXhIn = (nint)xh, pGates = (nint)gates;
                    var bias = n._cellBias64!;
                    ManagedThreads.For(bias.Length / 16, chunk =>
                    {
                        var x = (float*)pXhIn;
                        var g = (float*)pGates;
                        for (int j = chunk * 16; j < chunk * 16 + 16; j++)
                        {
                            var wj = rows.AsSpan(j * ldxh, ldxh);
                            for (int r = 0; r < b; r++)
                            {
                                double s = bias[j];
                                var xr = x + r * ldxh;
                                for (int i = 0; i < ldxh; i++)
                                    s += (double)wj[i] * xr[i];
                                g[r * ldg + j] = (float)s;
                            }
                        }
                    });
                }
                else
                {
                    Gemm.Run(xh, b, ldxh, n._cell, cellBias, gates, ldg, ct);
                }

                for (int r = 0; r < b; r++)
                {
                    for (int p = 0; p < h / 4; p++)
                        Act.LstmCell(gates + r * ldg + p * 16, cell + r * h + p * 4, xh + r * ldxh + emb + p * 4);
                }

                Gemm.Run(xh + emb, b, ldxh, n._attnIn, null, target, _ldTarget, ct);

                // SoftDotAttention per row: scores over the real positions, softmax, the weighted context next to h.
                nint pTarget = (nint)target, pWh = (nint)wh, pXh = (nint)xh;
                var attn = ArrayPool<double>.Shared.Rent(b * _steps);
                try
                {
                    ManagedThreads.For(b, r => Attend(r, (float*)pTarget, (float*)pWh, (float*)pXh, attn));
                    Gemm.Run(wh, b, 2 * h, n._attnOut, null, target, _ldTarget, ct);
                    for (int r = 0; r < b; r++)
                    {
                        for (int j = 0; j < h; j++)
                            target[r * _ldTarget + j] = MathF.Tanh(target[r * _ldTarget + j]);
                    }

                    Gemm.Run(target, b, _ldTarget, n._out, outBias, logits, _ldOut, ct);
                    nint pLogits = (nint)logits;
                    ManagedThreads.For(b, r => Mix(r, (float*)pLogits, attn));
                }
                finally
                {
                    ArrayPool<double>.Shared.Return(attn);
                }
            }

            return _result;
        }

        /// <summary>Row r's attention: log-softmax scores kept in <paramref name="attn"/>, the weighted context into wh.</summary>
        private void Attend(int r, float* target, float* wh, float* xh, double[] attn)
        {
            int h = _h, steps = _steps, source = r % _batch, len = (int)_lengths[source];
            var la = attn.AsSpan(r * steps, len);
            var q = new ReadOnlySpan<float>(target + r * _ldTarget, h);
            double max = double.NegativeInfinity;
            for (int t = 0; t < len; t++)
            {
                var k = _ctx.AsSpan((source * steps + t) * h, h);
                double s = 0;
                for (int i = 0; i < h; i++)
                    s += (double)k[i] * q[i];
                la[t] = s;
                max = Math.Max(max, s);
            }

            double sum = 0;
            for (int t = 0; t < len; t++)
                sum += Math.Exp(la[t] - max);
            double lse = max + Math.Log(sum);
            var weighted = new Span<float>(wh + r * 2 * h, h);
            Span<double> acc = stackalloc double[h];
            acc.Clear();
            for (int t = 0; t < len; t++)
            {
                la[t] -= lse;
                double wt = Math.Exp(la[t]);
                var k = _ctx.AsSpan((source * steps + t) * h, h);
                for (int i = 0; i < h; i++)
                    acc[i] += wt * k[i];
            }

            for (int i = 0; i < h; i++)
                weighted[i] = (float)acc[i];
            new ReadOnlySpan<float>(xh + r * (_net._c.Emb + h) + _net._c.Emb, h).CopyTo(new Span<float>(wh + r * 2 * h + h, h));
        }

        /// <summary>
        /// Row r's output: log_softmax(dec2vocab) + log(1 − σ(copy)), mixed (logsumexp) with σ(copy) · the attention over the
        /// source characters (the UPOS step left out, renormalized), scattered onto their ids. Columns nothing is copied to
        /// keep the vocabulary score, as torch's logsumexp with −1e12 gives; DeltaVocab columns use &lt;UNK&gt;'s.
        /// </summary>
        private void Mix(int r, float* logits, double[] attn)
        {
            int vocab = _net._c.Vocab, steps = _steps, len = (int)_lengths[r % _batch], cols = Columns, first = _first;
            var row = new ReadOnlySpan<float>(logits + r * _ldOut, _net._out.N);
            double max = double.NegativeInfinity;
            for (int v = 0; v < vocab; v++)
                max = Math.Max(max, row[v]);
            double sum = 0;
            for (int v = 0; v < vocab; v++)
                sum += Math.Exp(row[v] - max);
            double lse = max + Math.Log(sum);
            var output = _result.AsSpan(r * cols, cols);
            if (!_net._c.Copy)
            {
                // get_log_prob alone: log_softmax over the vocabulary.
                for (int v = 0; v < vocab; v++)
                    output[v] = (float)(row[v] - lse);
                return;
            }

            double copyLogit = row[vocab];
            double logNoCopy = -Math.Log(Math.Exp(copyLogit) + 1);
            double logCopy = Math.Min(copyLogit, 0) - Math.Log(1 + Math.Exp(-Math.Abs(copyLogit))); // logsigmoid

            for (int v = 0; v < cols; v++)
                output[v] = (float)((v < vocab ? row[v] : row[Seq2SeqConfig.UnkId]) - lse + logNoCopy);
            if (r >= _batch)
                return; // a later beam's row: Stanza's scatter_add of the unrepeated source reaches only the first batch rows

            // The attention over the source positions (first..len-1) renormalized: log a_t − log Σ a_t.
            var la = attn.AsSpan(r * steps, len);
            double amax = double.NegativeInfinity;
            for (int t = first; t < len; t++)
                amax = Math.Max(amax, la[t]);
            double asum = 0;
            for (int t = first; t < len; t++)
                asum += Math.Exp(la[t] - amax);
            double alse = amax + Math.Log(asum);

            // copied[id] = Σ_t exp(log_copy_prob_t − mx), mx = the largest log_copy_prob (logCopy + amax − alse).
            var copied = ArrayPool<double>.Shared.Rent(cols);
            try
            {
                Array.Clear(copied, 0, cols);
                for (int t = first; t < len; t++)
                    copied[_ids[r * _width + t - first]] += Math.Exp(la[t] - amax);
                double mx = logCopy + amax - alse;
                for (int v = 0; v < cols; v++)
                {
                    if (copied[v] > 0)
                    {
                        double a = Math.Log(copied[v]) + mx, b = output[v];
                        double m = Math.Max(a, b);
                        output[v] = (float)(m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m)));
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(copied);
            }
        }

        public void Reorder(int[] rows)
        {
            int h = _h, ldxh = _net._c.Emb + h;
            var oldH = new float[_rows * h];
            var oldC = _cell.AsSpan(0, _rows * h).ToArray();
            for (int r = 0; r < _rows; r++)
                _xh.AsSpan(r * ldxh + _net._c.Emb, h).CopyTo(oldH.AsSpan(r * h));
            for (int r = 0; r < _rows; r++)
            {
                oldH.AsSpan(rows[r] * h, h).CopyTo(_xh.AsSpan(r * ldxh + _net._c.Emb));
                oldC.AsSpan(rows[r] * h, h).CopyTo(_cell.AsSpan(r * h));
            }
        }

        public void Dispose()
        {
            var pool = ArrayPool<float>.Shared;
            pool.Return(_ctx);
            pool.Return(_xh);
            pool.Return(_cell);
            pool.Return(_gates);
            pool.Return(_target);
            pool.Return(_wh);
            pool.Return(_logits);
        }
    }

    public void Dispose()
    {
    }
}
