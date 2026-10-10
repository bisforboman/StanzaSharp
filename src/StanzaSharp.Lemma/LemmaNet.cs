using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Lemma;

/// <summary>The lemmatizer's seq2seq network, per <see cref="Backend"/>: <c>LemmaNet</c> or <see cref="ManagedLemmaNet"/>.</summary>
internal interface ILemmaNet : IDisposable
{
    /// <summary>Seq2SeqModel.encode on one batch, rows sorted longest first; the decoder then runs one step per call.</summary>
    /// <param name="ids">[batch, width] source ids (<c>&lt;SOS&gt;</c> chars <c>&lt;EOS&gt;</c>, padded with 0); ids past the
    /// vocabulary are DeltaVocab characters, embedded as <c>&lt;UNK&gt;</c> but copyable.</param>
    /// <param name="posIds">[batch] UPOS ids, the encoder's first step.</param>
    /// <param name="lengths">Each row's encoder length: its source length + 1 (the UPOS step).</param>
    ILemmaDecoder Encode(long[] ids, int batch, int width, long[] posIds, long[] lengths, CancellationToken ct);
}

/// <summary>One batch's decoder state (the per-call state of <see cref="ILemmaNet"/>).</summary>
internal interface ILemmaDecoder : IDisposable
{
    /// <summary>[batch, 3] edit classifier logits.</summary>
    float[] EditLogits { get; }

    /// <summary>Width of a <see cref="Step"/> row: the vocabulary, widened to the batch's largest source id + 1.</summary>
    int Columns { get; }

    /// <summary>
    /// Seq2SeqModel.decode for one step: feeds each row's previous character (<c>&lt;SOS&gt;</c> first) and returns
    /// [batch, <see cref="Columns"/>] log-probs, the vocabulary mixed with the copy distribution. The array may be reused
    /// by the next call.
    /// </summary>
    float[] Step(long[] previous, CancellationToken ct);
}

/// <summary>The network's sizes, from the checkpoint config.</summary>
internal readonly record struct LemmaConfig(int Vocab, int PosVocab, int Emb, int PosDim, int Hidden)
{
    public const int PadId = 0, UnkId = 1, SosId = 2, EosId = 3; // seq2seq_constant.py

    public static LemmaConfig From(JsonNode config) => new(
        config["vocab_size"]!.GetValue<int>(), config["pos_vocab_size"]!.GetValue<int>(), config["emb_dim"]!.GetValue<int>(),
        config["pos_dim"]!.GetValue<int>(), config["hidden_dim"]!.GetValue<int>());
}

/// <summary>
/// Managed twin of <c>LemmaNet</c> (<see cref="Backend.Managed"/>). The encoder is a packed <see cref="ManagedLstm"/>;
/// each decoder step is three GEMMs over the batch (the LSTM cell with x and h in one K, linear_in; linear_out) and one for
/// dec2vocab with the copy gate as an extra output, with the attention and the copy mix per row in between. Sums over
/// attention positions and the vocabulary run in double.
/// </summary>
internal sealed unsafe class ManagedLemmaNet : ILemmaNet
{
    private readonly LemmaConfig _c;
    private readonly float[] _emb, _posEmb;
    private readonly ManagedLstm _encoder;
    private readonly PackedMatrix _cell, _attnIn, _attnOut, _out;
    private readonly float[] _cellBias, _outBias;
    private readonly float[] _edit0, _edit0Bias, _edit2, _edit2Bias;

    /// <summary>
    /// Study only (tools/lemma_divergence.py, the benchmark's <c>lemma-divergence --double-gates</c>): the decoder's
    /// LSTMCell gate sums in double (plain loops over the pool) instead of the GEMM. Read when a net is constructed. Off: the shipped path.
    /// </summary>
    internal static bool DoubleGates;
    private readonly float[]? _cellRows;  // with DoubleGates: [4·hidden, emb + hidden], rows in GateOrder
    private readonly double[]? _cellBias64;

    public ManagedLemmaNet(Checkpoint ckpt, LemmaConfig c)
    {
        _c = c;
        var model = ckpt.Root["model"]!;
        float[] T(string key) => ckpt.Tensor<float>(model[key] ?? throw new KeyNotFoundException($"Checkpoint has no weight '{key}'"));
        int h = c.Hidden;
        if (h % 4 != 0 || c.PosDim != c.Emb)
            throw new NotSupportedException("The managed lemmatizer needs hidden_dim % 4 == 0 and pos_dim == emb_dim");
        _emb = T("embedding.weight");
        _posEmb = T("pos_embedding.weight");
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
        // dec2vocab and the copy gate read the same h_tilde: one GEMM, the gate is column Vocab.
        _out = new PackedMatrix([.. T("dec2vocab.weight"), .. T("copy_gate.weight")], c.Vocab + 1, h);
        _outBias = new float[_out.PaddedN];
        T("dec2vocab.bias").CopyTo(_outBias, 0);
        _outBias[c.Vocab] = T("copy_gate.bias")[0];

        (_edit0, _edit0Bias, _edit2, _edit2Bias) = (T("edit_clf.0.weight"), T("edit_clf.0.bias"), T("edit_clf.2.weight"), T("edit_clf.2.bias"));
    }

    public ILemmaDecoder Encode(long[] ids, int batch, int width, long[] posIds, long[] lengths, CancellationToken ct) =>
        new Decoder(this, ids, batch, width, posIds, lengths, ct);

    /// <summary>One batch: the encoder output and the decoder's buffers, rented for the batch and returned on Dispose.</summary>
    private sealed class Decoder : ILemmaDecoder
    {
        private readonly ManagedLemmaNet _net;
        private readonly long[] _ids, _lengths;
        private readonly int _batch, _width, _h, _ldTarget, _ldOut;
        private readonly float[] _ctx;    // [batch, width + 1, hidden]: the encoder output, real positions only
        private readonly float[] _xh;     // [batch, emb + hidden]: the cell's input and h
        private readonly float[] _cell;   // [batch, hidden]
        private readonly float[] _gates;  // [batch, 4·hidden]
        private readonly float[] _target; // [batch, ldTarget]: linear_in(h), then h_tilde
        private readonly float[] _wh;     // [batch, 2·hidden]: the attention-weighted context and h
        private readonly float[] _logits; // [batch, ldOut]: dec2vocab and the copy gate
        private readonly float[] _result;

        public float[] EditLogits { get; }
        public int Columns { get; }

        public Decoder(ManagedLemmaNet net, long[] ids, int batch, int width, long[] posIds, long[] lengths, CancellationToken ct)
        {
            (_net, _ids, _lengths, _batch, _width) = (net, ids, lengths, batch, width);
            var c = net._c;
            int h = _h = c.Hidden, half = h / 2, steps = width + 1, emb = c.Emb;
            _ldTarget = net._attnIn.PaddedN;
            _ldOut = net._out.PaddedN;
            Columns = (int)Math.Max(c.Vocab, ids.Max() + 1);
            var pool = ArrayPool<float>.Shared;
            _ctx = pool.Rent(batch * steps * h);
            _xh = pool.Rent(batch * (emb + h));
            _cell = pool.Rent(batch * h);
            _gates = pool.Rent(batch * net._cell.PaddedN);
            _target = pool.Rent(batch * _ldTarget);
            _wh = pool.Rent(batch * 2 * h);
            _logits = pool.Rent(batch * _ldOut);
            _result = new float[batch * Columns];
            EditLogits = new float[batch * 3];

            // The packed encoder input: step 0 the UPOS embedding, then the characters (<UNK>'s embedding past the vocabulary).
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
                    for (int r = 0; r < batchSizes[t]; r++)
                    {
                        long id = t == 0 ? posIds[r] : ids[r * width + t - 1];
                        var table = t == 0 ? net._posEmb : net._emb;
                        if (t > 0 && id >= c.Vocab)
                            id = LemmaConfig.UnkId;
                        table.AsSpan((int)id * emb, emb).CopyTo(x.AsSpan((start[t] + r) * emb));
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
                    Edit(h0, EditLogits.AsSpan(r * 3, 3));
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
            int inner = n._edit0Bias.Length;
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
                double s = n._edit2Bias[k];
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
            int b = _batch, h = _h, emb = c.Emb, ldxh = emb + h;
            for (int r = 0; r < b; r++)
            {
                long id = previous[r] >= c.Vocab ? LemmaConfig.UnkId : previous[r];
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
                    Gemm.Run(xh, b, ldxh, n._cell, cellBias, gates, ldg, ct);
                for (int r = 0; r < b; r++)
                    for (int p = 0; p < h / 4; p++)
                        Act.LstmCell(gates + r * ldg + p * 16, cell + r * h + p * 4, xh + r * ldxh + emb + p * 4);
                Gemm.Run(xh + emb, b, ldxh, n._attnIn, null, target, _ldTarget, ct);

                // SoftDotAttention per row: scores over the real positions, softmax, the weighted context next to h.
                nint pTarget = (nint)target, pWh = (nint)wh, pXh = (nint)xh;
                var attn = ArrayPool<double>.Shared.Rent(b * (_width + 1));
                try
                {
                    ManagedThreads.For(b, r => Attend(r, (float*)pTarget, (float*)pWh, (float*)pXh, attn));
                    Gemm.Run(wh, b, 2 * h, n._attnOut, null, target, _ldTarget, ct);
                    for (int r = 0; r < b; r++)
                        for (int j = 0; j < h; j++)
                            target[r * _ldTarget + j] = MathF.Tanh(target[r * _ldTarget + j]);
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
            int h = _h, steps = _width + 1, len = (int)_lengths[r];
            var la = attn.AsSpan(r * steps, len);
            var q = new ReadOnlySpan<float>(target + r * _ldTarget, h);
            double max = double.NegativeInfinity;
            for (int t = 0; t < len; t++)
            {
                var k = _ctx.AsSpan((r * steps + t) * h, h);
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
                var k = _ctx.AsSpan((r * steps + t) * h, h);
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
            int vocab = _net._c.Vocab, steps = _width + 1, len = (int)_lengths[r], cols = Columns;
            var row = new ReadOnlySpan<float>(logits + r * _ldOut, vocab + 1);
            double max = double.NegativeInfinity;
            for (int v = 0; v < vocab; v++)
                max = Math.Max(max, row[v]);
            double sum = 0;
            for (int v = 0; v < vocab; v++)
                sum += Math.Exp(row[v] - max);
            double lse = max + Math.Log(sum);
            double copyLogit = row[vocab];
            double logNoCopy = -Math.Log(Math.Exp(copyLogit) + 1);
            double logCopy = Math.Min(copyLogit, 0) - Math.Log(1 + Math.Exp(-Math.Abs(copyLogit))); // logsigmoid

            var output = _result.AsSpan(r * cols, cols);
            for (int v = 0; v < cols; v++)
                output[v] = (float)((v < vocab ? row[v] : row[LemmaConfig.UnkId]) - lse + logNoCopy);

            // The attention over source positions 1..len-1 renormalized: log a_t − log Σ a_t.
            var la = attn.AsSpan(r * steps, len);
            double amax = double.NegativeInfinity;
            for (int t = 1; t < len; t++)
                amax = Math.Max(amax, la[t]);
            double asum = 0;
            for (int t = 1; t < len; t++)
                asum += Math.Exp(la[t] - amax);
            double alse = amax + Math.Log(asum);
            // copied[id] = Σ_t exp(log_copy_prob_t − mx), mx = the largest log_copy_prob (logCopy + amax − alse).
            var copied = ArrayPool<double>.Shared.Rent(cols);
            try
            {
                Array.Clear(copied, 0, cols);
                for (int t = 1; t < len; t++)
                    copied[_ids[r * _width + t - 1]] += Math.Exp(la[t] - amax);
                double mx = logCopy + amax - alse;
                for (int v = 0; v < cols; v++)
                    if (copied[v] > 0)
                    {
                        double a = Math.Log(copied[v]) + mx, b = output[v];
                        double m = Math.Max(a, b);
                        output[v] = (float)(m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m)));
                    }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(copied);
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
