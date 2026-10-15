using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Nn;

/// <summary>seq2seq_model.py <c>Seq2SeqModel</c> (encode, decode) on TorchSharp: the lemmatizer's and German MWT's model.</summary>
internal sealed class Seq2SeqNet : ISeq2SeqNet
{
    // Scalars kept alive for TorchSharp (see Nn.Scalars): ids, and the masking values the seq2seq model uses.
    private static readonly Scalar PadScalar = Seq2SeqConfig.PadId, UnkScalar = Seq2SeqConfig.UnkId, MaskedScore = -1e12, Epsilon = 1e-12;

    private readonly int _vocabSize;
    private readonly bool _usesPos;
    private readonly Embedding _embedding;
    private readonly Embedding? _posEmbedding;
    private readonly LSTM _encoder;
    private readonly LSTMCell _decoderCell;
    private readonly Linear _attnIn, _attnOut, _dec2vocab;
    private readonly Linear? _copyGate; // null without copy (older lemmatizers)
    private readonly Linear? _editHidden, _editOutput;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public Seq2SeqNet(Checkpoint ckpt, Seq2SeqConfig c)
    {
        var model = ckpt.Root["model"]!;
        _vocabSize = c.Vocab;
        _usesPos = c.UsesPos;
        _embedding = nn.Embedding(c.Vocab, c.Emb, padding_idx: Seq2SeqConfig.PadId).LoadFrom(ckpt, model, "embedding.");
        if (c.UsesPos)
            _posEmbedding = nn.Embedding(c.PosVocab, c.PosDim, padding_idx: Seq2SeqConfig.PadId).LoadFrom(ckpt, model, "pos_embedding.");
        _encoder = nn.LSTM(c.Emb, c.Hidden / 2, numLayers: 1, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "encoder.");
        _decoderCell = nn.LSTMCell(c.Emb, c.Hidden).LoadFrom(ckpt, model, "decoder.lstm_cell.");
        _attnIn = nn.Linear(c.Hidden, c.Hidden, hasBias: false).LoadFrom(ckpt, model, "decoder.attention_layer.linear_in.");
        _attnOut = nn.Linear(c.Hidden * 2, c.Hidden, hasBias: false).LoadFrom(ckpt, model, "decoder.attention_layer.linear_out.");
        _dec2vocab = nn.Linear(c.Hidden, c.Vocab).LoadFrom(ckpt, model, "dec2vocab.");
        if (c.Edit)
        {
            _editHidden = nn.Linear(c.Hidden, c.Hidden / 2).LoadFrom(ckpt, model, "edit_clf.0.");
            _editOutput = nn.Linear(c.Hidden / 2, 3).LoadFrom(ckpt, model, "edit_clf.2.");
        }

        if (c.Copy)
            _copyGate = nn.Linear(c.Hidden, 1).LoadFrom(ckpt, model, "copy_gate.");
    }

    public ISeq2SeqDecoder Encode(long[] ids, int batch, int width, long[]? posIds, long[] lengths, CancellationToken ct, int beam = 1) =>
        new Decoder(this, ids, batch, width, posIds, lengths, beam);

    /// <summary>The batch's tensors live in one dispose scope, freed with the decoder; each step's temporaries in their own.</summary>
    private sealed class Decoder : ISeq2SeqDecoder
    {
        private readonly Seq2SeqNet _net;
        private readonly DisposeScope _scope;
        private readonly Tensor _src, _ctx, _srcMask;
        private Tensor _h, _c;
        private readonly int _batch;

        public float[] EditLogits { get; } = [];
        public int Columns { get; }

        public Decoder(Seq2SeqNet net, long[] ids, int batch, int width, long[]? posIds, long[] lengths, int beam)
        {
            _net = net;
            _batch = batch * beam;
            using var noGrad = torch.no_grad();
            _scope = NewDisposeScope();
            try
            {
                var device = net._device;
                _src = torch.tensor(ids, [batch, width], device: device);
                Columns = net._copyGate != null ? (int)Math.Max(net._vocabSize, ids.Max() + 1) : net._vocabSize;
                using var vocabSize = net._vocabSize.ToScalar();

                // embed: characters past the trained vocabulary embed as <UNK>; the POS embedding (if any) goes in front.
                var encInputs = net._embedding.forward(_src.masked_fill(_src.ge(vocabSize), UnkScalar));
                _srcMask = _src.eq(PadScalar);
                if (net._usesPos)
                {
                    var pos = torch.tensor(posIds!, device: device);
                    encInputs = cat([net._posEmbedding!.forward(pos).unsqueeze(1), encInputs], 1);
                    _srcMask = cat([torch.zeros([batch, 1], ScalarType.Bool, device: device), _srcMask], 1);
                }

                // encode
                using var lens = torch.tensor(lengths);
                var packed = nn.utils.rnn.pack_padded_sequence(encInputs, lens, batch_first: true, enforce_sorted: true);
                var (packedOut, hn, cn) = net._encoder.call(packed, null);
                (_ctx, _) = nn.utils.rnn.pad_packed_sequence(packedOut, batch_first: true);
                _h = cat([hn[1], hn[0]], 1);
                _c = cat([cn[1], cn[0]], 1);
                if (net._editHidden != null)
                    EditLogits = net._editOutput!.forward(F.relu(net._editHidden.forward(_h))).ToArray<float>();

                // Beam search: h_in, src_mask, hn and cn repeated beam times; src (for the copy) is not, as in Stanza.
                if (beam > 1)
                {
                    (_ctx, _srcMask) = (_ctx.repeat(beam, 1, 1), _srcMask.repeat(beam, 1));
                    (_h, _c) = (_h.repeat(beam, 1), _c.repeat(beam, 1));
                }
            }
            catch
            {
                _scope.Dispose();
                throw;
            }
        }

        public float[] Step(long[] previous, CancellationToken ct)
        {
            using var noGrad = torch.no_grad();
            using var stepScope = NewDisposeScope();
            using var vocabSize = _net._vocabSize.ToScalar();
            var preds = torch.tensor(previous, [_batch, 1], device: _net._device);
            var decInputs = _net._embedding.forward(preds.masked_fill(preds.ge(vocabSize), UnkScalar));
            var logProbs = _net.Decode(decInputs, ref _h, ref _c, _ctx, _srcMask, _src);
            _h.MoveToOuterDisposeScope();
            _c.MoveToOuterDisposeScope();
            return logProbs.ToArray<float>();
        }

        public void Reorder(int[] rows)
        {
            using var noGrad = torch.no_grad();
            using var stepScope = NewDisposeScope();
            var index = torch.tensor(rows.Select(r => (long)r).ToArray(), device: _net._device);
            _h = _h.index_select(0, index).MoveToOuterDisposeScope();
            _c = _c.index_select(0, index).MoveToOuterDisposeScope();
        }

        public void Dispose() => _scope.Dispose();
    }

    /// <summary>
    /// Seq2SeqModel.decode for one step: LSTMAttention with SoftDotAttention, then the vocabulary
    /// distribution mixed with the copy distribution over the source characters.
    /// </summary>
    private Tensor Decode(Tensor decInputs, ref Tensor h, ref Tensor c, Tensor ctx, Tensor ctxMask, Tensor src)
    {
        // LSTMAttention.forward (batch_first) over a single step.
        var input = decInputs.transpose(0, 1);
        (h, c) = _decoderCell.forward(input[0], (h, c));
        var attn = torch.bmm(ctx, _attnIn.forward(h).unsqueeze(2)).squeeze(2).masked_fill(ctxMask, MaskedScore);
        attn = F.log_softmax(attn, 1);
        var attnW = torch.exp(attn);
        var weighted = torch.bmm(attnW.view(attnW.shape[0], 1, attnW.shape[1]), ctx).squeeze(1);
        var hTilde = torch.tanh(_attnOut.forward(cat([weighted, h], 1)));
        var hOut = cat([hTilde], 0).view(1, hTilde.shape[0], hTilde.shape[1]).transpose(0, 1);
        var logAttn = stack([attn], 0).transpose(0, 1);

        long batch = hOut.shape[0], steps = hOut.shape[1];
        var logits = _dec2vocab.forward(hOut.contiguous().view(batch * steps, -1)).view(batch, steps, -1);
        var logProbs = F.log_softmax(logits.view(-1, _vocabSize), 1).view(batch, steps, _vocabSize);
        if (_copyGate == null)
            return logProbs;

        // Copy: renormalize attention (without the POS position: it can't be copied), then scatter it onto the source character ids.
        var copyLogit = _copyGate.forward(hOut);
        if (_usesPos)
            logAttn = logAttn[.., .., 1..];
        logAttn = F.log_softmax(logAttn, -1);
        var logCopyProb = F.logsigmoid(copyLogit).add(logAttn, Scalars.One);
        var mx = logCopyProb.max(-1, keepdim: true).values;
        var copyProb = torch.exp(logCopyProb - mx);
        long vocab = Math.Max(_vocabSize, src.max().ToArray<long>()[0] + 1);
        var scattered = src.unsqueeze(1).expand(src.shape[0], copyProb.shape[1], src.shape[1]);
        var copied = torch.zeros([batch, steps, vocab], device: _device).scatter_add(-1, scattered, copyProb);
        var zeroMask = copied.eq(Scalars.Zero);
        var logCopied = torch.log(copied.masked_fill(zeroMask, Epsilon)).add(mx, Scalars.One).masked_fill(zeroMask, MaskedScore);

        var logNoCopy = -torch.log(torch.exp(copyLogit).add(Scalars.One, Scalars.One));
        if (vocab > _vocabSize) // characters new to the vocabulary reuse the <UNK> score
            logProbs = cat([logProbs, logProbs[.., .., Seq2SeqConfig.UnkId].unsqueeze(2).expand(batch, steps, vocab - _vocabSize)], 2);
        logProbs = logProbs.add(logNoCopy, Scalars.One);
        return torch.logsumexp(stack([logCopied, logProbs]), 0);
    }

    public void Dispose()
    {
        nn.Module?[] modules = [_embedding, _posEmbedding, _encoder, _decoderCell, _attnIn, _attnOut, _dec2vocab, _editHidden, _editOutput, _copyGate];
        foreach (var m in modules)
            m?.Dispose();
    }
}
