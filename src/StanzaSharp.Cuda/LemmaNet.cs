using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Lemma;

/// <summary>seq2seq_model.py <c>Seq2SeqModel</c> (encode, decode) on TorchSharp (today's code).</summary>
internal sealed class LemmaNet : ILemmaNet
{
    // Scalars kept alive for TorchSharp (see Nn.Scalars): ids, and the masking values the seq2seq model uses.
    private static readonly Scalar PadScalar = LemmaConfig.PadId, UnkScalar = LemmaConfig.UnkId, MaskedScore = -1e12, Epsilon = 1e-12;

    private readonly int _vocabSize;
    private readonly Embedding _embedding, _posEmbedding;
    private readonly LSTM _encoder;
    private readonly LSTMCell _decoderCell;
    private readonly Linear _attnIn, _attnOut, _dec2vocab, _editHidden, _editOutput, _copyGate;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public LemmaNet(Checkpoint ckpt, LemmaConfig c)
    {
        var model = ckpt.Root["model"]!;
        _vocabSize = c.Vocab;
        _embedding = nn.Embedding(c.Vocab, c.Emb, padding_idx: LemmaConfig.PadId).LoadFrom(ckpt, model, "embedding.");
        _posEmbedding = nn.Embedding(c.PosVocab, c.PosDim, padding_idx: LemmaConfig.PadId).LoadFrom(ckpt, model, "pos_embedding.");
        _encoder = nn.LSTM(c.Emb, c.Hidden / 2, numLayers: 1, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "encoder.");
        _decoderCell = nn.LSTMCell(c.Emb, c.Hidden).LoadFrom(ckpt, model, "decoder.lstm_cell.");
        _attnIn = nn.Linear(c.Hidden, c.Hidden, hasBias: false).LoadFrom(ckpt, model, "decoder.attention_layer.linear_in.");
        _attnOut = nn.Linear(c.Hidden * 2, c.Hidden, hasBias: false).LoadFrom(ckpt, model, "decoder.attention_layer.linear_out.");
        _dec2vocab = nn.Linear(c.Hidden, c.Vocab).LoadFrom(ckpt, model, "dec2vocab.");
        _editHidden = nn.Linear(c.Hidden, c.Hidden / 2).LoadFrom(ckpt, model, "edit_clf.0.");
        _editOutput = nn.Linear(c.Hidden / 2, 3).LoadFrom(ckpt, model, "edit_clf.2.");
        _copyGate = nn.Linear(c.Hidden, 1).LoadFrom(ckpt, model, "copy_gate.");
    }

    public ILemmaDecoder Encode(long[] ids, int batch, int width, long[] posIds, long[] lengths, CancellationToken ct) =>
        new Decoder(this, ids, batch, width, posIds, lengths);

    /// <summary>The batch's tensors live in one dispose scope, freed with the decoder; each step's temporaries in their own.</summary>
    private sealed class Decoder : ILemmaDecoder
    {
        private readonly LemmaNet _net;
        private readonly DisposeScope _scope;
        private readonly Tensor _src, _ctx, _srcMask;
        private Tensor _h, _c;
        private readonly int _batch;

        public float[] EditLogits { get; }
        public int Columns { get; }

        public Decoder(LemmaNet net, long[] ids, int batch, int width, long[] posIds, long[] lengths)
        {
            _net = net;
            _batch = batch;
            using var noGrad = torch.no_grad();
            _scope = NewDisposeScope();
            try
            {
                var device = net._device;
                _src = torch.tensor(ids, [batch, width], device: device);
                var pos = torch.tensor(posIds, device: device);
                Columns = (int)Math.Max(net._vocabSize, ids.Max() + 1);
                using var vocabSize = net._vocabSize.ToScalar();

                // embed: characters past the trained vocabulary embed as <UNK>; the POS embedding goes in front.
                var embedSrc = _src.masked_fill(_src.ge(vocabSize), UnkScalar);
                var encInputs = cat([net._posEmbedding.forward(pos).unsqueeze(1), net._embedding.forward(embedSrc)], 1);
                _srcMask = cat([torch.zeros([batch, 1], ScalarType.Bool, device: device), _src.eq(PadScalar)], 1);

                // encode
                using var lens = torch.tensor(lengths);
                var packed = nn.utils.rnn.pack_padded_sequence(encInputs, lens, batch_first: true, enforce_sorted: true);
                var (packedOut, hn, cn) = net._encoder.call(packed, null);
                (_ctx, _) = nn.utils.rnn.pad_packed_sequence(packedOut, batch_first: true);
                _h = cat([hn[1], hn[0]], 1);
                _c = cat([cn[1], cn[0]], 1);
                EditLogits = net._editOutput.forward(F.relu(net._editHidden.forward(_h))).ToArray<float>();
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

        // Copy: renormalize attention without the POS position, then scatter it onto the source character ids.
        var copyLogit = _copyGate.forward(hOut);
        logAttn = F.log_softmax(logAttn[.., .., 1..], -1);
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
            logProbs = cat([logProbs, logProbs[.., .., LemmaConfig.UnkId].unsqueeze(2).expand(batch, steps, vocab - _vocabSize)], 2);
        logProbs = logProbs.add(logNoCopy, Scalars.One);
        return torch.logsumexp(stack([logCopied, logProbs]), 0);
    }

    public void Dispose()
    {
        nn.Module[] modules = [_embedding, _posEmbedding, _encoder, _decoderCell, _attnIn, _attnOut, _dec2vocab, _editHidden, _editOutput, _copyGate];
        foreach (var m in modules)
            m.Dispose();
    }
}

/// <summary>Loads a <see cref="Lemmatizer"/> on either backend.</summary>
internal static class LemmatizerLoad
{
    extension(Lemmatizer)
    {
        /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/lemma/combined_nocharlm</c>.</summary>
        /// <param name="device">Where the model runs on TorchSharp; CPU by default.</param>
        /// <param name="backend">The network's backend; the managed one ignores <paramref name="device"/>.</param>
        public static Lemmatizer Load(string basePath, Device? device = null, Backend backend = Backend.TorchSharp) =>
            backend == Backend.Managed ? Lemmatizer.LoadManaged(basePath) : Weights.On(device, () => new Lemmatizer(Checkpoint.Load(basePath), (ckpt, sizes) => new LemmaNet(ckpt, sizes)));
    }
}
