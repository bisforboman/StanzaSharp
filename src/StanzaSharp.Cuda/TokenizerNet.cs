using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Tokenize;

/// <summary>
/// Port of stanza/models/tokenization/model.py <c>Tokenizer</c> for the options the English model
/// uses: char embedding + features, a biLSTM, and the hierarchical second biLSTM with MWT heads.
/// Output per character: log-probabilities of [continue, token end, sentence end, MWT end, MWT+sentence end].
/// </summary>
internal sealed class TokenizerNet : ITokenizerNet
{
    private static readonly Scalar NegativeInfinity = double.NegativeInfinity; // kept alive for TorchSharp (see Nn.Scalars)

    private readonly Embedding _embeddings;
    private readonly LSTM _rnn, _rnn2;
    private readonly Linear _tokClf, _sentClf, _tokClf2, _sentClf2;
    private readonly Linear? _mwtClf, _mwtClf2; // without use_mwt (English eslspok and the biomedical tokenizers): none
    private readonly double _hierInvTemp;
    private readonly int _charDim; // the charlm's hidden size for a _charlm model, else 0
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public TokenizerNet(Checkpoint ckpt)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        Require(config["conv_res"] == null, "conv_res");
        Require(config["hierarchical"]!.GetValue<bool>(), "non-hierarchical models");

        int vocab = config["vocab_size"]!.GetValue<int>();
        int emb = config["emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["rnn_layers"]!.GetValue<int>();
        int inputDim = (int)ckpt.Shape(model["rnn.weight_ih_l0"])[1]; // embedding + the charlm's hidden size (if any) + features
        _charDim = inputDim - emb - config["feat_dim"]!.GetValue<int>();
        _hierInvTemp = config["hier_invtemp"]!.GetValue<double>();

        _embeddings = nn.Embedding(vocab, emb, padding_idx: 0).LoadFrom(ckpt, model, "embeddings.");
        _rnn = nn.LSTM(inputDim, hidden, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "rnn.");
        _rnn2 = nn.LSTM(hidden * 2, hidden, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "rnn2.");
        _tokClf = nn.Linear(hidden * 2, 1).LoadFrom(ckpt, model, "tok_clf.");
        _sentClf = nn.Linear(hidden * 2, 1).LoadFrom(ckpt, model, "sent_clf.");
        bool useMwt = config["use_mwt"]!.GetValue<bool>();
        _mwtClf = useMwt ? nn.Linear(hidden * 2, 1).LoadFrom(ckpt, model, "mwt_clf.") : null;
        _tokClf2 = nn.Linear(hidden * 2, 1, hasBias: false).LoadFrom(ckpt, model, "tok_clf2.");
        _sentClf2 = nn.Linear(hidden * 2, 1, hasBias: false).LoadFrom(ckpt, model, "sent_clf2.");
        _mwtClf2 = useMwt ? nn.Linear(hidden * 2, 1, hasBias: false).LoadFrom(ckpt, model, "mwt_clf2.") : null;
    }

    public float[] Forward(long[] ids, float[] feats, float[]? charlm, int rows, int width, long[] lengths, CancellationToken ct)
    {
        if ((_charDim > 0) != (charlm != null))
            throw new ArgumentException(_charDim > 0 ? "This tokenizer reads charlm states" : "This tokenizer reads no charlm states", nameof(charlm));
        using var _ = torch.no_grad();
        using var units = torch.tensor(ids, [rows, width], device: _device);
        using var featTensor = torch.tensor(feats, [rows, width, feats.Length / (rows * width)], device: _device);
        using var charTensor = charlm == null ? null : torch.tensor(charlm.AsSpan(0, rows * width * _charDim).ToArray(), [rows, width, _charDim], device: _device);
        using var pred = Forward(units, featTensor, lengths, charTensor);
        return pred.ToArray<float>();
    }

    /// <param name="units">[batch, len] int64 character ids.</param>
    /// <param name="feats">[batch, len, feat_dim] float32 features.</param>
    /// <param name="lengths">Per-row length both LSTMs are packed at, as model.forward does.</param>
    /// <param name="charlm">[batch, len, charlm hidden] forward charlm states for a <c>_charlm</c> model; else null.</param>
    /// <returns>[batch, len, 5] log-probabilities.</returns>
    public Tensor Forward(Tensor units, Tensor feats, long[] lengths, Tensor? charlm = null)
    {
        using var scope = NewDisposeScope();
        var emb = cat(charlm is null ? [_embeddings.forward(units), feats] : [_embeddings.forward(units), charlm, feats], 2);
        var inp = Rnn.RunPacked(_rnn, emb, lengths);

        var tok0 = _tokClf.forward(inp);
        var sent0 = _sentClf.forward(inp);
        var mwt0 = _mwtClf?.forward(inp);

        // inp * (1 - sigmoid(-tok0 * hier_invtemp)), with 1 - x computed as TorchSharp's operator does: -x + 1.
        using var invTemp = _hierInvTemp.ToScalar();
        var inp2 = inp * sigmoid((-tok0).mul(invTemp)).neg().add(Scalars.One, Scalars.One);
        var hid2 = Rnn.RunPacked(_rnn2, inp2, lengths);
        tok0 = tok0.add(_tokClf2.forward(hid2), Scalars.One);
        sent0 = sent0.add(_sentClf2.forward(hid2), Scalars.One);

        var tok = F.logsigmoid(tok0);
        var sent = F.logsigmoid(sent0);
        var nonsent = F.logsigmoid(-sent0);
        if (mwt0 is null)
        {
            // [nontok, tok + nonsent, tok + sent], padded to the five classes with -inf (they never win the argmax).
            var none = full_like(tok, NegativeInfinity);
            return cat([F.logsigmoid(-tok0), tok.add(nonsent, Scalars.One), tok.add(sent, Scalars.One), none, none], 2).MoveToOuterDisposeScope();
        }

        mwt0 = mwt0.add(_mwtClf2!.forward(hid2), Scalars.One);
        var mwt = F.logsigmoid(mwt0);
        var nonmwt = F.logsigmoid(-mwt0);
        Tensor Sum(Tensor a, Tensor b, Tensor c) => a.add(b, Scalars.One).add(c, Scalars.One);
        var pred = cat([F.logsigmoid(-tok0), Sum(tok, nonsent, nonmwt), Sum(tok, sent, nonmwt), Sum(tok, nonsent, mwt), Sum(tok, sent, mwt)], 2);
        return pred.MoveToOuterDisposeScope();
    }

    private static void Require(bool ok, string what)
    {
        if (!ok)
            throw new NotSupportedException($"Tokenizer checkpoint uses {what}, which is not ported");
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module?[] { _embeddings, _rnn, _rnn2, _tokClf, _sentClf, _mwtClf, _tokClf2, _sentClf2, _mwtClf2 })
            m?.Dispose();
    }
}

/// <summary>Loads a <see cref="Tokenizer"/> on either backend.</summary>
internal static class TokenizerLoad
{
    extension(Tokenizer)
    {
        /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/tokenize/combined_nocharlm</c>.</summary>
        /// <param name="device">Where the model runs on TorchSharp; CPU by default.</param>
        /// <param name="backend">The network's backend; the managed one ignores <paramref name="device"/>.</param>
        /// <param name="charlm">The forward charlm of a <c>_charlm</c> model (TorchSharp; the managed backend takes its own, <see cref="Tokenizer.LoadManaged"/>).</param>
        public static Tokenizer Load(string basePath, Device? device = null, Backend backend = Backend.TorchSharp, CharLanguageModel? charlm = null) =>
            backend == Backend.Managed ? Tokenizer.LoadManaged(basePath) : Weights.On(device, () => new Tokenizer(Checkpoint.Load(basePath), ckpt => new TokenizerNet(ckpt), charlm));
    }
}
