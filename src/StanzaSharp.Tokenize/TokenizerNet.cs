using System.Text.Json.Nodes;
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
internal sealed class TokenizerNet : IDisposable
{
    private readonly Embedding _embeddings;
    private readonly LSTM _rnn, _rnn2;
    private readonly Linear _tokClf, _sentClf, _mwtClf, _tokClf2, _sentClf2, _mwtClf2;
    private readonly double _hierInvTemp;

    public TokenizerNet(Checkpoint ckpt)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        Require(config["conv_res"] == null, "conv_res");
        Require(config["use_dictionary"]?.GetValue<bool>() != true, "use_dictionary");
        Require(config["charlm"]?.GetValue<bool>() != true, "charlm");
        Require(config["hierarchical"]!.GetValue<bool>(), "non-hierarchical models");
        Require(config["use_mwt"]!.GetValue<bool>(), "models without use_mwt");

        int vocab = config["vocab_size"]!.GetValue<int>();
        int emb = config["emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["rnn_layers"]!.GetValue<int>();
        int inputDim = emb + config["feat_dim"]!.GetValue<int>();
        _hierInvTemp = config["hier_invtemp"]!.GetValue<double>();

        _embeddings = nn.Embedding(vocab, emb, padding_idx: 0).LoadFrom(ckpt, model, "embeddings.");
        _rnn = nn.LSTM(inputDim, hidden, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "rnn.");
        _rnn2 = nn.LSTM(hidden * 2, hidden, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "rnn2.");
        _tokClf = nn.Linear(hidden * 2, 1).LoadFrom(ckpt, model, "tok_clf.");
        _sentClf = nn.Linear(hidden * 2, 1).LoadFrom(ckpt, model, "sent_clf.");
        _mwtClf = nn.Linear(hidden * 2, 1).LoadFrom(ckpt, model, "mwt_clf.");
        _tokClf2 = nn.Linear(hidden * 2, 1, hasBias: false).LoadFrom(ckpt, model, "tok_clf2.");
        _sentClf2 = nn.Linear(hidden * 2, 1, hasBias: false).LoadFrom(ckpt, model, "sent_clf2.");
        _mwtClf2 = nn.Linear(hidden * 2, 1, hasBias: false).LoadFrom(ckpt, model, "mwt_clf2.");
    }

    /// <param name="units">[batch, len] int64 character ids.</param>
    /// <param name="feats">[batch, len, feat_dim] float32 features.</param>
    /// <returns>[batch, len, 5] log-probabilities.</returns>
    public Tensor Forward(Tensor units, Tensor feats)
    {
        using var scope = NewDisposeScope();
        var emb = cat([_embeddings.forward(units), feats], 2);
        var (inp, _, _) = _rnn.forward(emb);

        var tok0 = _tokClf.forward(inp);
        var sent0 = _sentClf.forward(inp);
        var mwt0 = _mwtClf.forward(inp);

        var inp2 = inp * (1 - sigmoid(-tok0 * _hierInvTemp));
        var (hid2, _, _) = _rnn2.forward(inp2);
        tok0 = tok0 + _tokClf2.forward(hid2);
        sent0 = sent0 + _sentClf2.forward(hid2);
        mwt0 = mwt0 + _mwtClf2.forward(hid2);

        var tok = F.logsigmoid(tok0);
        var sent = F.logsigmoid(sent0);
        var nonsent = F.logsigmoid(-sent0);
        var mwt = F.logsigmoid(mwt0);
        var nonmwt = F.logsigmoid(-mwt0);
        var pred = cat([F.logsigmoid(-tok0), tok + nonsent + nonmwt, tok + sent + nonmwt, tok + nonsent + mwt, tok + sent + mwt], 2);
        return pred.MoveToOuterDisposeScope();
    }

    private static void Require(bool ok, string what)
    {
        if (!ok)
            throw new NotSupportedException($"Tokenizer checkpoint uses {what}, which is not ported");
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _embeddings, _rnn, _rnn2, _tokClf, _sentClf, _mwtClf, _tokClf2, _sentClf2, _mwtClf2 })
            m.Dispose();
    }
}
