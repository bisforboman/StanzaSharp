using System.Buffers;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Pos;

/// <summary>stanza/models/pos/model.py <c>Tagger.forward</c> on TorchSharp.</summary>
internal sealed class PosNet : IPosNet
{
    private readonly Pretrain? _pretrain;
    private readonly CharLanguageModel? _charlmForward, _charlmBackward;
    private readonly CharacterModel? _charModel;
    private readonly Linear? _transChar;
    private readonly Embedding _wordEmb, _uposEmb;
    private readonly Linear? _transPretrained;
    private readonly HighwayLstm _lstm;
    private readonly Linear _uposHid, _xposHid, _featsHid;
    private readonly Tensor _uposClfWeight, _uposClfBias; // upos_clf in float64 (see Forward)
    private readonly Biaffine[] _xposClf; // one per part of a composite XPOS vocabulary
    private readonly Biaffine[] _featsClf;
    private readonly Device _device = Weights.Device; // the device the model was loaded on
    private readonly int _transformerDim;

    /// <param name="pretrain">The word vectors, or null for a <c>_nopretrain</c> tagger (no trans_pretrained columns).</param>
    /// <param name="transformerDim">The transformer's hidden size for a <c>_electra-large</c> tagger (the last input columns), else 0.</param>
    public PosNet(Checkpoint ckpt, int words, int upos, XposVocab xpos, int[] feats, Pretrain? pretrain, CharLanguageModel? charlmForward, CharLanguageModel? charlmBackward,
        int transformerDim = 0)
    {
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int biaff = config["deep_biaff_hidden_dim"]!.GetValue<int>();
        int compositeBiaff = config["composite_deep_biaff_hidden_dim"]!.GetValue<int>();
        int tagEmb = config["tag_emb_dim"]!.GetValue<int>();
        int transformed = config["transformed_dim"]!.GetValue<int>();
        int wordEmb = config["word_emb_dim"]!.GetValue<int>();
        int inputSize = wordEmb + (pretrain == null ? 0 : transformed);
        if (_charlmForward != null)
        {
            inputSize += _charlmForward.HiddenDim + _charlmBackward!.HiddenDim;
        }
        else
        {
            _charModel = new CharacterModel(ckpt, model, config, ckpt.Root["vocab"]!["char"]!, "charmodel.",
                bidirectional: config["char_bidirectional"]?.GetValue<bool>() == true, attention: true);
            _transChar = nn.Linear(_charModel.OutputDim, transformed, hasBias: false).LoadFrom(ckpt, model, "trans_char.");
            inputSize += transformed;
        }

        _wordEmb = nn.Embedding(words, wordEmb, padding_idx: 0).LoadFrom(ckpt, model, "word_emb.");
        _uposEmb = nn.Embedding(upos, tagEmb, padding_idx: 0).LoadFrom(ckpt, model, "upos_emb.");
        _transPretrained = pretrain == null ? null : nn.Linear(pretrain.Dim, transformed, hasBias: false).LoadFrom(ckpt, model, "trans_pretrained.");
        _transformerDim = transformerDim;
        inputSize += transformerDim;
        _lstm = new HighwayLstm(ckpt, model, "taggerlstm", inputSize, hidden, config["num_layers"]!.GetValue<int>());

        _uposHid = nn.Linear(hidden * 2, biaff).LoadFrom(ckpt, model, "upos_hid.");
        using (var weight = ckpt.ToTensor(model["upos_clf.weight"]))
        using (var bias = ckpt.ToTensor(model["upos_clf.bias"]))
            (_uposClfWeight, _uposClfBias) = (weight.to(ScalarType.Float64), bias.to(ScalarType.Float64));
        int xposHid = xpos.IsComposite ? compositeBiaff : biaff;
        _xposHid = nn.Linear(hidden * 2, xposHid).LoadFrom(ckpt, model, "tag_hid.xpos.");
        _xposClf = xpos.IsComposite
            ? [.. xpos.Sizes.Select((n, i) => new Biaffine(ckpt, model, $"tag_clf.xpos.{i}.", xposHid, tagEmb, n))]
            : [new Biaffine(ckpt, model, "tag_clf.xpos.", xposHid, tagEmb, xpos.Sizes[0])];
        _featsHid = nn.Linear(hidden * 2, compositeBiaff).LoadFrom(ckpt, model, "tag_hid.feats.");
        _featsClf = feats.Select((n, i) => new Biaffine(ckpt, model, $"tag_clf.feats.{i}.", compositeBiaff, tagEmb, n)).ToArray();
    }

    public PosOutput Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] wordIds, long[] pretrainIds,
        CharlmCache? charlms, IReadOnlyList<Sentence?>? cacheKeys, CancellationToken ct, float[][]? transformer = null)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        int batch = sentences.Count, width = sentences.Max(s => s.Count);
        var lengths = sentences.Select(s => (long)s.Count).ToArray();

        // Nothing is padded: the input is built straight in the packed order of Stanza's pack_padded_sequence, so a
        // long sentence in a batch of 250 costs only its own rows. Word k of the batch (sentences in order) is
        // packed row packedRow[k], and packed row p holds word wordOf[p].
        var offsets = new int[batch];
        for (int i = 1; i < batch; i++)
            offsets[i] = offsets[i - 1] + sentences[i - 1].Count;
        var wordOf = Rnn.PackedOrder(lengths).Select(x => (long)offsets[x / width] + x % width).ToArray();
        var packedRow = new long[wordOf.Length];
        for (int p = 0; p < wordOf.Length; p++)
            packedRow[wordOf[p]] = p;
        var toPacked = torch.tensor(wordOf, device: _device);
        Tensor Packed(Tensor t)
        {
            using (t)
                return t.index_select(0, toPacked);
        }

        var words = _wordEmb.forward(torch.tensor(wordOf.Select(k => wordIds[k]).ToArray(), device: _device));
        Tensor[] pretrained = _pretrain == null
            ? []
            : [_transPretrained!.forward(_pretrain.Embeddings[torch.tensor(wordOf.Select(k => pretrainIds[k]).ToArray(), device: _device)])];
        Tensor[] chars;
        if (_charModel != null)
        {
            chars = [_transChar!.forward(Packed(_charModel.ForwardWords(sentences.Select(s => (IReadOnlyList<long[]>)s.Select(_charModel.CharIds).ToList()).ToList())))];
        }
        else
        {
            var repsForward = _charlmForward!.BuildCharRepresentation(sentences);
            ct.ThrowIfCancellationRequested();
            var repsBackward = _charlmBackward!.BuildCharRepresentation(sentences);
            if (cacheKeys != null)
            {
                for (int i = 0; i < batch; i++)
                {
                    if (cacheKeys[i] is { } key)
                        charlms!.TryAdd(key, repsForward[i], repsBackward[i]);
                }
            }

            chars = [Packed(cat(repsForward, 0)), Packed(cat(repsBackward, 0))];
        }

        ct.ThrowIfCancellationRequested();
        if (transformer != null)
            chars = [.. chars, Packed(TransformerRows.Tensor(transformer, _transformerDim, _device))];
        var input = cat([words, .. pretrained, .. chars], 1);
        foreach (var t in (Tensor[])[words, .. pretrained, .. chars])
            t.Dispose();
        var packed = Rnn.Pack(input, lengths);
        input.Dispose();

        // The heads see the words in sentence order.
        var output = _lstm.Forward(packed, disposeInput: true, ct).data.index_select(0, torch.tensor(packedRow, device: _device));

        ct.ThrowIfCancellationRequested();

        // upos_clf sums in double, rounded to float once, as the managed net does: German logits reach |380|, where a float
        // sum's last bits depend on libtorch's CPU kernels (Intel AVX-512 drifted 1.2e-4 from Stanza, AMD 6.1e-5; round 26).
        var uposScores = F.linear(F.relu(_uposHid.forward(output)).to(ScalarType.Float64), _uposClfWeight, _uposClfBias).to(ScalarType.Float32);
        var uposIds = uposScores.argmax(1);
        var parent = _uposEmb.forward(uposIds);
        var xposHid = F.relu(_xposHid.forward(output));
        var xposIds = _xposClf.Select(c => c.Forward(xposHid, parent).argmax(1).ToArray<long>()).ToArray();
        ct.ThrowIfCancellationRequested();
        var featsHid = F.relu(_featsHid.forward(output));
        var featIds = _featsClf.Select(c => c.Forward(featsHid, parent).argmax(1).ToArray<long>()).ToArray();
        return new(uposScores.ToArray<float>(), uposIds.ToArray<long>(), xposIds, featIds);
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _wordEmb, _uposEmb, _uposHid, _xposHid, _featsHid })
            m.Dispose();
        _transPretrained?.Dispose();
        _uposClfWeight.Dispose();
        _uposClfBias.Dispose();
        _lstm.Dispose();
        _charModel?.Dispose();
        _transChar?.Dispose();
        foreach (var c in _xposClf.Concat(_featsClf))
            c.Dispose();
    }
}

/// <summary>Loads a <see cref="PosTagger"/> on TorchSharp.</summary>
internal static class PosTaggerLoad
{
    extension(PosTagger)
    {
        /// <summary>
        /// Loads e.g. <c>models/converted/en/pos/combined_charlm</c> on TorchSharp. The pretrain and charlms are
        /// shared with the other processors, so the caller owns them. A <c>_nocharlm</c> model takes none.
        /// </summary>
        /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
        /// <param name="transformer">The transformer a <c>_electra-large</c> tagger reads (shared, the caller owns it).</param>
        public static PosTagger Load(string basePath, Pretrain? pretrain, CharLanguageModel? charlmForward, CharLanguageModel? charlmBackward, Device? device = null,
            TransformerEmbedder? transformer = null) =>
            Weights.On(device, () =>
            {
                var ckpt = Checkpoint.Load(basePath);
                return new PosTagger(ckpt, pretrain, (charlm, feats) =>
                {
                    if (charlm && (charlmForward == null || charlmBackward == null || !charlmForward.IsForward || charlmBackward.IsForward))
                        throw new ArgumentException("This tagger needs the forward charlm, then the backward one");
                    var (words, upos, xpos) = PosTagger.Counts(ckpt);
                    return new PosNet(ckpt, words, upos, xpos, feats, PosTagger.UsesPretrain(ckpt) ? pretrain : null, charlm ? charlmForward : null, charlm ? charlmBackward : null,
                        ckpt.Root["config"]!["bert_model"] == null ? 0 : transformer?.Net.Config.Hidden ?? 0);
                }, transformer);
            });
    }
}
