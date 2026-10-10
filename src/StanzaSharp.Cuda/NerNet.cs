using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Ner;

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
                cacheKeys?[i] is { } key && charlms!.TryGet(key, out var reps, _device) ? reps : ((Tensor, Tensor)?)null).ToList();
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

/// <summary>Loads a <see cref="NerTagger"/> on TorchSharp.</summary>
internal static class NerTaggerLoad
{
    extension(NerTagger)
    {
        /// <summary>
        /// Loads e.g. <c>models/converted/en/ner/ontonotes-ww-multi_charlm</c> on TorchSharp. The pretrain and charlms are
        /// shared with the other processors, so the caller owns them. A <c>_nocharlm</c> model takes none.
        /// </summary>
        /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
        public static NerTagger Load(string basePath, Pretrain pretrain, CharLanguageModel? charlmForward, CharLanguageModel? charlmBackward, Device? device = null) =>
            Weights.On(device, () =>
            {
                var ckpt = Checkpoint.Load(basePath);
                return new NerTagger(ckpt, pretrain, (tags, tagset, charlm) =>
                {
                    if (charlm && (charlmForward == null || charlmBackward == null || !charlmForward.IsForward || charlmBackward.IsForward))
                        throw new ArgumentException("This NER model needs the forward charlm, then the backward one");
                    return charlm ? new NerNet(ckpt, NerTagger.DeltaCount(ckpt), tags, tagset, pretrain, charlmForward, charlmBackward)
                        : new NerNet(ckpt, NerTagger.DeltaCount(ckpt), tags, tagset, pretrain, null, null);
                });
            });
    }
}
