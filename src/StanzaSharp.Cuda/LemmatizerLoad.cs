using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Lemma;

/// <summary>Loads a <see cref="Lemmatizer"/> on either backend.</summary>
internal static class LemmatizerLoad
{
    extension(Lemmatizer)
    {
        /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/lemma/combined_nocharlm</c>.</summary>
        /// <param name="device">Where the model runs on TorchSharp; CPU by default.</param>
        /// <param name="backend">The network's backend; the managed one ignores <paramref name="device"/>.</param>
        /// <param name="forward">The forward charlm of a <c>_charlm</c> model (TorchSharp; the managed backend takes its own, <see cref="Lemmatizer.LoadManaged"/>).</param>
        /// <param name="backward">Its backward charlm.</param>
        /// <param name="pretrain">The pretrain of a model with contextual classifiers (English <c>combined_charlm</c>); null for the others.</param>
        public static Lemmatizer Load(string basePath, Device? device = null, Backend backend = Backend.TorchSharp, CharLanguageModel? forward = null,
            CharLanguageModel? backward = null, Pretrain? pretrain = null) =>
            backend == Backend.Managed
                ? Lemmatizer.LoadManaged(basePath)
                : Weights.On(device, () => new Lemmatizer(Checkpoint.Load(basePath), (ckpt, sizes) => new Seq2SeqNet(ckpt, sizes), forward, backward,
                    pretrain, (ckpt, model) => new LemmaClassifierNet(ckpt, model, pretrain!,
                        forward ?? throw new ArgumentException("The contextual classifiers read both charlms", nameof(forward)),
                        backward ?? throw new ArgumentException("The contextual classifiers read both charlms", nameof(backward)))));
    }
}

/// <summary>lstm_model.py <c>LemmaClassifierLSTM.forward</c> on TorchSharp.</summary>
internal sealed class LemmaClassifierNet : ILemmaClassifierNet
{
    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Embedding _delta, _upos;
    private readonly LSTM _lstm;
    private readonly Linear _hidden, _output;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public LemmaClassifierNet(Checkpoint ckpt, JsonNode model, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        Embedding Emb(string prefix)
        {
            var shape = ckpt.Shape(model[prefix + "weight"]);
            return nn.Embedding(shape[0], shape[1]).LoadFrom(ckpt, model, prefix);
        }

        Linear Lin(string prefix)
        {
            var shape = ckpt.Shape(model[prefix + "weight"]);
            return nn.Linear(shape[1], shape[0]).LoadFrom(ckpt, model, prefix);
        }

        _delta = Emb("delta_embedding.");
        _upos = Emb("upos_emb.");
        var ih = ckpt.Shape(model["lstm.weight_ih_l0"]);
        var hh = ckpt.Shape(model["lstm.weight_hh_l0"]);
        _lstm = nn.LSTM(ih[1], hh[1], numLayers: 1, bidirectional: true, batchFirst: true).LoadFrom(ckpt, model, "lstm.");
        (_hidden, _output) = (Lin("mlp.0."), Lin("mlp.2."));
    }

    public float[] Forward(IReadOnlyList<IReadOnlyList<string>> sentences, long[] ids, long[] deltaIds, long[] uposIds, int width, int[] positions, CancellationToken ct)
    {
        using var noGrad = torch.no_grad();
        using var scope = NewDisposeScope();
        int n = sentences.Count;
        var embedded = _pretrain.Embeddings[torch.tensor(ids, [n, width], device: _device)]
            .add(_delta.forward(torch.tensor(deltaIds, [n, width], device: _device)), Scalars.One);
        var upos = _upos.forward(torch.tensor(uposIds, [n, width], device: _device));
        var forward = CharReps(_charlmForward, sentences, width);
        ct.ThrowIfCancellationRequested();
        var backward = CharReps(_charlmBackward, sentences, width);
        ct.ThrowIfCancellationRequested();

        // Stanza packs every row at the padded width, so the LSTM runs unpacked: padding reaches the backward direction.
        var (output, _, _) = _lstm.call(cat([embedded, upos, forward, backward], 2));
        var rows = torch.tensor(positions.Select((p, i) => (long)(i * width + p)).ToArray(), device: _device);
        var reps = output.reshape(n * width, output.shape[2]).index_select(0, rows);
        return _output.forward(F.relu(_hidden.forward(reps))).ToArray<float>();
    }

    /// <summary>build_char_representation padded by pad_sequence: [batch, width, dim], zeros after each sentence.</summary>
    private Tensor CharReps(CharLanguageModel charlm, IReadOnlyList<IReadOnlyList<string>> sentences, int width)
    {
        var reps = charlm.BuildCharRepresentation(sentences);
        var result = torch.zeros([sentences.Count, width, charlm.HiddenDim], device: _device);
        for (int i = 0; i < sentences.Count; i++)
            result[i].narrow(0, 0, sentences[i].Count).copy_(reps[i]);
        return result;
    }

    public void Dispose()
    {
        nn.Module[] modules = [_delta, _upos, _lstm, _hidden, _output];
        foreach (var m in modules)
            m.Dispose();
    }
}
