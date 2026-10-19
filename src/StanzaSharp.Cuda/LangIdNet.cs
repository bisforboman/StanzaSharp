using StanzaSharp.Nn;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp;

/// <summary>stanza/models/langid/model.py <c>LangIDBiLSTM.forward</c> on TorchSharp: embedding, biLSTM, Linear, sum over positions.</summary>
internal sealed class LangIdNet : ILangIdNet
{
    private readonly Embedding _embeddings;
    private readonly LSTM _lstm;
    private readonly Linear _toTag;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public LangIdNet(Checkpoint ckpt)
    {
        var root = ckpt.Root;
        var model = root["model_state_dict"]!;
        int emb = root["embedding_dim"]!.GetValue<int>(), hidden = root["hidden_dim"]!.GetValue<int>();
        var chars = Checkpoint.StringToId(root["char_to_idx"]!);
        var tags = Checkpoint.StringToId(root["tag_to_idx"]!);
        _embeddings = nn.Embedding(chars.Count, emb, padding_idx: chars["<PAD>"]).LoadFrom(ckpt, model, "char_embeds.");
        _lstm = nn.LSTM(emb, hidden, numLayers: root["num_layers"]!.GetValue<int>(), batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "lstm.");
        _toTag = nn.Linear(2 * hidden, tags.Count).LoadFrom(ckpt, model, "hidden_to_tag.");
    }

    public float[] Forward(long[] ids, int rows, int width, CancellationToken ct)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var x = tensor(ids, [rows, width], device: _device);
        var (output, _, _) = _lstm.forward(_embeddings.forward(x));
        return _toTag.forward(output).sum(1).ToArray<float>();
    }

    public void Dispose()
    {
        _embeddings.Dispose();
        _lstm.Dispose();
        _toTag.Dispose();
    }
}
