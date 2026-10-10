using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Mwt;

/// <summary>stanza/models/mwt/character_classifier.py on TorchSharp: embedding, biLSTM encoder, Linear → ReLU → Linear.</summary>
internal sealed class MwtNet : IMwtNet
{
    private readonly Embedding _embedding;
    private readonly LSTM _encoder;
    private readonly Linear _hidden, _output;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public MwtNet(Checkpoint ckpt, JsonNode config, int padId)
    {
        var model = ckpt.Root["model"]!;
        int emb = config["emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["num_layers"]!.GetValue<int>();
        _embedding = nn.Embedding(config["vocab_size"]!.GetValue<int>(), emb, padding_idx: padId).LoadFrom(ckpt, model, "embedding.");
        _encoder = nn.LSTM(emb, hidden / 2, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "encoder.");
        _hidden = nn.Linear(hidden, hidden).LoadFrom(ckpt, model, "output_layer.0.");
        _output = nn.Linear(hidden, 2).LoadFrom(ckpt, model, "output_layer.2.");
    }

    public float[] Forward(long[] ids, int rows, int width, long[] lengths)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var src = torch.tensor(ids, [rows, width], device: _device);
        var encoded = Rnn.RunPacked(_encoder, _embedding.forward(src), lengths);
        return _output.forward(nn.functional.relu(_hidden.forward(encoded))).ToArray<float>();
    }

    public void Dispose()
    {
        _embedding.Dispose();
        _encoder.Dispose();
        _hidden.Dispose();
        _output.Dispose();
    }
}

/// <summary>Loads a <see cref="MwtExpander"/> on either backend.</summary>
internal static class MwtExpanderLoad
{
    extension(MwtExpander)
    {
        /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/mwt/combined</c>.</summary>
        /// <param name="device">Where the model runs on TorchSharp; CPU by default.</param>
        /// <param name="backend">The network's backend; the managed one ignores <paramref name="device"/>.</param>
        public static MwtExpander Load(string basePath, Device? device = null, Backend backend = Backend.TorchSharp) =>
            backend == Backend.Managed ? MwtExpander.LoadManaged(basePath) : Weights.On(device, () => new MwtExpander(Checkpoint.Load(basePath), (ckpt, config, padId) => new MwtNet(ckpt, config, padId)));
    }
}
