using System.Text.Json.Nodes;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// Port of stanza/models/common/hlstm.py <c>HighwayLSTM</c> with learned initial states (as the
/// tagger uses it): stacked packed biLSTM layers, each followed by
/// <c>h + sigmoid(gate(x)) * tanh(highway(x))</c>.
/// </summary>
internal sealed class HighwayLstm : IDisposable
{
    private readonly LSTM[] _lstm;
    private readonly Linear[] _highway, _gate;
    private readonly Tensor _hInit, _cInit;
    private readonly int _hidden;

    /// <param name="name">Module name in the state dict, e.g. <c>taggerlstm</c>; initial states are <c>{name}_h_init</c>/<c>{name}_c_init</c>.</param>
    public HighwayLstm(Checkpoint ckpt, JsonNode stateDict, string name, int inputSize, int hidden, int layers)
    {
        _hidden = hidden;
        _lstm = new LSTM[layers];
        _highway = new Linear[layers];
        _gate = new Linear[layers];
        int inSize = inputSize;
        for (int l = 0; l < layers; l++)
        {
            _lstm[l] = nn.LSTM(inSize, hidden, batchFirst: true, bidirectional: true).LoadFrom(ckpt, stateDict, $"{name}.lstm.{l}.lstm.");
            _highway[l] = nn.Linear(inSize, hidden * 2).LoadFrom(ckpt, stateDict, $"{name}.highway.{l}.");
            _gate[l] = nn.Linear(inSize, hidden * 2).LoadFrom(ckpt, stateDict, $"{name}.gate.{l}.");
            inSize = hidden * 2;
        }
        _hInit = ckpt.ToTensor(stateDict[$"{name}_h_init"]);
        _cInit = ckpt.ToTensor(stateDict[$"{name}_c_init"]);
    }

    /// <param name="input">[batch, maxLen, inputSize]</param>
    /// <param name="disposeInput">Dispose <paramref name="input"/> once the first layer is done with it.</param>
    /// <returns>[batch, maxLen, 2 * hidden]; positions past a sentence's length are meaningless.</returns>
    /// <remarks>
    /// The batch is padded, so each layer's tensors can be large (hundreds of MB for a batch padded to one long
    /// sentence): every layer frees its intermediates and its input as soon as it is done, and the elementwise
    /// steps run in place.
    /// </remarks>
    public Tensor Forward(Tensor input, long[] lengths, bool disposeInput = false)
    {
        long batch = input.shape[0];
        var x = input;
        for (int l = 0; l < _lstm.Length; l++)
        {
            Tensor next;
            using (NewDisposeScope())
            {
                var h0 = _hInit.narrow(0, 2 * l, 2).expand(2, batch, _hidden).contiguous();
                var c0 = _cInit.narrow(0, 2 * l, 2).expand(2, batch, _hidden).contiguous();
                var h = Rnn.RunPacked(_lstm[l], x, lengths, (h0, c0));
                var gate = _gate[l].forward(x).sigmoid_();
                gate.mul_(_highway[l].forward(x).tanh_());
                next = h.add_(gate, Scalars.One).MoveToOuterDisposeScope();
            }
            if (l > 0 || disposeInput)
                x.Dispose();
            x = next;
        }
        return x;
    }

    public void Dispose()
    {
        foreach (var m in _lstm.Cast<nn.Module>().Concat(_highway).Concat(_gate))
            m.Dispose();
        _hInit.Dispose();
        _cInit.Dispose();
    }
}
