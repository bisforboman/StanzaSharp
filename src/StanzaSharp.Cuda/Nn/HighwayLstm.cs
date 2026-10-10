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
    /// <param name="disposeInput">Dispose <paramref name="input"/> once it is packed.</param>
    /// <param name="cancellationToken">Checked between layers.</param>
    /// <returns>[batch, maxLen, 2 * hidden], zero past each sentence's length.</returns>
    public Tensor Forward(Tensor input, long[] lengths, bool disposeInput = false, CancellationToken cancellationToken = default)
    {
        using var scope = NewDisposeScope();
        long width = input.shape[1];
        using var lens = torch.tensor(lengths);
        var packed = nn.utils.rnn.pack_padded_sequence(input, lens, batch_first: true, enforce_sorted: false);
        if (disposeInput)
            input.Dispose();
        var (padded, _) = nn.utils.rnn.pad_packed_sequence(Forward(packed, disposeInput: true, cancellationToken), batch_first: true, total_length: width);
        return padded.MoveToOuterDisposeScope();
    }

    /// <param name="input">A packed batch, data [words, inputSize] (pack_padded_sequence or <see cref="Rnn.Pack"/>).</param>
    /// <param name="disposeInput">Dispose <paramref name="input"/> once the first layer is done with it.</param>
    /// <param name="cancellationToken">Checked between layers. On cancellation the last layer's output stays in the caller's dispose scope.</param>
    /// <returns>The last layer's output, packed like <paramref name="input"/>: data [words, 2 * hidden].</returns>
    /// <remarks>
    /// The gate and highway layers run on the packed rows, so padding costs nothing here (a batch padded to one
    /// long sentence used to make each layer's tensors hundreds of MB). Every layer frees its intermediates and
    /// its input as soon as it is done, and the elementwise steps run in place on the LSTM's output.
    /// </remarks>
    public nn.utils.rnn.PackedSequence Forward(nn.utils.rnn.PackedSequence input, bool disposeInput = false,
        CancellationToken cancellationToken = default)
    {
        long batch = input.batch_sizes[0].item<long>();
        var x = input;
        for (int l = 0; l < _lstm.Length; l++)
        {
            if (l > 0)
                cancellationToken.ThrowIfCancellationRequested();
            nn.utils.rnn.PackedSequence next;
            using (NewDisposeScope())
            {
                var h0 = _hInit.narrow(0, 2 * l, 2).expand(2, batch, _hidden).contiguous();
                var c0 = _cInit.narrow(0, 2 * l, 2).expand(2, batch, _hidden).contiguous();
                (next, _, _) = _lstm[l].call(x, (h0, c0));
                var gate = _gate[l].forward(x.data).sigmoid_();
                gate.mul_(_highway[l].forward(x.data).tanh_());
                next.data.add_(gate, Scalars.One);
                next.MoveToOuterDisposeScope();
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
