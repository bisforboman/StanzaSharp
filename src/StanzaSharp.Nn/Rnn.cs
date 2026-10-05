using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

public static class Rnn
{
    /// <summary>
    /// Runs a batch-first LSTM over padded sequences of differing lengths, like PyTorch's
    /// pack_padded_sequence → lstm → pad_packed_sequence, so padding never reaches either direction.
    /// </summary>
    /// <param name="input">[batch, maxLen, dim]</param>
    /// <param name="state">Optional initial (h0, c0), each [layers * directions, batch, hidden].</param>
    /// <returns>[batch, maxLen, hidden * directions], zero past each sequence's length.</returns>
    public static Tensor RunPacked(LSTM lstm, Tensor input, long[] lengths, (Tensor, Tensor)? state = null)
    {
        using var scope = NewDisposeScope();
        using var lens = torch.tensor(lengths);
        var packed = nn.utils.rnn.pack_padded_sequence(input, lens, batch_first: true, enforce_sorted: false);
        var (output, _, _) = lstm.call(packed, state);
        var (padded, _) = nn.utils.rnn.pad_packed_sequence(output, batch_first: true, total_length: input.shape[1]);
        return padded.MoveToOuterDisposeScope();
    }

    /// <summary>
    /// Stacks per-sentence [len_i, dim] tensors into [batch, maxLen, dim], zero-padded
    /// (torch.nn.utils.rnn.pad_sequence).
    /// </summary>
    public static Tensor PadSequence(IReadOnlyList<Tensor> rows) =>
        nn.utils.rnn.pad_sequence(rows, batch_first: true);
}
