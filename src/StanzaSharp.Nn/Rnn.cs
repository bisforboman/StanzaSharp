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
    /// Like <see cref="RunPacked"/>, but returns only the outputs at <paramref name="positions"/>[i] of each
    /// sequence i, read straight from the packed output. That skips the padded [batch, maxLen, hidden]
    /// tensor, which is mostly padding when one sequence is much longer than the rest.
    /// </summary>
    /// <returns>One [positions[i].Length, hidden * directions] tensor per sequence.</returns>
    public static List<Tensor> RunPackedAt(LSTM lstm, Tensor input, long[] lengths, IReadOnlyList<long[]> positions, (Tensor, Tensor)? state = null)
    {
        using var scope = NewDisposeScope();
        using var lens = torch.tensor(lengths);
        var packed = nn.utils.rnn.pack_padded_sequence(input, lens, batch_first: true, enforce_sorted: false);
        var (output, _, _) = lstm.call(packed, state);

        // Packed rows are time-major: step t holds batch_sizes[t] rows, the longest sequences first,
        // so sequence i (rank r in the sorted order) is row start[t] + r at step t.
        var batchSizes = output.batch_sizes.data<long>().ToArray();
        var sorted = output.sorted_indices!.data<long>().ToArray();
        var rank = new long[sorted.Length];
        for (int r = 0; r < sorted.Length; r++)
            rank[sorted[r]] = r;
        var start = new long[batchSizes.Length];
        for (int t = 1; t < start.Length; t++)
            start[t] = start[t - 1] + batchSizes[t - 1];

        var result = new List<Tensor>(positions.Count);
        for (int i = 0; i < positions.Count; i++)
        {
            var rows = positions[i].Select(t => start[t] + rank[i]).ToArray();
            result.Add(output.data.index_select(0, torch.tensor(rows)).MoveToOuterDisposeScope());
        }
        return result;
    }

    /// <summary>
    /// Stacks per-sentence [len_i, dim] tensors into [batch, maxLen, dim], zero-padded
    /// (torch.nn.utils.rnn.pad_sequence).
    /// </summary>
    public static Tensor PadSequence(IReadOnlyList<Tensor> rows) =>
        nn.utils.rnn.pad_sequence(rows, batch_first: true);
}
