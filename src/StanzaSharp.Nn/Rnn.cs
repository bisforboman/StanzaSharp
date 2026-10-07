using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

internal static class Rnn
{
    /// <summary>
    /// Runs a batch-first LSTM over padded sequences of differing lengths, like PyTorch's
    /// pack_padded_sequence → lstm → pad_packed_sequence, so padding never reaches either direction.
    /// </summary>
    /// <param name="input">[batch, maxLen, dim]</param>
    /// <param name="state">Optional initial (h0, c0), each [layers * directions, batch, hidden].</param>
    /// <returns>[batch, maxLen, hidden * directions], zero past each sequence's length.</returns>
    /// <remarks>Lengths stay on the CPU, as pack_padded_sequence requires, whatever the input's device.</remarks>
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

        // batch_sizes is always on the CPU; sorted_indices is on the input's device.
        // Packed rows are time-major: step t holds batch_sizes[t] rows, the longest sequences first,
        // so sequence i (rank r in the sorted order) is row start[t] + r at step t.
        var batchSizes = output.batch_sizes.ToArray<long>();
        var sorted = output.sorted_indices!.ToArray<long>();
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
            result.Add(output.data.index_select(0, torch.tensor(rows, device: output.data.device)).MoveToOuterDisposeScope());
        }
        return result;
    }

    /// <summary>
    /// The row order of pack_padded_sequence(input, lengths, batch_first: true, enforce_sorted: false): for each
    /// packed row, the index of the input row it holds in the flattened [batch * maxLen] input.
    /// </summary>
    public static long[] PackedOrder(long[] lengths)
    {
        long width = lengths.Max();
        using var lens = torch.tensor(lengths);
        using var index = torch.arange(lengths.Length * width).view(lengths.Length, width, 1);
        using var packed = nn.utils.rnn.pack_padded_sequence(index, lens, batch_first: true, enforce_sorted: false);
        return packed.data.ToArray<long>();
    }

    /// <summary>
    /// pack_padded_sequence(input, lengths, batch_first: true, enforce_sorted: false) for an input given as its
    /// packed rows (<paramref name="data"/>, rows in <see cref="PackedOrder"/>), without ever making the padded
    /// [batch, maxLen, dim] input.
    /// </summary>
    /// <remarks>
    /// TorchSharp can't make a PackedSequence from data, so this packs a one-wide zero input, then points the
    /// result's data at <paramref name="data"/>'s storage in place (<c>set_</c>; the PackedSequence holds the same
    /// tensor, so nothing is copied). The caller may dispose <paramref name="data"/>; its memory lives on in the
    /// result. (Packing an expanded zero input with enforce_sorted: true would avoid the probe, but TorchSharp
    /// counts the two undefined index tensors of such a PackedSequence as live forever.)
    /// </remarks>
    public static nn.utils.rnn.PackedSequence Pack(Tensor data, long[] lengths)
    {
        using var lens = torch.tensor(lengths);
        using var probe = torch.zeros(lengths.Length, lengths.Max(), 1, dtype: data.dtype, device: data.device);
        var packed = nn.utils.rnn.pack_padded_sequence(probe, lens, batch_first: true, enforce_sorted: false);
        packed.data.set_(data);
        return packed;
    }

    /// <summary>
    /// Stacks per-sentence [len_i, dim] tensors into [batch, maxLen, dim], zero-padded
    /// (torch.nn.utils.rnn.pad_sequence).
    /// </summary>
    public static Tensor PadSequence(IReadOnlyList<Tensor> rows) =>
        nn.utils.rnn.pad_sequence(rows, batch_first: true);
}
