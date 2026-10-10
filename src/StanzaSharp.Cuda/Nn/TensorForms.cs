using TorchSharp;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>The TorchSharp form of <see cref="Pretrain"/>: the [count, dim] embedding matrix as a tensor.</summary>
internal sealed class PretrainTensor(Tensor embeddings) : IPretrainVectors
{
    private float[]? _cpuCopy;

    public Tensor Embeddings { get; } = embeddings;

    /// <summary>The CPU tensor's own memory (not a copy); on another device, a copy made on first use.</summary>
    public ReadOnlySpan<float> CpuVectors() => Embeddings.device_type == DeviceType.CPU
        ? System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(Embeddings.bytes)
        : _cpuCopy ??= Embeddings.ToArray<float>(); // a race only makes the same copy twice

    public void Dispose() => Embeddings.Dispose();
}

/// <summary>The TorchSharp form of a <see cref="CharlmCache"/> entry.</summary>
internal sealed class CharlmTensors(Tensor forward, Tensor backward) : ICharlmReps
{
    public Tensor Forward { get; } = forward;
    public Tensor Backward { get; } = backward;

    public (float[] Forward, float[] Backward) ToArrays() => (Forward.ToArray<float>(), Backward.ToArray<float>());

    public void Dispose()
    {
        Forward.Dispose();
        Backward.Dispose();
    }
}

/// <summary>The tensor halves of <see cref="Pretrain"/> and <see cref="CharlmCache"/>.</summary>
internal static class TensorForms
{
    extension(Pretrain)
    {
        /// <summary>Loads e.g. <c>models/converted/en/pretrain/conll17</c> as a tensor.</summary>
        /// <param name="device">Where the embedding matrix lives; CPU by default.</param>
        public static Pretrain Load(string basePath, Device? device = null) =>
            Weights.On(device, () => new Pretrain(Checkpoint.Load(basePath), (ckpt, emb) => new PretrainTensor(ckpt.ToTensor(emb))));
    }

    extension(Pretrain pretrain)
    {
        /// <summary>[vocab, dim] embedding matrix (TorchSharp form only).</summary>
        public Tensor Embeddings => (pretrain.Native as PretrainTensor)?.Embeddings
            ?? throw new InvalidOperationException("This pretrain was loaded for the managed backend (LoadManaged) and has no tensor");
    }

    extension(CharlmCache cache)
    {
        /// <summary>
        /// Keeps a sentence's [words, dim] representations if they fit under <see cref="CharlmCache.MaxWords"/>.
        /// When kept, the cache owns them: it detaches them from their dispose scope and disposes them
        /// itself. When not kept, nothing happens and they stay with the caller (and its dispose scope).
        /// </summary>
        /// <returns>Whether the cache kept them.</returns>
        public bool TryAdd(Sentence sentence, Tensor forward, Tensor backward) =>
            cache.TryAdd(sentence, (int)forward.shape[0], () => new CharlmTensors(forward.DetachFromDisposeScope(), backward.DetachFromDisposeScope()));

        /// <summary>
        /// A sentence's [words, dim] tensors, owned by the cache (don't dispose them). A managed producer's arrays are
        /// converted once, on <paramref name="device"/> (the reader's; CPU by default), and kept with the entry.
        /// </summary>
        public bool TryGet(Sentence sentence, out (Tensor Forward, Tensor Backward) reps, Device? device = null)
        {
            if (!cache.TryGetEntry(sentence, out var e))
            {
                reps = default;
                return false;
            }
            if (e.Native is null)
            {
                Tensor Convert(float[] data)
                {
                    using var cpu = torch.tensor(data, [e.Words, data.Length / e.Words]);
                    return (device == null || device.type == DeviceType.CPU ? cpu.clone() : cpu.to(device)).DetachFromDisposeScope();
                }
                e.Native = new CharlmTensors(Convert(e.ForwardData!), Convert(e.BackwardData!));
            }
            var t = (CharlmTensors)e.Native;
            reps = (t.Forward, t.Backward);
            return true;
        }
    }
}
