using System.Buffers;
using System.Diagnostics;
using StanzaSharp;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

/// <summary>
/// The managed transformer encoder's time per call by sentence length (#169, docs/performance/round-48-joint-encoder-pass.md):
/// one sentence, two in one <see cref="ITransformerNet.Encode"/> call and two calls; with <c>--lora</c>, a frozen and a LoRA
/// sentence in one <see cref="ITransformerNet.EncodeJoint"/> sweep against a pass of each net.
/// </summary>
internal static class EncoderScaling
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark encoder-scaling MODEL_DIR [--threads N] [--runs N] [--lora CHECKPOINT]

          MODEL_DIR    a transformer's files (config.json, pytorch_model.bin), e.g.
                       models/stanza/en/transformer/google/electra-large-discriminator
          --lora C     a classifier checkpoint with LoRA adapters (bert_lora), e.g. models/stanza/en/sentiment/sstplus_electra-large
          --runs N     calls per measurement, the median reported (default 25)
        """;

    public static int Run(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        string? lora = null;
        int runs = 25;
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--threads" when i + 1 < args.Length: ManagedThreads.Count = int.Parse(args[++i]); break;
                case "--runs" when i + 1 < args.Length: runs = int.Parse(args[++i]); break;
                case "--lora" when i + 1 < args.Length: lora = args[++i]; break;
                default:
                    Console.Error.WriteLine(Usage);
                    return 2;
            }
        }

        using var frozen = ManagedTransformer.Load(args[0]);
        ITransformerNet? adapted = null;
        if (lora != null)
        {
            var ckpt = Checkpoint.Load(lora);
            var p = ckpt.Root["params"]!;
            adapted = frozen.WithLora(LoraWeights.Read(ckpt, p["bert_lora"] ?? throw new InvalidDataException($"{lora} has no bert_lora"), p["config"]!));
        }

        GC.Collect();
        Console.WriteLine($"threads {ManagedThreads.Count}, {Gemm.Path}, median of {runs} calls");
        int[] Ids(int n) => [101, .. Enumerable.Range(0, n - 2).Select(i => 2000 + i), 102];
        double Time(Func<float[][]> call)
        {
            var times = new List<double>();
            for (int r = 0; r < runs; r++)
            {
                var watch = Stopwatch.StartNew();
                foreach (var states in call())
                    ArrayPool<float>.Shared.Return(states);
                times.Add(watch.Elapsed.TotalMilliseconds);
            }

            times.Sort();
            return times[times.Count / 2];
        }

        foreach (var n in new[] { 3, 8, 16, 32, 64, 128, 256 })
        {
            var one = Time(() => frozen.Encode([Ids(n)], 4));
            var two = Time(() => frozen.Encode([Ids(n), Ids(n)], 4));
            var line = $"{n,4} pieces: one sentence {one,7:F1} ms, two in one call {two,7:F1} ms, two calls {2 * one,7:F1} ms";
            if (adapted != null)
            {
                var separate = Time(() => [.. frozen.Encode([Ids(n)], 4), .. adapted.Encode([Ids(n)], 4)]);
                var joint = Time(() => [.. frozen.EncodeJoint([(frozen, [Ids(n)]), (adapted, [Ids(n)])], 4)!.SelectMany(g => g)]);
                line += $"; frozen + LoRA: a pass each {separate,7:F1} ms, one sweep {joint,7:F1} ms";
            }

            Console.WriteLine(line);
        }

        adapted?.Dispose();
        return 0;
    }
}
