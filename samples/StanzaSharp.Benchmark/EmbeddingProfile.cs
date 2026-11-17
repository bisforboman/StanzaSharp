using System.Buffers;
using System.Diagnostics;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using StanzaSharp;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

/// <summary>
/// Where the managed encoder's time goes on a sentence-transformers model (issue #281,
/// docs/performance/round-51-small-width-encoder.md). The texts are tools/embedding_spike.py's 935 (no reference file
/// needed): the golden files' sentences, those twelve at a time and each file whole, cut to max_seq_length.
/// <list type="bullet">
/// <item><c>run</c>: the whole encoder (and ONNX Runtime with <c>--onnx</c>) over every text in batches, longest first,
/// and a SHA-256 of every output bit, to show a kernel change keeps the bits.</item>
/// <item><c>stages</c>: the same batches through each stage of a layer on its own (the fused QKV, attention, the
/// output dense, the two LayerNorms, the intermediate dense, GELU, the output dense), each timed, on the real kernels
/// with the model's shapes.</item>
/// <item><c>gemm</c>: one GEMM's GFLOP/s at M × K → N (the processors' small shapes too).</item>
/// <item><c>attention</c>: one head of one N-piece sentence (head width D, 12 heads), in ns per query–key pair.</item>
/// </list>
/// </summary>
internal static unsafe class EmbeddingProfile
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark embedding-profile run|stages MODEL_DIR [--batch N] [--runs N] [--threads N] [--path P] [--onnx]
               StanzaSharp.Benchmark embedding-profile gemm M K N [--runs N] [--threads N] [--path P]
               StanzaSharp.Benchmark embedding-profile attention N D [--runs N] [--path P]

          MODEL_DIR   a sentence-transformers model (config.json, pytorch_model.bin, tokenizer.json, sentence_bert_config.json;
                      onnx/model.onnx for --onnx), or a WordPiece transformer folder as ModelDownloader writes it
          --batch N   texts per forward call (default 32), longest first
          --runs N    timed runs after one warm-up (default 5; the median is reported)
          --threads N the managed pool's size and ONNX Runtime's intra-op threads (default 1)
          --path P    Gemm.Path: Vector512, Vector256, Vector128 or Scalar (default: the CPU's)
          --onnx      run: also time ONNX Runtime on the same batches (padded, with a mask)
          --save FILE run: write every piece's last state (float32, text by text, longest first) to FILE
        """;

    public static int Run(string[] args)
    {
        if (args.Length < 2 || args[0] is not ("run" or "stages" or "gemm" or "attention"))
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        int batch = 32, runs = 5, positional = args[0] switch { "gemm" => 4, "attention" => 3, _ => 2 };
        bool onnx = false;
        string? save = null;
        ManagedThreads.Count = 1;
        for (int i = positional; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--batch" when i + 1 < args.Length: batch = int.Parse(args[++i]); break;
                case "--runs" when i + 1 < args.Length: runs = int.Parse(args[++i]); break;
                case "--threads" when i + 1 < args.Length: ManagedThreads.Count = int.Parse(args[++i]); break;
                case "--path" when i + 1 < args.Length: Gemm.Path = Enum.Parse<KernelPath>(args[++i]); break;
                case "--onnx": onnx = true; break;
                case "--save" when i + 1 < args.Length: save = args[++i]; break;
                default: Console.Error.WriteLine(Usage); return 2;
            }
        }

        Console.WriteLine($"Threads {ManagedThreads.Count}, {Gemm.Path}, {Environment.ProcessorCount} logical cores");
        if (args[0] == "gemm")
            return GemmShape(int.Parse(args[1]), int.Parse(args[2]), int.Parse(args[3]), runs);
        if (args[0] == "attention")
            return AttentionShape(int.Parse(args[1]), int.Parse(args[2]), runs);

        string modelDir = args[1];
        // A sentence-transformers model is a bare BertModel state dict cut at max_seq_length; any other transformer folder
        // (ModelDownloader's, e.g. ELECTRA-large) keeps its config's weight prefix and is cut at its positions.
        string sentenceConfig = Path.Combine(modelDir, "sentence_bert_config.json");
        var config = TransformerConfig.Load(Path.Combine(modelDir, "config.json"));
        int maxLength = config.MaxPositions;
        if (File.Exists(sentenceConfig))
        {
            config = config with { Prefix = "" };
            maxLength = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(sentenceConfig))!["max_seq_length"]!.GetValue<int>();
        }

        var tokenizer = WordPiece.Load(Path.Combine(modelDir, "tokenizer.json"));
        var ids = Texts().Select(t =>
        {
            var all = tokenizer.Encode([t]).Ids;
            return all.Length <= maxLength ? all : [.. all.AsSpan(0, maxLength - 1), all[^1]];
        }).ToList();
        var order = Enumerable.Range(0, ids.Count).OrderByDescending(s => ids[s].Length).ToArray();
        var batches = Enumerable.Range(0, (order.Length + batch - 1) / batch)
            .Select(b => order.Skip(b * batch).Take(batch).Select(s => ids[s]).ToList())
            .ToList();
        Console.WriteLine($"{ids.Count} texts, {ids.Sum(i => i.Length)} pieces, batch {batch} ({batches.Count} calls), {runs} timed runs");

        using var encoder = ManagedTransformer.Load(config, Checkpoint.Load(Path.Combine(modelDir, "pytorch_model.bin")));
        if (args[0] == "stages")
            return Stages(config, batches, runs);

        string? hash = null;
        var managed = Median(runs, () =>
        {
            using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            foreach (var b in batches)
            {
                var states = encoder.Encode(b, 1);
                for (int s = 0; s < b.Count; s++)
                {
                    sha.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(states[s].AsSpan(0, b[s].Length * config.Hidden)));
                    ArrayPool<float>.Shared.Return(states[s]);
                }
            }

            hash = Convert.ToHexString(sha.GetHashAndReset())[..16];
        });
        if (save != null)
        {
            using var file = File.Create(save);
            foreach (var b in batches)
            {
                var states = encoder.Encode(b, 1);
                for (int s = 0; s < b.Count; s++)
                {
                    file.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(states[s].AsSpan(0, b[s].Length * config.Hidden)));
                    ArrayPool<float>.Shared.Return(states[s]);
                }
            }
        }

        Console.WriteLine($"managed: {managed.Median:F3} s (runs {string.Join(", ", managed.Times.Select(t => t.ToString("F3")))}); states sha256 {hash}");
        if (onnx)
        {
            using var options = new SessionOptions { IntraOpNumThreads = ManagedThreads.Count, InterOpNumThreads = 1 };
            using var session = new InferenceSession(Path.Combine(modelDir, "onnx", "model.onnx"), options);
            var ort = Median(runs, () =>
            {
                foreach (var b in batches)
                    RunOnnx(session, b);
            });
            Console.WriteLine($"onnx:    {ort.Median:F3} s (runs {string.Join(", ", ort.Times.Select(t => t.ToString("F3")))}); managed / ORT {managed.Median / ort.Median:F2}x");
        }

        return 0;
    }

    // tools/embedding_spike.py's texts without --languages.
    private static List<string> Texts()
    {
        string root = AppContext.BaseDirectory;
        while (!Directory.Exists(Path.Combine(root, "tests", "golden")))
            root = Path.GetDirectoryName(root) ?? throw new DirectoryNotFoundException("tests/golden");
        var golden = Path.Combine(root, "tests", "golden");
        var files = new[] { Path.Combine(golden, "pipeline.conllu") }
            .Concat(Directory.GetFiles(golden, "validation*.conllu").OrderBy(f => f, StringComparer.Ordinal));
        var perFile = files.Select(f => File.ReadLines(f).Where(l => l.StartsWith("# text = ", StringComparison.Ordinal)).Select(l => l["# text = ".Length..]).ToList())
            .ToList();
        var texts = perFile.SelectMany(f => f).ToList();
        foreach (var f in perFile)
        {
            for (int i = 0; i < f.Count; i += 12)
                texts.Add(string.Join(' ', f.Skip(i).Take(12)));
        }

        texts.AddRange(perFile.Select(f => string.Join(' ', f)));
        return texts;
    }

    private static (double Median, List<double> Times) Median(int runs, Action run)
    {
        run();
        var times = new List<double>();
        for (int r = 0; r < runs; r++)
        {
            var timer = Stopwatch.StartNew();
            run();
            times.Add(timer.Elapsed.TotalSeconds);
        }

        var sorted = times.Order().ToList();
        return (sorted[sorted.Count / 2], times);
    }

    private static void RunOnnx(InferenceSession session, List<int[]> batch)
    {
        int width = batch.Max(i => i.Length);
        var inputIds = new long[batch.Count * width];
        var mask = new long[batch.Count * width];
        for (int b = 0; b < batch.Count; b++)
        {
            for (int t = 0; t < batch[b].Length; t++)
            {
                inputIds[(b * width) + t] = batch[b][t];
                mask[(b * width) + t] = 1;
            }
        }

        long[] shape = [batch.Count, width];
        using var idsValue = OrtValue.CreateTensorValueFromMemory(inputIds, shape);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, shape);
        using var typesValue = OrtValue.CreateTensorValueFromMemory(new long[batch.Count * width], shape);
        using var runOptions = new RunOptions();
        var inputs = new Dictionary<string, OrtValue> { ["input_ids"] = idsValue, ["attention_mask"] = maskValue, ["token_type_ids"] = typesValue };
        using var outputs = session.Run(runOptions, inputs, ["last_hidden_state"]);
    }

    // Each stage of a layer over every batch, timed on its own (random weights of the model's shapes: the kernels' time
    // doesn't depend on the values), times the layer count.
    [System.Diagnostics.CodeAnalysis.SuppressMessage("Security", "CA5394:Do not use insecure randomness", Justification = "Seeded random weights for a timing; nothing secret is drawn.")]
    private static int Stages(TransformerConfig config, List<List<int[]>> batches, int runs)
    {
        int h = config.Hidden, inter = config.Intermediate, heads = config.Heads, d = h / heads;
        var random = new Random(1);
        PackedMatrix Matrix(int n, int k) => new([.. Enumerable.Range(0, n * k).Select(_ => (float)((random.NextDouble() - 0.5) * 0.1))], n, k);
        float[] Vector(int n)
        {
            var v = GC.AllocateArray<float>(n, pinned: true); // P() takes their address
            for (int i = 0; i < n; i++)
                v[i] = (float)((random.NextDouble() - 0.5) * 4);
            return v;
        }

        PackedMatrix qkvW = Matrix(3 * h, h), outW = Matrix(h, h), upW = Matrix(inter, h), downW = Matrix(h, inter);
        float[] qkvB = Vector(qkvW.PaddedN), outB = Vector(h), upB = Vector(upW.PaddedN), downB = Vector(h), gamma = Vector(h), beta = Vector(h);
        int maxRows = batches.Max(b => b.Sum(s => s.Length));
        float[] x = Vector(maxRows * h), qkv = Vector(maxRows * 3 * h), context = Vector(maxRows * h), tmp = Vector(maxRows * h), up = Vector(maxRows * inter), upSource = Vector(maxRows * inter);
        var stages = new (string Name, Action<int, int[]> Run)[]
        {
            ($"QKV dense ({h} → {3 * h})", (rows, _) => Gemm.Run(P(x), rows, h, qkvW, P(qkvB), P(qkv), 3 * h, rowInvariant: true)),
            ($"attention ({heads} heads of {d})", (rows, starts) => Attention(P(qkv), P(context), starts, h, heads, d)),
            ($"attention output dense ({h} → {h})", (rows, _) => Gemm.Run(P(context), rows, h, outW, P(outB), P(tmp), h, rowInvariant: true)),
            ("LayerNorm + residual (×2)", (rows, _) =>
            {
                for (int r = 0; r < rows; r++)
                    ManagedTransformer.LayerNormRow(P(x) + (r * h), P(tmp) + (r * h), h, P(gamma), P(beta), 1e-12);
                for (int r = 0; r < rows; r++)
                    ManagedTransformer.LayerNormRow(P(x) + (r * h), P(tmp) + (r * h), h, P(gamma), P(beta), 1e-12);
            }),
            ($"intermediate dense ({h} → {inter})", (rows, _) => Gemm.Run(P(x), rows, h, upW, P(upB), P(up), inter, rowInvariant: true)),
            ("GELU (with a copy of its input)", (rows, _) =>
            {
                upSource.AsSpan(0, rows * inter).CopyTo(up);
                ManagedTransformer.GeluBlock(P(up), rows * inter);
            }),
            ($"output dense ({inter} → {h})", (rows, _) => Gemm.Run(P(upSource), rows, inter, downW, P(downB), P(tmp), h, rowInvariant: true)),
        };

        var starts = batches.Select(b =>
        {
            var s = new int[b.Count + 1];
            for (int i = 0; i < b.Count; i++)
                s[i + 1] = s[i] + b[i].Length;
            return s;
        }).ToList();
        long pairs = batches.Sum(b => b.Sum(s => (long)s.Length * s.Length));
        Console.WriteLine($"  attention: {pairs:N0} query-key pairs per head and layer ({pairs * heads * config.Layers:N0} in all)");
        double total = 0;
        var results = new List<(string, double)>();
        foreach (var (name, run) in stages)
        {
            var t = Median(runs, () =>
            {
                for (int b = 0; b < batches.Count; b++)
                    run(starts[b][^1], starts[b]);
            }).Median * config.Layers;
            results.Add((name, t));
            total += t;
        }

        foreach (var (name, t) in results)
            Console.WriteLine($"  {name,-36} {t:F3} s  {100 * t / total,5:F1} %");
        Console.WriteLine($"  {"sum",-36} {total:F3} s");
        GC.KeepAlive(x);
        return 0;

        float* P(float[] a) => (float*)System.Runtime.CompilerServices.Unsafe.AsPointer(ref a[0]);
    }

    // ManagedTransformer.Attention, which is private: the same per-sentence-and-head tasks.
    private static void Attention(float* qkv, float* context, int[] starts, int h, int heads, int d)
    {
        double scale = 1 / Math.Sqrt(d);
        bool blocked = d % 8 == 0 && Gemm.Path != KernelPath.Scalar;
        nint pq = (nint)qkv, pc = (nint)context;
        ManagedThreads.For((starts.Length - 1) * heads, task =>
        {
            int s = task / heads, head = task % heads, r0 = starts[s], n = starts[s + 1] - r0;
            if (blocked)
                ManagedTransformer.AttentionBlocked((float*)pq + ((long)r0 * 3 * h) + (head * d), (float*)pc + ((long)r0 * h) + (head * d), n, h, d, scale);
            else
                ManagedTransformer.AttentionRows((float*)pq + ((long)r0 * 3 * h) + (head * d), (float*)pc + ((long)r0 * h) + (head * d), n, h, d, scale);
        });
    }

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Security", "CA5394:Do not use insecure randomness", Justification = "Seeded random weights for a timing; nothing secret is drawn.")]
    private static int AttentionShape(int n, int d, int runs)
    {
        int h = 12 * d;
        var random = new Random(1);
        var qkv = GC.AllocateArray<float>(n * 3 * h, pinned: true);
        for (int i = 0; i < qkv.Length; i++)
            qkv[i] = (float)((random.NextDouble() - 0.5) * 4);
        var context = GC.AllocateArray<float>(n * h, pinned: true);
        int reps = (int)Math.Max(1, 2e8 / ((double)n * n * d));
        bool blocked = d % 8 == 0 && Gemm.Path != KernelPath.Scalar;
        var t = Median(runs, () =>
        {
            fixed (float* q = qkv, c = context)
            {
                for (int r = 0; r < reps; r++)
                {
                    if (blocked)
                        ManagedTransformer.AttentionBlocked(q, c, n, h, d, 1 / Math.Sqrt(d));
                    else
                        ManagedTransformer.AttentionRows(q, c, n, h, d, 1 / Math.Sqrt(d));
                }
            }
        }).Median;
        Console.WriteLine($"attention n {n}, d {d}: {t / reps * 1e6:F1} µs, {t / reps / ((double)n * n) * 1e9:F2} ns per query-key pair");
        return 0;
    }

    [System.Diagnostics.CodeAnalysis.SuppressMessage("Security", "CA5394:Do not use insecure randomness", Justification = "Seeded random weights for a timing; nothing secret is drawn.")]
    private static int GemmShape(int m, int k, int n, int runs)
    {
        var random = new Random(1);
        var w = new PackedMatrix([.. Enumerable.Range(0, n * k).Select(_ => (float)random.NextDouble())], n, k);
        var a = Enumerable.Range(0, m * k).Select(_ => (float)random.NextDouble()).ToArray();
        var bias = new float[w.PaddedN];
        var c = new float[m * w.PaddedN];
        int reps = (int)Math.Max(1, 2e9 / (2.0 * m * k * n));
        var t = Median(runs, () =>
        {
            fixed (float* pa = a, pb = bias, pc = c)
            {
                for (int r = 0; r < reps; r++)
                    Gemm.Run(pa, m, k, w, pb, pc, w.PaddedN, rowInvariant: true);
            }
        }).Median;
        Console.WriteLine($"GEMM {m} × {k} → {n}: {t / reps * 1e6:F1} µs, {2.0 * m * k * n * reps / t / 1e9:F1} GFLOP/s");
        return 0;
    }
}
