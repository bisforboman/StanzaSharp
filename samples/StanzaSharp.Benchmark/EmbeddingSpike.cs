using System.Buffers;
using System.Diagnostics;
using System.Text.Json.Nodes;
using Microsoft.ML.OnnxRuntime;
using StanzaSharp;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

/// <summary>
/// The sentence-embedding spike (issue #268, docs/embedding-spike.md): sentence-transformers/all-MiniLM-L6-v2 on the
/// managed encoder (<see cref="ManagedTransformer"/>) with mean pooling, L2 normalisation and truncation at the model's
/// max_seq_length, against tools/embedding_spike.py's sentence-transformers vectors and against ONNX Runtime running the
/// model's ONNX export on the same ids (as samples/StanzaSharp.Rag's MiniLmEmbeddingGenerator did before it moved to the managed encoder), and both timed.
/// </summary>
internal static class EmbeddingSpike
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark embedding-spike MODEL_DIR REFERENCE [--backend managed|onnx|both] [--batch N] [--runs N]
                                                     [--threads N] [--path P] [--limit N]

          MODEL_DIR   sentence-transformers/all-MiniLM-L6-v2 at 1110a243 (config.json, pytorch_model.bin, tokenizer.json,
                      sentence_bert_config.json, 1_Pooling/config.json, onnx/model.onnx)
          REFERENCE   tools/embedding_spike.py's output, without .safetensors/.json
          --backend B managed, onnx or both (default); for memory figures run one backend per process
          --batch N   texts per forward call (default: 32, sentence-transformers' encode default), longest first in both
          --runs N    timed runs over all texts after one warm-up (default: 3; the median is reported)
          --threads N the managed pool's size and ONNX Runtime's intra-op threads (default: ManagedThreads.Default)
          --path P    Gemm.Path: Vector512, Vector256, Vector128 or Scalar (default: the CPU's)
          --limit N   only the first N texts
        """;

    private const double Tolerance = 1e-4;

    public static int Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        string modelDir = args[0], reference = args[1], backend = "both";
        int batch = 32, runs = 3, limit = int.MaxValue;
        ManagedThreads.Count = ManagedThreads.Default;
        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--backend" when i + 1 < args.Length && args[i + 1] is "managed" or "onnx" or "both": backend = args[++i]; break;
                case "--batch" when i + 1 < args.Length: batch = int.Parse(args[++i]); break;
                case "--runs" when i + 1 < args.Length: runs = int.Parse(args[++i]); break;
                case "--threads" when i + 1 < args.Length: ManagedThreads.Count = int.Parse(args[++i]); break;
                case "--path" when i + 1 < args.Length: Gemm.Path = Enum.Parse<KernelPath>(args[++i]); break;
                case "--limit" when i + 1 < args.Length: limit = int.Parse(args[++i]); break;
                default: Console.Error.WriteLine(Usage); return 2;
            }
        }

        // sentence-transformers' settings: max_seq_length, and mean pooling (the only mode this spike runs).
        int maxLength = JsonNode.Parse(File.ReadAllText(Path.Combine(modelDir, "sentence_bert_config.json")))!["max_seq_length"]!.GetValue<int>();
        var pooling = JsonNode.Parse(File.ReadAllText(Path.Combine(modelDir, "1_Pooling", "config.json")))!;
        if (!pooling["pooling_mode_mean_tokens"]!.GetValue<bool>() || pooling["pooling_mode_cls_token"]!.GetValue<bool>())
            throw new InvalidDataException("The spike runs mean pooling only");

        var texts = JsonNode.Parse(File.ReadAllText(reference + ".json"))!["texts"]!.AsArray().Select(t => t!.GetValue<string>()).Take(limit).ToList();
        var refFile = SafeTensorFile.Load(reference + ".safetensors");

        // Tokenize: the whole text is one "word" (BertPreTokenizer splits it at white space and punctuation as it would
        // the text), then sentence-transformers' truncation (longest_first on one sequence: the last pieces go, [SEP] stays).
        var tokenizer = WordPiece.Load(Path.Combine(modelDir, "tokenizer.json"));
        var ids = texts.Select(t =>
        {
            var all = tokenizer.Encode([t]).Ids;
            return all.Length <= maxLength ? all : [.. all.AsSpan(0, maxLength - 1), tokenizer.Sep];
        }).ToList();
        int idMismatches = 0;
        for (int s = 0; s < texts.Count; s++)
        {
            var expected = refFile.Read<long>($"s{s}.ids");
            if (!expected.Select(v => (int)v).SequenceEqual(ids[s]) && ++idMismatches <= 5)
                Console.WriteLine($"  ids differ in text {s}: {texts[s]}\n    python {string.Join(',', expected)}\n    C#     {string.Join(',', ids[s])}");
        }

        int pieces = ids.Sum(i => i.Length);
        Console.WriteLine($"Tokenizer: {texts.Count} texts, {pieces} pieces (max {ids.Max(i => i.Length)}, {ids.Count(i => i.Length == maxLength)} truncated "
            + $"to {maxLength}); {idMismatches} differ from sentence-transformers'");
        Console.WriteLine($"Threads {ManagedThreads.Count} (cores: {Environment.ProcessorCount} logical), {Gemm.Path}, batch {batch}, {runs} timed runs");

        // Longest first, as sentence-transformers' encode orders its batches; the same batches for both backends.
        var order = Enumerable.Range(0, ids.Count).OrderByDescending(s => ids[s].Length).ToArray();
        float[][]? managed = null, onnx = null;
        if (backend is "managed" or "both")
        {
            var load = Stopwatch.StartNew();
            var config = TransformerConfig.Load(Path.Combine(modelDir, "config.json")) with { Prefix = "" }; // a bare BertModel's state dict
            using var encoder = ManagedTransformer.Load(config, Checkpoint.Load(Path.Combine(modelDir, "pytorch_model.bin")));
            GC.Collect();
            Console.WriteLine($"managed: load {load.Elapsed.TotalMilliseconds:F0} ms, {config.Layers} layers, hidden {config.Hidden}, "
                + $"working set {Environment.WorkingSet / 1e6:F0} MB");
            managed = Time("managed", runs, pieces, texts.Count, () => Batches(order, batch, ids, b => EncodeManaged(encoder, b)));
        }

        if (backend is "onnx" or "both")
        {
            var load = Stopwatch.StartNew();
            using var options = new SessionOptions { IntraOpNumThreads = ManagedThreads.Count, InterOpNumThreads = 1 };
            using var session = new InferenceSession(Path.Combine(modelDir, "onnx", "model.onnx"), options);
            GC.Collect();
            Console.WriteLine($"onnx:    load {load.Elapsed.TotalMilliseconds:F0} ms, ONNX Runtime {OrtEnv.Instance().GetVersionString()}, "
                + $"working set {Environment.WorkingSet / 1e6:F0} MB");
            onnx = Time("onnx", runs, pieces, texts.Count, () => Batches(order, batch, ids, b => EncodeOnnx(session, b)));
        }

        bool ok = idMismatches == 0;
        foreach (var (name, vectors) in new[] { ("managed", managed), ("onnx", onnx) })
        {
            if (vectors == null)
                continue;
            var (f32, at) = MaxDiff(texts.Count, s => vectors[s], s => [.. refFile.Read<float>($"s{s}.emb").Select(v => (double)v)]);
            var (f64, _) = MaxDiff(texts.Count, s => vectors[s], s => refFile.Read<double>($"s{s}.emb.f64"));
            Console.WriteLine($"  {name,-8} vs sentence-transformers f32: max |diff| {f32:E2} ({at}); vs f64 {f64:E2}; "
                + $"within {Tolerance:E0}: {(f32 <= Tolerance ? "yes" : "NO")}");
            ok &= f32 <= Tolerance;
        }

        if (managed != null && onnx != null)
        {
            var (diff, at) = MaxDiff(texts.Count, s => managed[s], s => [.. onnx[s].Select(v => (double)v)]);
            Console.WriteLine($"  managed vs onnx: max |diff| {diff:E2} ({at})");
        }

        return ok ? 0 : 1;
    }

    // One warm-up run (whose vectors are returned), then the timed runs; prints the median, the peak working set and the
    // managed allocations per run.
    private static float[][] Time(string name, int runs, int pieces, int texts, Func<float[][]> run)
    {
        var first = Stopwatch.StartNew();
        var vectors = run();
        double warmup = first.Elapsed.TotalSeconds;
        var times = new List<double>();
        long allocated = GC.GetTotalAllocatedBytes(true);
        for (int r = 0; r < runs; r++)
        {
            var timer = Stopwatch.StartNew();
            run();
            times.Add(timer.Elapsed.TotalSeconds);
        }

        times.Sort();
        double median = times[times.Count / 2];
        Console.WriteLine($"{name + ":",-8} {median:F3} s per run (warm-up {warmup:F3} s; runs {string.Join(", ", times.Select(t => t.ToString("F3")))}), "
            + $"{texts / median:F0} texts/s, {pieces / median:F0} pieces/s; peak working set {Process.GetCurrentProcess().PeakWorkingSet64 / 1e6:F0} MB, "
            + $"{(GC.GetTotalAllocatedBytes(true) - allocated) / Math.Max(1, runs) / 1e6:F1} MB managed allocations per run");
        return vectors;
    }

    private static float[][] Batches(int[] order, int batch, List<int[]> ids, Func<List<int[]>, float[][]> encode)
    {
        var vectors = new float[ids.Count][];
        for (int b = 0; b < order.Length; b += batch)
        {
            var members = order.AsSpan(b, Math.Min(batch, order.Length - b)).ToArray();
            var encoded = encode([.. members.Select(s => ids[s])]);
            for (int i = 0; i < members.Length; i++)
                vectors[members[i]] = encoded[i];
        }

        return vectors;
    }

    // The last hidden state of every piece (no padding: each text runs alone in attention), mean-pooled and normalised.
    private static float[][] EncodeManaged(ManagedTransformer encoder, List<int[]> batch)
    {
        int h = encoder.Config.Hidden;
        var states = encoder.Encode(batch, 1);
        try
        {
            return [.. batch.Select((ids, s) => Pool(states[s].AsSpan(0, ids.Length * h), ids.Length, h))];
        }
        finally
        {
            if (encoder.RentsOutputs)
            {
                foreach (var array in states)
                    ArrayPool<float>.Shared.Return(array);
            }
        }
    }

    // ONNX Runtime on one padded batch with its attention mask, as samples/StanzaSharp.Rag ran it.
    private static float[][] EncodeOnnx(InferenceSession session, List<int[]> batch)
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
        using var outputs = session.Run(runOptions,
            new Dictionary<string, OrtValue> { ["input_ids"] = idsValue, ["attention_mask"] = maskValue, ["token_type_ids"] = typesValue },
            ["last_hidden_state"]);
        var hidden = outputs[0].GetTensorDataAsSpan<float>();
        int h = (int)(hidden.Length / ((long)batch.Count * width));
        var result = new float[batch.Count][];
        for (int b = 0; b < batch.Count; b++)
            result[b] = Pool(hidden.Slice(b * width * h, batch[b].Length * h), batch[b].Length, h);
        return result;
    }

    // sentence-transformers' Pooling (mean over the unmasked pieces) and Normalize (x / max(‖x‖, 1e-12)), in double.
    private static float[] Pool(ReadOnlySpan<float> states, int n, int h)
    {
        var sum = new double[h];
        for (int t = 0; t < n; t++)
        {
            var row = states.Slice(t * h, h);
            for (int d = 0; d < h; d++)
                sum[d] += row[d];
        }

        double norm = 0;
        for (int d = 0; d < h; d++)
        {
            sum[d] /= n;
            norm += sum[d] * sum[d];
        }

        double scale = 1 / Math.Max(Math.Sqrt(norm), 1e-12);
        return [.. sum.Select(v => (float)(v * scale))];
    }

    private static (double Max, string At) MaxDiff(int texts, Func<int, float[]> actual, Func<int, double[]> expected)
    {
        double max = 0;
        string at = "";
        for (int s = 0; s < texts; s++)
        {
            var (a, e) = (actual(s), expected(s));
            for (int d = 0; d < e.Length; d++)
            {
                double diff = Math.Abs(a[d] - e[d]);
                if (diff > max)
                    (max, at) = (diff, $"text {s} dim {d}, value {e[d]:F4}");
            }
        }

        return (max, at);
    }
}
