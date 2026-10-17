using System.Diagnostics;
using StanzaSharp;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

/// <summary>
/// The managed transformer encoder spike (docs/transformer-plan.md): google/electra-large-discriminator on the
/// golden sentences' words against tools/transformer_spike.py's reference hidden states, plus speed and memory.
/// </summary>
internal static class TransformerSpike
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark transformer-spike MODEL_DIR REFERENCE [--threads N] [--batch N] [--runs N] [--dump FILE] [--ids FILE]

          MODEL_DIR   google/electra-large-discriminator's files (tokenizer.json, pytorch_model.bin)
          REFERENCE   tools/transformer_spike.py's output, without .safetensors/.json
          --batch N   sentences per forward call, like Stanza's 128 (default)
          --runs N    timed runs over all sentences after one warm-up (default: 1)
          --dump FILE write every sentence's ids and kept layers (s{i}.ids, s{i}.h{L}) as safetensors, for
                      tools/transformer_spike_tags.py
          --ids FILE  instead of the golden words: the id sequences s{i}.ids of FILE (tools/transformer_spike_tags.py
                      record); runs and dumps them, no drift report
        """;

    public static int Run(string[] args, string root)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        string modelDir = args[0], reference = args[1];
        int batch = 128, runs = 1;
        string? dump = null, idsFile = null;
        for (int i = 2; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--threads" when i + 1 < args.Length: ManagedThreads.Count = int.Parse(args[++i]); break;
                case "--batch" when i + 1 < args.Length: batch = int.Parse(args[++i]); break;
                case "--runs" when i + 1 < args.Length: runs = int.Parse(args[++i]); break;
                case "--invariance": break;
                case "--dump" when i + 1 < args.Length: dump = args[++i]; break;
                case "--ids" when i + 1 < args.Length: idsFile = args[++i]; break;
                default: Console.Error.WriteLine(Usage); return 2;
            }
        }

        var sentences = new List<string[]>();
        foreach (var file in new[] { "pipeline.conllu" }.Concat(Directory.GetFiles(Path.Combine(root, "tests", "golden"), "validation*.conllu")
                     .Select(Path.GetFileName)
                     .Order(StringComparer.Ordinal)))
        {
            sentences.AddRange(GoldenWords(Path.Combine(root, "tests", "golden", file!)));
        }

        var load = Stopwatch.StartNew();
        var tokenizer = WordPiece.Load(Path.Combine(modelDir, "tokenizer.json"));
        var weights = Path.Combine(modelDir, "pytorch_model.pt");
        if (!File.Exists(weights))
            File.CreateSymbolicLink(weights, "pytorch_model.bin");
        var encoder = ManagedElectra.Load(Checkpoint.Load(weights), heads: 16);
        GC.Collect();
        Console.WriteLine($"Load: {load.Elapsed.TotalSeconds:F1} s, {encoder.Layers} layers, hidden {encoder.Hidden}, "
            + $"working set {Environment.WorkingSet / 1e6:F0} MB, managed threads {ManagedThreads.Count}, {Gemm.Path}");

        var encoded = sentences.Select(tokenizer.Encode).ToList();
        if (idsFile != null)
        {
            var given = SafeTensorFile.Load(idsFile);
            encoded = [.. Enumerable.Range(0, given.Tensors.Count).Select(s => (given.Read<long>($"s{s}.ids").Select(v => (int)v).ToArray(), Array.Empty<int>()))];
        }

        if (args.Contains("--invariance"))
        {
            // Bits of each of the first 64 sentences alone vs inside one 64-sentence batch, with and without rowInvariant.
            var first = encoded.Take(64).Select(e => e.Ids).ToList();
            foreach (bool invariant in new[] { false, true })
            {
                var together = encoder.Forward(first, [24], rowInvariant: invariant);
                int differ = 0;
                for (int s = 0; s < first.Count; s++)
                {
                    var alone = encoder.Forward([first[s]], [24], rowInvariant: invariant)[0][0];
                    if (!alone.AsSpan().SequenceEqual(together[s][0]))
                        differ++;
                }

                Console.WriteLine($"rowInvariant {invariant}: {differ} of {first.Count} sentences differ in some bit alone vs batched");
            }

            return 0;
        }

        var refFile = SafeTensorFile.Load(reference + ".safetensors");
        if (idsFile != null)
        {
            var dumpOnly = Forward(encoder, encoded, [21, 22, 23, 24], batch);
            WriteSafetensors(dump!, encoded.Select((e, s) => (s, e)).SelectMany(t => new[] { 21, 22, 23, 24 }.Select((l, k) => ($"s{t.s}.h{l}", "F32",
                new long[] { t.e.Ids.Length, encoder.Hidden }, System.Runtime.InteropServices.MemoryMarshal.AsBytes(dumpOnly[t.s][k].AsSpan()).ToArray()))
                .Prepend(($"s{t.s}.ids", "I64", new long[] { t.e.Ids.Length },
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.e.Ids.Select(v => (long)v).ToArray().AsSpan()).ToArray()))));
            Console.WriteLine($"Dumped {encoded.Count} sequences to {dump}");
            return 0;
        }

        int compared = 0, idMismatches = 0;
        for (int s = 0; refFile.Tensors.ContainsKey($"s{s}.ids"); s++, compared++)
        {
            var ids = refFile.Read<long>($"s{s}.ids");
            var words = refFile.Read<long>($"s{s}.words");
            if (!ids.Select(v => (int)v).SequenceEqual(encoded[s].Ids) || !words.Select(v => (int)v).SequenceEqual(encoded[s].Words))
            {
                idMismatches++;
                if (idMismatches <= 5)
                    Console.WriteLine($"  ids differ in sentence {s}: {string.Join(' ', sentences[s])}\n    python {string.Join(',', ids)}\n    C#     {string.Join(',', encoded[s].Ids)}");
            }
        }

        int pieces = encoded.Sum(e => e.Ids.Length), wordCount = sentences.Sum(s => s.Length);
        Console.WriteLine($"Tokenizer: {sentences.Count} sentences, {wordCount} words, {pieces} pieces; {idMismatches} of {compared} reference sentences differ");

        int[] keep = [21, 22, 23, 24];
        float[][][] outputs = Forward(encoder, encoded, keep, batch); // warm-up, and the outputs compared below
        if (dump != null)
        {
            WriteSafetensors(dump, encoded.Select((e, s) => (s, e)).SelectMany(t => keep.Select((l, k) => ($"s{t.s}.h{l}", "F32",
                new long[] { t.e.Ids.Length, encoder.Hidden }, System.Runtime.InteropServices.MemoryMarshal.AsBytes(outputs[t.s][k].AsSpan()).ToArray()))
                .Prepend(($"s{t.s}.ids", "I64", new long[] { t.e.Ids.Length },
                    System.Runtime.InteropServices.MemoryMarshal.AsBytes(t.e.Ids.Select(v => (long)v).ToArray().AsSpan()).ToArray()))));
        }

        var times = new List<double>();
        long allocated = GC.GetTotalAllocatedBytes(true);
        for (int r = 0; r < runs; r++)
        {
            var run = Stopwatch.StartNew();
            Forward(encoder, encoded, keep, batch);
            times.Add(run.Elapsed.TotalSeconds);
        }

        times.Sort();
        double median = times[times.Count / 2];
        Console.WriteLine($"Forward: {median:F2} s for {wordCount} words ({wordCount / median:F0} words/s, {pieces / median:F0} pieces/s), "
            + $"peak working set {Process.GetCurrentProcess().PeakWorkingSet64 / 1e6:F0} MB, {(GC.GetTotalAllocatedBytes(true) - allocated) / runs / 1e6:F0} MB allocated per run");

        // Drift: every kept layer, and Stanza's mixes (pos: (h21 + h22 + h23) / 4; depparse: its 4-layer bert_layer_mix).
        for (int k = 0; k < keep.Length; k++)
        {
            var (abs, rel, at) = Drift(refFile, compared, s => $"s{s}.h{keep[k]}", s => outputs[s][k]);
            var (abs64, _, _) = Drift(refFile, refFile.Tensors.Keys.Count(t => t.EndsWith($".h{keep[k]}.f64", StringComparison.Ordinal)),
                s => $"s{s}.h{keep[k]}.f64", s => outputs[s][k]);
            var (py64, _, _) = Drift(refFile, refFile.Tensors.Keys.Count(t => t.EndsWith($".h{keep[k]}.f64", StringComparison.Ordinal)),
                s => $"s{s}.h{keep[k]}.f64", s => refFile.Read<float>($"s{s}.h{keep[k]}"));
            Console.WriteLine($"  hidden_states[{keep[k]}]: max |C# - torch f32| {abs:E2} (largest relative {rel:E2}, {at}); "
                + $"vs torch f64 {abs64:E2}; torch f32 vs f64 {py64:E2}");
        }

        var mix = Drift(refFile, compared, null, s => Mix(outputs[s]), s => Mix(keep.Select(l => refFile.Read<float>($"s{s}.h{l}")).ToArray()));
        Console.WriteLine($"  Stanza pos mix (h21 + h22 + h23) / 4: max |C# - torch f32| {mix.Abs:E2}");
        return idMismatches == 0 ? 0 : 1;
    }

    private static void WriteSafetensors(string path, IEnumerable<(string Name, string Dtype, long[] Shape, byte[] Data)> tensors)
    {
        var list = tensors.ToList();
        var header = new System.Text.Json.Nodes.JsonObject();
        long offset = 0;
        foreach (var (name, dtype, shape, data) in list)
        {
            header[name] = new System.Text.Json.Nodes.JsonObject
            {
                ["dtype"] = dtype,
                ["shape"] = new System.Text.Json.Nodes.JsonArray([.. shape.Select(v => (System.Text.Json.Nodes.JsonNode)v)]),
                ["data_offsets"] = new System.Text.Json.Nodes.JsonArray(offset, offset + data.Length),
            };
            offset += data.Length;
        }

        var json = System.Text.Encoding.UTF8.GetBytes(header.ToJsonString());
        int padded = (json.Length + 7) / 8 * 8;
        using var file = File.Create(path);
        file.Write(BitConverter.GetBytes((long)padded));
        file.Write(json);
        for (int i = json.Length; i < padded; i++)
            file.WriteByte((byte)' ');
        foreach (var t in list)
            file.Write(t.Data);
    }

    private static float[] Mix(float[][] layers)
    {
        var mix = new float[layers[0].Length];
        for (int i = 0; i < mix.Length; i++)
            mix[i] = (layers[0][i] + layers[1][i] + layers[2][i]) / 4;
        return mix;
    }

    private static float[][][] Forward(ManagedElectra encoder, List<(int[] Ids, int[] Words)> encoded, int[] keep, int batch)
    {
        var all = new List<float[][]>();
        for (int b = 0; b < encoded.Count; b += batch)
            all.AddRange(encoder.Forward(encoded.Skip(b).Take(batch).Select(e => e.Ids).ToList(), keep));
        return [.. all];
    }

    private static (double Abs, double Rel, string At) Drift(SafeTensorFile refFile, int sentences, Func<int, string>? key, Func<int, float[]> actual,
        Func<int, float[]>? expected = null)
    {
        double abs = 0, rel = 0;
        string at = "";
        for (int s = 0; s < sentences; s++)
        {
            var a = actual(s);
            double[] e = expected != null ? [.. expected(s).Select(v => (double)v)]
                : key!(s).EndsWith(".f64", StringComparison.Ordinal) ? refFile.Read<double>(key(s)) : [.. refFile.Read<float>(key(s)).Select(v => (double)v)];
            for (int i = 0; i < e.Length; i++)
            {
                double d = Math.Abs(a[i] - e[i]);
                if (d > abs)
                {
                    abs = d;
                    at = $"s{s} piece {i / 1024} unit {i % 1024}, value {e[i]:F3}";
                }

                rel = Math.Max(rel, d / Math.Max(1, Math.Abs(e[i])));
            }
        }

        return (abs, rel, at);
    }

    private static IEnumerable<string[]> GoldenWords(string path)
    {
        var current = new List<string>();
        foreach (var line in File.ReadLines(path))
        {
            if (line.Length == 0)
            {
                if (current.Count > 0)
                    yield return [.. current];
                current.Clear();
            }
            else if (!line.StartsWith('#'))
            {
                var cols = line.Split('\t');
                if (!cols[0].Contains('-') && !cols[0].Contains('.'))
                    current.Add(cols[1]);
            }
        }

        if (current.Count > 0)
            yield return [.. current];
    }
}
