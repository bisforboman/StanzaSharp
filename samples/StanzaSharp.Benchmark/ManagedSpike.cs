using System.Diagnostics;
using StanzaSharp;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using StanzaSharp.Pos;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

/// <summary>
/// Issue #29, Phase 0: the managed charlm and highway biLSTM against TorchSharp, for values and speed, on the POS
/// tagger's batches. docs/managed-backend-spike.md has the results.
/// <code>
/// StanzaSharp.Benchmark managed-spike check [--models DIR]
/// StanzaSharp.Benchmark managed-spike speed [--models DIR] [--threads N] [--runs N] [--copies N] [--impl both|torch|managed] [--path vector256|vector128|scalar]
/// StanzaSharp.Benchmark managed-spike concurrent [--callers N] (other options as speed)
/// </code>
/// </summary>
internal static class ManagedSpike
{
    private const int BatchSize = 250, MaximumTokens = 5000; // the tagger's batching (PosTagger.Process)

    public static int Run(string[] args, string repoRoot, Func<int, string> buildText)
    {
        string mode = args.FirstOrDefault() ?? "speed", modelDir = Path.Combine("models", "converted", "en"), impl = "both";
        int threads = Environment.ProcessorCount, runs = 3, copies = 8, callers = 4;
        for (int i = 1; i < args.Length; i++)
            switch (args[i])
            {
                case "--models": modelDir = args[++i]; break;
                case "--threads": threads = int.Parse(args[++i]); break;
                case "--runs": runs = int.Parse(args[++i]); break;
                case "--copies": copies = int.Parse(args[++i]); break;
                case "--impl": impl = args[++i]; break;
                case "--callers": callers = int.Parse(args[++i]); break;
                case "--path": Gemm.Path = Enum.Parse<KernelPath>(args[++i], ignoreCase: true); break;
                default: Console.Error.WriteLine($"Unknown option {args[i]}"); return 2;
            }
        torch.set_num_threads(threads);
        ManagedThreads.Count = threads;
        using var _ = torch.no_grad();
        Console.WriteLine($"threads {threads}, kernel path {Gemm.Path} (detected {Gemm.Detect()}), {System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture}, " +
                          $"AVX2 {System.Runtime.Intrinsics.X86.Avx2.IsSupported}, FMA {System.Runtime.Intrinsics.X86.Fma.IsSupported}, " +
                          $"AdvSimd {System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported}, Vector512 {System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated}");

        if (mode == "gemm")
            return GemmSpeed();

        using var nlp = Pipeline.Load(modelDir, new PipelineOptions { Processors = "tokenize,mwt" });
        var m = new Models(modelDir);
        return mode switch
        {
            "check" => Check(m, nlp, repoRoot),
            "concurrent" => Concurrent(m, nlp, buildText(copies), callers, impl),
            _ => Speed(m, nlp, buildText(copies), runs, impl),
        };
    }

    /// <summary>The models of both implementations, plus the tagger's input layers and heads (TorchSharp).</summary>
    private sealed class Models
    {
        public readonly CharLanguageModel Forward, Backward;
        public readonly ManagedCharLanguageModel ManagedForward, ManagedBackward;
        public readonly HighwayLstm Highway;
        public readonly ManagedHighwayLstm ManagedHighway;
        public readonly PosTagger Tagger;
        public readonly Pretrain Pretrain;
        public readonly float[] PretrainEmb, WordEmb, TransPretrained;
        public readonly Dictionary<string, int> WordVocab;
        public readonly int WordUnk, WordDim, Transformed;
        public readonly Linear UposHid, UposClf, XposHid, FeatsHid;
        public readonly Embedding UposEmb;
        public readonly Biaffine XposClf;
        public readonly Biaffine[] FeatsClf;
        public readonly string[] Upos, Xpos;
        public readonly (string Key, string[] Values)[] Feats;

        public Models(string dir)
        {
            Forward = CharLanguageModel.Load(Path.Combine(dir, "forward_charlm/1billion"));
            Backward = CharLanguageModel.Load(Path.Combine(dir, "backward_charlm/1billion"));
            ManagedForward = ManagedCharLanguageModel.Load(Path.Combine(dir, "forward_charlm/1billion"));
            ManagedBackward = ManagedCharLanguageModel.Load(Path.Combine(dir, "backward_charlm/1billion"));
            Pretrain = Pretrain.Load(Path.Combine(dir, "pretrain/conll17"));
            PretrainEmb = Pretrain.Embeddings.data<float>().ToArray();
            var path = Path.Combine(dir, "pos/combined_charlm");
            Tagger = PosTagger.Load(path, Pretrain, Forward, Backward);
            var ckpt = Checkpoint.Load(path);
            var model = ckpt.Root["model"]!;
            var vocab = ckpt.Root["vocab"]!;
            var config = ckpt.Root["config"]!;
            Highway = new HighwayLstm(ckpt, model, "taggerlstm", 2248, 200, 2);
            ManagedHighway = new ManagedHighwayLstm(ckpt, model, "taggerlstm", 2248, 200, 2);
            WordVocab = Checkpoint.UnitToId(vocab["word"]);
            WordUnk = WordVocab["<UNK>"];
            WordEmb = ckpt.Tensor<float>(model["word_emb.weight"]);
            WordDim = (int)ckpt.Shape(model["word_emb.weight"])[1];
            TransPretrained = ckpt.Tensor<float>(model["trans_pretrained.weight"]);
            Transformed = (int)ckpt.Shape(model["trans_pretrained.weight"])[0];

            string[] Strings(System.Text.Json.Nodes.JsonNode n) => n.AsArray().Select(x => x!.GetValue<string>()).ToArray();
            Upos = Strings(vocab["upos"]!["_id2unit"]!);
            Xpos = Strings(vocab["xpos"]!["_id2unit"]!);
            Feats = vocab["feats"]!["_id2unit"]!.AsObject().Select(kv => (kv.Key, Strings(kv.Value!))).ToArray();
            int biaff = config["deep_biaff_hidden_dim"]!.GetValue<int>(), composite = config["composite_deep_biaff_hidden_dim"]!.GetValue<int>();
            int tagEmb = config["tag_emb_dim"]!.GetValue<int>();
            UposHid = nn.Linear(400, biaff).LoadFrom(ckpt, model, "upos_hid.");
            UposClf = nn.Linear(biaff, Upos.Length).LoadFrom(ckpt, model, "upos_clf.");
            UposEmb = nn.Embedding(Upos.Length, tagEmb, padding_idx: 0).LoadFrom(ckpt, model, "upos_emb.");
            XposHid = nn.Linear(400, biaff).LoadFrom(ckpt, model, "tag_hid.xpos.");
            XposClf = new Biaffine(ckpt, model, "tag_clf.xpos.", biaff, tagEmb, Xpos.Length);
            FeatsHid = nn.Linear(400, composite).LoadFrom(ckpt, model, "tag_hid.feats.");
            FeatsClf = Feats.Select((f, i) => new Biaffine(ckpt, model, $"tag_clf.feats.{i}.", composite, tagEmb, f.Values.Length)).ToArray();
        }
    }

    private sealed record Batch(List<IReadOnlyList<string>> Original, List<IReadOnlyList<string>> Words, long[] Lengths, long[] WordOf, long[] PackedRow);

    /// <summary>Sentences of the text after tokenize and mwt, in the tagger's batches, with its packed order.</summary>
    private static List<Batch> Batches(Pipeline nlp, string text)
    {
        var sentences = nlp.Process(text).Sentences.Select(s => (IReadOnlyList<string>)s.Words.Select(w => w.Text).ToList()).Where(s => s.Count > 0).ToList();
        var batches = new List<Batch>();
        for (int b = 0, end; b < sentences.Count; b = end)
        {
            int words = 0;
            for (end = b; end < sentences.Count && end - b < BatchSize && (end == b || words + sentences[end].Count <= MaximumTokens); end++)
                words += sentences[end].Count;
            var original = sentences.GetRange(b, end - b);
            var simplified = original.Select(s => (IReadOnlyList<string>)s.Select(PosTagger.SimplifyPunct).ToList()).ToList();
            var lengths = simplified.Select(s => (long)s.Count).ToArray();
            int width = (int)lengths.Max();
            var offsets = new long[lengths.Length];
            for (int i = 1; i < offsets.Length; i++)
                offsets[i] = offsets[i - 1] + lengths[i - 1];
            var wordOf = Rnn.PackedOrder(lengths).Select(x => offsets[x / width] + x % width).ToArray();
            var packedRow = new long[wordOf.Length];
            for (int p = 0; p < wordOf.Length; p++)
                packedRow[wordOf[p]] = p;
            batches.Add(new Batch(original, simplified, lengths, wordOf, packedRow));
        }
        return batches;
    }

    private static float[] Flatten(List<Tensor> reps)
    {
        var result = reps.SelectMany(r => r.data<float>().ToArray()).ToArray();
        foreach (var r in reps)
            r.Dispose();
        return result;
    }

    /// <summary>The tagger's LSTM input [words, 2248] in packed order: word emb, trans_pretrained(pretrain), forward and backward charlm.</summary>
    private static float[] Input(Models m, Batch batch, float[] forward, float[] backward)
    {
        var flat = batch.Words.SelectMany(s => s).ToArray();
        int n = flat.Length, wd = m.WordDim, tr = m.Transformed, pd = 100, ch = 1024, width = wd + tr + 2 * ch;
        var input = new float[n * width];
        Parallel.For(0, n, p =>
        {
            int k = (int)batch.WordOf[p];
            var lower = PyString.Lower(flat[k]);
            var row = input.AsSpan(p * width, width);
            m.WordEmb.AsSpan(m.WordVocab.GetValueOrDefault(lower, m.WordUnk) * wd, wd).CopyTo(row);
            var pre = m.PretrainEmb.AsSpan(m.Pretrain.UnitToId(lower) * pd, pd);
            for (int j = 0; j < tr; j++)
            {
                float s = 0;
                var w = m.TransPretrained.AsSpan(j * pd, pd);
                for (int q = 0; q < pd; q++)
                    s += pre[q] * w[q];
                row[wd + j] = s;
            }
            forward.AsSpan(k * ch, ch).CopyTo(row[(wd + tr)..]);
            backward.AsSpan(k * ch, ch).CopyTo(row[(wd + tr + ch)..]);
        });
        return input;
    }

    /// <summary>The tagger's heads on LSTM output rows in packed order: tags per word in sentence order, and the UPOS logits.</summary>
    private static ((string, string, string?)[] Tags, float[] Logits) Heads(Models m, Batch batch, float[] packedOutput)
    {
        using var scope = NewDisposeScope();
        var output = torch.tensor(packedOutput, [batch.PackedRow.Length, 400]).index_select(0, torch.tensor(batch.PackedRow));
        var uposScores = m.UposClf.forward(nn.functional.relu(m.UposHid.forward(output)));
        var uposIds = uposScores.argmax(1);
        var parent = m.UposEmb.forward(uposIds);
        var xpos = m.XposClf.Forward(nn.functional.relu(m.XposHid.forward(output)), parent).argmax(1).data<long>().ToArray();
        var featsHid = nn.functional.relu(m.FeatsHid.forward(output));
        var feats = m.FeatsClf.Select(c => c.Forward(featsHid, parent).argmax(1).data<long>().ToArray()).ToArray();
        var upos = uposIds.data<long>().ToArray();
        var tags = new (string, string, string?)[upos.Length];
        for (int k = 0; k < upos.Length; k++)
        {
            var parts = Enumerable.Range(0, feats.Length).Where(f => feats[f][k] != 2).Select(f => $"{m.Feats[f].Key}={m.Feats[f].Values[feats[f][k]]}").ToList();
            tags[k] = (m.Upos[upos[k]], m.Xpos[xpos[k]], parts.Count == 0 ? null : string.Join('|', parts));
        }
        return (tags, uposScores.data<float>().ToArray());
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new InvalidOperationException($"lengths {a.Length} vs {b.Length}");
        float max = 0;
        for (int i = 0; i < a.Length; i++)
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        return max;
    }

    private static int Check(Models m, Pipeline nlp, string repoRoot)
    {
        var golden = Path.Combine(repoRoot, "tests", "golden");
        // The golden charlm tensors (Python Stanza, 3 sentences, each alone).
        var tensors = SafeTensorFile.Load(Path.Combine(golden, "intermediates.safetensors"));
        var index = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(Path.Combine(golden, "intermediates.json")))!["sentences"]!.AsArray();
        foreach (var (name, lm, tlm) in new[] { ("forward", m.ManagedForward, m.Forward), ("backward", m.ManagedBackward, m.Backward) })
        {
            float dm = 0, dt = 0;
            for (int i = 0; i < index.Count; i++)
            {
                IReadOnlyList<string> sentence = index[i]!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList();
                var expected = tensors.Read<float>($"s{i}.charlm_{name}");
                dm = Math.Max(dm, MaxDiff(expected, lm.BuildCharRepresentation([sentence])[0]));
                dt = Math.Max(dt, MaxDiff(expected, Flatten(tlm.BuildCharRepresentation([sentence]))));
            }
            Console.WriteLine($"golden charlm {name}: max |diff| managed {dm:E2}, TorchSharp {dt:E2}");
        }
        var files = new[] { "corpus.txt" }.Concat(Directory.GetFiles(golden, "validation*.txt").Select(Path.GetFileName).Order(StringComparer.Ordinal)!).ToList();
        float baseline = 0, charF = 0, charB = 0, highway = 0, highwayFull = 0, logits = 0;
        int words = 0, sentences = 0, tagDiffs = 0, tagDiffsReplica = 0;
        foreach (var file in files)
        {
            foreach (var batch in Batches(nlp, File.ReadAllText(Path.Combine(golden, file!))))
            {
                sentences += batch.Words.Count;
                // Charlm: TorchSharp vs managed, on the tagger's batch.
                var tf = Flatten(m.Forward.BuildCharRepresentation(batch.Words));
                var tb = Flatten(m.Backward.BuildCharRepresentation(batch.Words));
                var mf = m.ManagedForward.BuildCharRepresentation(batch.Words).SelectMany(x => x).ToArray();
                var mb = m.ManagedBackward.BuildCharRepresentation(batch.Words).SelectMany(x => x).ToArray();
                charF = Math.Max(charF, MaxDiff(tf, mf));
                // Baseline: TorchSharp itself, one sentence at a time vs batched (the charlm is not batch-invariant).
                var alone = batch.Words.SelectMany(s => Flatten(m.Forward.BuildCharRepresentation([s]))).ToArray();
                baseline = Math.Max(baseline, MaxDiff(tf, alone));
                charB = Math.Max(charB, MaxDiff(tb, mb));

                // Highway on the same (TorchSharp) input; then everything managed (charlm + highway) through the heads.
                var input = Input(m, batch, tf, tb);
                float[] torchOut;
                using (var scope = NewDisposeScope())
                    torchOut = m.Highway.Forward(Rnn.Pack(torch.tensor(input, [batch.WordOf.Length, input.Length / batch.WordOf.Length]), batch.Lengths)).data.data<float>().ToArray();
                highway = Math.Max(highway, MaxDiff(torchOut, m.ManagedHighway.Forward(input, batch.Lengths)));
                var managedOut = m.ManagedHighway.Forward(Input(m, batch, mf, mb), batch.Lengths);
                highwayFull = Math.Max(highwayFull, MaxDiff(torchOut, managedOut));

                var reference = m.Tagger.Predict(batch.Original, out var refLogits).SelectMany(t => t).ToArray();
                var (replica, replicaLogits) = Heads(m, batch, torchOut);
                var (managed, managedLogits) = Heads(m, batch, managedOut);
                words += reference.Length;
                tagDiffsReplica += reference.Zip(replica).Count(x => x.First != x.Second);
                tagDiffs += reference.Zip(managed).Count(x => x.First != x.Second);
                logits = Math.Max(logits, MaxDiff(refLogits.SelectMany(x => x).ToArray(), managedLogits));
                _ = replicaLogits;
            }
        }
        Console.WriteLine($"{files.Count} files, {sentences} sentences, {words} words");
        Console.WriteLine($"charlm forward  max |managed - TorchSharp| {charF:E2}");
        Console.WriteLine($"charlm forward  max |TorchSharp alone - TorchSharp batched| {baseline:E2} (baseline)");
        Console.WriteLine($"charlm backward max |managed - TorchSharp| {charB:E2}");
        Console.WriteLine($"highway (same input) max |diff| {highway:E2}");
        Console.WriteLine($"highway (managed charlm + highway) max |diff| {highwayFull:E2}");
        Console.WriteLine($"UPOS logits (managed charlm + highway, TorchSharp heads) max |diff| {logits:E2}");
        Console.WriteLine($"words whose UPOS/XPOS/feats differ from PosTagger: managed {tagDiffs}, TorchSharp replica {tagDiffsReplica}");
        return tagDiffs == 0 && tagDiffsReplica == 0 ? 0 : 1;
    }

    /// <summary>GEMM alone, managed vs torch.nn.functional.linear, on the charlm's and the highway's shapes.</summary>
    private static int GemmSpeed()
    {
        var rng = new Random(1);
        foreach (var (m, k, n) in new[] { (250, 1024, 4096), (32, 1024, 4096), (1, 1024, 4096), (3000, 2248, 2400), (250, 200, 800) })
        {
            var a = Enumerable.Range(0, m * k).Select(_ => (float)rng.NextDouble()).ToArray();
            var w = Enumerable.Range(0, n * k).Select(_ => (float)rng.NextDouble()).ToArray();
            var packed = new PackedMatrix(w, n, k);
            using var ta = torch.tensor(a, [m, k]);
            using var tw = torch.tensor(w, [n, k]);
            double flops = 2.0 * m * k * n;
            int reps = (int)Math.Max(3, 2e10 / flops);
            double Gflops(Action f)
            {
                f();
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < reps; i++)
                    f();
                return flops * reps / sw.Elapsed.TotalSeconds / 1e9;
            }
            var managed = Gflops(() => Gemm.Run(a, m, packed, null));
            var torchRate = Gflops(() => nn.functional.linear(ta, tw).Dispose());
            Console.WriteLine($"[{m}, {k}] x [{k}, {n}]: managed {managed,6:F1} GFLOP/s, torch {torchRate,6:F1} GFLOP/s");
        }
        return 0;
    }

    /// <summary>Both managed charlms over every batch, into one pooled output buffer (as a caller would use them).</summary>
    private static void ManagedCharlmOver(Models m, List<Batch> batches, CancellationToken ct = default)
    {
        foreach (var b in batches)
        {
            int words = b.WordOf.Length, h = m.ManagedForward.HiddenDim;
            var buffer = System.Buffers.ArrayPool<float>.Shared.Rent(words * 2 * h);
            try
            {
                m.ManagedForward.BuildCharRepresentation(b.Words, buffer.AsSpan(0, words * 2 * h), 2 * h, ct);
                m.ManagedBackward.BuildCharRepresentation(b.Words, buffer.AsSpan(h, words * 2 * h - h), 2 * h, ct);
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(buffer);
            }
        }
    }

    private static void ManagedHighwayOver(Models m, List<Batch> batches, List<float[]> inputs, CancellationToken ct = default)
    {
        for (int i = 0; i < batches.Count; i++)
        {
            int size = batches[i].WordOf.Length * m.ManagedHighway.OutputSize;
            var output = System.Buffers.ArrayPool<float>.Shared.Rent(size);
            m.ManagedHighway.Forward(inputs[i], batches[i].Lengths, output.AsSpan(0, size), ct);
            System.Buffers.ArrayPool<float>.Shared.Return(output);
        }
    }

    /// <summary>
    /// N callers at once, each running the charlms and the highway over all batches (TorchSharp: each caller gets its
    /// own libtorch team; managed: the shared pool). Reports the wall time and the mean time per call.
    /// </summary>
    private static int Concurrent(Models m, Pipeline nlp, string text, int callers, string impl)
    {
        var batches = Batches(nlp, text);
        var inputs = batches.Select(b => Input(m, b, Flatten(m.Forward.BuildCharRepresentation(b.Words)), Flatten(m.Backward.BuildCharRepresentation(b.Words)))).ToList();
        void Torch()
        {
            using var _ = torch.no_grad();
            foreach (var b in batches)
                foreach (var lm in new[] { m.Forward, m.Backward })
                    foreach (var t in lm.BuildCharRepresentation(b.Words))
                        t.Dispose();
            for (int i = 0; i < batches.Count; i++)
                using (NewDisposeScope())
                    m.Highway.Forward(Rnn.Pack(torch.tensor(inputs[i], [batches[i].WordOf.Length, 2248]), batches[i].Lengths), disposeInput: true);
        }
        void Managed()
        {
            ManagedCharlmOver(m, batches);
            ManagedHighwayOver(m, batches, inputs);
        }
        foreach (var which in new[] { "torch", "managed" }.Where(w => impl == "both" || impl == w))
        {
            Action call = which == "torch" ? Torch : Managed;
            call(); // warm-up, on this thread
            var perCall = new double[callers];
            using var start = new Barrier(callers + 1);
            var threads = Enumerable.Range(0, callers).Select(c => new Thread(() =>
            {
                start.SignalAndWait();
                var sw = Stopwatch.StartNew();
                call();
                perCall[c] = sw.Elapsed.TotalSeconds;
            })).ToList();
            threads.ForEach(t => t.Start());
            start.SignalAndWait();
            var wall = Stopwatch.StartNew();
            threads.ForEach(t => t.Join());
            Console.WriteLine($"{which,-8} callers {callers}: wall {wall.Elapsed.TotalSeconds,7:F2} s, per call mean {perCall.Average(),7:F2} s " +
                              $"(min {perCall.Min():F2}, max {perCall.Max():F2}), throughput {callers / wall.Elapsed.TotalSeconds:F3} calls/s, " +
                              $"OS threads {Process.GetCurrentProcess().Threads.Count}");
        }
        return 0;
    }

    private static int Speed(Models m, Pipeline nlp, string text, int runs, string impl)
    {
        var batches = Batches(nlp, text);
        long chars = batches.Sum(b => b.Words.Sum(s => s.Sum(w => w.EnumerateRunes().Count() + 1) + 1));
        Console.WriteLine($"{batches.Count} batches, {batches.Sum(b => b.Words.Count)} sentences, {batches.Sum(b => b.WordOf.Length)} words, {chars} charlm steps per direction");
        var inputs = batches.Select(b => Input(m, b, Flatten(m.Forward.BuildCharRepresentation(b.Words)), Flatten(m.Backward.BuildCharRepresentation(b.Words)))).ToList();

        double Time(Action a)
        {
            GC.Collect();
            long allocated = GC.GetTotalAllocatedBytes(precise: true);
            var sw = Stopwatch.StartNew();
            a();
            double s = sw.Elapsed.TotalSeconds;
            Console.WriteLine($"    (managed heap allocations {(GC.GetTotalAllocatedBytes(precise: true) - allocated) / 1048576.0:F1} MB)");
            return s;
        }
        void TorchCharlm()
        {
            foreach (var b in batches)
                foreach (var lm in new[] { m.Forward, m.Backward })
                    foreach (var t in lm.BuildCharRepresentation(b.Words))
                        t.Dispose();
        }
        void TorchHighway()
        {
            for (int i = 0; i < batches.Count; i++)
                using (NewDisposeScope())
                    m.Highway.Forward(Rnn.Pack(torch.tensor(inputs[i], [batches[i].WordOf.Length, 2248]), batches[i].Lengths), disposeInput: true);
        }
        void ManagedCharlm() => ManagedCharlmOver(m, batches);
        void ManagedHighway() => ManagedHighwayOver(m, batches, inputs);

        var results = new Dictionary<string, List<double>>();
        void Add(string key, double s)
        {
            (results.TryGetValue(key, out var l) ? l : results[key] = []).Add(s);
            Console.WriteLine($"  {key,-18} {s,8:F2} s");
        }
        for (int run = 0; run <= runs; run++) // run 0 is the warm-up
        {
            Console.WriteLine(run == 0 ? "warm-up" : $"run {run}");
            var order = run % 2 == 0 ? new[] { "torch", "managed" } : ["managed", "torch"];
            foreach (var which in order.Where(w => impl == "both" || impl == w))
            {
                var charlm = Time(which == "torch" ? TorchCharlm : ManagedCharlm);
                var hw = Time(which == "torch" ? TorchHighway : ManagedHighway);
                if (run > 0)
                {
                    Add($"{which} charlm", charlm);
                    Add($"{which} highway", hw);
                }
            }
        }
        Console.WriteLine("medians:");
        foreach (var (key, list) in results.OrderBy(kv => kv.Key))
            Console.WriteLine($"  {key,-18} {list.Order().ElementAt(list.Count / 2),8:F2} s   ({string.Join(", ", list.Select(x => x.ToString("F2")))})");
        Console.WriteLine($"peak working set {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576} MB");
        return 0;
    }
}
