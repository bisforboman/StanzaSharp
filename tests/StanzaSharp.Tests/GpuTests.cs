using System.Text.Json.Nodes;
using StanzaSharp.Constituency;
using StanzaSharp.Nn;
using StanzaSharp.Pos;
using StanzaSharp.Tokenize;
using TorchSharp;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// The pipeline on CUDA, compiled only in STANZASHARP_CUDA=1 builds. GPU kernels round differently from
/// CPU ones, so these compare with a tolerance and report how much of the (CPU) golden data the GPU still
/// reproduces exactly; see docs/gpu.md. Each runs with TF32 on (libtorch's default for cuDNN) and off.
/// </summary>
public class GpuTests(ITestOutputHelper output)
{
    [CudaTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Intermediates_StayCloseToGolden(bool tf32)
    {
        using var tf32Scope = Tf32(tf32);
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var words = index.Select(e => (IReadOnlyList<string>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>()).ToList()).ToList();
        var tagged = index.Select(e => (IReadOnlyList<(string, string)>)e!["words"]!.AsArray().Select(w => w!.GetValue<string>())
            .Zip(e["xpos"]!.AsArray().Select(t => t!.GetValue<string>())).ToList()).ToList();

        // Max |diff| against the golden tensors. Without TF32 the CPU tests' tolerances hold.
        void Check(string what, string key, float[] actual, float cpuTolerance)
        {
            var expected = golden.Read<float>(key);
            Assert.Equal(expected.Length, actual.Length);
            float max = expected.Zip(actual, (e, a) => Math.Abs(e - a)).Max();
            float tolerance = tf32 ? 50 * cpuTolerance : cpuTolerance;
            output.WriteLine($"{what,-26} max |diff| {max:E2}");
            Assert.True(max <= tolerance, $"{what}: max |diff| {max} > {tolerance}");
        }

        using var tokenizer = Tokenizer.Load(Repo.Model("tokenize/combined_nocharlm"), torch.CUDA);
        using var pretrain = Pretrain.Load(Repo.Model("pretrain/conll17"), torch.CUDA);
        using var forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"), torch.CUDA);
        using var backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"), torch.CUDA);
        using var tagger = PosTagger.Load(Repo.Model("pos/combined_charlm"), pretrain, forward, backward, torch.CUDA);
        using var parser = ConstituencyParser.Load(Repo.Model("constituency/ptb3-revised_charlm"), pretrain, forward, backward, torch.CUDA);

        var repsForward = forward.BuildCharRepresentation(words);
        var repsBackward = backward.BuildCharRepresentation(words);
        tagger.Predict(words, out var uposLogits);
        var scores = new List<List<float[]>>();
        var trees = parser.Parse(tagged, scores);
        for (int i = 0; i < words.Count; i++)
        {
            Check($"s{i} tokenizer logits", $"s{i}.tokenize.pred", tokenizer.ParagraphLogits(index[i]!["text"]!.GetValue<string>(), out _), 1e-4f);
            using (var rep = repsForward[i])
                Check($"s{i} charlm forward", $"s{i}.charlm_forward", rep.ToArray<float>(), 1e-4f);
            using (var rep = repsBackward[i])
                Check($"s{i} charlm backward", $"s{i}.charlm_backward", rep.ToArray<float>(), 1e-4f);
            Check($"s{i} upos logits", $"s{i}.pos.upos_logits", uposLogits[i], 1e-3f);
            Check($"s{i} transition scores", $"s{i}.constituency.scores", scores[i].SelectMany(r => r).ToArray(), 1e-3f);
            Assert.Equal(index[i]!["tree"]!.GetValue<string>(), trees[i]?.ToString());
        }
    }

    /// <param name="set">"constituency": tests/golden/*.conllu (the default processors); "depparse": tests/golden/depparse (adds lemma and depparse).</param>
    [CudaTheory]
    [InlineData("constituency", true)]
    [InlineData("constituency", false)]
    [InlineData("depparse", true)]
    [InlineData("depparse", false)]
    public void Pipeline_MostlyMatchesGoldenConllu(string set, bool tf32)
    {
        using var tf32Scope = Tf32(tf32);
        var (dir, processors) = set == "depparse" ? (Path.Combine(Repo.Golden, "depparse"), "tokenize,mwt,pos,lemma,depparse") : (Repo.Golden, Pipeline.AllProcessors);
        using var nlp = Pipeline.Load(Repo.Models, processors, torch.CUDA);
        var files = Directory.GetFiles(dir, "validation*.conllu").Order().Append(Path.Combine(dir, set == "depparse" ? "corpus.conllu" : "pipeline.conllu")).ToList();
        int sentences = 0, same = 0, lines = 0, lineDiffs = 0, treeDiffs = 0;
        output.WriteLine($"{"file",-34}{"sentences",10}{"identical",10}{"lines",8}{"differ",8}{"trees",8}");
        foreach (var conllu in files)
        {
            var name = Path.GetFileName(conllu);
            var txt = Path.Combine(Repo.Golden, name == "pipeline.conllu" ? "corpus.txt" : Path.ChangeExtension(name, ".txt"));
            var actual = Conllu.Write(nlp.Process(File.ReadAllText(txt)));
            var golden = File.ReadAllText(conllu);
            if (name == "validation_nonbmp.conllu") // as in PipelineTests
                (actual, golden) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(golden));

            // Sentence blocks line up because tokenization matches; word lines carry tags, lemmas and heads,
            // "# constituency" the tree.
            var expected = golden.TrimEnd().Split("\n\n");
            var got = actual.TrimEnd().Split("\n\n");
            Assert.Equal(expected.Length, got.Length);
            int n = expected.Length, s = 0, l = 0, ld = 0, td = 0;
            for (int i = 0; i < n; i++)
            {
                var (e, a) = (expected[i].Split('\n'), got[i].Split('\n'));
                Assert.Equal(e.Length, a.Length);
                s += expected[i] == got[i] ? 1 : 0;
                for (int k = 0; k < e.Length; k++)
                {
                    if (!e[k].StartsWith('#'))
                    {
                        l++;
                        if (e[k] != a[k])
                        {
                            ld++;
                            output.WriteLine($"  {name} sentence {i}:\n  CPU {e[k]}\n  GPU {a[k]}");
                        }
                    }
                    else if (e[k].StartsWith("# constituency") && e[k] != a[k])
                    {
                        td++;
                        output.WriteLine($"  {name} sentence {i}:\n  CPU {e[k]}\n  GPU {a[k]}");
                    }
                }
            }
            output.WriteLine($"{name,-34}{n,10}{s,10}{l,8}{ld,8}{td,8}");
            (sentences, same, lines, lineDiffs, treeDiffs) = (sentences + n, same + s, lines + l, lineDiffs + ld, treeDiffs + td);
        }
        output.WriteLine($"{"total",-34}{sentences,10}{same,10}{lines,8}{lineDiffs,8}{treeDiffs,8}");
        output.WriteLine($"identical: sentences {100.0 * same / sentences:F2}%, word lines {100.0 * (lines - lineDiffs) / lines:F3}%, trees {100.0 * (sentences - treeDiffs) / sentences:F2}%");
        // Well below what an RTX 3080 gives (docs/gpu.md): these catch breakage, not rounding.
        Assert.True(lineDiffs <= 0.005 * lines, "word lines differ from the CPU golden data on more than 0.5% of words");
        Assert.True(treeDiffs <= 0.05 * sentences, "trees differ from the CPU golden data on more than 5% of sentences");
    }

    /// <summary>Sets TF32 for cuDNN and cuBLAS (process-wide; GPU tests share this class, so they run one at a time).</summary>
    private static Restore Tf32(bool on)
    {
        var old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32);
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = on;
        return new Restore(() => (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = old);
    }

    private sealed class Restore(Action undo) : IDisposable
    {
        public void Dispose() => undo();
    }
}

/// <summary>A theory that is skipped unless the converted models and a CUDA device (and CUDA libtorch) are available.</summary>
public sealed class CudaTheoryAttribute : TheoryAttribute
{
    public CudaTheoryAttribute()
    {
        if (!Directory.Exists(Repo.Models))
            Skip = "models/converted/en not found; run setup.ps1 -Models";
        else if (!torch.cuda.is_available())
            Skip = "CUDA not available; build with STANZASHARP_CUDA=1 on a machine with an NVIDIA GPU";
    }
}
