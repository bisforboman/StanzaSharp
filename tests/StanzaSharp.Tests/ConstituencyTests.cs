using System.Text.Json.Nodes;
using StanzaSharp.Constituency;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class ConstituencyTests(ITestOutputHelper output)
{
    private sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain;
        public readonly CharLanguageModel? Forward, Backward;
        public readonly ConstituencyParser Parser;

        public Models(bool managed = false)
        {
            Pretrain = Repo.LoadPretrain(managed);
            if (managed)
            {
                Parser = ConstituencyParser.LoadManaged(Repo.Model("constituency/ptb3-revised_charlm"), Pretrain,
                    ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion")), ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion")));
                return;
            }
            Forward = CharLanguageModel.Load(Repo.Model("forward_charlm/1billion"));
            Backward = CharLanguageModel.Load(Repo.Model("backward_charlm/1billion"));
            Parser = ConstituencyParser.Load(Repo.Model("constituency/ptb3-revised_charlm"), Pretrain, Forward, Backward);
        }

        public void Dispose()
        {
            Parser.Dispose();
            Forward?.Dispose();
            Backward?.Dispose();
            Pretrain.Dispose();
        }
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void TransitionScores_MatchGoldenIntermediates(bool managed)
    {
        using var models = new Models(managed);
        var golden = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e =>
        {
            var words = e!["words"]!.AsArray().Select(w => w!.GetValue<string>());
            var tags = e["xpos"]!.AsArray().Select(t => t!.GetValue<string>());
            return (IReadOnlyList<(string, string)>)words.Zip(tags).ToList();
        }).ToList();

        // One batch; the golden scores were computed one sentence at a time.
        var scores = new List<List<float[]>>();
        var trees = models.Parser.Parse(sentences, scores);
        for (int i = 0; i < sentences.Count; i++)
        {
            Assert.Equal(index[i]!["tree"]!.GetValue<string>(), trees[i]?.ToString());
            Assert.Equal(golden[$"s{i}.constituency.scores"].Shape[0], scores[i].Count);
            float diff = TokenizerTests.AssertClose(golden.Read<float>($"s{i}.constituency.scores"), scores[i].SelectMany(r => r).ToArray(), 1e-3f, $"s{i}");
            output.WriteLine($"s{i}: {scores[i].Count} steps, transition scores max |diff| {diff:E2}");
        }
    }

    /// <summary>
    /// Near-ties: both backends parse every golden file (pipeline.conllu and validation*.conllu, Stanza's words and XPOS,
    /// charlms computed) with identical decisions at every step. Reports the per-step score drift between the backends and,
    /// per backend, how close each decision was: the margin between the best legal transition and the next legal one
    /// (steps with one legal transition decide nothing), plus the raw top-2 margin of the whole row, and the closest calls.
    /// Guards that the smallest margin stays at least 3 × the score drift between the backends.
    /// </summary>
    [ModelFact]
    public void NearTies_StayClearOfBackendDrift()
    {
        var files = new[] { "pipeline.conllu" }.Concat(Directory.GetFiles(Repo.Golden, "validation*.conllu").Select(Path.GetFileName).Order()).ToList();
        var sentences = new List<(string File, int Index, IReadOnlyList<(string, string)> Words)>();
        foreach (var file in files)
        {
            var doc = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, file!)));
            for (int i = 0; i < doc.Sentences.Count; i++)
                sentences.Add((file!, i, doc.Sentences[i].Words.Select(w => (w.Text, w.Xpos!)).ToList()));
        }

        var steps = new[] { new List<List<(float[] Row, bool[] Legal)>>(), new List<List<(float[] Row, bool[] Legal)>>() };
        var trees = new List<Tree?>[2];
        IReadOnlyList<Transition> transitions = [];
        foreach (bool managed in new[] { false, true })
        {
            using var models = new Models(managed);
            transitions = models.Parser.Transitions;
            var mine = steps[managed ? 1 : 0];
            mine.AddRange(sentences.Select(_ => new List<(float[] Row, bool[] Legal)>()));
            trees[managed ? 1 : 0] = models.Parser.Parse(sentences.Select(s => s.Words).ToList(), onStep: (i, row, legal) => mine[i].Add((row, legal)));
        }
        Assert.Equal(trees[0].Select(t => t?.ToString()), trees[1].Select(t => t?.ToString()));

        static (int Best, int Second, float Margin) Decision(float[] row, bool[]? legal)
        {
            int best = -1, second = -1;
            for (int j = 0; j < row.Length; j++)
            {
                if (legal != null && !legal[j])
                    continue;
                if (best < 0 || row[j] > row[best])
                    (best, second) = (j, best);
                else if (second < 0 || row[j] > row[second])
                    second = j;
            }
            return (best, second, second < 0 ? float.PositiveInfinity : row[best] - row[second]);
        }
        string Name(int j) => transitions[j].Kind == TransitionKind.Open ? $"Open({transitions[j].Label})" : transitions[j].Kind.ToString();

        float drift = 0;
        int total = 0, decisions = 0;
        var margins = new[] { new List<float>(), new List<float>() };
        var raw = new[] { new List<float>(), new List<float>() };
        var calls = new List<(float Min, string What)>();
        for (int i = 0; i < sentences.Count; i++)
        {
            Assert.Equal(steps[0][i].Count, steps[1][i].Count);
            for (int k = 0; k < steps[0][i].Count; k++)
            {
                var (t, m) = (steps[0][i][k], steps[1][i][k]);
                Assert.Equal(t.Legal, m.Legal);
                total++;
                for (int j = 0; j < t.Row.Length; j++)
                    drift = Math.Max(drift, Math.Abs(t.Row[j] - m.Row[j]));
                var (dt, dm) = (Decision(t.Row, t.Legal), Decision(m.Row, m.Legal));
                Assert.Equal(dt.Best, dm.Best);
                raw[0].Add(Decision(t.Row, null).Margin);
                raw[1].Add(Decision(m.Row, null).Margin);
                if (dt.Second < 0)
                    continue;
                decisions++;
                margins[0].Add(dt.Margin);
                margins[1].Add(dm.Margin);
                calls.Add((Math.Min(dt.Margin, dm.Margin), $"{sentences[i].File} sentence {sentences[i].Index} step {k}: {Name(dt.Best)} over {Name(dt.Second)}, " +
                    $"margin TorchSharp {dt.Margin:E2}, managed {dm.Margin:E2}" + (dt.Second != dm.Second ? $" (managed's runner-up {Name(dm.Second)})" : "")));
            }
        }
        output.WriteLine($"{sentences.Count} sentences, {total} steps, {decisions} with two or more legal transitions; scores max |diff| between backends {drift:E2}");
        string Counts(List<float> m) => $"< 1e-2: {m.Count(x => x < 1e-2f)}, < 1e-3: {m.Count(x => x < 1e-3f)}, < 1e-4: {m.Count(x => x < 1e-4f)}, min {m.Min():E2}";
        output.WriteLine($"decision margins TorchSharp {Counts(margins[0])}");
        output.WriteLine($"decision margins managed    {Counts(margins[1])}");
        output.WriteLine($"raw top-2 margins TorchSharp {Counts(raw[0])}");
        output.WriteLine($"raw top-2 margins managed    {Counts(raw[1])}");
        foreach (var (_, what) in calls.OrderBy(c => c.Min).Take(10))
            output.WriteLine(what);

        // The guard: a decision flips only if its margin is below the margin difference, at most twice the score drift.
        // Measured: smallest margin 3.6e-4 = 4.9 × the drift (7.3e-5). k = 3 keeps 50% over that bound and fails once a
        // kernel change grows the drift by 60% or a change moves the closest call 40% closer (docs/backends.md, "constituency").
        const float K = 3;
        float smallest = Math.Min(margins[0].Min(), margins[1].Min());
        Assert.True(smallest >= K * drift, $"smallest decision margin {smallest:E2} is under {K} × the backends' score drift {drift:E2}");
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Pipeline_MatchesGoldenTrees(bool managed)
    {
        // The tagger's charlm cache feeds the parser here; the test above computes the charlms itself.
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize,mwt,pos,constituency", Backend = Repo.PipelineBackendFor(managed) });
        var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu")));

        Assert.Equal(golden.Sentences.Select(s => s.Constituency?.ToString()), doc.Sentences.Select(s => s.Constituency?.ToString()));
    }
}
