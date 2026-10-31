using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using StanzaSharp.Constituency;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class ConstituencyTests(ITestOutputHelper output, ConstituencyTests.Fixture fixture) : IClassFixture<ConstituencyTests.Fixture>
{
    /// <summary>Each backend's and language's <see cref="Models"/>, loaded on first use and shared by the class's tests (read-only).</summary>
    public sealed class Fixture : IDisposable
    {
        private readonly ModelCache<Models> _models = new((managed, lang) => new Models(managed, lang));

        internal Models Get(bool managed = false, string lang = "en") => _models.Get(managed, lang);

        public void Dispose() => _models.Dispose();
    }

    internal sealed class Models : IDisposable
    {
        public readonly Pretrain Pretrain;
        public readonly CharLanguageModel? Forward, Backward;
        public readonly ConstituencyParser Parser;

        public Models(bool managed, string lang)
        {
            Pretrain = Repo.LoadPretrain(managed, lang);
            if (managed)
            {
                Parser = ConstituencyParser.LoadManaged(Repo.ModelOf(lang, "constituency"), Pretrain,
                    ManagedCharLanguageModel.Load(Repo.Charlm(lang, forward: true)), ManagedCharLanguageModel.Load(Repo.Charlm(lang, forward: false)));
                return;
            }

            Forward = CharLanguageModel.Load(Repo.Charlm(lang, forward: true));
            Backward = CharLanguageModel.Load(Repo.Charlm(lang, forward: false));
            Parser = ConstituencyParser.Load(Repo.ModelOf(lang, "constituency"), Pretrain, Forward, Backward);
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
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void TransitionScores_MatchGoldenIntermediates(bool managed, string lang)
    {
        var models = fixture.Get(managed, lang);
        var golden = SafeTensorFile.Load(Path.Combine(Repo.GoldenFor(lang), "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "intermediates.json")))!["sentences"]!.AsArray();
        var sentences = index.Select(e =>
        {
            var words = e!["words"]!.AsArray().Select(w => w!.GetValue<string>());

            // The tags the parser reads: English's XPOS; German's UPOS (retag_method), the golden tree's preterminals.
            var tags = lang == "en"
                ? e["xpos"]!.AsArray().Select(t => t!.GetValue<string>())
                : Preterminals(Tree.Parse(e["tree"]!.GetValue<string>()));
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
    /// Reviewed near-ties of <see cref="NearTies_StayClearOfBackendDrift"/>: decisions whose margin is under 3 × the largest
    /// difference between the backends' margins, accepted at 2 × (1.5 × on Arm64) after a review (golden file, sentence
    /// index, step). Add an entry only after checking that both backends and Stanza make the same decision there; remove
    /// one that clears 3 ×.
    /// </summary>
    private static readonly (string Lang, string File, int Sentence, int Step, string Review)[] KnownNearTies =
    [
        // Stanford's re-uploaded English tagger (2026-10-30): Open(NP) over Open(ADVP), margin 9.79e-5 on TorchSharp (CI's
        // x64 settings), 1.03e-4 managed, against a largest margin difference of 3.81e-5 (2.6 ×; on the Arm64 runners 1.64 ×
        // and 1.80 ×). Same decision as Stanza; the owner accepted it on 2026-10-31.
        ("en", "validation_multilingual.conllu", 13, 7, "2026-10-31"),
    ];

    /// <summary>
    /// Near-ties: both backends parse every golden file (pipeline.conllu and validation*.conllu, Stanza's words and XPOS,
    /// charlms computed) with identical decisions at every step. Reports the per-step score drift between the backends and,
    /// per backend, how close each decision was: the margin between the best legal transition and the next legal one
    /// (steps with one legal transition decide nothing), plus the raw top-2 margin of the whole row, and the closest calls.
    /// Guards that every margin stays at least 3 × the largest difference between the backends' margins, except the reviewed
    /// near-ties in <see cref="KnownNearTies"/> (at least 2 ×, 1.5 × on Arm64), and reports where the largest score difference
    /// is. German's parser reads
    /// UPOS (retag_method) and allows longer unary chains (unary_limit 8).
    /// </summary>
    [ModelTheory]
    [InlineData("en")]
    [German]
    public void NearTies_StayClearOfBackendDrift(string lang)
    {
        var golden = Repo.GoldenFor(lang);
        var files = new[] { "pipeline.conllu" }.Concat(Directory.GetFiles(golden, "validation*.conllu").Select(Path.GetFileName).Order()).ToList();
        var sentences = new List<(string File, int Index, IReadOnlyList<(string, string)> Words)>();
        foreach (var file in files)
        {
            var doc = Conllu.Read(File.ReadAllText(Path.Combine(golden, file!)));
            for (int i = 0; i < doc.Sentences.Count; i++)
                sentences.Add((file!, i, doc.Sentences[i].Words.Select(w => (w.Text, (lang == "en" ? w.Xpos : w.Upos)!)).ToList()));
        }

        var steps = new[] { new List<List<(float[] Row, bool[] Legal)>>(), new List<List<(float[] Row, bool[] Legal)>>() };
        var trees = new List<Tree?>[2];
        IReadOnlyList<Transition> transitions = [];
        foreach (bool managed in new[] { false, true })
        {
            var models = fixture.Get(managed, lang);
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

        float drift = 0, marginDiff = 0, legalDrift = 0;
        string driftAt = "";
        var sentenceDrift = new float[sentences.Count];
        int total = 0, decisions = 0;
        var margins = new[] { new List<float>(), new List<float>() };
        var raw = new[] { new List<float>(), new List<float>() };
        var calls = new List<(float Min, string What, (string, string, int, int) Where)>();
        for (int i = 0; i < sentences.Count; i++)
        {
            Assert.Equal(steps[0][i].Count, steps[1][i].Count);
            for (int k = 0; k < steps[0][i].Count; k++)
            {
                var (t, m) = (steps[0][i][k], steps[1][i][k]);
                Assert.Equal(t.Legal, m.Legal);
                total++;
                for (int j = 0; j < t.Row.Length; j++)
                {
                    float d = Math.Abs(t.Row[j] - m.Row[j]);
                    sentenceDrift[i] = Math.Max(sentenceDrift[i], d);
                    if (d > drift)
                        (drift, driftAt) = (d, $"{sentences[i].File} sentence {sentences[i].Index} step {k} {Name(j)}{(t.Legal[j] ? "" : " (illegal)")}: {t.Row[j]} vs {m.Row[j]}, row's largest |score| {t.Row.Max(Math.Abs)}");
                    if (t.Legal[j])
                        legalDrift = Math.Max(legalDrift, d);
                }

                var (dt, dm) = (Decision(t.Row, t.Legal), Decision(m.Row, m.Legal));
                Assert.Equal(dt.Best, dm.Best);
                raw[0].Add(Decision(t.Row, null).Margin);
                raw[1].Add(Decision(m.Row, null).Margin);
                if (dt.Second < 0)
                    continue;
                decisions++;
                marginDiff = Math.Max(marginDiff, Math.Abs(dt.Margin - dm.Margin));
                margins[0].Add(dt.Margin);
                margins[1].Add(dm.Margin);
                calls.Add((Math.Min(dt.Margin, dm.Margin), $"{sentences[i].File} sentence {sentences[i].Index} step {k}: {Name(dt.Best)} over {Name(dt.Second)}, "
                    + $"margin TorchSharp {dt.Margin:E2}, managed {dm.Margin:E2}" + (dt.Second != dm.Second ? $" (managed's runner-up {Name(dm.Second)})" : ""), (lang, sentences[i].File, sentences[i].Index, k)));
            }
        }

        output.WriteLine($"{lang}: {sentences.Count} sentences, {total} steps, {decisions} with two or more legal transitions; scores max |diff| between backends {drift:E2}, decision margins max |diff| {marginDiff:E2}");
        output.WriteLine($"largest score difference at {driftAt}; over legal transitions only {legalDrift:E2}");
        foreach (int i in Enumerable.Range(0, sentences.Count).OrderByDescending(i => sentenceDrift[i]).Take(5))
            output.WriteLine($"  drift {sentenceDrift[i]:E2}: {sentences[i].File} sentence {sentences[i].Index}");
        string Counts(List<float> m) => $"< 1e-2: {m.Count(x => x < 1e-2f)}, < 1e-3: {m.Count(x => x < 1e-3f)}, < 1e-4: {m.Count(x => x < 1e-4f)}, min {m.Min():E2}";
        output.WriteLine($"decision margins TorchSharp {Counts(margins[0])}");
        output.WriteLine($"decision margins managed    {Counts(margins[1])}");
        output.WriteLine($"raw top-2 margins TorchSharp {Counts(raw[0])}");
        output.WriteLine($"raw top-2 margins managed    {Counts(raw[1])}");
        foreach (var (_, what, _) in calls.OrderBy(c => c.Min).Take(10))
            output.WriteLine(what);

        // The guard: besides identical decisions (asserted above), every margin must be at least K × the largest difference
        // between the backends' margins, the quantity that can flip a decision; a reviewed near-tie in KnownNearTies needs
        // KnownK × (the floor the owner accepted on 2026-10-31). A new model that makes another decision a near-tie fails
        // here until that decision is reviewed and listed, instead of the global factor moving to fit it.
        // On Arm64 KnownK is 1.5 (owner's decision, 2026-10-31): there the reference itself moves. Arm64 libtorch's kernels
        // aren't pinned like x64's (ATEN_CPU_CAPABILITY, MKL_CBWR), and TorchSharp's closest call was 1.06e-4 (macOS) and
        // 1.03e-4 (Windows on Arm) where x64 gives 9.79e-5, while the managed margins equal x64 AVX2's to three digits at
        // all ten closest calls; the margin differences were 6.29e-5 and 5.72e-5 (ratios 1.64 and 1.80). The next closest
        // call there (3.6e-4) still clears K = 3 (5.8 ×).
        const float K = 3;
        float knownK = RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? 1.5f : 2;
        var listed = KnownNearTies.Where(n => n.Lang == lang).Select(n => (n.Lang, n.File, n.Sentence, n.Step)).ToHashSet();
        var failures = calls.Where(c => c.Min < (listed.Contains(c.Where) ? knownK : K) * marginDiff).OrderBy(c => c.Min).ToList();
        foreach (var where in listed.Where(w => !calls.Any(c => c.Where == w && c.Min < K * marginDiff)))
            output.WriteLine($"KnownNearTies entry {where} is no longer under {K} × {marginDiff:E2} here: remove it if that holds on CI too");
        Assert.True(failures.Count == 0,
            $"{failures.Count} decision(s) under the bound ({K} × the largest margin difference {marginDiff:E2}, {knownK} × for KnownNearTies): "
            + string.Join("; ", failures.Select(f => f.What)) + ". If a new model made them near-ties, review each and add it to KnownNearTies.");
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Pipeline_MatchesGoldenTrees(bool managed, string lang)
    {
        // The tagger's charlm cache feeds the parser here; the test above computes the charlms itself.
        using var nlp = Pipeline.Load(Repo.ModelsFor(lang), new PipelineOptions { Language = lang, Processors = "tokenize,mwt,pos,constituency", Backend = Repo.PipelineBackendFor(managed) });
        var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "corpus.txt")));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "pipeline.conllu")));

        Assert.Equal(golden.Sentences.Select(s => s.Constituency?.ToString()), doc.Sentences.Select(s => s.Constituency?.ToString()));
    }

    private static IEnumerable<string> Preterminals(Tree tree) =>
        tree.IsPreterminal ? [tree.Label] : tree.Children.SelectMany(Preterminals);
}
