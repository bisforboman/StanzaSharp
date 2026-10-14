using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Tokenize;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

public class TokenizerTests(ITestOutputHelper output, SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    // managed: Backend.Managed (issue #29), else TorchSharp. Every golden test runs on both.
    private static Tokenizer Load(bool managed, string lang = "en") =>
        Tokenizer.Load(Repo.ModelOf(lang, "tokenize"), backend: Repo.Backend(managed));

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Logits_MatchGoldenIntermediates(bool managed, string lang)
    {
        using var tokenizer = Load(managed, lang);
        var golden = SafeTensorFile.Load(Path.Combine(Repo.GoldenFor(lang), "intermediates.safetensors"));
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "intermediates.json")))!;

        foreach (var entry in index["sentences"]!.AsArray())
        {
            int i = entry!["sentence"]!.GetValue<int>();
            var expected = golden.Read<float>($"s{i}.tokenize.pred");
            var actual = tokenizer.ParagraphLogits(entry["text"]!.GetValue<string>(), out var shape);
            Assert.Equal(golden[$"s{i}.tokenize.pred"].Shape, shape);
            output.WriteLine($"sentence {i}: max |diff| {AssertClose(expected, actual, 1e-4f, $"sentence {i}"):E2}");
        }
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Process_MatchesGoldenTokensAndSentences(bool managed, string lang)
    {
        using var tokenizer = Load(managed, lang);
        var text = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "corpus.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "pipeline.conllu")));

        var doc = tokenizer.Process(text);

        Assert.Equal(Describe(golden), Describe(doc));
        foreach (var (g, a) in golden.Sentences.SelectMany(s => s.Tokens).Zip(doc.Sentences.SelectMany(s => s.Tokens)))
        {
            if (g.IsMultiWord)
                Assert.True(a.IsMwtCandidate, $"'{a.Text}' at {a.StartChar} should be marked as a multi-word token");
        }
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Process_MatchesGoldenOnLongParagraphsAndManyBatches(bool managed, string lang)
    {
        using var tokenizer = Load(managed, lang);
        var text = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "tokenize_stress.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "tokenize_stress.conllu")));
        Assert.Equal(Describe(golden), Describe(tokenizer.Process(text)));
    }

    [ModelTheory]
    [InlineData(false, "en")]
    [InlineData(true, "en")]
    [German(false)]
    [German(true)]
    public void Process_ReplacesTokensLongerThanMaxSeqlen(bool managed, string lang)
    {
        using var tokenizer = Load(managed, lang);
        var text = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "long_token.txt"));
        var golden = Conllu.Read(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "long_token.conllu")));
        Assert.Equal(Describe(golden), Describe(tokenizer.Process(text)));
    }

    /// <summary>All processors see <c>&lt;UNK&gt;</c> (tests/golden/long_token*.conllu), in bulk too; pretokenized tokens stay, as in Stanza.</summary>
    [ModelTheory]
    [InlineData("default", false, "en")]
    [InlineData("default_fast", false, "en")]
    [InlineData("default", true, "en")]
    [InlineData("default_fast", true, "en")]
    [German("default", false)]
    [German("default_fast", false)]
    [German("default", true)]
    [German("default_fast", true)]
    public void Pipeline_LongTokens_MatchGolden(string package, bool managed, string lang)
    {
        var nlp = pipelines.Get(package, managed, lang: lang);
        var text = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "long_token.txt"));
        var golden = File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), package == "default" ? "long_token.conllu" : "long_token.fast.conllu"));
        Assert.Equal(golden, Conllu.Write(nlp.Process(text)));
        Assert.Equal(golden, Conllu.Write(nlp.Process([text])[0]));
        var url = new string('x', 300);
        Assert.Equal(url, nlp.Process([["see", url]]).Sentences[0].Tokens[1].Text);
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Process_HandlesEmptyAndWhitespaceText(bool managed)
    {
        using var tokenizer = Load(managed);
        Assert.Empty(tokenizer.Process("").Sentences);
        Assert.Empty(tokenizer.Process(" \n\n \t").Sentences);
    }

    /// <summary>
    /// The e-mail/URL regex's matches, one per alternative of its pattern, each containing only the one literal of
    /// <see cref="Tokenizer.MaskLiterals"/> it needs: the list misses none of these, and each literal is needed.
    /// </summary>
    [Theory]
    [InlineData("write to jane.doe+news@example.io today", "@")]
    [InlineData("\"john.smith\"@[192.168.1.1]", "@")]
    [InlineData("x@[1.2.3.ab:Z9]", "@")]
    [InlineData("see https://example.io/page?q=1", "://")]
    [InlineData("see http://ab.io", "://")]
    [InlineData("go to www.example.io now", "www.")]
    [InlineData("go to www.ab.io", "www.")]
    [InlineData("site.gov.uk", ".gov")]
    [InlineData("site.org.uk", ".org")]
    [InlineData("site.edu.au", ".edu")]
    [InlineData("site.net.au", ".net")]
    [InlineData("bbc.co.uk and site.com.au", ".co")]
    public void MaskLiterals_EveryAlternativeNeedsOne(string text, string literal)
    {
        Assert.Matches(Tokenizer.MaskRe(), text);
        Assert.Contains(literal, text);
        string[] all = ["@", "://", "www.", ".gov", ".org", ".edu", ".net", ".co"];
        Assert.All(all.Where(l => l != literal && !literal.Contains(l)), l => Assert.DoesNotContain(l, text));
        Assert.True(text.AsSpan().ContainsAny(Tokenizer.MaskLiterals));
    }

    /// <summary>Where <see cref="Tokenizer.MaskRe"/> matches, a literal is present: the golden texts and a fuzz.</summary>
    [Fact]
    public void MaskLiterals_PresentWhereverTheRegexMatches()
    {
        int matched = 0;
        void Check(string s)
        {
            if (!Tokenizer.MaskRe().IsMatch(s))
                return;
            matched++;
            Assert.True(s.AsSpan().ContainsAny(Tokenizer.MaskLiterals), s);
        }

        foreach (var file in Directory.GetFiles(Repo.Golden, "*.txt"))
        {
            foreach (var paragraph in File.ReadAllText(file).Split("\n\n"))
                Check(paragraph);
        }

        int golden = matched;
        string[] pieces =
        [
            "http", "https", "://", ":/", "//", "www", "www.", "WWW.", ".", "..", "@", "@@", "com", "co", "gov", "org", "edu", "net",
            "Com", "a", "ab", "Z", "9", "255", "1", "-", "_", "+", "\"", "\\", " ", "\n", "[", "]", ":", "é", "日", "x.y",
        ];
        var random = new Random(16);
        for (int i = 0; i < 200_000; i++)
        {
            var s = new System.Text.StringBuilder();
            for (int n = random.Next(1, 12); n > 0; n--)
                s.Append(pieces[random.Next(pieces.Length)]);
            Check(s.ToString());
        }

        output.WriteLine($"matches: {golden} golden paragraphs, {matched - golden} fuzz strings");
        Assert.True(golden > 0 && matched - golden > 1000);
    }

    /// <summary>One line per sentence with its text, then each token as text[start,end) plus its trailing space.</summary>
    private static List<string> Describe(Document doc) =>
        doc.Sentences.Select(s => $"{s.SentId}: {s.Text} || " + string.Join(" ",
            s.Tokens.Select(t => $"{t.Text}[{t.StartChar},{t.EndChar}){Escape(t.SpaceAfter)}"))).ToList();

    private static string Escape(string space) => space == " " ? "" : "<" + space.Replace("\n", "\\n") + ">";

    /// <summary>
    /// The difference allowed from a reference value <paramref name="expected"/>: <paramref name="tolerance"/>, except that
    /// 1e-4 grows with large values, 1e-4 × max(1, |expected| / 100) (owner's decision, 2026-10-14). Float32 can't hold
    /// 1e-4 above |x| ≈ 300 (one ulp is 3.05e-5 from 256 up, so 1e-4 is about three roundings there), and German UPOS
    /// logits reach |380| (round 26). Nothing changes up to |x| = 100; other tolerances (1e-3) stay absolute.
    /// </summary>
    internal static double Allowed(float tolerance, double expected) =>
        tolerance == 1e-4f ? 1e-4 * Math.Max(1, Math.Abs(expected) / 100) : tolerance;

    /// <summary>Each value within <see cref="Allowed"/> of the expected one.</summary>
    /// <returns>The largest difference.</returns>
    internal static float AssertClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int worst = 0, over = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = Math.Abs(expected[i] - actual[i]);
            if (d > Math.Abs(expected[worst] - actual[worst]))
                worst = i;
            if (over < 0 && !(d <= Allowed(tolerance, expected[i])))
                over = i;
        }

        if (over >= 0)
            Assert.Fail($"{what}: |diff| {Math.Abs(expected[over] - actual[over])} at {over}, over {Allowed(tolerance, expected[over]):E2} (expected {expected[over]}, got {actual[over]})");
        return Math.Abs(expected[worst] - actual[worst]);
    }
}
