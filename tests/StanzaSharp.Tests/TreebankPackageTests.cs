using System.Text.Json.Nodes;
using StanzaSharp.Mwt;

namespace StanzaSharp.Tests;

/// <summary>
/// The treebank packages (tokenize to depparse) against tests/golden/&lt;package&gt; (make_golden.py --treebank-only):
/// English ewt, gum, lines, partut, atis, childes and eslspok (no mwt, the identity lemmatizer), German gsd and hdt; and
/// <c>combined</c>, whose models are the default package's, against depparse/.
/// </summary>
public class TreebankPackageTests
{
    [ModelTheory]
    [EnglishTreebank("combined", false)]
    [EnglishTreebank("combined", true)]
    [EnglishTreebank("ewt", false)]
    [EnglishTreebank("ewt", true)]
    [EnglishTreebank("gum", false)]
    [EnglishTreebank("gum", true)]
    [EnglishTreebank("lines", false)]
    [EnglishTreebank("lines", true)]
    [EnglishTreebank("partut", false)]
    [EnglishTreebank("partut", true)]
    [EnglishTreebank("atis", false)]
    [EnglishTreebank("atis", true)]
    [EnglishTreebank("childes", false)]
    [EnglishTreebank("childes", true)]
    [EnglishTreebank("eslspok", false)]
    [EnglishTreebank("eslspok", true)]
    [German("gsd", false)]
    [German("gsd", true)]
    [German("hdt", false)]
    [German("hdt", true)]
    [German("combined", false)]
    [German("combined", true)]
    public void Process_MatchesEveryGoldenFile(string package, bool managed, string lang)
    {
        // Loaded here, not shared: a pipeline per row would otherwise stay in memory until the class ends.
        using var nlp = Pipeline.Load(Repo.ModelsFor(lang, package), new PipelineOptions { Language = lang, Package = package, Backend = Repo.PipelineBackendFor(managed) });
        var golden = Repo.GoldenFor(lang);
        var dir = Path.Combine(golden, package == "combined" ? "depparse" : package);
        var files = new[] { Path.Combine(golden, "corpus.txt") }.Concat(Directory.GetFiles(golden, "validation*.txt").Order()).ToList();
        Assert.Equal(13, files.Count);
        var failures = new List<string>();
        foreach (var txt in files)
        {
            var name = Path.GetFileName(txt) == "corpus.txt" ? "corpus" : Path.GetFileNameWithoutExtension(txt);
            var text = File.ReadAllText(txt);
            var actual = Conllu.Write(nlp.Process(text));
            var expected = File.ReadAllText(Path.Combine(dir, name + ".conllu"));

            // Outside the BMP, C# offsets are UTF-16 indices and Stanza's are code points.
            if (PipelineTests.HasNonBmp(text))
                (actual, expected) = (PipelineTests.StripOffsets(actual), PipelineTests.StripOffsets(expected));
            var e = expected.Split("\n\n");
            var g = actual.Split("\n\n");
            int i = 0;
            while (i < Math.Min(e.Length, g.Length) && e[i] == g[i]) i++;
            if (i < e.Length || i < g.Length)
                failures.Add($"{name}, sentence {i}:\n--- expected\n{e.ElementAtOrDefault(i)}\n--- actual\n{g.ElementAtOrDefault(i)}");
        }

        Assert.True(failures.Count == 0, string.Join("\n\n", failures));
    }

    /// <summary>
    /// Their MWT models with and without the dictionary: English's character classifiers (partut's a seq2seq), German's
    /// seq2seq models (hdt's has its own vocabulary and max_dec_len 12).
    /// </summary>
    [ModelTheory]
    [EnglishTreebank("ewt", false)]
    [EnglishTreebank("ewt", true)]
    [EnglishTreebank("gum", false)]
    [EnglishTreebank("gum", true)]
    [EnglishTreebank("lines", false)]
    [EnglishTreebank("lines", true)]
    [EnglishTreebank("partut", false)]
    [EnglishTreebank("partut", true)]
    [EnglishTreebank("atis", false)]
    [EnglishTreebank("atis", true)]
    [EnglishTreebank("childes", false)]
    [EnglishTreebank("childes", true)]
    [German("gsd", false)]
    [German("gsd", true)]
    [German("hdt", false)]
    [German("hdt", true)]
    public void Mwt_MatchesGolden(string package, bool managed, string lang)
    {
        using var mwt = MwtExpander.Load(Repo.ModelOf(lang, "mwt", package), backend: Repo.Backend(managed));
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), package, "mwt.json")))!.AsArray();
        var words = golden.Select(g => g!["word"]!.GetValue<string>()).ToList();
        var expanded = mwt.Expand(words);
        var predicted = mwt.Predict(words);
        for (int i = 0; i < words.Count; i++)
        {
            Assert.Equal(Words(golden[i]!["pipeline"]), Split(expanded[i], words[i]));
            Assert.Equal(Words(golden[i]!["model"]), Split(predicted[i], words[i]));
        }
    }

    private static List<string> Words(JsonNode? node) => node!.AsArray().Select(w => w!.GetValue<string>()).ToList();

    /// <summary>How set_mwt_expansions turns an expansion into words.</summary>
    private static List<string> Split(string expansion, string token)
    {
        var words = expansion.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
        return words.Count <= 1 ? [token] : words;
    }
}
