namespace StanzaSharp.Tests;

/// <summary>
/// German's treebank packages (tokenize to depparse). <c>combined</c>'s models are the default package's, so its golden
/// data is depparse/ (make_golden.py --lang de).
/// </summary>
public class TreebankPackageTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    [ModelTheory]
    [German("combined", false)]
    [German("combined", true)]
    public void Process_MatchesEveryGoldenFile(string package, bool managed, string lang)
    {
        var nlp = pipelines.Get(package, managed, lang: lang);
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
}
