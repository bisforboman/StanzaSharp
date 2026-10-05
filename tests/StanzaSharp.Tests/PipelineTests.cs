namespace StanzaSharp.Tests;

public class PipelineTests
{
    [ModelFact]
    public void Process_ReproducesGoldenConlluExactly()
    {
        using var nlp = Pipeline.Load(Repo.Models);
        var doc = nlp.Process(File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt")));
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));
        Assert.Equal(golden, Conllu.Write(doc));
    }

    [ModelFact]
    public void Load_RunsASubsetOfProcessors()
    {
        using var nlp = Pipeline.Load(Repo.Models, "tokenize,mwt");
        var doc = nlp.Process("I don't know.");
        var words = doc.Sentences.Single().Words.ToList();
        Assert.Equal(["I", "do", "n't", "know", "."], words.Select(w => w.Text));
        Assert.All(words, w => Assert.Null(w.Upos));
        Assert.Null(doc.Sentences[0].Constituency);
    }

    [Fact]
    public void Load_RejectsUnknownOrIncompleteProcessorLists()
    {
        var dir = Repo.Root; // any existing directory: validation happens before loading
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, "tokenize,lemma"));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, "tokenize,constituency"));
        Assert.Throws<ArgumentException>(() => Pipeline.Load(dir, ""));
        Assert.Throws<DirectoryNotFoundException>(() => Pipeline.Load(Path.Combine(dir, "no-such-dir")));
    }
}
