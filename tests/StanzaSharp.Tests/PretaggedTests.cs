using System.Text.RegularExpressions;

namespace StanzaSharp.Tests;

/// <summary><see cref="PipelineOptions.Pretagged"/>, Stanza's <c>pretagged=True</c> (tests/golden/pretagged).</summary>
public class PretaggedTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    // make_golden.py's PRETAGGED_CASES: input, processors, package, fields removed from every word.
    public static readonly TheoryData<string, string, string, string, string, bool> Cases = Rows(
        ("depparse", "fast/corpus.conllu", "depparse", "default", ""),
        ("constituency", "fast/corpus.conllu", "constituency", "default", "sentiment"),
        ("lemma", "process_document/input.conllu", "lemma", "default", ""),
        ("all", "process_document/input.conllu", "lemma,depparse,constituency", "default", ""),
        ("all.fast", "process_document/input.conllu", "lemma,depparse", "default_fast", ""),
        ("validation", "fast/validation.conllu", "lemma,depparse,constituency", "default", "sentiment"),
        ("depparse_nolemma", "fast/corpus.conllu", "depparse", "default", "lemma"),
        ("depparse_noxpos", "fast/corpus.conllu", "depparse", "default", "xpos"),
        ("lemma_noupos", "fast/corpus.conllu", "lemma", "default", "upos,lemma"));

    private static TheoryData<string, string, string, string, string, bool> Rows(params (string, string, string, string, string)[] cases)
    {
        var data = new TheoryData<string, string, string, string, string, bool>();
        foreach (var managed in new[] { false, true })
        {
            foreach (var (name, input, processors, package, strip) in cases)
                data.Add(name, input, processors, package, strip, managed);
        }

        return data;
    }

    [ModelTheory]
    [MemberData(nameof(Cases))]
    public void Process_MatchesStanzasPretaggedPipeline(string name, string input, string processors, string package, string strip, bool managed)
    {
        // "sentiment": without the comments, which Stanza would keep before the new # constituency (make_golden.py).
        var fields = strip.Split(',', StringSplitOptions.RemoveEmptyEntries);
        var text = File.ReadAllText(Path.Combine(Repo.Golden, input));
        if (fields.Contains("sentiment"))
            text = Regex.Replace(text, @"^# sentiment = .*\n", "", RegexOptions.Multiline);
        var doc = Conllu.Read(text);
        foreach (var word in doc.Sentences.SelectMany(s => s.Words))
        {
            foreach (var field in fields)
            {
                switch (field)
                {
                    case "sentiment": break;
                    case "upos": word.Upos = null; break;
                    case "xpos": word.Xpos = null; break;
                    case "lemma": word.Lemma = null; break;
                    default: throw new ArgumentException(field);
                }
            }
        }

        var nlp = pipelines.Get(package, managed, processors, pretagged: true);
        var actual = Conllu.Write(nlp.Process(doc));
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, "pretagged", name + ".conllu"));
        Assert.True(golden == actual, $"{name}:\n--- expected\n{golden}\n--- actual\n{actual}");
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DepparseAlone_OnStanzasTags_ReproducesThePipeline(bool managed)
    {
        // pipeline.conllu's tags and lemmas are the full pipeline's, so parsing them again gives its heads byte for byte.
        var golden = File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));
        var doc = Conllu.Read(golden);
        foreach (var word in doc.Sentences.SelectMany(s => s.Words))
            (word.Head, word.Deprel) = (null, null);
        var actual = Conllu.Write(pipelines.Get(managed: managed, processors: "depparse", pretagged: true).Process(doc));
        Assert.True(golden == actual, $"--- expected\n{golden}\n--- actual\n{actual}");
    }

    [Fact]
    public void SelectModels_DropsOnlyLemmaDepparseAndConstituencysRequirements()
    {
        Assert.Throws<ArgumentException>(() => Pipeline.SelectModels("default", "depparse", false, "options"));
        Assert.Equal(["depparse", "mwt", "tokenize"], Pipeline.SelectModels("default", "depparse", false, "options", pretagged: true).Keys.Order());
        Assert.Equal(["constituency", "lemma", "mwt", "tokenize"],
            Pipeline.SelectModels("default", "lemma,constituency", false, "options", pretagged: true).Keys.Order());
        Assert.Equal(["mwt", "ner", "tokenize"], Pipeline.SelectModels("default", "ner", false, "options", pretagged: true).Keys.Order());

        // pos keeps its requirements, which the tokenizer that always loads meets; the package's processors are unchanged.
        Assert.Equal(["depparse", "mwt", "pos", "tokenize"], Pipeline.SelectModels("default", "pos,depparse", false, "options", pretagged: true).Keys.Order());
        Assert.Equal(Pipeline.SelectModels("default", null, false, "options").Keys.Order(), Pipeline.SelectModels("default", null, false, "options", pretagged: true).Keys.Order());
        Assert.Equal(["depparse", "mwt", "tokenize"], Pipeline.SelectModels("default", "depparse", true, "options", pretagged: true).Keys.Order());
        Assert.Throws<ArgumentException>(() => Pipeline.SelectModels("default_fast", "constituency", false, "options", pretagged: true));
    }

    [Fact]
    public void FilesFor_ListsOnlyWhatLoadReads()
    {
        var files = ModelDownloader.FilesFor("depparse", pretagged: true).Select(f => f.Path).Order();
        Assert.Equal(["backward_charlm/1billion.pt", "depparse/combined_charlm.pt", "forward_charlm/1billion.pt", "mwt/combined.pt", "pretrain/conll17.pt",
            "tokenize/combined_nocharlm.pt"], files);
        Assert.Contains("pos/combined_charlm.pt", ModelDownloader.FilesFor("depparse").Select(f => f.Path));
        Assert.Equal(["lemma/combined_nocharlm.pt", "mwt/combined.pt", "tokenize/combined_nocharlm.pt"],
            ModelDownloader.FilesFor("lemma", pretagged: true).Select(f => f.Path).Order());
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void TextOverloads_WithoutTheTagger_FailInTheParsers(bool managed)
    {
        // As Stanza's tokenize,depparse pretagged pipeline: the text is tokenized, then the parser finds no tags.
        var depparse = pipelines.Get(managed: managed, processors: "depparse", pretagged: true);
        Assert.Contains("POS tagger", Assert.Throws<InvalidOperationException>(() => depparse.Process("A short text.")).Message);
        Assert.Throws<InvalidOperationException>(() => depparse.Process([["A", "pretokenized", "text"]]));
        Assert.Throws<InvalidOperationException>(() => depparse.Process(["One text.", "Two texts."]));
        Assert.Empty(depparse.Process("").Sentences);

        var constituency = pipelines.Get(managed: managed, processors: "constituency", pretagged: true);
        Assert.Throws<InvalidOperationException>(() => constituency.Process("A short text."));

        // The lemmatizer reads a missing UPOS as "_", as Stanza's tokenize,lemma pipeline does.
        var lemma = pipelines.Get(managed: managed, processors: "lemma", pretagged: true);
        Assert.All(lemma.Process("The cats were sleeping.").Sentences[0].Words, w => Assert.NotNull(w.Lemma));
    }

    [ModelTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void UntaggedWord_Throws(bool managed)
    {
        var doc = Conllu.Read(File.ReadAllText(Path.Combine(Repo.Golden, "fast", "corpus.conllu")));
        var word = doc.Sentences[1].Words.First();
        (word.Upos, word.Xpos) = (null, null);
        Assert.Throws<InvalidOperationException>(() => pipelines.Get(managed: managed, processors: "depparse", pretagged: true).Process(doc));
        Assert.Throws<InvalidOperationException>(() => pipelines.Get(managed: managed, processors: "constituency", pretagged: true).Process(doc));
    }

    [ModelFact]
    public void Load_WithoutPretagged_StillRequiresTheTagger()
    {
        var e = Assert.Throws<ArgumentException>(() => Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "depparse" }));
        Assert.Contains("requires", e.Message);
    }
}
