namespace StanzaSharp.Tests;

public class CoreTests
{
    private static string GoldenConllu => File.ReadAllText(Path.Combine(Repo.Golden, "pipeline.conllu"));

    [Fact]
    public void Conllu_RoundTripsGoldenPipelineOutput()
    {
        var text = GoldenConllu;
        Assert.Equal(text, Conllu.Write(Conllu.Read(text)));
    }

    [Fact]
    public void Conllu_ReadsTokensWordsAndSpacing()
    {
        var doc = Conllu.Read(GoldenConllu);
        Assert.Equal(10, doc.Sentences.Count);

        // "I don't think it's ..." : MWT tokens expand to two words, offsets on both levels.
        var s = doc.Sentences[2];
        var dont = s.Tokens[1];
        Assert.True(dont.IsMultiWord);
        Assert.Equal(("don't", 70, 75), (dont.Text, dont.StartChar, dont.EndChar));
        Assert.Equal(["do", "n't"], dont.Words.Select(w => w.Text));
        Assert.Equal((2, "AUX", "VBP", 70, 72), (dont.Words[0].Id, dont.Words[0].Upos, dont.Words[0].Xpos, dont.Words[0].StartChar, dont.Words[0].EndChar));
        Assert.Equal(" ", dont.SpaceAfter);
        Assert.Equal("", s.Tokens.Single(t => t.Text == "today").SpaceAfter);

        // Paragraph break after "He was elected president in 2008."
        Assert.Equal("\n\n", doc.Sentences[1].Tokens[^1].SpaceAfter);

        // Parse tree leaves line up with the words.
        Assert.Equal(s.Words.Select(w => w.Text), s.Constituency!.Leaves().Select(l => l.Label));
    }

    [Fact]
    public void Tree_ParsesAndFormats()
    {
        const string text = "(ROOT (S (NP (PRP He)) (VP (VBD left)) (. .)))";
        var tree = Tree.Parse(text);
        Assert.Equal(text, tree.ToString());
        Assert.Equal("ROOT", tree.Label);
        Assert.True(tree.Children[0].Children[0].Children[0].IsPreterminal); // (PRP He)
        Assert.False(tree.Children[0].Children[0].IsPreterminal); // (NP ...)
        Assert.Equal(["He", "left", "."], tree.Leaves().Select(l => l.Label));
        Assert.Equal("(NP (-LRB- -LRB-) (NN x) (-RRB- -RRB-))",
            new Tree("NP", [new Tree("-LRB-", [new Tree("(")]), new Tree("NN", [new Tree("x")]), new Tree("-RRB-", [new Tree(")")])]).ToString());
        Assert.Throws<FormatException>(() => Tree.Parse("(ROOT (S"));
        Assert.Throws<FormatException>(() => Tree.Parse("(ROOT) x"));
    }

    [Fact]
    public void SafeTensors_ReadsGoldenIntermediates()
    {
        var file = SafeTensorFile.Load(Path.Combine(Repo.Golden, "intermediates.safetensors"));
        Assert.Equal("1.15.0", file.Metadata["stanza"]);
        Assert.Equal([7L, 1024L], file["s0.charlm_forward"].Shape);

        var data = file.Read<float>("s0.charlm_forward");
        Assert.Equal(7 * 1024, data.Length);
        Assert.Equal(-0.003412561723962426f, data[0]);
        Assert.Equal(-0.010616631247103214f, data[^1]);
        Assert.Throws<InvalidOperationException>(() => file.Read<long>("s0.charlm_forward"));
    }

    [ModelFact]
    public void Checkpoint_LoadsConvertedPosModel()
    {
        var ckpt = Checkpoint.Load(Repo.Model("pos/combined_charlm"));
        var node = ckpt.Root["model"]!["upos_clf.weight"];
        Assert.Equal([21L, 400L], ckpt.Shape(node));

        // Values checked against torch.load of the original .pt file.
        var w = ckpt.Tensor<float>(node);
        Assert.Equal(-0.603345513343811f, w[0]);
        Assert.Equal(0.04959297925233841f, w[20 * 400 + 399]);
        Assert.Equal(-1123.825, w.Sum(x => (double)x), tolerance: 0.01);

        Assert.Equal(200, ckpt.Root["config"]!["hidden_dim"]!.GetValue<int>());
    }

    [ModelFact]
    public void Checkpoint_LoadsConstituencyConfig()
    {
        var ckpt = Checkpoint.Load(Repo.Model("constituency/ptb3-revised_charlm"));
        var p = ckpt.Root["params"]!;
        Assert.Equal("IN_ORDER", p["config"]!["transition_scheme"]!.GetValue<string>());
        Assert.Equal(29, p["transitions"]!.AsArray().Count);
        Assert.Equal([2048L, 2268L], ckpt.Shape(p["model"]!["word_lstm.weight_ih_l0"]));
    }
}
