using System.Runtime.Intrinsics;
using System.Text.Json.Nodes;
using StanzaSharp.Lemma;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using StanzaSharp.Tokenize;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// The per-character charlm input of the <c>_charlm</c> tokenizers and lemmatizers on every <see cref="KernelPath"/> this
/// machine runs natively: <see cref="IPerCharLanguageModel.PerChar"/> on both backends, the tokenizer's network with
/// charlm states, and the lemmatizer's seq2seq model (its encoder reading them) against Stanza's output. In the managed
/// kernels' collection, since they set <see cref="Gemm.Path"/>. The tokenizer and lemmatizer rows need English ewt's
/// <c>_charlm</c> models (models/stanza/en, CI's models-nocharlm cache: golden shards only; elsewhere they skip, and the
/// no-skip checks exempt this class).
/// </summary>
[Collection(ManagedKernelsCollection.Name)]
public class CharlmInputTests(ITestOutputHelper output)
{
    public static TheoryData<string> Paths() =>
        [.. Enum.GetValues<KernelPath>().Where(p => p switch
        {
            KernelPath.Vector512 => Vector512.IsHardwareAccelerated,
            KernelPath.Vector256 => Vector256.IsHardwareAccelerated,
            _ => true,
        }).Select(p => p.ToString())];

    private static T With<T>(string path, Func<T> body)
    {
        var (beforePath, beforeThreads) = (Gemm.Path, ManagedThreads.Count);
        (Gemm.Path, ManagedThreads.Count) = (Enum.Parse<KernelPath>(path), 4);
        try
        {
            return body();
        }
        finally
        {
            (Gemm.Path, ManagedThreads.Count) = (beforePath, beforeThreads);
        }
    }

    /// <summary>
    /// Both charlms (the backward one also left to right) over ragged sequences with <c>&lt;PAD&gt;</c> units, the lemma's
    /// <c>"\n" + word + " "</c> and a tokenizer row: the managed states equal TorchSharp's, and only the asked-for rows are written.
    /// </summary>
    [ModelTheory]
    [MemberData(nameof(Paths))]
    public void PerChar_ManagedMatchesTorchSharp(string path)
    {
        foreach (var direction in new[] { "forward", "backward" })
        {
            using var reference = CharLanguageModel.Load(Repo.Model($"{direction}_charlm/1billion"));
            var managed = ManagedCharLanguageModel.Load(Repo.Model($"{direction}_charlm/1billion"));
            int[] Ids(IPerCharLanguageModel lm, IEnumerable<string> units) => [.. units.Select(u => lm.Id(u))];
            string[][] texts =
            [
                ["\n", .. "Hello".Select(c => c.ToString()), " "],
                [.. "Dr. Smith's 3rd x@y.com".Select(c => c.ToString()), "<PAD>"],
                ["<PAD>", "<PAD>", "<PAD>"],
                ["\n", "𝔘", "ä", " "],
                ["a"],
            ];
            int[] firstRow = [0, 9, 40, 50, 60];
            int h = managed.HiddenDim, rows = 64;
            float[] Run(IPerCharLanguageModel lm)
            {
                var o = new float[rows * h];
                Array.Fill(o, float.NaN);
                lm.PerChar([.. texts.Select(t => Ids(lm, t))], firstRow, o, h, default);
                return o;
            }

            var expected = Run(reference);
            var actual = With(path, () => Run(managed));
            for (int r = 0; r < rows; r++)
            {
                bool used = Enumerable.Range(0, texts.Length).Any(i => r >= firstRow[i] && r < firstRow[i] + texts[i].Length);
                Assert.Equal(used, !float.IsNaN(actual[r * h]));
                Assert.Equal(used, !float.IsNaN(expected[r * h]));
            }

            var keep = expected.Select((_, i) => !float.IsNaN(expected[i])).ToArray();
            output.WriteLine($"{path} {direction}: max |diff| "
                + $"{TokenizerTests.AssertClose([.. expected.Where((_, i) => keep[i])], [.. actual.Where((_, i) => keep[i])], 1e-4f, $"{path} {direction}"):E2}");
        }
    }

    /// <summary>The <c>_charlm</c> tokenizer's network, both backends on the same batch and charlm states (one row at full width).</summary>
    [CharlmModelsTheory("tokenize/ewt_charlm")]
    [MemberData(nameof(Paths))]
    public void TokenizerNet_WithCharlm_ManagedMatchesTorchSharp(string path)
    {
        var ckpt = Checkpoint.Load(Path.Combine(Repo.StanzaModels, "tokenize", "ewt_charlm"));
        using var reference = new TokenizerNet(ckpt);
        var managed = new ManagedTokenizerNet(ckpt);
        long[] lengths = [40, 12, 1, 25];
        int width = 40, vocab = (int)ckpt.Shape(ckpt.Root["model"]!["embeddings.weight"])[0], featDim = ckpt.Root["config"]!["feat_dim"]!.GetValue<int>();
        var rng = new Random(5);
        var ids = Enumerable.Range(0, lengths.Length * width).Select(_ => (long)rng.Next(vocab)).ToArray();
        var feats = Enumerable.Range(0, lengths.Length * width * featDim).Select(_ => rng.Next(4) == 0 ? 1f : 0f).ToArray();
        var charlm = Enumerable.Range(0, lengths.Length * width * 1024).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        var expected = reference.Forward(ids, feats, charlm, lengths.Length, width, lengths, default);
        var actual = With(path, () => managed.Forward(ids, feats, charlm, lengths.Length, width, lengths, default));
        output.WriteLine($"{path}: max |diff| {TokenizerTests.AssertClose(expected, actual, 1e-4f, path):E2}");
    }

    /// <summary>The <c>_charlm</c> lemmatizer's seq2seq model on the managed backend, on every path: Stanza's decoded strings and edits (seq2seq.json).</summary>
    [CharlmModelsTheory("lemma/ewt_charlm")]
    [MemberData(nameof(Paths))]
    public void Lemmatizer_WithCharlm_MatchesGoldenOnEveryPath(string path)
    {
        var golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "models", "tokenize+lemma=ewt_charlm", "seq2seq.json")))!.AsArray();
        var words = golden.Select(g => new Word { Text = (string)g!["word"]!, Upos = (string?)g["upos"] }).ToList();
        using var lemmatizer = Lemmatizer.LoadManaged(Path.Combine(Repo.StanzaModels, "lemma", "ewt_charlm"),
            ManagedCharLanguageModel.Load(Repo.Model("forward_charlm/1billion")), ManagedCharLanguageModel.Load(Repo.Model("backward_charlm/1billion")));
        var actual = With(path, () => lemmatizer.Predict(words));
        Assert.Equal(golden.Select(g => (string)g!["seq2seq"]!), actual.Decoded);
        Assert.Equal(golden.Select(g => (int)g!["edit"]!), actual.Edits);
    }
}

/// <summary>A theory skipped where the converted models or the named Stanza model (models/stanza/en, e.g. <c>lemma/ewt_charlm</c>) are missing.</summary>
public sealed class CharlmModelsTheoryAttribute : TheoryAttribute
{
    public CharlmModelsTheoryAttribute(string model)
    {
        if (!Directory.Exists(Repo.Models))
            Skip = "models/converted/en not found; run setup.ps1 -Models";
        else if (!File.Exists(Path.Combine(Repo.StanzaModels, model + ".pt")))
            Skip = $"{model}.pt not found in {Repo.StanzaModels}; run stanzasharp download models/stanza/en --package ewt --model tokenize=ewt_charlm,lemma=ewt_charlm";
    }
}
