using System.Text.Json.Nodes;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="PipelineOptions.TokenizePostprocessor"/> against Stanza's <c>tokenize_postprocessor</c> with the same function
/// (tests/golden/postprocess, <c>make_golden.py --postprocess-only</c>): each text alone, all in one bulk call, and with
/// <c>tokenize_no_ssplit</c>; byte for byte on both backends.
/// </summary>
public class PostprocessorTests
{
    private static readonly JsonNode Golden = JsonNode.Parse(File.ReadAllText(Path.Combine(Repo.Golden, "postprocess", "postprocess.json")))!;

    /// <summary>
    /// make_golden.py's postprocess: merges "New" "York", splits "cannot", expands "gonna" by hand, flags "wanna", keeps the
    /// tokenizer's flags, and merges the last two sentences.
    /// </summary>
    internal static IEnumerable<IEnumerable<PostprocessorToken>> Postprocess(IReadOnlyList<IReadOnlyList<PostprocessorToken>> sentences)
    {
        var result = new List<List<PostprocessorToken>>();
        foreach (var sentence in sentences)
        {
            var tokens = new List<PostprocessorToken>();
            for (int i = 0; i < sentence.Count; i++)
            {
                var token = sentence[i];
                if (token.Text == "New" && i + 1 < sentence.Count && sentence[i + 1] == new PostprocessorToken("York"))
                {
                    tokens.Add("New York");
                    i++;
                }
                else if (token.Text == "cannot")
                {
                    tokens.AddRange(["can", "not"]);
                }
                else if (token.Text == "gonna")
                {
                    tokens.Add(new("gonna", Expansion: ["gon", "na"]));
                }
                else if (token.Text == "wanna")
                {
                    tokens.Add(new("wanna", IsMultiWord: true));
                }
                else
                {
                    tokens.Add(token);
                }
            }

            result.Add(tokens);
        }

        if (result.Count >= 2)
        {
            result[^2].AddRange(result[^1]);
            result.RemoveAt(result.Count - 1);
        }

        return result;
    }

    private static string Expected(string conllu) => conllu.Length > 0 ? conllu + "\n" : conllu; // Stanza's "{:C}"

    private static Pipeline Load(bool managed, bool splitSentences = true) => Pipeline.Load(Repo.Models, new PipelineOptions
    {
        Backend = Repo.PipelineBackendFor(managed),
        TokenizePostprocessor = Postprocess,
        SplitSentences = splitSentences,
    });

    [ModelTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void Process_MatchesStanzasPostprocessor(bool managed)
    {
        var texts = Golden["texts"]!.AsArray().Select(t => t!.GetValue<string>()).ToList();
        using (var nlp = Load(managed))
        {
            var alone = Golden["alone"]!.AsArray();
            for (int i = 0; i < texts.Count; i++)
                Assert.Equal(Expected(alone[i]!.GetValue<string>()), Conllu.Write(nlp.Process(texts[i])));
            var bulk = Golden["bulk"]!.AsArray();
            var docs = nlp.Process(texts);
            for (int i = 0; i < texts.Count; i++)
                Assert.Equal(Expected(bulk[i]!.GetValue<string>()), Conllu.Write(docs[i]));

            // Pretokenized input is not postprocessed, as in Stanza: "New" and "York" stay two tokens.
            var pretokenized = nlp.Process([["New", "York", "cannot", "wait"]]);
            Assert.Equal(["New", "York", "cannot", "wait"], pretokenized.Sentences[0].Tokens.Select(t => t.Text));
        }

        using var noSsplit = Load(managed, splitSentences: false);
        Assert.Equal(Expected(Golden["no_ssplit"]!.GetValue<string>()), Conllu.Write(noSsplit.Process(texts[0])));
    }

    [ModelFact]
    public void ATokenNotInTheText_Throws()
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions
        {
            Processors = "tokenize",
            TokenizePostprocessor = sentences => sentences.Select(s => s.Select(t => (PostprocessorToken)(t.Text + "x"))),
        });
        var e = Assert.Throws<InvalidOperationException>(() => nlp.Process("Hello world."));
        Assert.Contains("|Hellox|", e.Message);
    }
}
