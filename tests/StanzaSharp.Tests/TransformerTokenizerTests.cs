using System.Text.Json;
using StanzaSharp.Nn;
using Xunit.Abstractions;

namespace StanzaSharp.Tests;

/// <summary>
/// <see cref="WordPiece"/> and <see cref="TransformerInput"/> against tests/golden/transformer/tokenizer.json:
/// transformers' ids and word ids, and Stanza's fix_blank_tokens and convert_to_position_list, for every golden sentence's
/// words, simplify_punct'ed words and tokens and for make_golden.py's TRANSFORMER_TOKENIZER_CASES.
/// </summary>
[Trait("Backend", "Managed")]
public class TransformerTokenizerTests(ITestOutputHelper output)
{
    private static string TokenizerJson => Path.Combine(TransformerFactAttribute.Directory, "tokenizer.json");

    [TransformerFact("tokenizer.json")]
    public void MatchesTransformersAndStanzaOnEveryGoldenCase() =>
        AssertGolden(WordPiece.LoadDirectory(TransformerFactAttribute.Directory), Path.Combine(Repo.Golden, "transformer", "tokenizer.json"), minCases: 1000);

    /// <summary>German's transformer: a vocab.txt tokenizer that lowercases but keeps accents (docs/german-accurate-plan.md).</summary>
    [GermanTransformerFact("vocab.txt", "tokenizer_config.json")]
    public void German_MatchesTransformersAndStanzaOnEveryGoldenCase() =>
        AssertGolden(WordPiece.LoadDirectory(GermanTransformerFactAttribute.Directory), Path.Combine(Repo.GoldenFor("de"), "transformer", "tokenizer.json"),
            minCases: 200);

    /// <summary>dbmdz/bert-base-german-cased: a cased vocab.txt tokenizer (accents kept), on make_golden's DBMDZ_TOKENIZER_CASES too.</summary>
    [DbmdzFact("vocab.txt", "tokenizer_config.json")]
    public void Dbmdz_MatchesTransformersAndStanzaOnEveryGoldenCase() =>
        AssertGolden(WordPiece.LoadDirectory(DbmdzFactAttribute.Directory),
            Path.Combine(Repo.GoldenFor("de"), "transformer", "tokenizer_dbmdz_bert-base-german-cased.json"), minCases: 200);

    [DbmdzFact("config.json", "tokenizer_config.json")]
    public void Dbmdz_ConfigReadsBert()
    {
        // BertForMaskedLM: the weights under bert., positions from 0, max_len instead of model_max_length.
        var config = TransformerConfig.Load(Path.Combine(DbmdzFactAttribute.Directory, "config.json"));
        Assert.Equal((768, 12, 12, 3072, 512, 0, "bert."), (config.Hidden, config.Heads, config.Layers, config.Intermediate, config.MaxPositions,
            config.PositionOffset, config.Prefix));
        Assert.Equal(512, TransformerEmbedder.ReadMaxLength(DbmdzFactAttribute.Directory));
    }

    /// <summary>roberta-base's byte-level BPE (#172), as Stanza loads it (add_prefix_space=True), on make_golden's ROBERTA_TOKENIZER_CASES too.</summary>
    [RobertaFact("tokenizer.json")]
    public void Roberta_MatchesTransformersAndStanzaOnEveryGoldenCase() =>
        AssertGolden(ByteLevelBpe.Load(Path.Combine(RobertaFactAttribute.Directory, "tokenizer.json")),
            Path.Combine(Repo.Golden, "transformer", "tokenizer_roberta-base.json"), minCases: 1000);

    [RobertaFact("config.json", "tokenizer_config.json")]
    public void Roberta_ConfigReadsPositionsFromTwo()
    {
        // max_position_embeddings 514 less RoBERTa's offset (pad_token_id + 1): 512 pieces, the weights under roberta.
        var config = TransformerConfig.Load(Path.Combine(RobertaFactAttribute.Directory, "config.json"));
        Assert.Equal((768, 12, 12, 3072, 512, 2, "roberta."), (config.Hidden, config.Heads, config.Layers, config.Intermediate, config.MaxPositions,
            config.PositionOffset, config.Prefix));
        Assert.Equal(512, TransformerEmbedder.ReadMaxLength(RobertaFactAttribute.Directory));
    }

    [GermanTransformerFact("config.json", "tokenizer_config.json")]
    public void German_ConfigAndMaxLengthFromTheOlderKeys()
    {
        // No position_embedding_type (transformers' default: absolute) and max_len instead of model_max_length.
        var config = TransformerConfig.Load(Path.Combine(GermanTransformerFactAttribute.Directory, "config.json"));
        Assert.Equal((768, 12, 12, 3072), (config.Hidden, config.Heads, config.Layers, config.Intermediate));
        Assert.Equal(512, TransformerEmbedder.ReadMaxLength(GermanTransformerFactAttribute.Directory));
    }

    private void AssertGolden(ITransformerTokenizer tokenizer, string goldenPath, int minCases)
    {
        using var golden = JsonDocument.Parse(File.ReadAllBytes(goldenPath));
        int cases = 0, failures = 0;
        foreach (var c in golden.RootElement.GetProperty("cases").EnumerateArray())
        {
            cases++;
            var words = c.GetProperty("words").EnumerateArray().Select(w => w.GetString()!).ToArray();
            var ids = c.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray();
            var wordIds = c.GetProperty("word_ids").EnumerateArray().Select(i => i.GetInt32()).ToArray();
            int[] positions = c.TryGetProperty("positions", out var p) ? [.. p.EnumerateArray().Select(i => i.GetInt32())] : [.. Enumerable.Range(0, words.Length + 2)];
            string[] fixedWords = c.TryGetProperty("fixed", out var f) ? [.. f.EnumerateArray().Select(w => w.GetString()!)] : words;

            var input = TransformerInput.Build(tokenizer, words);
            var (encodedIds, owners) = tokenizer.Encode(fixedWords);
            if ((!input.Ids.SequenceEqual(ids) || !owners.SequenceEqual(wordIds) || !input.Positions.SequenceEqual(positions) || !input.Words.SequenceEqual(fixedWords)
                || !encodedIds.SequenceEqual(ids))
                && ++failures <= 10)
            {
                output.WriteLine($"[{string.Join(", ", words.Select(Escape))}]\n  ids      {string.Join(',', ids)}\n  C#       {string.Join(',', input.Ids)}"
                    + $"\n  word ids {string.Join(',', wordIds)}\n  C#       {string.Join(',', owners)}"
                    + $"\n  positions {string.Join(',', positions)}\n  C#        {string.Join(',', input.Positions)}");
            }
        }

        output.WriteLine($"{cases} cases, {failures} differ");
        Assert.True(cases > minCases);
        Assert.Contains(golden.RootElement.GetProperty("cases").EnumerateArray(), c => c.GetProperty("ids").GetArrayLength() > 512);
        Assert.Equal(0, failures);
    }

    [TransformerFact("tokenizer.json")]
    public void AddedTokensAreMatchedInTheRawWord()
    {
        var tokenizer = WordPiece.Load(TokenizerJson);
        Assert.Equal([tokenizer.Cls, tokenizer.Cls, tokenizer.Pad, tokenizer.Sep], tokenizer.Encode(["[CLS][PAD]"]).Ids);
        Assert.Equal([tokenizer.Cls, tokenizer.Unk, tokenizer.Sep], tokenizer.Encode(["[UNK]"]).Ids);
        Assert.Equal([-1, 0, 0, 0, -1], tokenizer.Encode(["a[SEP]b"]).Words);
    }

    [TransformerFact("tokenizer.json")]
    public void AnEmptySentenceIsClsAndSep()
    {
        // Python: tokenizer([[]], is_split_into_words=True) is [101, 102]; convert_to_position_list gives [0, 1].
        var input = TransformerInput.Build(WordPiece.Load(TokenizerJson), []);
        Assert.Equal([101, 102], input.Ids);
        Assert.Equal([0, 1], input.Positions);
    }

    private static string Escape(string s) => string.Concat(s.Select(c => c < ' ' || c > '~' ? $"\\u{(int)c:x4}" : c.ToString()));
}
