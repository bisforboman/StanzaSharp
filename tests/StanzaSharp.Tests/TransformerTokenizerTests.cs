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
    private static string TokenizerJson => Path.Combine(ModelDownloader.TransformerDirectory(Repo.StanzaModels, TransformerFactAttribute.Electra), "tokenizer.json");

    [TransformerFact("tokenizer.json")]
    public void MatchesTransformersAndStanzaOnEveryGoldenCase()
    {
        var tokenizer = WordPiece.Load(TokenizerJson);
        using var golden = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Repo.Golden, "transformer", "tokenizer.json")));
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
        Assert.True(cases > 1000);
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

    private static string Escape(string s) => string.Concat(s.Select(c => c < ' ' || c > '~' ? $"\\u{(int)c:x4}" : c.ToString()));
}
