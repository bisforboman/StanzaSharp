using System.Collections.Concurrent;
using TasmanianDevil.Analyzer.Context;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// A TasmanianDevil <see cref="ITokenNormalizer"/> from StanzaSharp's tokenizer, tagger and lemmatizer, so
/// <see cref="LemmaContextAwareEnhancer"/> matches context words by their real lemma ("children" and "child", "was" and
/// "be", Swedish "barnen" and "barn") in any language Stanza has a lemmatizer for, instead of English Porter stems.
/// </summary>
/// <remarks>
/// <para><see cref="Tokenize"/> gives Stanza's tokens with their offsets. <see cref="Normalize"/> gives a word's lowercased
/// lemma (a multi-word token's lemmas joined by spaces, "don't" → "do not"), always from the word tagged alone, so a word
/// normalizes the same wherever it appears and whatever was analyzed before. <see cref="Tokenize"/> lemmatizes the
/// text's new words in one bulk call, and the results are cached (up to 100,000 words).</para>
/// <para>Stop words are words tagged as function words (<c>DET</c>, <c>ADP</c>, <c>PRON</c>, <c>AUX</c>, <c>CCONJ</c>,
/// <c>SCONJ</c>, <c>PART</c>), so no word list is needed per language; punctuation is <c>PUNCT</c>, <c>SYM</c> or a
/// token without a letter or digit.</para>
/// <para>The pipeline needs <c>tokenize,mwt,pos,lemma</c> (mwt where the language has it); more processors only cost time
/// (a bulk call gives every word the lemma and tag it gets alone; only the parser and sentiment, which this doesn't read,
/// can differ in bulk). The normalizer doesn't dispose it; it is thread-safe, like <see cref="Pipeline"/>'s
/// Process.</para>
/// </remarks>
public sealed class StanzaTokenNormalizer : ITokenNormalizer
{
    private const int MaxCached = 100_000;
    private static readonly HashSet<string> FunctionTags = ["DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ", "PART"];
    private static readonly HashSet<string> PunctuationTags = ["PUNCT", "SYM"];

    private readonly Pipeline _pipeline;
    private readonly ConcurrentDictionary<string, Word> _words = new(StringComparer.Ordinal);

    /// <summary>Creates a normalizer that lemmatizes with <paramref name="pipeline"/>.</summary>
    /// <param name="pipeline">A loaded pipeline with the <c>lemma</c> processor (and its requirements).</param>
    public StanzaTokenNormalizer(Pipeline pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        if (!pipeline.HasLemma)
            throw new ArgumentException("The pipeline has no lemma processor; load it with Processors = \"tokenize,mwt,pos,lemma\".", nameof(pipeline));
        _pipeline = pipeline;
    }

    /// <inheritdoc />
    public IReadOnlyList<(string Token, int Start, int End)> Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (string.IsNullOrWhiteSpace(text))
            return [];
        var tokens = new List<(string Token, int Start, int End)>();
        foreach (var token in _pipeline.Process(text).Sentences.SelectMany(s => s.Tokens))
        {
            if (token.StartChar is int start && token.EndChar is int end)
                tokens.Add((text[start..end], start, end));
        }

        // The text's new words, each tagged alone, in one call.
        var unknown = tokens.Select(t => t.Token).Where(t => !_words.ContainsKey(t)).Distinct(StringComparer.Ordinal).ToList();
        if (unknown.Count > 0)
        {
            var documents = _pipeline.Process(unknown);
            for (var i = 0; i < unknown.Count; i++)
                Remember(unknown[i], Describe(unknown[i], documents[i]));
        }

        return tokens;
    }

    /// <inheritdoc />
    public string Normalize(string token) => Lookup(token).Lemma;

    /// <inheritdoc />
    public bool IsStopWord(string token) => Lookup(token).IsStopWord;

    /// <inheritdoc />
    public bool IsPunctuation(string token) => Lookup(token).IsPunctuation;

    private Word Lookup(string token)
    {
        ArgumentNullException.ThrowIfNull(token);
        if (_words.TryGetValue(token, out var word))
            return word;
        word = string.IsNullOrWhiteSpace(token) ? new Word(token.Trim().ToLowerInvariant(), false, true) : Describe(token, _pipeline.Process(token));
        Remember(token, word);
        return word;
    }

    private void Remember(string token, Word word)
    {
        if (_words.Count < MaxCached)
            _words.TryAdd(token, word);
    }

    private static Word Describe(string token, Document document)
    {
        var words = document.Sentences.SelectMany(s => s.Words).ToList();
        if (words.Count == 0)
            return new Word(token.ToLowerInvariant(), false, !token.Any(char.IsLetterOrDigit));
        var lemma = string.Join(' ', words.Select(w => w.Lemma ?? w.Text)).ToLowerInvariant();
        var punctuation = words.All(w => w.Upos != null && PunctuationTags.Contains(w.Upos)) || !token.Any(char.IsLetterOrDigit);
        var stop = !punctuation && words.All(w => w.Upos != null && FunctionTags.Contains(w.Upos));
        return new Word(lemma, stop, punctuation);
    }

    private sealed record Word(string Lemma, bool IsStopWord, bool IsPunctuation);
}
