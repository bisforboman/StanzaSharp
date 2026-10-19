using System.Text.RegularExpressions;

namespace StanzaSharp;

/// <summary>
/// A processed text, as <c>Pipeline.Process</c> returns it: sentences of tokens, each token expanding to one or
/// more words. Like Stanza's <c>Document</c>.
/// </summary>
public sealed class Document
{
    /// <summary>The original text the document was made from; null for a document read from CoNLL-U.</summary>
    public string? Text { get; set; }

    /// <summary>
    /// The text's language as the language identifier saw it, a UD code such as <c>en</c> or <c>de</c> (Stanza's
    /// <c>doc.lang</c>). Set by <c>MultilingualPipeline.Process</c>; null otherwise. Neither <see cref="Conllu"/> nor
    /// <see cref="Json"/> writes or reads it, as in Stanza.
    /// </summary>
    public string? Lang { get; set; }

    /// <summary>The sentences, in text order.</summary>
    public List<Sentence> Sentences { get; } = [];

    /// <summary>Named entities of every sentence, in order (Stanza's <c>doc.ents</c>).</summary>
    public IEnumerable<Entity> Entities => Sentences.SelectMany(s => s.Entities);

    /// <summary>The number of tokens in all sentences (Stanza's <c>doc.num_tokens</c>).</summary>
    public int TokenCount => Sentences.Sum(s => s.Tokens.Count);

    /// <summary>The number of words in all sentences, after multi-word token expansion (Stanza's <c>doc.num_words</c>).</summary>
    public int WordCount => Sentences.Sum(s => s.Tokens.Sum(t => t.Words.Count));
}

/// <summary>One sentence: its tokens, and the sentence-level results (parse tree, sentiment, entities).</summary>
public sealed class Sentence
{
    /// <summary>
    /// The sentence's text, a slice of <see cref="Document.Text"/> (CoNLL-U's <c># text</c>). Every <c>Pipeline.Process</c>
    /// overload sets it on every sentence, so it is never null there (use <c>sentence.Text!</c>). It is null only for a
    /// sentence read by <see cref="Conllu.Read"/> without a <c># text</c> comment, or one built by hand.
    /// </summary>
    public string? Text { get; set; }

    /// <summary>The sentence id: its 0-based index in the document, as a string (CoNLL-U's <c># sent_id</c>).</summary>
    public string? SentId { get; set; }

    /// <summary>The surface tokens, in order.</summary>
    public List<Token> Tokens { get; } = [];

    /// <summary>The constituency parse, rooted at <c>ROOT</c>; null unless the constituency parser ran.</summary>
    public Tree? Constituency { get; set; }

    /// <summary>Stanza's sentiment label: 0 negative, 1 neutral, 2 positive; null unless sentiment ran.</summary>
    public int? Sentiment { get; set; }

    /// <summary>Named entities built from the tokens' <see cref="Token.Ner"/> tags (Stanza's <c>sentence.ents</c>).</summary>
    public List<Entity> Entities { get; } = [];

    /// <summary>Syntactic words (after MWT expansion), which POS and the parser operate on.</summary>
    public IEnumerable<Word> Words => Tokens.SelectMany(t => t.Words);

    /// <summary>
    /// The dependency arcs, one per word in order (Stanza's <c>sentence.dependencies</c>); empty unless every word has a
    /// <see cref="Word.Head"/> and a <see cref="Word.Deprel"/> and the ids run 1 to n, as when the dependency parser ran.
    /// </summary>
    /// <exception cref="InvalidOperationException">A word's head is not a word of the sentence.</exception>
    public IReadOnlyList<Dependency> Dependencies
    {
        get
        {
            // Stanza builds them only when every word has a head and a relation and the ids run 1..n (is_complete_words).
            var words = Words.ToList();
            if (words.Count == 0 || words.Any(w => w.Head == null || w.Deprel == null) || words[^1].Id != words.Count || words.Count < Tokens.Count)
                return [];
            return [.. words.Select(w => new Dependency(
                w.Head == 0 ? null : w.Head > 0 && w.Head <= words.Count && words[w.Head!.Value - 1].Id == w.Head
                    ? words[w.Head!.Value - 1]
                    : throw new InvalidOperationException($"Word {w.Id}'s head {w.Head} is not a word of the sentence"),
                w.Deprel!, w))];
        }
    }

    /// <summary>
    /// The dependencies one per line as Stanza's <c>dependencies_string()</c> prints them: (word text, head id, relation) as
    /// a Python tuple, e.g. <c>('born', 0, 'root')</c>; empty when there are none.
    /// </summary>
    public string DependenciesString() =>
        string.Join("\n", Dependencies.Select(d => $"({PyString.Repr(d.Dependent.Text)}, {d.Head?.Id ?? 0}, {PyString.Repr(d.Relation)})"));

    /// <summary>Each token's <see cref="Token.ToString"/>, one per line (Stanza's <c>tokens_string()</c>).</summary>
    public string TokensString() => string.Join("\n", Tokens);

    /// <summary>Each word's <see cref="Word.ToString"/>, one per line (Stanza's <c>words_string()</c>).</summary>
    public string WordsString() => string.Join("\n", Words);
}

/// <summary>
/// A token as a <c>PipelineOptions.TokenizePostprocessor</c> sees and returns it (Stanza's <c>tokenize_postprocessor</c>):
/// its text, and whether it is a multi-word token. Stanza's forms map to it: <c>"text"</c> is <c>new("text")</c> (a string
/// converts implicitly), <c>("text", True)</c> is <c>new("text", IsMultiWord: true)</c> and <c>("text", ["a", "b"])</c> is
/// <c>new("text", Expansion: ["a", "b"])</c>.
/// </summary>
/// <param name="Text">The token's text. A returned token must be found in the original text after the one before it, so the
/// postprocessor can merge, split or regroup tokens and sentences but not change their characters.</param>
/// <param name="IsMultiWord">The tokenizer's multi-word flag (Stanza's <c>MWT=Yes</c>); returned true without an
/// <see cref="Expansion"/>, the MWT model expands the token.</param>
/// <param name="Expansion">The token's words, given by hand: the MWT model leaves it alone.</param>
public sealed record PostprocessorToken(string Text, bool IsMultiWord = false, IReadOnlyList<string>? Expansion = null)
{
    /// <summary>A token that is not multi-word.</summary>
    public static implicit operator PostprocessorToken(string text) => new(text);
}

/// <summary>A dependency arc (Stanza's <c>(head, deprel, word)</c> entries of <c>sentence.dependencies</c>).</summary>
/// <param name="Head">The head word; null for the root (Stanza's word <c>ROOT</c>, id 0).</param>
/// <param name="Relation">The universal dependency relation, e.g. <c>nsubj</c>.</param>
/// <param name="Dependent">The dependent word.</param>
public sealed record Dependency(Word? Head, string Relation, Word Dependent);

/// <summary>A surface token from the tokenizer. A multi-word token ("don't") has several words.</summary>
public sealed class Token
{
    /// <summary>The token's text as it appears in the input.</summary>
    public required string Text { get; set; }

    /// <summary>Offset of the token's first character in <see cref="Document.Text"/> (a UTF-16 index).</summary>
    public int? StartChar { get; set; }

    /// <summary>Offset just past the token's last character in <see cref="Document.Text"/> (a UTF-16 index).</summary>
    public int? EndChar { get; set; }

    /// <summary>Whitespace following the token in the original text: "" for none.</summary>
    public string SpaceAfter { get; set; } = " ";

    /// <summary>Whitespace before the token; only set on a document's first token, as in Stanza.</summary>
    public string SpacesBefore { get; set; } = "";

    /// <summary>The syntactic words: one for most tokens, several for a multi-word token ("do", "n't").</summary>
    public List<Word> Words { get; } = new(1); // most tokens are one word: 32 bytes instead of an empty list growing to 4

    /// <summary>Whether the token expands to more than one word.</summary>
    public bool IsMultiWord => Words.Count > 1;

    /// <summary>The tokenizer predicted a multi-word token here (Stanza's <c>MWT=Yes</c>); the MWT stage expands it.</summary>
    internal bool IsMwtCandidate { get; set; }

    /// <summary>The token's named-entity tag in BIOES form, e.g. <c>B-ORG</c> or <c>O</c>; null if NER has not run.</summary>
    public string? Ner { get; set; }

    /// <summary>Document.set_mwt_expansions for one token: new words, offsets found by matching them in the token.</summary>
    internal void SetWords(string expansion)
    {
        var words = expansion.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length <= 1)
            words = [Text];

        IsMwtCandidate = false;
        Words.Clear();
        foreach (var w in words)
            Words.Add(new Word { Text = w });

        if (words.Length == 1)
        {
            (Words[0].StartChar, Words[0].EndChar) = (StartChar, EndChar);
            return;
        }

        if (StartChar is not int start)
            return;
        var match = Regex.Match(Text, "^" + string.Join(@"\s*", words.Select(w => $"({Regex.Escape(w)})")) + "$");
        if (!match.Success)
            return;
        for (int i = 0; i < words.Length; i++)
        {
            var g = match.Groups[i + 1];
            (Words[i].StartChar, Words[i].EndChar) = (start + g.Index, start + g.Index + g.Length);
        }
    }

    /// <summary>The token and its words on one line, as Stanza's <c>pretty_print()</c>, e.g. <c>&lt;Token id=1-2;words=[...]&gt;</c>.</summary>
    public override string ToString()
    {
        var id = Words.Count switch { 0 => "", 1 => $"{Words[0].Id}", _ => $"{Words[0].Id}-{Words[^1].Id}" };
        return $"<Token id={id};words=[{string.Join(", ", Words)}]>";
    }
}

/// <summary>A named entity: a span of tokens with a type such as <c>PERSON</c> (Stanza's <c>Span</c>).</summary>
/// <param name="Text">The entity's text, e.g. <c>Barack Obama</c>.</param>
/// <param name="Type">The OntoNotes type, e.g. <c>PERSON</c>, <c>ORG</c>, <c>GPE</c> or <c>DATE</c>.</param>
/// <param name="StartChar">Offset of the first character in <see cref="Document.Text"/>.</param>
/// <param name="EndChar">Offset just past the last character in <see cref="Document.Text"/>.</param>
/// <param name="Tokens">The tokens the entity spans.</param>
public sealed record Entity(string Text, string Type, int? StartChar, int? EndChar, IReadOnlyList<Token> Tokens)
{
    /// <summary>
    /// Sentence.build_ents: replaces <paramref name="sentence"/>'s entities with the spans its tokens' BIOES
    /// tags describe (ner/utils.py decode_from_bioes). The text is <paramref name="docText"/> between the
    /// offsets when both are known, else the tokens joined with their whitespace.
    /// </summary>
    internal static void Build(Sentence sentence, string? docText)
    {
        sentence.Entities.Clear();
        var tokens = sentence.Tokens;
        int start = -1, end = -1;
        string type = "";

        void Flush()
        {
            if (start < 0)
                return;
            var span = tokens.GetRange(start, end - start + 1);
            int? from = span[0].StartChar, to = span[^1].EndChar;
            var text = docText != null && from != null && to != null
                ? docText[from.Value..to.Value]
                : string.Concat(span.SelectMany(t => new[] { t.Text, t.SpaceAfter }).SkipLast(1));
            sentence.Entities.Add(new Entity(text, type, from, to, span));
            start = -1;
        }

        for (int i = 0; i < tokens.Count; i++)
        {
            var tag = tokens[i].Ner ?? "O";
            var prefix = tag.Length >= 2 ? tag[..2] : "";
            if (tag == "O")
            {
                Flush();
            }
            else if (prefix is "B-" or "S-")
            {
                Flush();
                (start, end, type) = (i, i, tag[2..]);
                if (prefix == "S-")
                    Flush();
            }
            else if (prefix is "I-" or "E-")
            {
                // An I-/E- without a B- starts the entity right there, as in Stanza.
                if (start < 0)
                    start = i;
                (end, type) = (i, tag[2..]);
                if (prefix == "E-")
                    Flush();
            }
        }

        Flush();
    }
}

/// <summary>A syntactic word, with the annotations the processors add (CoNLL-U's word lines).</summary>
public sealed class Word
{
    /// <summary>1-based index within the sentence.</summary>
    public int Id { get; set; }

    /// <summary>The word's text; for a word of a multi-word token, its expanded form (e.g. <c>n't</c>).</summary>
    public required string Text { get; set; }

    /// <summary>The lemma, e.g. <c>bear</c> for <c>born</c>; null unless the lemmatizer ran.</summary>
    public string? Lemma { get; set; }

    /// <summary>The universal part-of-speech tag, e.g. <c>VERB</c>; null unless the tagger ran.</summary>
    public string? Upos { get; set; }

    /// <summary>The Penn Treebank tag, e.g. <c>VBN</c>; null unless the tagger ran.</summary>
    public string? Xpos { get; set; }

    /// <summary>Universal morphological features, e.g. <c>Tense=Past|VerbForm=Part</c>; null if none or untagged.</summary>
    public string? Feats { get; set; }

    /// <summary>The <see cref="Id"/> of the word's dependency head, 0 for the root; null unless the dependency parser ran.</summary>
    public int? Head { get; set; }

    /// <summary>The universal dependency relation to the head, e.g. <c>nsubj:pass</c>; null unless the dependency parser ran.</summary>
    public string? Deprel { get; set; }

    /// <summary>Offset of the word's first character in <see cref="Document.Text"/>, when it can be located.</summary>
    public int? StartChar { get; set; }

    /// <summary>Offset just past the word's last character in <see cref="Document.Text"/>, when it can be located.</summary>
    public int? EndChar { get; set; }

    /// <summary>
    /// The word on one line, as Stanza's <c>pretty_print()</c>: its id, text, lemma, UPOS, XPOS, feats, head and relation,
    /// those that are set, e.g. <c>&lt;Word id=4;text=born;lemma=bear;upos=VERB;...;head=0;deprel=root&gt;</c>.
    /// </summary>
    public override string ToString()
    {
        var fields = new (string Key, object? Value)[]
        {
            ("id", Id), ("text", Text), ("lemma", Lemma), ("upos", Upos), ("xpos", Xpos), ("feats", Feats), ("head", Head), ("deprel", Deprel),
        };
        return $"<Word {string.Join(";", fields.Where(f => f.Value != null).Select(f => $"{f.Key}={f.Value}"))}>";
    }
}
