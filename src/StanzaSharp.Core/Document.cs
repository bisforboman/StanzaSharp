namespace StanzaSharp;

/// <summary>A processed text: sentences of tokens, each token expanding to one or more words.</summary>
public sealed class Document
{
    public string? Text { get; set; }
    public List<Sentence> Sentences { get; } = [];
}

public sealed class Sentence
{
    public string? Text { get; set; }
    public string? SentId { get; set; }
    public List<Token> Tokens { get; } = [];
    public Tree? Constituency { get; set; }

    /// <summary>Syntactic words (after MWT expansion), which POS and the parser operate on.</summary>
    public IEnumerable<Word> Words => Tokens.SelectMany(t => t.Words);
}

/// <summary>A surface token from the tokenizer. A multi-word token ("don't") has several words.</summary>
public sealed class Token
{
    public required string Text { get; set; }
    public int? StartChar { get; set; }
    public int? EndChar { get; set; }

    /// <summary>Whitespace following the token in the original text: "" for none.</summary>
    public string SpaceAfter { get; set; } = " ";

    public List<Word> Words { get; } = [];
    public bool IsMultiWord => Words.Count > 1;

    /// <summary>The tokenizer predicted a multi-word token here (Stanza's <c>MWT=Yes</c>); the MWT stage expands it.</summary>
    public bool IsMwtCandidate { get; set; }
}

public sealed class Word
{
    /// <summary>1-based index within the sentence.</summary>
    public int Id { get; set; }
    public required string Text { get; set; }
    public string? Lemma { get; set; }
    public string? Upos { get; set; }
    public string? Xpos { get; set; }
    public string? Feats { get; set; }
    public int? Head { get; set; }
    public string? Deprel { get; set; }
    public int? StartChar { get; set; }
    public int? EndChar { get; set; }
}
