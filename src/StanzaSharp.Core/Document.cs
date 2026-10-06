namespace StanzaSharp;

/// <summary>A processed text: sentences of tokens, each token expanding to one or more words.</summary>
public sealed class Document
{
    public string? Text { get; set; }
    public List<Sentence> Sentences { get; } = [];

    /// <summary>Named entities of every sentence, in order (Stanza's <c>doc.ents</c>).</summary>
    public IEnumerable<Entity> Entities => Sentences.SelectMany(s => s.Entities);
}

public sealed class Sentence
{
    public string? Text { get; set; }
    public string? SentId { get; set; }
    public List<Token> Tokens { get; } = [];
    public Tree? Constituency { get; set; }

    /// <summary>Named entities built from the tokens' <see cref="Token.Ner"/> tags (Stanza's <c>sentence.ents</c>).</summary>
    public List<Entity> Entities { get; } = [];

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

    /// <summary>Whitespace before the token; only set on a document's first token, as in Stanza.</summary>
    public string SpacesBefore { get; set; } = "";

    public List<Word> Words { get; } = [];
    public bool IsMultiWord => Words.Count > 1;

    /// <summary>The tokenizer predicted a multi-word token here (Stanza's <c>MWT=Yes</c>); the MWT stage expands it.</summary>
    public bool IsMwtCandidate { get; set; }

    /// <summary>The token's named-entity tag in BIOES form, e.g. <c>B-ORG</c> or <c>O</c>; null if NER has not run.</summary>
    public string? Ner { get; set; }
}

/// <summary>A named entity: a span of tokens with a type such as <c>PERSON</c> (Stanza's <c>Span</c>).</summary>
public sealed record Entity(string Text, string Type, int? StartChar, int? EndChar, IReadOnlyList<Token> Tokens)
{
    /// <summary>
    /// Sentence.build_ents: replaces <paramref name="sentence"/>'s entities with the spans its tokens' BIOES
    /// tags describe (ner/utils.py decode_from_bioes). The text is <paramref name="docText"/> between the
    /// offsets when both are known, else the tokens joined with their whitespace.
    /// </summary>
    public static void Build(Sentence sentence, string? docText)
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
                Flush();
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
