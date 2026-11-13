namespace StanzaSharp.Subtitles;

/// <summary>A dependency a line break separates although a style rule keeps its two words on one line.</summary>
/// <param name="Relation">The UD relation (<c>det</c>, <see langword="case"/>, ...).</param>
/// <param name="Dependent">The dependent word's text.</param>
/// <param name="Head">The head word's text.</param>
/// <param name="Rule">What the style guide asks, in words.</param>
/// <param name="Weight">The rule's cost in a candidate break's score.</param>
public sealed record SplitRelation(string Relation, string Dependent, string Head, string Rule, double Weight);

/// <summary>A place a cue's text can break into two lines, scored.</summary>
/// <param name="Top">The first line.</param>
/// <param name="Bottom">The second line.</param>
/// <param name="Splits">The kept-together relations the break separates.</param>
/// <param name="Score">Lower is better: the rules' weights, the lines' imbalance and overflow, less a bonus for breaking
/// after punctuation or before a conjunction or preposition.</param>
public sealed record BreakOption(string Top, string Bottom, IReadOnlyList<SplitRelation> Splits, double Score);

/// <summary>A line break that separates what a style rule keeps together.</summary>
/// <param name="Cue">The cue.</param>
/// <param name="Line">The break's place: after the cue's line <c>Line</c> (1-based).</param>
/// <param name="Lines">The cue's lines without formatting.</param>
/// <param name="Splits">The relations the break separates.</param>
/// <param name="Suggestion">The best-scoring two-line break, when it splits less than the present one (null if none
/// does).</param>
/// <param name="FitsOneLine">Whether the cue's text fits on one line, which the style guides prefer.</param>
public sealed record Finding(Cue Cue, int Line, IReadOnlyList<string> Lines, IReadOnlyList<SplitRelation> Splits,
    BreakOption? Suggestion, bool FitsOneLine);

/// <summary>
/// Checks where cues break into lines against the dependency parse of their sentences. Netflix's timed text style
/// guides ask for a break after punctuation and before conjunctions and prepositions, and not to separate a noun from
/// its article or adjective, a first name from a last name, a verb from its subject pronoun or a prepositional verb
/// from its preposition; each of those is a relation whose two words should share a line (<see cref="Rule"/>).
/// </summary>
public static class BreakChecker
{
    /// <summary>The usual limit of Netflix's guides for Latin-script languages, English and Swedish among them.</summary>
    public const int DefaultMaxChars = 42;

    /// <summary>
    /// The style rule a dependency falls under when a break separates its words, or null when breaking there is fine.
    /// <paramref name="dependentEndsLine"/> says whether the dependent is the last word before the break.
    /// </summary>
    public static (string Rule, double Weight)? Rule(Word dependent, Word head, bool dependentEndsLine)
    {
        var before = dependent.Id < head.Id; // a pre-nominal adjective, determiner or possessive
        return dependent.Deprel switch
        {
            "det" or "det:poss" or "det:predet" when before => ("keep an article or determiner with its noun", 10),
            "nmod:poss" when before => ("keep a possessive with its noun", 10),
            "amod" when before => ("keep an adjective with its noun", 10),
            "nummod" when before => ("keep a number with its noun", 8),
            "compound" => ("keep a compound together", 8),
            "flat" or "flat:name" => ("keep a first name with the last name", 10),
            "fixed" => ("keep a fixed expression together", 10),
            "case" => ("keep a preposition with its noun (break before the preposition)", 8),
            "compound:prt" => ("keep a verb with its particle or preposition", 10),
            "cc" or "mark" when dependentEndsLine => ("break before a conjunction, not after it", 6),
            _ => null,
        };
    }

    /// <summary>
    /// Whether a break between <paramref name="left"/> and <paramref name="right"/> (adjacent words of one sentence)
    /// separates a subject pronoun from its verb: one is the pronoun, the other its verb or the verb's auxiliary or
    /// copula ("we | will call", "ska | vi ringa").
    /// </summary>
    public static bool SeparatesSubjectPronoun(Word left, Word right) => IsVerbOf(left, right) || IsVerbOf(right, left);

    private static bool IsVerbOf(Word pronoun, Word verb) =>
        pronoun.Upos == "PRON" && pronoun.Deprel is "nsubj" or "nsubj:pass" or "expl"
        && (verb.Id == pronoun.Head || (verb.Head == pronoun.Head && verb.Deprel is "aux" or "aux:pass" or "cop"));

    /// <summary>The flagged line breaks of every cue, in the file's order.</summary>
    public static List<Finding> Check(Transcript transcript, Document document, int maxChars = DefaultMaxChars)
    {
        var findings = new List<Finding>();
        var tokens = Placed(document);
        for (var c = 0; c < transcript.Cues.Count; c++)
        {
            var lines = transcript.Lines[c];
            if (lines.Count < 2)
                continue;
            var cue = CueTokens(tokens, lines);
            var text = transcript.Text[lines[0].Start..lines[^1].End];
            for (var l = 1; l < lines.Count; l++)
            {
                // A break before a speaker's dash is the dialogue format, not a choice.
                if (lines[l].Dialogue)
                    continue;
                var at = cue.FindIndex(t => t.Token.StartChar >= lines[l].Start);
                if (at <= 0 || cue[at - 1].Token.EndChar > lines[l - 1].End)
                    continue;
                var splits = Splits(cue, at);
                if (splits.Count == 0)
                    continue;

                // Only a two-line cue gets a suggestion; one of more lines breaks the guides' two-line limit anyway.
                var best = lines.Count == 2 ? Options(cue, transcript.Text, lines[0].Start, lines[1].End, maxChars).MinBy(o => o.Score) : null;
                if (best != null && (best.Splits.Count >= splits.Count || best.Top == lines[0].Text))
                    best = null;
                findings.Add(new Finding(transcript.Cues[c], l, [.. lines.Select(x => x.Text)], splits, best,
                    !lines.Any(x => x.Dialogue) && text.Length <= maxChars));
            }
        }

        return findings;
    }

    /// <summary>The tokens that lie within a cue's lines, in order.</summary>
    public static List<PlacedToken> CueTokens(IReadOnlyList<PlacedToken> tokens, IReadOnlyList<CueLine> lines) =>
        [.. tokens.Where(t => t.Token.StartChar >= lines[0].Start && t.Token.EndChar <= lines[^1].End)];

    /// <summary>Every place the cue's text can break into two lines, scored.</summary>
    public static IEnumerable<BreakOption> Options(IReadOnlyList<PlacedToken> cue, string text, int start, int end, int maxChars)
    {
        for (var at = 1; at < cue.Count; at++)
        {
            var left = cue[at - 1].Token;
            var right = cue[at].Token;
            if (left.EndChar >= right.StartChar)
                continue; // no space between the tokens: "dog" and ".", "do" and "n't"
            var top = text[start..left.EndChar!.Value];
            var bottom = text[right.StartChar!.Value..end];
            var splits = Splits(cue, at);
            var score = splits.Sum(s => s.Weight)
                + Math.Abs(top.Length - bottom.Length) / 4.0
                + (top.Length > bottom.Length ? 1 : 0) // the guides prefer a bottom-heavy pyramid
                + (at == 1 ? 3 : 0) // and not a lone word on the top line
                + (Math.Max(0, top.Length - maxChars) + Math.Max(0, bottom.Length - maxChars)) * 5.0;
            if (cue[at - 1].Sentence != cue[at].Sentence || (left.Words[^1].Upos == "PUNCT" && left.Text is not ("(" or "[" or "\"" or "“" or "„")))
                score -= 4; // after punctuation
            else if (right.Words[0].Deprel is "cc" or "mark" or "case")
                score -= 2; // before a conjunction or preposition
            yield return new BreakOption(top, bottom, splits, score);
        }
    }

    /// <summary>The kept-together relations whose words lie on both sides of a break before <c>cue[at]</c>.</summary>
    public static List<SplitRelation> Splits(IReadOnlyList<PlacedToken> cue, int at)
    {
        var splits = new List<SplitRelation>();
        var lastLeft = cue[at - 1].Token.Words[^1];
        for (var i = 0; i < cue.Count; i++)
        {
            foreach (var word in cue[i].Token.Words)
            {
                if (word.Head is not > 0)
                    continue;
                var j = 0;
                while (j < cue.Count && (cue[j].Sentence != cue[i].Sentence || !cue[j].Token.Words.Any(w => w.Id == word.Head)))
                    j++;
                if (j == cue.Count || (i < at) == (j < at))
                    continue;
                var headWord = cue[j].Token.Words.First(w => w.Id == word.Head);
                if (Rule(word, headWord, ReferenceEquals(word, lastLeft)) is { } rule)
                    splits.Add(new SplitRelation(word.Deprel!, word.Text, headWord.Text, rule.Rule, rule.Weight));
            }
        }

        var right = cue[at].Token.Words[0];
        if (cue[at - 1].Sentence == cue[at].Sentence && SeparatesSubjectPronoun(lastLeft, right))
        {
            var (pronoun, verb) = IsVerbOf(lastLeft, right) ? (lastLeft, right) : (right, lastLeft);
            splits.Add(new SplitRelation(pronoun.Deprel!, pronoun.Text, verb.Text, "keep a subject pronoun with its verb", 10));
        }

        return splits;
    }

    /// <summary>The document's tokens that carry offsets, each with its sentence's index.</summary>
    public static List<PlacedToken> Placed(Document document) =>
        [.. document.Sentences.SelectMany((s, i) => s.Tokens.Where(t => t.StartChar != null && t.EndChar != null).Select(t => new PlacedToken(t, i)))];
}

/// <summary>A token and the index of its sentence in the document.</summary>
/// <param name="Token">The token.</param>
/// <param name="Sentence">Its sentence's index.</param>
public sealed record PlacedToken(Token Token, int Sentence);
