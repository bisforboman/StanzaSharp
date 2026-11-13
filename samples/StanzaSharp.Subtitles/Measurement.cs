namespace StanzaSharp.Subtitles;

/// <summary>
/// How the checker judges a file's own line breaks against breaks made at the midpoint of the same subtitles. On
/// professionally made subtitles the flag rate should be low and on midpoint breaks high; the checker's best break
/// should agree with the human one more often than the midpoint does.
/// </summary>
/// <param name="Cues">The file's cues.</param>
/// <param name="Compared">The two-line cues compared: not a dialogue, the break between two tokens.</param>
/// <param name="Skipped">Cues of two or more lines left out: dialogues and those of more than two lines.</param>
/// <param name="HumanFlagged">Compared cues whose own break the checker flags.</param>
/// <param name="MidpointFlagged">Compared cues whose midpoint break the checker flags.</param>
/// <param name="SameBreak">Compared cues whose own break is the midpoint break.</param>
/// <param name="HumanFlaggedDiffering">Of the cues whose own break is not the midpoint one, those flagged.</param>
/// <param name="MidpointFlaggedDiffering">Of the cues whose own break is not the midpoint one, those whose midpoint
/// break is flagged.</param>
/// <param name="BestIsHuman">Compared cues whose best-scoring break (<see cref="BreakChecker.Options"/>) is their own
/// break.</param>
/// <param name="OverLimit">Compared cues with a line over the limit.</param>
/// <param name="HumanRelations">How many own breaks split each relation.</param>
/// <param name="MidpointRelations">How many midpoint breaks split each relation.</param>
public sealed record MeasureResult(
    int Cues, int Compared, int Skipped, int HumanFlagged, int MidpointFlagged, int SameBreak, int HumanFlaggedDiffering,
    int MidpointFlaggedDiffering, int BestIsHuman, int OverLimit, IReadOnlyDictionary<string, int> HumanRelations,
    IReadOnlyDictionary<string, int> MidpointRelations)
{
    /// <summary>Compared cues whose own break is not the midpoint one.</summary>
    public int Differing => Compared - SameBreak;
}

/// <summary>Measures the checker on a file of human-made breaks (#267): the file's breaks against midpoint ones.</summary>
public static class Measurement
{
    /// <summary>
    /// The break nearest the middle of <paramref name="text"/>: the space that leaves the two lines' lengths closest,
    /// the shorter top line on a tie (the guides prefer a bottom-heavy pyramid). Null without a space.
    /// </summary>
    public static (string Top, string Bottom)? Midpoint(string text)
    {
        var best = -1;
        for (var i = 0; i < text.Length; i++)
        {
            if (text[i] == ' ' && (best < 0 || Math.Abs((2 * i) + 1 - text.Length) < Math.Abs((2 * best) + 1 - text.Length)))
                best = i;
        }

        return best < 0 ? null : (text[..best], text[(best + 1)..]);
    }

    /// <summary>
    /// Compares the cues' own breaks with midpoint breaks; <paramref name="document"/> is the parse of the cues'
    /// <see cref="Transcript.Text"/>.
    /// </summary>
    public static MeasureResult Measure(IReadOnlyList<Cue> cues, Document document, int maxChars = BreakChecker.DefaultMaxChars)
    {
        var human = Transcript.Build(cues);
        var tokens = BreakChecker.Placed(document);

        // The same cues broken at the midpoint: the same text, so the same parse, with other line breaks.
        var compared = new List<int>();
        var midCues = new List<Cue>(cues.Count);
        var skipped = 0;
        for (var c = 0; c < cues.Count; c++)
        {
            var lines = human.Lines[c];
            if (lines.Count == 2 && !lines.Any(l => l.Dialogue) && Breaks(tokens, lines)
                && Midpoint(Srt.Join(lines.Select(l => l.Text))) is var (top, bottom))
            {
                compared.Add(c);
                midCues.Add(cues[c] with { Lines = [top, bottom] });
                continue;
            }

            if (lines.Count >= 2)
                skipped++;
            midCues.Add(cues[c] with { Lines = [.. lines.Select(l => l.Text)] });
        }

        var mid = Transcript.Build(midCues);
        if (mid.Text != human.Text)
            throw new InvalidOperationException("The midpoint breaks changed the text.");

        var humanFlags = Flags(human, document, maxChars);
        var midFlags = Flags(mid, document, maxChars);
        int humanFlagged = 0, midFlagged = 0, same = 0, humanDiff = 0, midDiff = 0, bestIsHuman = 0, overLimit = 0;
        var humanRelations = new SortedDictionary<string, int>(StringComparer.Ordinal);
        var midRelations = new SortedDictionary<string, int>(StringComparer.Ordinal);
        foreach (var c in compared)
        {
            var lines = human.Lines[c];
            var isSame = lines[0].Text == mid.Lines[c][0].Text;
            var h = humanFlags.GetValueOrDefault(c);
            var m = midFlags.GetValueOrDefault(c);
            same += isSame ? 1 : 0;
            humanFlagged += h != null ? 1 : 0;
            midFlagged += m != null ? 1 : 0;
            humanDiff += !isSame && h != null ? 1 : 0;
            midDiff += !isSame && m != null ? 1 : 0;
            Count(humanRelations, h);
            Count(midRelations, m);
            overLimit += lines.Any(l => l.Text.Length > maxChars) ? 1 : 0;
            var cue = BreakChecker.CueTokens(tokens, lines);
            var best = BreakChecker.Options(cue, human.Text, lines[0].Start, lines[1].End, maxChars).MinBy(o => o.Score);
            bestIsHuman += best?.Top == lines[0].Text ? 1 : 0;
        }

        return new MeasureResult(cues.Count, compared.Count, skipped, humanFlagged, midFlagged, same, humanDiff, midDiff,
            bestIsHuman, overLimit, humanRelations, midRelations);
    }

    // Whether the cue's line break falls between two of the parse's tokens, as the checker needs.
    private static bool Breaks(IReadOnlyList<PlacedToken> tokens, IReadOnlyList<CueLine> lines)
    {
        var cue = BreakChecker.CueTokens(tokens, lines);
        var at = cue.FindIndex(t => t.Token.StartChar >= lines[1].Start);
        return at > 0 && cue[at - 1].Token.EndChar <= lines[0].End;
    }

    private static Dictionary<int, Finding> Flags(Transcript transcript, Document document, int maxChars)
    {
        var byCue = new Dictionary<Cue, int>(ReferenceEqualityComparer.Instance);
        for (var c = 0; c < transcript.Cues.Count; c++)
            byCue[transcript.Cues[c]] = c;
        return BreakChecker.Check(transcript, document, maxChars).GroupBy(f => byCue[f.Cue]).ToDictionary(g => g.Key, g => g.First());
    }

    private static void Count(SortedDictionary<string, int> counts, Finding? finding)
    {
        foreach (var relation in finding?.Splits.Select(s => s.Relation).Distinct() ?? [])
            counts[relation] = counts.GetValueOrDefault(relation) + 1;
    }
}
