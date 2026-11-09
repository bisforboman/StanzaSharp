using System.Text;

namespace StanzaSharp.Subtitles;

/// <summary>A cue's text line, placed in the transcript.</summary>
/// <param name="Start">The UTF-16 offset of the line's first character in <see cref="Transcript.Text"/>.</param>
/// <param name="Text">The line without formatting (and without a dialogue dash).</param>
/// <param name="Dialogue">Whether the line starts a speaker's turn (it began with a dash).</param>
public sealed record CueLine(int Start, string Text, bool Dialogue)
{
    /// <summary>The offset just past the line.</summary>
    public int End => Start + Text.Length;
}

/// <summary>
/// The text of all cues joined into one, so a sentence that runs over several cues is parsed whole: each line is
/// placed where it falls in the text, and each line break becomes a space whose offset maps back to its cue and line.
/// </summary>
public sealed class Transcript
{
    private Transcript(string text, IReadOnlyList<Cue> cues, IReadOnlyList<IReadOnlyList<CueLine>> lines)
    {
        Text = text;
        Cues = cues;
        Lines = lines;
    }

    /// <summary>All cues' lines without formatting, separated by single spaces.</summary>
    public string Text { get; }

    /// <summary>The cues, in the file's order.</summary>
    public IReadOnlyList<Cue> Cues { get; }

    /// <summary>Each cue's non-empty lines, placed in <see cref="Text"/>.</summary>
    public IReadOnlyList<IReadOnlyList<CueLine>> Lines { get; }

    /// <summary>Joins the cues' lines into one text.</summary>
    public static Transcript Build(IReadOnlyList<Cue> cues)
    {
        var text = new StringBuilder();
        var all = new List<IReadOnlyList<CueLine>>(cues.Count);
        foreach (var cue in cues)
        {
            var lines = new List<CueLine>();
            foreach (var raw in cue.Lines)
            {
                var dialogue = Srt.IsDialogueLine(Srt.PlainText(raw), out var plain);
                if (plain.Length == 0)
                    continue;
                if (text.Length > 0)
                    text.Append(' ');
                lines.Add(new CueLine(text.Length, plain, dialogue));
                text.Append(plain);
            }

            all.Add(lines);
        }

        return new Transcript(text.ToString(), cues, all);
    }
}
