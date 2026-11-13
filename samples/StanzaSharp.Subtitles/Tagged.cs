namespace StanzaSharp.Subtitles;

/// <summary>
/// Reads subtitles as tagged text, the format of the MuST-Cinema corpus and of EvalSubtitle: one sentence per line,
/// <c>&lt;eol&gt;</c> where a subtitle's text breaks into its next line and <c>&lt;eob&gt;</c> where the subtitle ends.
/// </summary>
public static class Tagged
{
    /// <summary>The marker of a line break inside a subtitle.</summary>
    public const string EndOfLine = "<eol>";

    /// <summary>The marker of a subtitle's end.</summary>
    public const string EndOfBlock = "<eob>";

    /// <summary>
    /// The subtitles of a tagged text as cues, numbered from 1 and without timing. A sentence whose last block has no
    /// <c>&lt;eob&gt;</c> continues into the next sentence's first block.
    /// </summary>
    public static List<Cue> Parse(string text)
    {
        var cues = new List<Cue>();
        var lines = new List<string>();
        var line = new List<string>();
        foreach (var sentence in text.TrimStart('﻿').ReplaceLineEndings("\n").Split('\n'))
        {
            foreach (var word in sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                switch (word)
                {
                    case EndOfLine:
                        EndLine();
                        break;
                    case EndOfBlock:
                        EndLine();
                        if (lines.Count > 0)
                            cues.Add(new Cue(cues.Count + 1, "", [.. lines]));
                        lines.Clear();
                        break;
                    default:
                        line.Add(word);
                        break;
                }
            }
        }

        EndLine();
        if (lines.Count > 0)
            cues.Add(new Cue(cues.Count + 1, "", [.. lines]));
        return cues;

        void EndLine()
        {
            if (line.Count > 0)
                lines.Add(string.Join(' ', line));
            line.Clear();
        }
    }
}
