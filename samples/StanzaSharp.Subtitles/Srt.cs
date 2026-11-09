using System.Text;
using System.Text.RegularExpressions;

namespace StanzaSharp.Subtitles;

/// <summary>One subtitle cue as an SRT file gives it.</summary>
/// <param name="Number">The cue's number (its counter line), or its position when the file has none.</param>
/// <param name="Timing">The timing line, as written (<c>00:01:02,000 --> 00:01:04,500</c>).</param>
/// <param name="Lines">The cue's text lines, as written (formatting tags included).</param>
public sealed record Cue(int Number, string Timing, IReadOnlyList<string> Lines);

/// <summary>Reads SubRip (.srt) files.</summary>
public static partial class Srt
{
    /// <summary>
    /// The cues of an SRT file: blocks separated by blank lines, each a counter line (optional), a timing line holding
    /// <c>--></c> and the text lines. A block without a timing line is skipped.
    /// </summary>
    public static List<Cue> Parse(string text)
    {
        var cues = new List<Cue>();
        var block = new List<string>();
        foreach (var line in text.TrimStart('﻿').ReplaceLineEndings("\n").Split('\n'))
        {
            if (line.Trim().Length == 0)
            {
                Flush();
                continue;
            }

            block.Add(line.TrimEnd());
        }

        Flush();
        return cues;

        void Flush()
        {
            var timing = block.FindIndex(l => l.Contains("-->", StringComparison.Ordinal));
            if (timing is 0 or 1)
            {
                var number = timing == 1 && int.TryParse(block[0].Trim(), out var n) ? n : cues.Count + 1;
                cues.Add(new Cue(number, block[timing].Trim(), block[(timing + 1)..]));
            }

            block.Clear();
        }
    }

    /// <summary>
    /// A line's text without formatting: HTML-like tags (<c>&lt;i&gt;</c>, <c>&lt;font ...&gt;</c>) and ASS overrides
    /// (<c>{\an8}</c>) removed, the spaces around it trimmed.
    /// </summary>
    public static string PlainText(string line) => Tags().Replace(line, "").Trim();

    /// <summary>
    /// Whether a line starts a speaker's turn in a dialogue cue (a leading hyphen or dash); <paramref name="rest"/> is
    /// the line without it.
    /// </summary>
    public static bool IsDialogueLine(string plain, out string rest)
    {
        if (plain.Length > 1 && plain[0] is '-' or '‐' or '–' or '—' && plain[1] != '-')
        {
            rest = plain[1..].TrimStart();
            return true;
        }

        rest = plain;
        return false;
    }

    /// <summary>A cue's lines joined back into one line, for display.</summary>
    public static string Join(IEnumerable<string> lines) => new StringBuilder().AppendJoin(" ", lines).ToString();

    [GeneratedRegex(@"<[^<>]*>|\{\\[^{}]*\}")]
    private static partial Regex Tags();
}
