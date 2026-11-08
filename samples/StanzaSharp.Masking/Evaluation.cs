using System.Text;
using System.Text.RegularExpressions;

namespace StanzaSharp.Masking;

/// <summary>A span that must be masked, from a labelled text; <see cref="Id"/> names the person of a PER span.</summary>
/// <param name="Start">The UTF-16 offset of the span's first character in the text without markup.</param>
/// <param name="Length">The span's length.</param>
/// <param name="Type">PER, LOC, ORG, PNR, ORGNR, PHONE, EMAIL or ADDRESS.</param>
/// <param name="Id">The person a PER span names (unique within its text); null for other types.</param>
/// <param name="Text">The span's text.</param>
public sealed record GoldSpan(int Start, int Length, string Type, string? Id, string Text)
{
    /// <summary>The offset just past the span.</summary>
    public int End => Start + Length;
}

/// <summary>How well one masking run did on labelled texts.</summary>
public sealed class Score
{
    /// <summary>Labelled spans per type.</summary>
    public Dictionary<string, int> Gold { get; } = [];

    /// <summary>Labelled spans per type with a letter or digit left unmasked.</summary>
    public Dictionary<string, int> Leaked { get; } = [];

    /// <summary>The leaked spans, as "file: text (TYPE)".</summary>
    public List<string> Leaks { get; } = [];

    /// <summary>Masked spans that overlap no labelled span, as "file: text (TYPE, source)".</summary>
    public List<string> OverMasked { get; } = [];

    /// <summary>People whose masked mentions got more than one placeholder, as "file: id: text [PH], ...".</summary>
    public List<string> Split { get; } = [];

    /// <summary>Pairs of mentions of different people under one placeholder, as "file: text (id) and text (id) as [PH]".</summary>
    public List<string> Joined { get; } = [];

    /// <summary>All masked spans.</summary>
    public int Masked { get; set; }

    /// <summary>Pairs of masked mentions of one person that share a placeholder.</summary>
    public int SamePersonSamePlaceholder { get; set; }

    /// <summary>Pairs of masked mentions of one person with different placeholders.</summary>
    public int SamePersonOtherPlaceholder { get; set; }

    /// <summary>Pairs of masked mentions of different people that share a placeholder.</summary>
    public int OtherPersonSamePlaceholder { get; set; }

    /// <summary>People with two or more masked mentions.</summary>
    public int People { get; set; }

    /// <summary>People with two or more masked mentions, all under one placeholder shared with no one else.</summary>
    public int PeopleWithOwnPlaceholder { get; set; }
}

/// <summary>
/// Scores masking on labelled texts (<c>eval/</c>): each span that must be masked is written inline as
/// <c>{TYPE:text}</c>, a person's as <c>{PER#id:text}</c>.
/// </summary>
public static partial class Evaluation
{
    /// <summary>The text without markup and its labelled spans.</summary>
    public static (string Text, List<GoldSpan> Spans) Parse(string labelled)
    {
        var text = new StringBuilder(labelled.Length);
        var spans = new List<GoldSpan>();
        var at = 0;
        foreach (Match m in Label().Matches(labelled))
        {
            text.Append(labelled, at, m.Index - at);
            var value = m.Groups["text"].Value;
            spans.Add(new GoldSpan(text.Length, value.Length, m.Groups["type"].Value,
                m.Groups["id"].Success ? m.Groups["id"].Value : null, value));
            text.Append(value);
            at = m.Index + m.Length;
        }

        text.Append(labelled, at, labelled.Length - at);
        var plain = text.ToString();
        if (plain.Contains('{') || plain.Contains('}'))
            throw new FormatException("A brace outside a well-formed {TYPE:text} label.");
        return (plain, spans);
    }

    /// <summary>Adds one text's result to <paramref name="score"/>.</summary>
    public static void Add(Score score, string name, string text, IReadOnlyList<GoldSpan> gold, IReadOnlyList<MaskSpan> masked)
    {
        var covered = new bool[text.Length];
        foreach (var span in masked)
            Array.Fill(covered, true, span.Start, span.Length);

        var leaked = new HashSet<GoldSpan>();
        foreach (var g in gold)
        {
            score.Gold[g.Type] = score.Gold.GetValueOrDefault(g.Type) + 1;
            if (Enumerable.Range(g.Start, g.Length).Any(i => char.IsLetterOrDigit(text[i]) && !covered[i]))
            {
                leaked.Add(g);
                score.Leaked[g.Type] = score.Leaked.GetValueOrDefault(g.Type) + 1;
                score.Leaks.Add($"{name}: {g.Text} ({g.Type})");
            }
        }

        score.Masked += masked.Count;
        foreach (var span in masked.Where(s => !gold.Any(g => g.Start < s.End && s.Start < g.End)))
            score.OverMasked.Add($"{name}: {span.Text} ({span.Type}, {span.Source})");

        // Each fully masked person mention's placeholder: that of the masked span covering most of it.
        var placeholders = Masker.Placeholders(masked);
        var people = new List<(string Id, string Placeholder, string Text)>();
        foreach (var g in gold.Where(g => g.Type == "PER" && g.Id != null && !leaked.Contains(g)))
        {
            var best = Enumerable.Range(0, masked.Count)
                .MaxBy(i => Math.Min(g.End, masked[i].End) - Math.Max(g.Start, masked[i].Start));
            people.Add((g.Id!, placeholders[best], g.Text));
        }

        for (var i = 0; i < people.Count; i++)
        {
            for (var j = i + 1; j < people.Count; j++)
            {
                var samePerson = people[i].Id == people[j].Id;
                var samePlaceholder = people[i].Placeholder == people[j].Placeholder;
                if (samePerson && samePlaceholder)
                {
                    score.SamePersonSamePlaceholder++;
                }
                else if (samePerson)
                {
                    score.SamePersonOtherPlaceholder++;
                }
                else if (samePlaceholder)
                {
                    score.OtherPersonSamePlaceholder++;
                    score.Joined.Add($"{name}: {people[i].Text} ({people[i].Id}) and {people[j].Text} ({people[j].Id}) as {people[i].Placeholder}");
                }
            }
        }

        foreach (var person in people.GroupBy(p => p.Id).Where(g => g.Count() >= 2))
        {
            score.People++;
            var own = person.Select(p => p.Placeholder).Distinct().ToList();
            if (own.Count > 1)
                score.Split.Add($"{name}: {person.Key}: " + string.Join(", ", person.Select(p => $"{p.Text} {p.Placeholder}")));
            if (own.Count == 1 && people.All(p => p.Id == person.Key || p.Placeholder != own[0]))
                score.PeopleWithOwnPlaceholder++;
        }
    }

    /// <summary>A table of the runs' scores side by side, one column per run.</summary>
    public static string Report(IReadOnlyList<(string Name, Score Score)> runs)
    {
        var report = new StringBuilder();
        var types = runs[0].Score.Gold.Keys.Order(StringComparer.Ordinal).ToList();
        report.Append("| | ").AppendJoin(" | ", runs.Select(r => r.Name)).AppendLine(" |");
        report.Append("|---|").AppendJoin("", runs.Select(_ => "---|")).AppendLine();
        Row("leaked spans (all types)", s => $"{s.Leaked.Values.Sum()} of {s.Gold.Values.Sum()}");
        foreach (var type in types)
            Row($"leaked {type}", s => $"{s.Leaked.GetValueOrDefault(type)} of {s.Gold[type]}");
        Row("masked spans overlapping no label", s => $"{s.OverMasked.Count} of {s.Masked}");
        Row("same person, same placeholder (pairs)", s => Ratio(s.SamePersonSamePlaceholder,
            s.SamePersonSamePlaceholder + s.SamePersonOtherPlaceholder));
        Row("placeholder shared by different people (pairs)", s => s.OtherPersonSamePlaceholder.ToString());
        Row("people with one placeholder of their own", s => $"{s.PeopleWithOwnPlaceholder} of {s.People}");
        return report.ToString();

        void Row(string label, Func<Score, string> cell) =>
            report.Append("| ").Append(label).Append(" | ").AppendJoin(" | ", runs.Select(r => cell(r.Score))).AppendLine(" |");

        static string Ratio(int a, int b) => b == 0 ? "-" : $"{a} of {b} ({100.0 * a / b:0}%)";
    }

    [GeneratedRegex(@"\{(?<type>[A-Z]+)(?:#(?<id>[\w-]+))?:(?<text>[^{}]+)\}")]
    private static partial Regex Label();
}
