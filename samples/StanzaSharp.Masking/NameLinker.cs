using System.Text.RegularExpressions;

namespace StanzaSharp.Masking;

/// <summary>
/// Links the mentions of each person in a document and finds the ones NER missed: every capitalized or all-caps
/// occurrence of a name part NER tagged somewhere (with a genitive <c>-s</c>, after an initial: <c>K. Lindqvist</c>) is
/// masked too, and the mentions of one person share one entity key, so they get one placeholder.
/// </summary>
/// <remarks>
/// <para>People are told apart by their full names: a mention with two or more name parts (<c>Karin Lindqvist</c>, or an
/// all-caps header <c>LINDQVIST, KARIN</c>) is one person; a shorter mention (<c>Karin</c>, <c>Karins</c>,
/// <c>K. Lindqvist</c>) joins the one full name it fits, keeps a key of its own where it fits several (<c>Lindqvists</c>
/// in a family of Lindqvists), and joins the other mentions of the same name where it fits none.</para>
/// <para>Places and organisations are propagated by their exact text only. Lowercase occurrences are never added: too
/// many Swedish names are also words (<c>sten</c>, <c>ek</c>, <c>rosa</c>).</para>
/// </remarks>
internal static partial class NameLinker
{
    // Name particles that carry no identity on their own.
    private static readonly HashSet<string> Particles = new(StringComparer.OrdinalIgnoreCase)
    {
        "af", "al", "bin", "da", "de", "del", "der", "di", "du", "el", "ibn", "la", "le", "van", "von",
    };

    /// <summary>The NER spans with entity keys, plus the propagated mentions that overlap none of <paramref name="taken"/>.</summary>
    public static List<MaskSpan> Link(string text, List<MaskSpan> ner, IReadOnlyList<MaskSpan> taken)
    {
        // A one-word piece cut off a longer span at a line break ("Maria Sten\nSamordnare") stays only if it is a name
        // elsewhere or written in capitals (the "KARIN" of a "LINDQVIST, KARIN" header).
        var pieces = ner.Select(s => Clean(text, s, taken).ToList()).ToList();
        var whole = new HashSet<string>(pieces.Where(p => p.Count == 1).SelectMany(p => p)
            .Where(s => s.Type == "PER").SelectMany(s => Split(s.Text).Parts), StringComparer.OrdinalIgnoreCase);
        ner = [.. pieces.SelectMany(p => p.Count == 1 ? p : p.Where(s => s.Text.Contains(' ') || IsAllCaps(s.Text)
            || whole.Contains(s.Text) || (s.Text[^1] is 's' && whole.Contains(s.Text[..^1]))))];

        // A person span right after an initial ("H. Saleh" tagged as "Saleh") takes the initial in.
        ner = [.. ner.Select(s => s.Type == "PER" && InitialBefore(text, s.Start) is int at
            ? s with { Start = at, Length = s.End - at, Text = text[at..s.End] }
            : s)];

        // A one-word place or organisation that is a person's name elsewhere in the text is that person ("Mikaels
        // arbetsgivare" tagged ORG next to "Mikael Lindqvist").
        var names = new HashSet<string>(ner.Where(s => s.Type == "PER").SelectMany(s => Split(s.Text).Parts),
            StringComparer.OrdinalIgnoreCase);
        ner = [.. ner.Select(s => s.Type != "PER" && Word().Match(s.Text) is { Success: true } w && w.Length == s.Length
            && (names.Contains(s.Text) || (s.Text[^1] is 's' && names.Contains(s.Text[..^1])))
                ? s with { Type = "PER" }
                : s)];
        var people = ner.Where(s => s.Type == "PER").ToList();
        var others = ner.Where(s => s.Type != "PER").ToList();

        // Name parts, genitives folded onto the base form where both occur.
        var mentions = Mentions(text, people);
        var parts = new HashSet<string>(mentions.SelectMany(m => m.Parts), StringComparer.OrdinalIgnoreCase);
        foreach (var m in mentions)
            m.Parts = [.. m.Parts.Select(p => BaseForm(p, parts))];
        var known = new HashSet<string>(mentions.SelectMany(m => m.Parts), StringComparer.OrdinalIgnoreCase);

        // All caps only for a name NER saw in all caps: SOL is the law, not Petra Sol.
        var shouted = new HashSet<string>(mentions.SelectMany(m => m.Parts).Where(IsAllCaps), StringComparer.OrdinalIgnoreCase);

        // Occurrences NER missed: runs of known parts (an initial may lead), not inside any span found so far.
        var occupied = new List<MaskSpan>(taken);
        occupied.AddRange(ner);
        foreach (var run in Runs(text, known, shouted))
        {
            if (occupied.Any(s => s.Start < run.End && run.Start < s.End))
                continue;
            var mention = new Mention([new MaskSpan(run.Start, run.End - run.Start, "PER", "name-link", text[run.Start..run.End])])
            {
                Parts = [.. run.Parts.Select(p => BaseForm(p, known))],
                Initials = run.Initials,
            };
            mentions.Add(mention);
            occupied.Add(mention.Spans[0]);
        }

        Resolve(mentions);
        var result = mentions.SelectMany(m => m.Spans.Select(s => s with { Entity = m.Key })).ToList();

        // Places and organisations: the same text again, as a whole word.
        foreach (var entity in others.ToList())
        {
            result.Add(entity with { Entity = entity.Text });
            if (entity.Text.Length < 3 || !char.IsUpper(entity.Text[0]))
                continue;
            foreach (Match m in Regex.Matches(text, $@"(?<![\p{{L}}\p{{M}}\p{{Nd}}]){Regex.Escape(entity.Text)}(?![\p{{L}}\p{{M}}\p{{Nd}}])"))
            {
                if (occupied.Any(s => s.Start < m.Index + m.Length && m.Index < s.End))
                    continue;
                var span = new MaskSpan(m.Index, m.Length, entity.Type, "name-link", m.Value, entity.Text);
                result.Add(span);
                occupied.Add(span);
            }
        }

        return result;
    }

    // Words that start or end a line next to a name without being one: letter closings, greetings.
    private static readonly HashSet<string> LetterWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "hej", "hejsan", "hälsningar", "mvh", "vh", "mvg", "tack", "hälsning",
    };

    // NER's span cut at line breaks (Stanza reads a single newline as a space, so "KARIN\nPersonnummer:" can be one
    // span), tabs and column gaps ("Ahmed Yusuf      K. Lind-Sjöberg" under a decision) and where it runs into a pattern's match ("Jonas Ek <jonas.ek@..."). Pieces that are a form's label
    // ("Personnummer:") or a letter's closing ("Mvh") are dropped.
    private static IEnumerable<MaskSpan> Clean(string text, MaskSpan span, IReadOnlyList<MaskSpan> patterns)
    {
        var start = span.Start;
        var end = span.End;
        foreach (var p in patterns.Where(p => p.Start < end && start < p.End && !(p.Start <= start && end <= p.End)))
        {
            if (p.Start > start)
                end = Math.Min(end, p.Start);
            else
                start = Math.Max(start, p.End);
        }

        var at = start;
        while (at < end)
        {
            var gap = Gap().Match(text, at, end - at);
            var stop = gap.Success ? gap.Index : end;
            var (a, b) = (at, stop);
            while (a < b && !char.IsLetterOrDigit(text[a]))
                a++;
            while (b > a && !char.IsLetterOrDigit(text[b - 1]) && text[b - 1] != '.')
                b--;
            var piece = text[a..b];
            var label = b < text.Length && text[b] == ':';
            if (piece.Length > 0 && !label && !LetterWords.Contains(piece))
                yield return span with { Start = a, Length = b - a, Text = piece };
            at = gap.Success ? gap.Index + gap.Length : end;
        }
    }

    // NER's person spans as mentions; an all-caps "SURNAME, FIRST NAME" header that NER split in two becomes one.
    private static List<Mention> Mentions(string text, List<MaskSpan> people)
    {
        var mentions = new List<Mention>();
        foreach (var span in people.OrderBy(s => s.Start))
        {
            var (parts, initials) = Split(span.Text);
            if (mentions.Count > 0 && mentions[^1].Spans is [var last] && IsAllCaps(last.Text) && IsAllCaps(span.Text)
                && text[last.End..span.Start] is ", " or "," or ",\n" or " ")
            {
                mentions[^1].Spans.Add(span);
                mentions[^1].Parts = [.. mentions[^1].Parts, .. parts];
                continue;
            }

            mentions.Add(new Mention([span]) { Parts = parts, Initials = initials });
        }

        return mentions;
    }

    // Full names are people; shorter mentions join the one full name that has all their parts and initials.
    private static void Resolve(List<Mention> mentions)
    {
        var people = new List<HashSet<string>>();
        var firsts = new List<string>();
        foreach (var m in mentions.Where(m => m.Parts.Length >= 2))
        {
            var set = new HashSet<string>(m.Parts, StringComparer.OrdinalIgnoreCase);
            var same = people.FindIndex(p => p.SetEquals(set));
            m.Person = same >= 0 ? same : people.Count;
            if (same < 0)
            {
                people.Add(set);
                firsts.Add(m.First);
            }
        }

        // A full name inside one longer full name with the same first name (Noah Korhonen in Noah Andersson Korhonen) is
        // that person; with another first name it is not (Abdirahman Nur is the father of Yusuf Abdirahman Nur).
        var merged = Enumerable.Range(0, people.Count).ToArray();
        for (var i = 0; i < people.Count; i++)
        {
            var supersets = Enumerable.Range(0, people.Count)
                .Where(j => j != i && people[i].IsProperSubsetOf(people[j])
                    && string.Equals(firsts[i], firsts[j], StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (supersets.Count == 1)
                merged[i] = supersets[0];
        }

        foreach (var m in mentions)
        {
            if (m.Person is int p)
            {
                m.Key = "person-" + merged[p];
                continue;
            }

            var fits = Enumerable.Range(0, people.Count)
                .Where(j => merged[j] == j && Fits(m, people[j]))
                .ToList();

            // A first name alone is the one person it is the first name of (Yusuf: Yusuf Abdirahman Nur, not Ahmed Yusuf).
            var byFirst = fits.Where(j => m.Parts.Length == 1 && m.Initials.Length == 0
                && string.Equals(firsts[j], m.Parts[0], StringComparison.OrdinalIgnoreCase)).ToList();
            if (fits.Count > 1 && byFirst.Count == 1)
                fits = byFirst;
            var name = string.Join(' ', m.Initials.Select(c => c + ".").Concat(m.Parts)).ToUpperInvariant();
            m.Key = fits.Count switch
            {
                _ when m.Parts.Length == 0 => "text-" + m.Spans[0].Text.ToUpperInvariant(),
                1 => "person-" + fits[0],
                0 => "name-" + string.Join(' ', m.Parts).ToUpperInvariant(),
                _ => "ambiguous-" + name,
            };
        }
    }

    private static bool Fits(Mention m, HashSet<string> person) =>
        m.Parts.Length > 0
        && m.Parts.All(person.Contains)
        && m.Initials.All(c => person.Any(p => !m.Parts.Contains(p, StringComparer.OrdinalIgnoreCase)
            && char.ToUpperInvariant(p[0]) == c));

    // "Karins" → "Karin" where "Karin" is a name part too.
    private static string BaseForm(string part, HashSet<string> parts) =>
        part.Length > 2 && part[^1] is 's' or 'S' && parts.Contains(part[..^1]) ? part[..^1] : part;

    private static (string[] Parts, char[] Initials) Split(string name)
    {
        var parts = new List<string>();
        var initials = new List<char>();
        foreach (Match m in Word().Matches(name))
        {
            if (m.Length == 1)
            {
                if (char.IsUpper(m.Value[0]))
                    initials.Add(m.Value[0]);
            }
            else if (!Particles.Contains(m.Value))
            {
                parts.Add(m.Value);
            }
        }

        return ([.. parts], [.. initials]);
    }

    // Runs of capitalized known name parts (or their genitive) separated by single spaces, an initial ("K. ") allowed
    // in front.
    private static IEnumerable<(int Start, int End, string[] Parts, char[] Initials)> Runs(string text, HashSet<string> known,
        HashSet<string> shouted)
    {
        var words = Word().Matches(text).Where(m => IsName(m.Value, known, shouted)).ToList();
        for (var i = 0; i < words.Count;)
        {
            var start = words[i].Index;
            var end = words[i].Index + words[i].Length;
            var parts = new List<string> { words[i].Value };
            var j = i + 1;
            while (j < words.Count && words[j].Index == end + 1 && text[end] == ' ')
            {
                parts.Add(words[j].Value);
                end = words[j].Index + words[j].Length;
                j++;
            }

            char[] initials = [];
            if (InitialBefore(text, start) is int at)
            {
                initials = [text[at]];
                start = at;
            }

            yield return (start, end, [.. parts], initials);
            i = j;
        }
    }

    private static bool IsName(string word, HashSet<string> known, HashSet<string> shouted)
    {
        if (word.Length < 2 || !char.IsUpper(word[0]) || Particles.Contains(word))
            return false;
        var name = known.Contains(word) ? word : word[^1] is 's' or 'S' && known.Contains(word[..^1]) ? word[..^1] : null;
        return name != null && (!IsAllCaps(word) || shouted.Contains(name));
    }

    // The start of an initial right before a name ("K. " in front of "Lindqvist"), or null.
    private static int? InitialBefore(string text, int start) =>
        start >= 3 && text[start - 1] == ' ' && text[start - 2] == '.' && char.IsUpper(text[start - 3])
        && (start == 3 || !(char.IsLetterOrDigit(text[start - 4]) || text[start - 4] == '.'))
            ? start - 3
            : null;

    private static bool IsAllCaps(string s) => s.Any(char.IsLetter) && !s.Any(char.IsLower);

    private sealed class Mention(List<MaskSpan> spans)
    {
        public List<MaskSpan> Spans { get; } = spans;

        public string[] Parts { get; set; } = [];

        public char[] Initials { get; set; } = [];

        public int? Person { get; set; }

        // The given name: the first part, or the first after the comma of a "LINDQVIST, KARIN" header.
        public string First => Spans.Count > 1 && Parts.Length > 1 ? Parts[Split(Spans[0].Text).Parts.Length] : Parts[0];

        public string Key { get; set; } = "";
    }

    // Where a span is cut: a line break, a tab, or a column gap of two or more spaces.
    [GeneratedRegex(@"\n|\t| {2,}")]
    private static partial Regex Gap();

    // A word: letters, joined by hyphens or apostrophes (Britt-Marie, O'Brien).
    [GeneratedRegex(@"(?<![\p{L}\p{M}\p{Nd}])\p{L}[\p{L}\p{M}]*(?:[-'’]\p{L}[\p{L}\p{M}]*)*(?![\p{L}\p{M}\p{Nd}])")]
    private static partial Regex Word();
}
