using System.Text.RegularExpressions;
using TasmanianDevil;

namespace StanzaSharp.TasmanianDevil;

/// <summary>A span of <see cref="StanzaNerRecognizer"/>'s results, before and after name linking.</summary>
/// <param name="Start">The UTF-16 offset of the span's first character.</param>
/// <param name="End">The offset just past the span.</param>
/// <param name="Type">The TasmanianDevil entity type (<c>PERSON</c>, <c>LOCATION</c>, ...).</param>
/// <param name="StanzaType">The NER model's own type (<c>PER</c>, <c>GPE</c>, ...).</param>
/// <param name="Person">The person a <c>PERSON</c> span is a mention of, unique within the text; null before linking.</param>
/// <param name="Added">Whether linking found the span (another occurrence of a name NER tagged elsewhere).</param>
internal sealed record LinkedSpan(int Start, int End, string Type, string StanzaType, string? Person = null, bool Added = false)
{
    public bool IsPerson => Type == PiiEntities.Person;
}

/// <summary>
/// Links the mentions of each person in a text and finds the ones NER missed (<see cref="StanzaNerRecognizer.LinkNames"/>):
/// every capitalized or all-caps occurrence of a name part NER tagged somewhere (with a genitive <c>-s</c>, after an
/// initial: <c>K. Lindqvist</c>) is a mention too, and the mentions of one person share one key.
/// </summary>
/// <remarks>
/// <para>People are told apart by their full names: a mention with two or more name parts (<c>Karin Lindqvist</c>, or an
/// all-caps header <c>LINDQVIST, KARIN</c>) is one person; a shorter mention (<c>Karin</c>, <c>Karins</c>,
/// <c>K. Lindqvist</c>) joins the one full name it fits, keeps a key of its own where it fits several (<c>Lindqvists</c>
/// in a family of Lindqvists), and joins the other mentions of the same name where it fits none. A full name inside a
/// longer one is the same person only with the same first name (Abdirahman Nur is the father of Yusuf Abdirahman Nur),
/// and a first name alone prefers the person it is the first name of.</para>
/// <para>Lowercase occurrences are never added (<c>sten</c>, <c>ek</c>, <c>rosa</c> are words), nor all-caps ones unless
/// NER saw the name in capitals (<c>SOL</c> is the social services act, not Petra Sol), nor occurrences in an e-mail
/// address. Places and organisations are repeated by their exact text; dates are left as NER gave them.</para>
/// <para>Measured on labelled Swedish texts (samples/StanzaSharp.Masking/eval): one placeholder per person for 92% of the
/// pairs of one person's mentions held-out (35% without), about as many leaks, a few more masks a reviewer removes.</para>
/// </remarks>
internal static partial class NameLinker
{
    // Name particles that carry no identity on their own.
    private static readonly HashSet<string> Particles = new(StringComparer.OrdinalIgnoreCase)
    {
        "af", "al", "bin", "da", "de", "del", "der", "di", "du", "el", "ibn", "la", "le", "van", "von",
    };

    // Words that start or end a line next to a name without being one: letter closings, greetings.
    private static readonly HashSet<string> LetterWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "hej", "hejsan", "hälsningar", "hälsning", "mvh", "vh", "mvg", "tack", "regards", "dear", "hi", "hello", "cheers",
        "sincerely", "thanks",
    };

    /// <summary>NER's spans cleaned, persons keyed, plus the mentions NER missed, in text order.</summary>
    public static List<LinkedSpan> Link(string text, IReadOnlyList<LinkedSpan> ner)
    {
        // A one-word piece cut off a longer span at a line break ("Maria Sten\nSamordnare") stays only if it is a name
        // elsewhere or written in capitals (the "KARIN" of a "LINDQVIST, KARIN" header).
        var pieces = ner.Select(s => Clean(text, s).ToList()).ToList();
        var whole = new HashSet<string>(pieces.Where(p => p.Count == 1).SelectMany(p => p).Where(s => s.IsPerson)
            .SelectMany(s => Split(Text(text, s)).Parts), StringComparer.OrdinalIgnoreCase);
        List<LinkedSpan> spans = [.. pieces.SelectMany(p => p.Count == 1
            ? p
            : p.Where(s => Text(text, s) is var t && (t.Contains(' ') || IsAllCaps(t) || IsKnown(t, whole))))];

        // A person span right after an initial ("H. Saleh" tagged as "Saleh") takes the initial in.
        spans = [.. spans.Select(s => s.IsPerson && InitialBefore(text, s.Start) is int at ? s with { Start = at } : s)];

        // A one-word place or organisation that is a person's name elsewhere in the text is that person ("Mikaels
        // arbetsgivare" tagged ORG next to "Mikael Lindqvist").
        var names = new HashSet<string>(spans.Where(s => s.IsPerson).SelectMany(s => Split(Text(text, s)).Parts),
            StringComparer.OrdinalIgnoreCase);
        spans = [.. spans.Select(s => !s.IsPerson && s.Type != PiiEntities.DateTime && Text(text, s) is var t
            && Word().Match(t) is { Success: true } w && w.Length == t.Length && IsKnown(t, names)
                ? s with { Type = PiiEntities.Person }
                : s)];
        var people = spans.Where(s => s.IsPerson).ToList();
        var others = spans.Where(s => !s.IsPerson).ToList();
        var personType = people.Count > 0 ? people[0].StanzaType : "";

        // Name parts, genitives folded onto the base form where both occur.
        var mentions = Mentions(text, people);
        var parts = new HashSet<string>(mentions.SelectMany(m => m.Parts), StringComparer.OrdinalIgnoreCase);
        foreach (var m in mentions)
            m.Parts = [.. m.Parts.Select(p => BaseForm(p, parts))];
        var known = new HashSet<string>(mentions.SelectMany(m => m.Parts), StringComparer.OrdinalIgnoreCase);

        // All caps only for a name NER saw in all caps.
        var shouted = new HashSet<string>(mentions.SelectMany(m => m.Parts).Where(IsAllCaps), StringComparer.OrdinalIgnoreCase);

        // Occurrences NER missed: runs of known parts (an initial may lead), outside every span found so far.
        var occupied = new List<LinkedSpan>(spans);
        foreach (var run in Runs(text, known, shouted))
        {
            if (occupied.Any(s => s.Start < run.End && run.Start < s.End) || InEmailAddress(text, run.Start, run.End))
                continue;
            var span = new LinkedSpan(run.Start, run.End, PiiEntities.Person, personType, Added: true);
            string[] runParts = [.. run.Parts.Select(p => BaseForm(p, known))];
            mentions.Add(new Mention([span]) { Parts = runParts, Initials = run.Initials, First = runParts[0] });
            occupied.Add(span);
        }

        Resolve(text, mentions);
        var result = mentions.SelectMany(m => m.Spans.Select(s => s with { Person = m.Key })).ToList();

        // Places and organisations: the same text again, as a whole word.
        foreach (var entity in others)
        {
            result.Add(entity);
            var value = Text(text, entity);
            if (entity.Type is not (PiiEntities.Location or PiiEntities.Organization) || value.Length < 3 || !char.IsUpper(value[0]))
                continue;
            for (var at = text.IndexOf(value, StringComparison.Ordinal); at >= 0; at = text.IndexOf(value, at + 1, StringComparison.Ordinal))
            {
                var end = at + value.Length;
                if ((at > 0 && IsWordChar(text[at - 1])) || (end < text.Length && IsWordChar(text[end]))
                    || occupied.Any(s => s.Start < end && at < s.End))
                {
                    continue;
                }

                var span = entity with { Start = at, End = end, Added = true };
                result.Add(span);
                occupied.Add(span);
            }
        }

        return [.. result.OrderBy(s => s.Start)];
    }

    // NER's span cut at line breaks (Stanza reads a single newline as a space, so "KARIN\nPersonnummer:" can be one
    // span), tabs and column gaps ("Ahmed Yusuf      K. Lind-Sjöberg"), and a person's where a character no name has
    // begins ("Jonas Ek <jonas.ek@..."). Pieces that are a form's label ("Personnummer:") or a letter's closing ("Mvh")
    // are dropped.
    private static IEnumerable<LinkedSpan> Clean(string text, LinkedSpan span)
    {
        var end = span.End;
        if (span.IsPerson && text.AsSpan(span.Start, span.End - span.Start).IndexOfAny(NotInNames) is int cut and >= 0)
            end = span.Start + cut;

        var at = span.Start;
        while (at < end)
        {
            var gap = Gap().Match(text, at, end - at);
            var stop = gap.Success ? gap.Index : end;
            var (a, b) = (at, stop);
            while (a < b && !char.IsLetterOrDigit(text[a]))
                a++;
            while (b > a && !char.IsLetterOrDigit(text[b - 1]) && text[b - 1] != '.')
                b--;
            var label = b < text.Length && text[b] == ':';
            if (a < b && !label && !LetterWords.Contains(text[a..b]))
                yield return span with { Start = a, End = b };
            at = gap.Success ? gap.Index + gap.Length : end;
        }
    }

    private static readonly System.Buffers.SearchValues<char> NotInNames =
        System.Buffers.SearchValues.Create("<>()[]{}@|/\\0123456789");

    // NER's person spans as mentions; an all-caps "SURNAME, FIRST NAME" header that NER split in two becomes one.
    private static List<Mention> Mentions(string text, List<LinkedSpan> people)
    {
        var mentions = new List<Mention>();
        foreach (var span in people.OrderBy(s => s.Start))
        {
            var value = Text(text, span);
            var (parts, initials) = Split(value);
            if (mentions.Count > 0 && mentions[^1].Spans is [var last] && IsAllCaps(Text(text, last)) && IsAllCaps(value)
                && last.End <= span.Start && text[last.End..span.Start] is ", " or "," or ",\n" or " ")
            {
                mentions[^1].Spans.Add(span);
                mentions[^1].First = parts.Length > 0 ? parts[0] : mentions[^1].First;
                mentions[^1].Parts = [.. mentions[^1].Parts, .. parts];
                continue;
            }

            mentions.Add(new Mention([span]) { Parts = parts, Initials = initials, First = parts.Length > 0 ? parts[0] : "" });
        }

        return mentions;
    }

    // Full names are people; shorter mentions join the one full name that has all their parts and initials.
    private static void Resolve(string text, List<Mention> mentions)
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

            var fits = Enumerable.Range(0, people.Count).Where(j => merged[j] == j && Fits(m, people[j])).ToList();
            var byFirst = fits.Where(j => m.Parts.Length == 1 && m.Initials.Length == 0
                && string.Equals(firsts[j], m.Parts[0], StringComparison.OrdinalIgnoreCase)).ToList();
            if (fits.Count > 1 && byFirst.Count == 1)
                fits = byFirst;
            m.Key = fits.Count switch
            {
                _ when m.Parts.Length == 0 => "text-" + Text(text, m.Spans[0]).ToUpperInvariant(),
                1 => "person-" + fits[0],
                0 => "name-" + string.Join(' ', m.Parts).ToUpperInvariant(),
                _ => "ambiguous-" + string.Join(' ', m.Initials.Select(c => c + ".").Concat(m.Parts)).ToUpperInvariant(),
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

    private static bool IsKnown(string word, HashSet<string> names) =>
        names.Contains(word) || (word.Length > 2 && word[^1] is 's' or 'S' && names.Contains(word[..^1]));

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

    // Whether [start, end) lies in a whitespace-delimited token holding an '@' (Karin.Lindqvist@example.se).
    private static bool InEmailAddress(string text, int start, int end)
    {
        var a = start;
        while (a > 0 && !char.IsWhiteSpace(text[a - 1]) && text[a - 1] is not ('<' or '(' or '['))
            a--;
        var b = end;
        while (b < text.Length && !char.IsWhiteSpace(text[b]) && text[b] is not ('>' or ')' or ']'))
            b++;
        return text.AsSpan(a, b - a).Contains('@');
    }

    private static bool IsWordChar(char c) => char.IsLetterOrDigit(c) || char.GetUnicodeCategory(c) is
        System.Globalization.UnicodeCategory.NonSpacingMark or System.Globalization.UnicodeCategory.SpacingCombiningMark;

    private static bool IsAllCaps(string s) => s.Any(char.IsLetter) && !s.Any(char.IsLower);

    private static string Text(string text, LinkedSpan span) => text[span.Start..span.End];

    private sealed class Mention(List<LinkedSpan> spans)
    {
        public List<LinkedSpan> Spans { get; } = spans;

        public string[] Parts { get; set; } = [];

        public char[] Initials { get; set; } = [];

        // The given name: the first part, or the first after the comma of a "LINDQVIST, KARIN" header.
        public string First { get; set; } = "";

        public int? Person { get; set; }

        public string Key { get; set; } = "";
    }

    // Where a span is cut: a line break, a tab, or a column gap of two or more spaces.
    [GeneratedRegex(@"\n|\t| {2,}")]
    private static partial Regex Gap();

    // A word: letters, joined by hyphens or apostrophes (Britt-Marie, O'Brien).
    [GeneratedRegex(@"(?<![\p{L}\p{M}\p{Nd}])\p{L}[\p{L}\p{M}]*(?:[-'’]\p{L}[\p{L}\p{M}]*)*(?![\p{L}\p{M}\p{Nd}])")]
    private static partial Regex Word();
}
