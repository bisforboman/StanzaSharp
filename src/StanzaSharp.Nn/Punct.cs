using System.Text.RegularExpressions;

namespace StanzaSharp.Nn;

/// <summary>
/// common/utils.py <c>simplify_punct</c>, which the tagger (pos/data.py load_doc) and the dependency parser apply to
/// their words: runs like "?!?" or "!!" become a single "?" or "!" (QUESTION_RE / EXCLAM_RE, including full-width and
/// small forms).
/// </summary>
internal static class Punct
{
    private const string QuestionMarks = "?？︖﹖⁇", AllMarks = QuestionMarks + "!！︕﹗‼";
    private static readonly Regex Question = new($"^[{QuestionMarks}][{AllMarks}]+$");
    private static readonly Regex Exclam = new($"^[!！︕﹗‼][{AllMarks}]+$");

    /// <summary>The word with a run of question or exclamation marks replaced by one "?" or "!"; other words unchanged.</summary>
    public static string Simplify(string word) => Exclam.Replace(Question.Replace(word, "?"), "!");
}
