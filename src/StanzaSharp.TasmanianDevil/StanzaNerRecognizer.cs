using TasmanianDevil;
using TasmanianDevil.Analyzer;
using TasmanianDevil.Anonymizer.Operators;

namespace StanzaSharp.TasmanianDevil;

/// <summary>
/// A TasmanianDevil recognizer for the entity types regular expressions can't reach (<c>PERSON</c>, <c>LOCATION</c>,
/// <c>ORGANIZATION</c>, <c>DATE_TIME</c>), found by StanzaSharp's NER: Stanza's supervised model for the pipeline's
/// language, in managed code, the same spans on every run.
/// </summary>
/// <remarks>
/// <para>The pipeline's entity types are mapped to TasmanianDevil's by <see cref="DefaultTypeMap"/>, which covers both
/// conventions of Stanza's models: English's OntoNotes types (<c>PERSON</c>, <c>ORG</c>, <c>GPE</c>, <c>LOC</c>,
/// <c>FAC</c>, <c>DATE</c>, <c>TIME</c>) and the <c>PER</c>, <c>LOC</c>, <c>ORG</c> (and Swedish <c>TME</c>) of most
/// other languages. Types it doesn't map (amounts, works of art, <c>MISC</c>, ...) are left out.</para>
/// <para>Stanza's tagger gives no confidence, so every span gets <see cref="Score"/> (0.85 by default, the score Python
/// Presidio's spaCy recognizer gives). Results carry the recognizer's name and id and the model's own type
/// (<c>stanzaType</c>) as metadata.</para>
/// <para>Register one recognizer per language, each with its own pipeline (<c>Processors = "tokenize,ner"</c> is all it
/// needs). Each analyzed text is one pipeline call (TasmanianDevil's batch APIs call recognizers per text). The
/// recognizer doesn't dispose the pipeline; <see cref="Pipeline"/>'s Process is thread-safe.</para>
/// <para>With <see cref="LinkNames"/>, the mentions of each person are linked across the text and the ones NER missed
/// are added; <see cref="PersonPlaceholders"/> then gives each person a numbered placeholder of their own.</para>
/// </remarks>
public sealed class StanzaNerRecognizer : EntityRecognizer
{
    private readonly Pipeline _pipeline;
    private readonly IReadOnlyDictionary<string, string> _typeMap;

    /// <summary>Creates a recognizer that tags <paramref name="supportedLanguage"/> text with <paramref name="pipeline"/>.</summary>
    /// <param name="pipeline">A loaded pipeline with the <c>ner</c> processor, of <paramref name="supportedLanguage"/>.</param>
    /// <param name="supportedLanguage">The analysis language TasmanianDevil selects this recognizer for (<c>en</c>, <c>sv</c>, ...).</param>
    /// <param name="typeMap">Stanza entity type → TasmanianDevil entity type; null is <see cref="DefaultTypeMap"/>.</param>
    /// <param name="score">The confidence every span gets, from 0 to 1.</param>
    public StanzaNerRecognizer(Pipeline pipeline, string supportedLanguage = "en", IReadOnlyDictionary<string, string>? typeMap = null,
        double score = 0.85)
        : base(SupportedEntitiesOf(typeMap ?? DefaultTypeMap), name: nameof(StanzaNerRecognizer), supportedLanguage: supportedLanguage)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentOutOfRangeException.ThrowIfLessThan(score, MinScore);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(score, MaxScore);
        if (!pipeline.HasNer)
            throw new ArgumentException("The pipeline has no ner processor; load it with Processors = \"tokenize,ner\".", nameof(pipeline));
        _pipeline = pipeline;
        _typeMap = typeMap ?? DefaultTypeMap;
        Score = score;
    }

    /// <summary>The mapping from Stanza's entity types (all of its NER models' conventions) to TasmanianDevil's.</summary>
    public static IReadOnlyDictionary<string, string> DefaultTypeMap { get; } = new Dictionary<string, string>
    {
        ["PERSON"] = PiiEntities.Person,
        ["PER"] = PiiEntities.Person,
        ["ORG"] = PiiEntities.Organization,
        ["GPE"] = PiiEntities.Location,
        ["LOC"] = PiiEntities.Location,
        ["FAC"] = PiiEntities.Location,
        ["DATE"] = PiiEntities.DateTime,
        ["TIME"] = PiiEntities.DateTime,
        ["TME"] = PiiEntities.DateTime,
    };

    /// <summary>The confidence every span gets.</summary>
    public double Score { get; }

    /// <summary>Whether to link each person's mentions across the analyzed text and add the ones NER missed. Off by default.</summary>
    /// <remarks>
    /// <para>Every capitalized or all-caps occurrence of a name part NER tagged somewhere in the text is a mention too,
    /// with its genitive and an initial in front (<c>Karins</c>, <c>K. Lindqvist</c>); lowercase words never are
    /// (<c>sten</c>, <c>ek</c>, <c>rosa</c> are also words), and all-caps ones only for a name NER saw in capitals. Each
    /// <c>PERSON</c> result carries the person it names in <see cref="PersonKey"/>: a full name is a person, a shorter
    /// mention joins the one full name it fits (<c>Karins</c>: Karin Lindqvist), and a surname shared by two people
    /// keeps a key of its own. Added results carry <see cref="LinkedKey"/>.</para>
    /// <para>NER's spans are also cut at line breaks, column gaps and characters no name has (<c>KARIN\nPersonnummer</c>,
    /// <c>Jonas Ek &lt;jonas.ek@...</c>), a one-word organisation or place that is a person's name in the text becomes
    /// that person (<c>Mikaels arbetsgivare</c>), and organisations and places are found again wherever their exact text
    /// recurs.</para>
    /// <para>On labelled Swedish texts it gave one placeholder per person for 92% of the pairs of one person's mentions
    /// (35% without), with about as many spans missed and a few more found that a reviewer removes
    /// (samples/StanzaSharp.Masking/eval/README.md in the repository).</para>
    /// </remarks>
    public bool LinkNames { get; init; }

    /// <summary>
    /// The metadata key of the person a <c>PERSON</c> result names, with <see cref="LinkNames"/>: a string, the same for
    /// every mention of one person in the analyzed text.
    /// </summary>
    public const string PersonKey = "stanzaPerson";

    /// <summary>The metadata key marking a result <see cref="LinkNames"/> added (<see langword="true"/>).</summary>
    public const string LinkedKey = "stanzaLinked";

    /// <inheritdoc />
    public override IReadOnlyList<RecognizerResult> Analyze(string text, IReadOnlyList<string> entities)
    {
        if (string.IsNullOrWhiteSpace(text))
            return [];
        var requested = entities is { Count: > 0 } ? new HashSet<string>(entities, StringComparer.Ordinal) : null;
        if (requested != null && !SupportedEntities.Any(requested.Contains))
            return [];

        var spans = new List<LinkedSpan>();
        foreach (var entity in _pipeline.Process(text).Entities)
        {
            if (!_typeMap.TryGetValue(entity.Type, out var type))
                continue;
            if (entity.StartChar is not int start || entity.EndChar is not int end || start < 0 || end > text.Length || start >= end)
                continue;
            spans.Add(new LinkedSpan(start, end, type, entity.Type));
        }

        // Linking reads every type (a name tagged ORG may be a person's), so the requested ones are picked after it.
        if (LinkNames)
            spans = NameLinker.Link(text, spans);

        var results = new List<RecognizerResult>();
        foreach (var span in spans)
        {
            if (requested != null && !requested.Contains(span.Type))
                continue;
            var metadata = new Dictionary<string, object>
            {
                [RecognizerResult.RecognizerNameKey] = Name,
                [RecognizerResult.RecognizerIdentifierKey] = Id,
                ["stanzaType"] = span.StanzaType,
            };
            if (span.Person != null)
                metadata[PersonKey] = span.Person;
            if (span.Added)
                metadata[LinkedKey] = true;
            results.Add(new RecognizerResult(span.Type, span.Start, span.End, Score, metadata));
        }

        return results;
    }

    /// <summary>
    /// Anonymization operators that replace each person in <paramref name="text"/> with a numbered placeholder of their
    /// own (<c>&lt;PERSON_1&gt;</c>, <c>&lt;PERSON_2&gt;</c>, ...), numbered in order of first mention, for one call of
    /// TasmanianDevil's <c>AnonymizerEngine.Anonymize</c> on the same text and results.
    /// </summary>
    /// <param name="text">The analyzed text.</param>
    /// <param name="results">The analyzer's results for <paramref name="text"/>.</param>
    /// <param name="operators">Operators for the other entity types (a <c>DEFAULT</c> entry included), kept as they are;
    /// null for none.</param>
    /// <returns><paramref name="operators"/> with the <c>PERSON</c> operator set to the numbered placeholders.</returns>
    /// <remarks>
    /// <para>The default <c>replace</c> operator gives every person the same <c>&lt;PERSON&gt;</c>. With
    /// <see cref="LinkNames"/>, the mentions of one person share a placeholder (<see cref="PersonKey"/>); without it, or
    /// for another recognizer's <c>PERSON</c> results, each distinct text gets one. A span the anonymizer forms
    /// otherwise (merging overlapping results) gets <c>&lt;PERSON&gt;</c>.</para>
    /// <code>
    /// var results = engine.Analyze(text);
    /// var anonymized = new AnonymizerEngine().Anonymize(text, results,
    ///     StanzaNerRecognizer.PersonPlaceholders(text, results), ConflictResolutionStrategy.MergeSimilarOrContained, false);
    /// </code>
    /// <para>The operators are made for this one text: build them again for the next.</para>
    /// </remarks>
    public static IReadOnlyDictionary<string, OperatorConfig> PersonPlaceholders(string text, IReadOnlyList<RecognizerResult> results,
        IReadOnlyDictionary<string, OperatorConfig>? operators = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(results);
        var numbers = new Dictionary<string, int>(StringComparer.Ordinal);
        var placeholders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var result in results.Where(r => r.EntityType == PiiEntities.Person && r.Start >= 0 && r.End <= text.Length
            && r.Start < r.End).OrderBy(r => r.Start))
        {
            var mention = text[result.Start..result.End];
            var person = result.RecognitionMetadata?.TryGetValue(PersonKey, out var key) == true && key is string id
                ? "person:" + id
                : "text:" + mention;
            if (!numbers.TryGetValue(person, out var n))
                numbers[person] = n = numbers.Count + 1;
            placeholders.TryAdd(mention, $"<{PiiEntities.Person}_{n}>");
        }

        Func<string, string> replace = mention => placeholders.GetValueOrDefault(mention, $"<{PiiEntities.Person}>");
        var all = operators == null
            ? new Dictionary<string, OperatorConfig>(StringComparer.Ordinal)
            : new Dictionary<string, OperatorConfig>(operators, StringComparer.Ordinal);
        all[PiiEntities.Person] = new OperatorConfig("custom", new Dictionary<string, object> { [CustomOperator.Lambda] = replace });
        return all;
    }

    private static List<string> SupportedEntitiesOf(IReadOnlyDictionary<string, string> typeMap)
    {
        ArgumentNullException.ThrowIfNull(typeMap);
        return [.. typeMap.Values.Distinct(StringComparer.Ordinal)];
    }
}
