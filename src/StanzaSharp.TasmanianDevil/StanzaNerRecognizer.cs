using TasmanianDevil;
using TasmanianDevil.Analyzer;

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

    /// <inheritdoc />
    public override IReadOnlyList<RecognizerResult> Analyze(string text, IReadOnlyList<string> entities)
    {
        if (string.IsNullOrWhiteSpace(text))
            return [];
        var requested = entities is { Count: > 0 } ? new HashSet<string>(entities, StringComparer.Ordinal) : null;
        if (requested != null && !SupportedEntities.Any(requested.Contains))
            return [];

        var results = new List<RecognizerResult>();
        foreach (var entity in _pipeline.Process(text).Entities)
        {
            if (!_typeMap.TryGetValue(entity.Type, out var type) || (requested != null && !requested.Contains(type)))
                continue;
            if (entity.StartChar is not int start || entity.EndChar is not int end || start < 0 || end > text.Length || start >= end)
                continue;
            results.Add(new RecognizerResult(type, start, end, Score, new Dictionary<string, object>
            {
                [RecognizerResult.RecognizerNameKey] = Name,
                [RecognizerResult.RecognizerIdentifierKey] = Id,
                ["stanzaType"] = entity.Type,
            }));
        }

        return results;
    }

    private static List<string> SupportedEntitiesOf(IReadOnlyDictionary<string, string> typeMap)
    {
        ArgumentNullException.ThrowIfNull(typeMap);
        return [.. typeMap.Values.Distinct(StringComparer.Ordinal)];
    }
}
