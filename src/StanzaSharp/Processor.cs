namespace StanzaSharp;

/// <summary>
/// The processor names that <see cref="PipelineOptions.Processors"/> and
/// <see cref="ModelDownloader.DownloadAsync(string, string, IProgress{string}?, CancellationToken)"/> accept, as
/// constants, so a typo fails at compile time. Join them with commas. The order doesn't matter: the pipeline always
/// runs them in Stanza's order (<see cref="Pipeline.AllProcessors"/>).
/// </summary>
/// <example>
/// <code>
/// var options = new PipelineOptions { Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos}" };
/// </code>
/// </example>
public static class Processor
{
    /// <summary>Tokenization and sentence splitting. Every other processor needs it.</summary>
    public const string Tokenize = "tokenize";

    /// <summary>Multi-word token expansion (e.g. "don't" → "do" + "n't"). Needs tokenize.</summary>
    public const string Mwt = "mwt";

    /// <summary>Part-of-speech tags (UPOS, XPOS) and morphological features. Needs tokenize and mwt.</summary>
    public const string Pos = "pos";

    /// <summary>Lemmas. Needs tokenize, mwt and pos.</summary>
    public const string Lemma = "lemma";

    /// <summary>Constituency trees (<c>default</c> package only). Needs tokenize, mwt and pos.</summary>
    public const string Constituency = "constituency";

    /// <summary>Dependency parses (each word's head and relation). Needs tokenize, mwt, pos and lemma.</summary>
    public const string Depparse = "depparse";

    /// <summary>Sentence sentiment: negative, neutral or positive. Needs tokenize.</summary>
    public const string Sentiment = "sentiment";

    /// <summary>Named entities. Needs tokenize.</summary>
    public const string Ner = "ner";
}
