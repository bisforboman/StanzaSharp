namespace StanzaSharp;

/// <summary>
/// Stanza's <c>MultilingualPipeline</c>: the <see cref="LanguageIdentifier"/> decides each text's language, and each
/// language's texts go through that language's <see cref="Pipeline"/> in one bulk call, as
/// <see cref="Pipeline.Process(IEnumerable{string})"/> does. Each returned document's <see cref="Document.Lang"/> is set.
/// </summary>
/// <remarks>
/// Unlike Stanza, which loads a pipeline for any language it detects, <see cref="Load"/> loads the configured languages'
/// pipelines up front, so <see cref="Process(IEnumerable{string})"/> changes nothing shared and may be called from several
/// threads at once. A text detected as a language without a configured pipeline (French, say, unless
/// <see cref="MultilingualOptions.Restrict"/> is on) comes back with only <see cref="Document.Text"/> and
/// <see cref="Document.Lang"/> set and no sentences, so the caller can route it elsewhere.
/// </remarks>
/// <example>
/// <code>
/// // models/stanza holds en/, de/ and multilingual/ (stanzasharp download models/stanza/multilingual --lang multilingual).
/// using var nlp = MultilingualPipeline.Load("models/stanza");
/// foreach (var doc in nlp.Process(["The weather is lovely.", "Das Wetter ist schön."]))
///     Console.WriteLine($"{doc.Lang}: {doc.Sentences[0].Words.First().Lemma}");
/// </code>
/// </example>
public sealed class MultilingualPipeline : IDisposable
{
    private readonly LanguageIdentifier _identifier;
    private readonly Dictionary<string, Pipeline> _pipelines;

    private MultilingualPipeline(LanguageIdentifier identifier, Dictionary<string, Pipeline> pipelines) =>
        (_identifier, _pipelines) = (identifier, pipelines);

    /// <summary>
    /// Loads the language identifier from <c>multilingual/</c> under <paramref name="modelRoot"/> and each configured
    /// language's pipeline from the language's folder there (<c>en/</c>, <c>de/</c>), e.g. <c>models/stanza</c>.
    /// </summary>
    /// <exception cref="ArgumentException">Two entries of <see cref="MultilingualOptions.Languages"/> name one language, or
    /// a language or option <see cref="Pipeline.Load"/> or <see cref="LanguageIdentifier.Load"/> rejects.</exception>
    /// <exception cref="FileNotFoundException">A model file is missing.</exception>
    public static MultilingualPipeline Load(string modelRoot, MultilingualOptions? options = null)
    {
        options ??= new MultilingualOptions();
        var languages = options.Languages ?? [new PipelineOptions { Language = Pipeline.English }, new PipelineOptions { Language = "de" }];
        if (languages.Count == 0)
            throw new ArgumentException("MultilingualOptions.Languages is empty: configure at least one language", nameof(options));
        if (languages.Any(l => l == null))
            throw new ArgumentException("MultilingualOptions.Languages holds a null entry", nameof(options));
        if (languages.GroupBy(l => l.Language).FirstOrDefault(g => g.Count() > 1) is { } twice)
            throw new ArgumentException($"MultilingualOptions.Languages has {twice.Count()} entries for language '{twice.Key}'", nameof(options));

        // Stanza's restrict: the configured languages become the subset unless the identifier's options name one.
        var identifierOptions = options.LanguageIdentifier ?? new LanguageIdentifierOptions();
        if (options.Restrict && identifierOptions.Languages is not { Count: > 0 })
        {
            identifierOptions = new LanguageIdentifierOptions
            {
                Languages = [.. languages.Select(l => l.Language)],
                Backend = identifierOptions.Backend,
                Threads = identifierOptions.Threads,
                VerifyChecksums = identifierOptions.VerifyChecksums,
            };
        }

        var identifier = LanguageIdentifier.Load(Path.Combine(modelRoot, ModelDownloader.Multilingual), identifierOptions);
        var pipelines = new Dictionary<string, Pipeline>();
        try
        {
            foreach (var language in languages)
                pipelines.Add(language.Language, Pipeline.Load(Path.Combine(modelRoot, language.Language), language));
            return new MultilingualPipeline(identifier, pipelines);
        }
        catch
        {
            foreach (var pipeline in pipelines.Values)
                pipeline.Dispose();
            identifier.Dispose();
            throw;
        }
    }

    /// <summary>The configured languages, whose texts are processed, in <see cref="MultilingualOptions.Languages"/>' order.</summary>
    public IReadOnlyList<string> Languages => [.. _pipelines.Keys];

    /// <summary>The language identifier the pipeline routes with.</summary>
    public LanguageIdentifier LanguageIdentifier => _identifier;

    /// <summary>Identifies <paramref name="text"/>'s language and processes it with that language's pipeline.</summary>
    /// <exception cref="ArgumentException"><paramref name="text"/> is empty: the identifier needs at least one character (Stanza fails on it too).</exception>
    public Document Process(string text) => Process(text, CancellationToken.None);

    /// <inheritdoc cref="Process(string)"/>
    /// <param name="text">The text.</param>
    /// <param name="cancellationToken">Cancels the call between batches.</param>
    public Document Process(string text, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Process([text], cancellationToken)[0]; // Stanza wraps a single text in a list: a bulk call of one
    }

    /// <summary>
    /// Identifies each text's language and processes each language's texts with its pipeline in one bulk call
    /// (<see cref="Pipeline.Process(IEnumerable{string})"/>: sentence ids continue across that language's documents, as in
    /// Stanza). The documents come back in the texts' order.
    /// </summary>
    /// <exception cref="ArgumentException">A text is empty: the identifier needs at least one character (Stanza fails on it too).</exception>
    public List<Document> Process(IEnumerable<string> texts) => Process(texts, CancellationToken.None);

    /// <inheritdoc cref="Process(IEnumerable{string})"/>
    /// <param name="texts">The texts.</param>
    /// <param name="cancellationToken">Cancels the call between batches.</param>
    public List<Document> Process(IEnumerable<string> texts, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var list = texts.ToList();
        if (list.FindIndex(t => t == null) is int missing and >= 0)
            throw new ArgumentException($"Text {missing} is null", nameof(texts));
        var languages = _identifier.Identify(list, cancellationToken);
        var result = new Document[list.Count];
        foreach (var group in Enumerable.Range(0, list.Count).GroupBy(i => languages[i]))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var indices = group.ToList();
            if (_pipelines.TryGetValue(group.Key, out var pipeline))
            {
                var docs = pipeline.Process(indices.Select(i => list[i]), cancellationToken);
                for (int k = 0; k < indices.Count; k++)
                    result[indices[k]] = docs[k];
            }
            else
            {
                foreach (int i in indices)
                    result[i] = new Document { Text = list[i] };
            }

            foreach (int i in indices)
                result[i].Lang = group.Key;
        }

        return [.. result];
    }

    /// <summary>Frees the identifier and every pipeline.</summary>
    public void Dispose()
    {
        foreach (var pipeline in _pipelines.Values)
            pipeline.Dispose();
        _identifier.Dispose();
    }
}

/// <summary>Settings for <see cref="MultilingualPipeline.Load"/>.</summary>
public sealed class MultilingualOptions
{
    /// <summary>
    /// One <see cref="PipelineOptions"/> per language to process (Stanza's <c>lang_configs</c>), each loaded from its
    /// <see cref="PipelineOptions.Language"/>'s folder; two entries for one language throw. Null: English and German, each
    /// with its default package and processors.
    /// </summary>
    public IReadOnlyList<PipelineOptions>? Languages { get; init; }

    /// <summary>
    /// Stanza's <c>restrict</c>: the identifier may answer only the configured languages (its
    /// <see cref="LanguageIdentifierOptions.Languages"/> subset), so every text is processed, a French one as English or
    /// German. Ignored when <see cref="LanguageIdentifier"/> names a subset of its own. Off by default, as in Stanza.
    /// </summary>
    public bool Restrict { get; init; }

    /// <summary>
    /// The identifier's settings (Stanza's <c>lang_id_config</c>); null for the defaults. Its
    /// <see cref="LanguageIdentifierOptions.Threads"/>, like each pipeline's <see cref="PipelineOptions.Threads"/>, is
    /// process-wide on the managed backend: the last one loaded applies, so set them alike.
    /// </summary>
    public LanguageIdentifierOptions? LanguageIdentifier { get; init; }
}
