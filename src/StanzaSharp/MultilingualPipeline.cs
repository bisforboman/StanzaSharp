namespace StanzaSharp;

/// <summary>
/// Stanza's <c>MultilingualPipeline</c>: the <see cref="LanguageIdentifier"/> decides each text's language, and each
/// language's texts go through that language's <see cref="Pipeline"/> in one bulk call, as
/// <see cref="Pipeline.Process(IEnumerable{string})"/> does. Each returned document's <see cref="Document.Lang"/> is set.
/// </summary>
/// <remarks>
/// <para>As in Stanza, a language's pipeline is loaded the first time a text in it comes, and at most
/// <see cref="MultilingualOptions.MaxLoadedPipelines"/> stay loaded (the least recently used one is freed first). By
/// default every language with a folder under the model root is used; <see cref="MultilingualOptions.Languages"/> names
/// the languages and their options instead. Nothing is ever downloaded: a text detected as a language without a folder
/// (or not configured) comes back with only <see cref="Document.Text"/> and <see cref="Document.Lang"/> set and no
/// sentences, so the caller can route it elsewhere. A language whose folder lacks a model file throws
/// <see cref="FileNotFoundException"/> at its first text.</para>
/// <para><see cref="Process(IEnumerable{string})"/> may be called from several threads at once. A pipeline another call
/// is still using is freed only when that call is done, so for a moment more than
/// <see cref="MultilingualOptions.MaxLoadedPipelines"/> can be in memory.</para>
/// </remarks>
/// <example>
/// <code>
/// // models/stanza holds multilingual/ (stanzasharp download models/stanza/multilingual --lang multilingual) and a
/// // folder per language (stanzasharp download models/stanza/fr --lang fr).
/// using var nlp = MultilingualPipeline.Load("models/stanza");
/// foreach (var doc in nlp.Process(["The weather is lovely.", "Das Wetter ist schön."]))
///     Console.WriteLine($"{doc.Lang}: {doc.Sentences[0].Words.First().Lemma}");
/// </code>
/// </example>
public sealed class MultilingualPipeline : IDisposable
{
    private readonly LanguageIdentifier _identifier;
    private readonly string _modelRoot;
    private readonly Dictionary<string, PipelineOptions> _options;
    private readonly string[] _languages;
    private readonly int _maxLoaded;
    private readonly Lock _gate = new();

    // The loaded (or loading) pipelines, least recently used first.
    private readonly Dictionary<string, Entry> _loaded = [];
    private readonly LinkedList<Entry> _recent = [];
    private bool _disposed;

    private MultilingualPipeline(LanguageIdentifier identifier, string modelRoot, Dictionary<string, PipelineOptions> options, string[] languages, int maxLoaded) =>
        (_identifier, _modelRoot, _options, _languages, _maxLoaded) = (identifier, modelRoot, options, languages, maxLoaded);

    /// <summary>
    /// Loads the language identifier from <c>multilingual/</c> under <paramref name="modelRoot"/> (e.g.
    /// <c>models/stanza</c>). Each language's pipeline is loaded from its folder there (<c>en/</c>, <c>fr/</c>) at its
    /// first text.
    /// </summary>
    /// <exception cref="ArgumentException">Two entries of <see cref="MultilingualOptions.Languages"/> name one language, or
    /// a language or option <see cref="Pipeline.Load"/> or <see cref="LanguageIdentifier.Load"/> rejects.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><see cref="MultilingualOptions.MaxLoadedPipelines"/> is less than 1.</exception>
    /// <exception cref="DirectoryNotFoundException">A configured language has no folder, or (without
    /// <see cref="MultilingualOptions.Languages"/>) no supported language has one.</exception>
    /// <exception cref="FileNotFoundException">The identifier's model file is missing.</exception>
    public static MultilingualPipeline Load(string modelRoot, MultilingualOptions? options = null)
    {
        options ??= new MultilingualOptions();
        if (options.MaxLoadedPipelines < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "MaxLoadedPipelines must be at least 1");
        var languages = options.Languages ?? Present(modelRoot);
        if (languages.Count == 0)
            throw new ArgumentException("MultilingualOptions.Languages is empty: configure at least one language", nameof(options));
        if (languages.Any(l => l == null))
            throw new ArgumentException("MultilingualOptions.Languages holds a null entry", nameof(options));
        if (languages.GroupBy(l => l.Language).FirstOrDefault(g => g.Count() > 1) is { } twice)
            throw new ArgumentException($"MultilingualOptions.Languages has {twice.Count()} entries for language '{twice.Key}'", nameof(options));
        foreach (var language in languages)
        {
            // What Pipeline.Load rejects (language, package, processors, models) throws now rather than at the first text.
            _ = ModelDownloader.FilesFor(language.Processors, language.Package, nameof(options), language.Language, language.Models, language.Pretagged, language.ModelPaths);
            if (!Directory.Exists(Path.Combine(modelRoot, language.Language)))
                throw new DirectoryNotFoundException($"No folder for language '{language.Language}': {Path.Combine(modelRoot, language.Language)}");
        }

        // Stanza's restrict: the configured languages become the subset unless the identifier's options name one.
        var identifierOptions = options.LanguageIdentifier ?? new LanguageIdentifierOptions();
        if (options.Restrict && identifierOptions.Languages is not { Count: > 0 })
        {
            identifierOptions = new LanguageIdentifierOptions
            {
                Languages = [.. languages.Select(l => Label(l.Language))],
                Backend = identifierOptions.Backend,
                Threads = identifierOptions.Threads,
                VerifyChecksums = identifierOptions.VerifyChecksums,
            };
        }

        // The languages found on disk include some the identifier never answers; they just stay out of the subset.
        var identifierDir = Path.Combine(modelRoot, ModelDownloader.Multilingual);
        var identifier = options.Languages == null
            ? LanguageIdentifier.LoadSubsetOfKnown(identifierDir, identifierOptions)
            : LanguageIdentifier.Load(identifierDir, identifierOptions);
        return new MultilingualPipeline(identifier, modelRoot, languages.ToDictionary(l => l.Language), [.. languages.Select(l => l.Language)], options.MaxLoadedPipelines);
    }

    /// <summary>The languages whose texts are processed: <see cref="MultilingualOptions.Languages"/>' in its order, else those found under the model root, sorted.</summary>
    public IReadOnlyList<string> Languages => Array.AsReadOnly(_languages);

    /// <summary>The language identifier the pipeline routes with.</summary>
    public LanguageIdentifier LanguageIdentifier => _identifier;

    /// <summary>The languages whose pipelines are loaded or loading, least recently used first (for tests).</summary>
    internal string[] LoadedLanguages
    {
        get
        {
            lock (_gate)
                return [.. _recent.Select(e => e.Language)];
        }
    }

    /// <summary>Identifies <paramref name="text"/>'s language and processes it with that language's pipeline.</summary>
    /// <exception cref="ArgumentException"><paramref name="text"/> is empty: the identifier needs at least one character (Stanza fails on it too).</exception>
    /// <exception cref="FileNotFoundException">The language's folder lacks a model file.</exception>
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
    /// Stanza), loading the pipeline first if it isn't. The documents come back in the texts' order.
    /// </summary>
    /// <exception cref="ArgumentException">A text is empty: the identifier needs at least one character (Stanza fails on it too).</exception>
    /// <exception cref="FileNotFoundException">A detected language's folder lacks a model file.</exception>
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
            if (_options.ContainsKey(Language(group.Key)))
            {
                // One language's pipeline at a time, so a call with more languages than MaxLoadedPipelines works too.
                var entry = Acquire(Language(group.Key));
                try
                {
                    var docs = entry.Pipeline.Value.Process(indices.Select(i => list[i]), cancellationToken);
                    for (int k = 0; k < indices.Count; k++)
                        result[indices[k]] = docs[k];
                }
                finally
                {
                    Release(entry);
                }
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

    /// <summary>Frees the identifier and every loaded pipeline; those still in use when the calls using them are done.</summary>
    public void Dispose()
    {
        List<Entry> free = [];
        lock (_gate)
        {
            if (_disposed)
                return;
            _disposed = true;
            foreach (var entry in _recent.ToList())
                Evict(entry, free);
        }

        Free(free);
        _identifier.Dispose();
    }

    // The identifier's label for a language (Norwegian Bokmål's is "no", Stanza's alias of nb) and back.
    private static string Label(string language) => language == "nb" ? "no" : language;

    private static string Language(string label) => label == "no" ? "nb" : label;

    // The supported languages with a folder under the model root, each with its default package and processors.
    private static List<PipelineOptions> Present(string modelRoot)
    {
        var present = Pipeline.SupportedPackages.Keys.Where(l => Directory.Exists(Path.Combine(modelRoot, l))).Order(StringComparer.Ordinal).ToList();
        if (present.Count == 0)
            throw new DirectoryNotFoundException($"No language folder under {modelRoot}: download one with \"stanzasharp download {Path.Combine(modelRoot, "<lang>")} --lang <lang>\"");
        return [.. present.Select(l => new PipelineOptions { Language = l })];
    }

    // A lease on the language's pipeline, which loads on first use (outside the lock, so other languages' calls go on).
    private Entry Acquire(string language)
    {
        Entry entry;
        List<Entry> free = [];
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_loaded.TryGetValue(language, out var found))
            {
                entry = found;
                _recent.Remove(entry.Node);
                _recent.AddLast(entry.Node);
            }
            else
            {
                var options = _options[language];
                entry = new Entry(language, new Lazy<Pipeline>(() => Pipeline.Load(Path.Combine(_modelRoot, language), options)));
                _loaded.Add(language, entry);
                _recent.AddLast(entry.Node);
                while (_loaded.Count > _maxLoaded)
                    Evict(_recent.First!.Value, free);
            }

            entry.Leases++;
        }

        Free(free);

        try
        {
            _ = entry.Pipeline.Value;
            return entry;
        }
        catch
        {
            // Not cached: the next text in the language tries again.
            lock (_gate)
            {
                if (_loaded.GetValueOrDefault(language) == entry)
                    Evict(entry, free);
            }

            Release(entry);
            throw;
        }
    }

    private void Release(Entry entry)
    {
        bool free;
        lock (_gate)
            free = --entry.Leases == 0 && entry.Evicted;
        if (free)
            entry.Free();
    }

    // Under the lock: out of the cache, into free (to be freed outside the lock) or freed when its last lease ends.
    private void Evict(Entry entry, List<Entry> free)
    {
        _loaded.Remove(entry.Language);
        _recent.Remove(entry.Node);
        entry.Evicted = true;
        if (entry.Leases == 0)
            free.Add(entry);
    }

    private static void Free(List<Entry> entries)
    {
        foreach (var entry in entries)
            entry.Free();
    }

    private sealed class Entry
    {
        public Entry(string language, Lazy<Pipeline> pipeline) => (Language, Pipeline, Node) = (language, pipeline, new(this));

        public string Language { get; }

        public Lazy<Pipeline> Pipeline { get; }

        public LinkedListNode<Entry> Node { get; }

        public int Leases { get; set; }

        public bool Evicted { get; set; }

        public void Free()
        {
            if (Pipeline.IsValueCreated)
                Pipeline.Value.Dispose();
        }
    }
}

/// <summary>Settings for <see cref="MultilingualPipeline.Load"/>.</summary>
public sealed class MultilingualOptions
{
    /// <summary>
    /// One <see cref="PipelineOptions"/> per language to process, each loaded from its
    /// <see cref="PipelineOptions.Language"/>'s folder at its first text; two entries for one language throw, and so does
    /// one whose folder is missing. Unlike Stanza's <c>lang_configs</c>, which only override options, the list is also the
    /// set of languages: a text in any other comes back unprocessed. Null: every supported language with a folder under
    /// the model root, each with its default package and processors.
    /// </summary>
    public IReadOnlyList<PipelineOptions>? Languages { get; init; }

    /// <summary>
    /// Stanza's <c>restrict</c>: the identifier may answer only the configured languages (its
    /// <see cref="LanguageIdentifierOptions.Languages"/> subset), so every text is processed, a French one as English or
    /// German. Ignored when <see cref="LanguageIdentifier"/> names a subset of its own. Off by default, as in Stanza.
    /// </summary>
    public bool Restrict { get; init; }

    /// <summary>
    /// How many languages' pipelines stay loaded (Stanza's <c>max_cache_size</c>, 10 by default): loading another frees
    /// the least recently used one. At least 1. A pipeline of a default package takes about 1 GB, so lower this on a
    /// small machine.
    /// </summary>
    public int MaxLoadedPipelines { get; init; } = 10;

    /// <summary>
    /// The identifier's settings (Stanza's <c>lang_id_config</c>); null for the defaults. Its
    /// <see cref="LanguageIdentifierOptions.Threads"/>, like each pipeline's <see cref="PipelineOptions.Threads"/>, is
    /// process-wide on the managed backend: the last one loaded applies (a pipeline loads at its language's first text), so
    /// set them alike.
    /// </summary>
    public LanguageIdentifierOptions? LanguageIdentifier { get; init; }
}
