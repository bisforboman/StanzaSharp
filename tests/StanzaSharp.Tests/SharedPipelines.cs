namespace StanzaSharp.Tests;

/// <summary>
/// A class fixture that hands out pipelines shared by every test class running at the same time: a pipeline is read-only
/// after Load and safe to share (CLAUDE.md, thread safety). Each is loaded on first use and disposed when the last class
/// that used it ends, so only what running classes need stays in memory (one default pipeline is ~950 MB). Tests that
/// check Load itself, options, logging or disposal load their own.
/// </summary>
public sealed class SharedPipelines : IDisposable
{
    private static readonly Dictionary<string, (Lazy<Pipeline> Pipeline, int Users)> Loaded = [];
    private readonly HashSet<string> _used = [];

    /// <summary>The pipeline for these options; <paramref name="processors"/> null is all of the package's.</summary>
    public Pipeline Get(string package = Pipeline.DefaultPackage, bool managed = true, string? processors = null, bool splitSentences = true, string lang = "en")
    {
        // The same set of processors in any order is the same pipeline.
        var set = string.Join(",", Pipeline.SelectModels(package, processors, addRequired: false, nameof(processors), lang).Keys.Order());
        var key = $"{lang} {package} {(managed ? "managed" : "torch")} {set} split={splitSentences}";
        Lazy<Pipeline> pipeline;
        lock (Loaded)
        {
            if (!Loaded.TryGetValue(key, out var entry))
            {
                entry = (new Lazy<Pipeline>(() => Pipeline.Load(Repo.ModelsFor(lang, package), new PipelineOptions
                {
                    Language = lang,
                    Package = package,
                    Processors = processors,
                    SplitSentences = splitSentences,
                    Backend = Repo.PipelineBackendFor(managed),
                })), 0);
            }

            if (_used.Add(key))
                entry.Users++;
            Loaded[key] = entry;
            pipeline = entry.Pipeline;
        }

        return pipeline.Value; // loaded outside the lock, once (Lazy), so classes needing other pipelines don't wait
    }

    public void Dispose()
    {
        lock (Loaded)
        {
            foreach (var key in _used)
            {
                var (pipeline, users) = Loaded[key];
                if (users > 1)
                {
                    Loaded[key] = (pipeline, users - 1);
                    continue;
                }

                Loaded.Remove(key);
                if (pipeline.IsValueCreated)
                    pipeline.Value.Dispose();
            }
        }
    }
}
