namespace StanzaSharp.Tests;

/// <summary>A class fixture's models per backend and language, each loaded on first use and disposed with the fixture.</summary>
internal sealed class ModelCache<T>(Func<bool, string, T> load) : IDisposable where T : IDisposable
{
    private readonly Dictionary<(bool, string), Lazy<T>> _models = [];

    public T Get(bool managed, string lang)
    {
        Lazy<T> models;
        lock (_models)
        {
            if (!_models.TryGetValue((managed, lang), out models!))
                _models[(managed, lang)] = models = new Lazy<T>(() => load(managed, lang));
        }

        return models.Value;
    }

    public void Dispose()
    {
        foreach (var models in _models.Values)
        {
            if (models.IsValueCreated)
                models.Value.Dispose();
        }
    }
}
