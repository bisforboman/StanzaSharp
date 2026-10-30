using System.Text.Json.Nodes;

namespace StanzaSharp;

/// <summary>
/// Stanza's resources.json for the supported languages, embedded as <c>stanza_resources.json</c>: each package's models,
/// each model's MD5 and its dependencies (the pretrains and charlms it reads). Pipeline.Packages, Pipeline.Dependencies
/// and ModelDownloader.Files are built from it. A language's entry is a snapshot of the models at its
/// ModelDownloader.Revisions commit, never Stanford's latest resources.json, which follows their re-uploads (it already
/// lists newer English pos and lemma models than en's commit has).
/// </summary>
internal static class StanzaResources
{
    internal const string ResourceName = "stanza_resources.json";

    private static readonly Lazy<JsonObject> Root = new(() =>
    {
        using var stream = Open();
        return JsonNode.Parse(stream)!.AsObject();
    });

    /// <summary>The embedded file, in Stanza's format (Stanza reads it as <c>resources_filepath</c>).</summary>
    internal static Stream Open() =>
        typeof(StanzaResources).Assembly.GetManifestResourceStream(ResourceName) ?? throw new InvalidOperationException($"{ResourceName} is not embedded");

    /// <summary>The Stanza release the file is from.</summary>
    internal static string Version => (string)Root.Value["stanza_version"]!;

    /// <summary>The languages the file has.</summary>
    internal static IEnumerable<string> Languages => Root.Value.Where(p => p.Value is JsonObject).Select(p => p.Key);

    /// <summary>The models of <paramref name="package"/> of <paramref name="language"/> (processor → model), without its optional ones.</summary>
    internal static Dictionary<string, string> Package(string language, string package) =>
        PackageNode(language, package).Where(p => p.Key != "optional").ToDictionary(p => p.Key, p => (string)p.Value!);

    /// <summary>The optional models of <paramref name="package"/> of <paramref name="language"/> (processor → model), maybe none.</summary>
    internal static Dictionary<string, string> Optional(string language, string package) =>
        (PackageNode(language, package)["optional"] as JsonObject)?.ToDictionary(p => p.Key, p => (string)p.Value!) ?? [];

    /// <summary>The shared files (<c>pretrain/conll17</c>, <c>forward_charlm/1billion</c>, ...) the model <paramref name="name"/> (<c>processor/model</c>) reads, in resources.json's order.</summary>
    internal static string[] Dependencies(string language, string name) =>
        Model(language, name)["dependencies"] is JsonArray dependencies
            ? [.. dependencies.Select(d => $"{(string)d!["model"]!}/{(string)d["package"]!}")]
            : [];

    /// <summary>The MD5 of the file of <paramref name="name"/> (<c>processor/model</c>, also a pretrain or a charlm).</summary>
    internal static string Md5(string language, string name) => (string)Model(language, name)["md5"]!;

    private static JsonObject Language(string language) =>
        Root.Value[language] as JsonObject ?? throw new KeyNotFoundException($"{ResourceName} has no language '{language}'");

    private static JsonObject PackageNode(string language, string package) =>
        Language(language)["packages"]?[package] as JsonObject ?? throw new KeyNotFoundException($"{ResourceName} has no package {language}/{package}");

    private static JsonObject Model(string language, string name)
    {
        var (processor, model) = (name[..name.IndexOf('/')], name[(name.IndexOf('/') + 1)..]);
        return Language(language)[processor]?[model] as JsonObject ?? throw new KeyNotFoundException($"{ResourceName} has no model {language} {name}");
    }
}
