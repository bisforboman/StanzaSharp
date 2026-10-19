using System.Reflection;
using StanzaSharp.Nn;
using Xunit.Sdk;

namespace StanzaSharp.Tests;

/// <summary>Paths into the repository, found by walking up from the test binaries to StanzaSharp.slnx.</summary>
public static class Repo
{
    public static readonly string Root = FindRoot();
    public static readonly string Golden = Path.Combine(Root, "tests", "golden");

    // STANZASHARP_MODELS overrides the location, e.g. for a git worktree without its own models/.
    public static readonly string Models =
        Environment.GetEnvironmentVariable("STANZASHARP_MODELS") is { Length: > 0 } dir ? dir : Path.Combine(Root, "models", "converted", "en");

    // The original Stanza download (<processor>/<name>.pt), found next to the converted models: models/stanza/en.
    public static readonly string StanzaModels = Path.GetFullPath(Path.Combine(Models, "..", "..", "stanza", "en"));

    public static string Model(string relativeBase) => Path.Combine(Models, relativeBase);

    /// <summary>
    /// A language's models: English's are <see cref="Models"/>; another language's are its folder next to them (e.g.
    /// models/converted/de), else next to <see cref="StanzaModels"/> (models/stanza/de, the .pt files CI caches).
    /// </summary>
    public static string ModelsFor(string lang)
    {
        if (lang == "en")
            return Models;
        var converted = Path.GetFullPath(Path.Combine(Models, "..", lang));
        return Directory.Exists(converted) ? converted : Path.GetFullPath(Path.Combine(StanzaModels, "..", lang));
    }

    /// <summary>
    /// A package's models: <see cref="ModelsFor(string)"/>, but English's treebank packages are only Stanza's .pt files
    /// (<see cref="StanzaModels"/>; CI caches them for the golden shards).
    /// </summary>
    public static string ModelsFor(string lang, string package) =>
        lang == "en" && package is not (Pipeline.DefaultPackage or "default_fast") ? StanzaModels : ModelsFor(lang);

    public static string Model(string lang, string relativeBase) => Path.Combine(ModelsFor(lang), relativeBase);

    /// <summary>The base path of <paramref name="processor"/>'s model in <paramref name="package"/> of <paramref name="lang"/>.</summary>
    public static string ModelOf(string lang, string processor, string package = Pipeline.DefaultPackage) =>
        Path.Combine(ModelsFor(lang, package), $"{processor}/{Pipeline.Packages[lang][package][processor]}");

    /// <summary>The forward or backward charlm's base path: English's 1billion, German's newswiki.</summary>
    public static string Charlm(string lang, bool forward) =>
        Model(lang, (forward ? "forward_charlm/" : "backward_charlm/") + (lang == "en" ? "1billion" : "newswiki"));

    /// <summary>The pretrain <paramref name="processor"/>'s model reads (German NER: fasttextwiki), loaded for the backend.</summary>
    internal static Pretrain LoadPretrainOf(bool managed, string lang, string processor, string package = Pipeline.DefaultPackage)
    {
        var name = Pipeline.Dependencies[lang][$"{processor}/{Pipeline.Packages[lang][package][processor]}"].Single(p => p.StartsWith("pretrain/"));
        return managed ? Pretrain.LoadManaged(Model(lang, name)) : Pretrain.Load(Model(lang, name));
    }

    /// <summary>A language's golden data: tests/golden for English, tests/golden/&lt;lang&gt; for the others.</summary>
    public static string GoldenFor(string lang) => lang == "en" ? Golden : Path.Combine(Golden, lang);

    /// <summary>The backend golden tests run on (issue #29): <see cref="Backend.Managed"/> or TorchSharp.</summary>
    internal static Backend Backend(bool managed) => managed ? Nn.Backend.Managed : Nn.Backend.TorchSharp;

    /// <summary>
    /// The shared word vectors in the backend's form: a plain array when managed, so no libtorch is needed
    /// (the linux-arm64 CI job runs the managed cases without one).
    /// </summary>
    internal static Pretrain LoadPretrain(bool managed, string lang = "en") =>
        managed ? Pretrain.LoadManaged(Model(lang, "pretrain/conll17")) : Pretrain.Load(Model(lang, "pretrain/conll17"));

    /// <summary>The same as a pipeline option.</summary>
    internal static PipelineBackend PipelineBackendFor(bool managed) => managed ? PipelineBackend.Managed : CudaBackend.Cpu;

    private static string FindRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "StanzaSharp.slnx")))
                return dir.FullName;
        }

        throw new InvalidOperationException("StanzaSharp.slnx not found above " + AppContext.BaseDirectory);
    }
}

/// <summary>A fact that is skipped when the converted models are missing.</summary>
public sealed class ModelFactAttribute : FactAttribute
{
    public ModelFactAttribute()
    {
        if (!Directory.Exists(Repo.Models))
            Skip = "models/converted/en not found; run setup.ps1 -Models";
    }
}

/// <summary>A theory that is skipped when the converted models are missing.</summary>
public sealed class ModelTheoryAttribute : TheoryAttribute
{
    public ModelTheoryAttribute()
    {
        if (!Directory.Exists(Repo.Models))
            Skip = "models/converted/en not found; run setup.ps1 -Models";
    }
}

/// <summary>
/// A German row of a golden theory: its arguments plus <c>"de"</c>, the theory's last parameter. Skipped, not failed, where
/// the German models are missing (CI caches them for the golden shards only; ci.yml's no-skip checks elsewhere exempt
/// these rows by their <c>lang: "de"</c>).
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
public sealed class GermanAttribute : DataAttribute
{
    private readonly object?[] _data;

    public GermanAttribute(params object?[] data)
    {
        _data = [.. data, "de"];
        if (!Directory.Exists(Repo.ModelsFor("de")))
            Skip = "German models not found (models/converted/de or models/stanza/de); run stanzasharp download --lang de";
    }

    public override IEnumerable<object?[]> GetData(MethodInfo testMethod) => [_data];
}

/// <summary>
/// An English treebank package's row of a <see cref="TreebankPackageTests"/> theory: its arguments plus <c>"en"</c>. Skipped,
/// not failed, where the package's .pt files are missing from <see cref="Repo.StanzaModels"/> (CI caches them for the golden
/// shards only; ci.yml's no-skip checks elsewhere exempt TreebankPackageTests).
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
public sealed class EnglishTreebankAttribute : DataAttribute
{
    private readonly object?[] _data;

    public EnglishTreebankAttribute(string package, params object?[] data)
    {
        _data = [package, .. data, "en"];
        var missing = ModelDownloader.FilesFor(null, package).Select(f => f.Path).Where(f => !File.Exists(Path.Combine(Repo.StanzaModels, f))).ToList();
        if (missing.Count > 0)
            Skip = $"{package}'s models not found in {Repo.StanzaModels} ({missing[0]}); run stanzasharp download models/stanza/en --package {package}";
    }

    public override IEnumerable<object?[]> GetData(MethodInfo testMethod) => [_data];
}

/// <summary>A German-only theory (its name says German), skipped where the German models are missing.</summary>
public sealed class GermanTheoryAttribute : TheoryAttribute
{
    public GermanTheoryAttribute()
    {
        if (!Directory.Exists(Repo.Models) || !Directory.Exists(Repo.ModelsFor("de")))
            Skip = "German models not found (models/converted/de or models/stanza/de); run stanzasharp download --lang de";
    }
}

/// <summary>A German-only fact (its name says German), skipped where the German models are missing.</summary>
public sealed class GermanFactAttribute : FactAttribute
{
    public GermanFactAttribute()
    {
        if (!Directory.Exists(Repo.Models) || !Directory.Exists(Repo.ModelsFor("de")))
            Skip = "German models not found (models/converted/de or models/stanza/de); run stanzasharp download --lang de";
    }
}

/// <summary>A theory that is skipped unless both the original .pt models and the converted ones are present.</summary>
public sealed class PtModelTheoryAttribute : TheoryAttribute
{
    public PtModelTheoryAttribute()
    {
        if (!Directory.Exists(Repo.Models) || !Directory.Exists(Repo.StanzaModels))
            Skip = "models/stanza/en or models/converted/en not found; run setup.ps1 -Models";
    }
}

/// <summary>A fact that is skipped when the original .pt models are missing.</summary>
public sealed class PtModelFactAttribute : FactAttribute
{
    public PtModelFactAttribute()
    {
        if (!Directory.Exists(Repo.StanzaModels))
            Skip = "models/stanza/en not found; run setup.ps1 -Models";
    }
}

/// <summary>
/// A fact that is skipped, not failed, when one of a transformer's files is missing from
/// <c>models/stanza/en/transformer/&lt;name&gt;/</c> (golden shard 1 downloads them; ci.yml's no-skip checks elsewhere exempt
/// the tests by name: <c>Transformer</c>).
/// </summary>
public sealed class TransformerFactAttribute : FactAttribute
{
    public const string Electra = "google/electra-large-discriminator";

    public static string Directory => ModelDownloader.TransformerDirectory(Repo.StanzaModels, Electra);

    public TransformerFactAttribute(params string[] files) => Skip = SkipReason(files);

    internal static string? SkipReason(string[] files)
    {
        var missing = files.FirstOrDefault(f => !File.Exists(Path.Combine(Directory, f)));
        return missing == null ? null : $"{Path.Combine(Directory, missing)} not found; run StanzaSharp.Cli download-transformer models/stanza/en {Electra}";
    }
}

/// <summary>
/// A test of roberta-base (#172: ner conll03_electra-large reads it), skipped without its files in
/// models/stanza/en/transformer/roberta-base/ (golden shard 4 downloads them; tools/ci-models.sh transformer).
/// </summary>
public sealed class RobertaFactAttribute : FactAttribute
{
    public const string Roberta = "roberta-base";

    public static string Directory => ModelDownloader.TransformerDirectory(Repo.StanzaModels, Roberta);

    public RobertaFactAttribute(params string[] files)
    {
        var missing = files.FirstOrDefault(f => !File.Exists(Path.Combine(Directory, f)));
        Skip = missing == null ? null : $"{Path.Combine(Directory, missing)} not found; run StanzaSharp.Cli download-transformer models/stanza/en {Roberta}";
    }
}

/// <summary>
/// A test of German's transformer (german-nlp-group/electra-base-german-uncased, #170), skipped without its files in
/// models/stanza/de/transformer/ (golden shard 4 downloads them; tools/ci-models.sh transformer).
/// </summary>
public sealed class GermanTransformerFactAttribute : FactAttribute
{
    public const string Electra = "german-nlp-group/electra-base-german-uncased";

    public static string Directory => ModelDownloader.TransformerDirectory(Path.GetFullPath(Path.Combine(Repo.StanzaModels, "..", "de")), Electra);

    public GermanTransformerFactAttribute(params string[] files)
    {
        var missing = files.FirstOrDefault(f => !File.Exists(Path.Combine(Directory, f)));
        if (missing != null)
            Skip = $"{Path.Combine(Directory, missing)} not found; run StanzaSharp.Cli download-transformer models/stanza/de {Electra}";
    }
}

/// <summary>
/// A fact that is skipped without German's dbmdz/bert-base-german-cased files (models/stanza/de/transformer/, next to the
/// English models, like <see cref="GermanTransformerFactAttribute"/>).
/// </summary>
public sealed class DbmdzFactAttribute : FactAttribute
{
    public const string Bert = "dbmdz/bert-base-german-cased";

    public static string Directory => ModelDownloader.TransformerDirectory(Path.GetFullPath(Path.Combine(Repo.StanzaModels, "..", "de")), Bert);

    public DbmdzFactAttribute(params string[] files)
    {
        var missing = files.FirstOrDefault(f => !File.Exists(Path.Combine(Directory, f)));
        if (missing != null)
            Skip = $"{Path.Combine(Directory, missing)} not found; run StanzaSharp.Cli download-transformer models/stanza/de {Bert}";
    }
}

/// <summary>
/// A test of FacebookAI/xlm-roberta-large (coref's transformer, docs/coref-plan.md), skipped without its files in
/// models/stanza/en/transformer/FacebookAI/xlm-roberta-large/ (golden shard 4 downloads them; tools/ci-models.sh transformer).
/// </summary>
public sealed class XlmRobertaFactAttribute : FactAttribute
{
    public const string XlmRoberta = "FacebookAI/xlm-roberta-large";

    public static string Directory => ModelDownloader.TransformerDirectory(Repo.StanzaModels, XlmRoberta);

    public XlmRobertaFactAttribute(params string[] files)
    {
        var missing = files.FirstOrDefault(f => !File.Exists(Path.Combine(Directory, f)));
        Skip = missing == null ? null : $"{Path.Combine(Directory, missing)} not found; run StanzaSharp.Cli download-transformer models/stanza/en {XlmRoberta}";
    }
}

/// <summary>A theory skipped like <see cref="XlmRobertaFactAttribute"/>.</summary>
public sealed class XlmRobertaTheoryAttribute : TheoryAttribute
{
    public XlmRobertaTheoryAttribute(params string[] files) => Skip = new XlmRobertaFactAttribute(files).Skip;
}

/// <summary>A theory skipped like <see cref="RobertaFactAttribute"/>.</summary>
public sealed class RobertaTheoryAttribute : TheoryAttribute
{
    public RobertaTheoryAttribute(params string[] files) => Skip = new RobertaFactAttribute(files).Skip;
}

/// <summary>A theory skipped like <see cref="TransformerFactAttribute"/>.</summary>
public sealed class TransformerTheoryAttribute : TheoryAttribute
{
    public TransformerTheoryAttribute(params string[] files) => Skip = TransformerFactAttribute.SkipReason(files);
}
