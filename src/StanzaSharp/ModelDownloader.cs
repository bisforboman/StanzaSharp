using System.Security.Cryptography;

namespace StanzaSharp;

/// <summary>
/// Downloads the Stanza models this version of StanzaSharp is verified against, from Stanza's Hugging Face repository
/// of the language (<c>stanfordnlp/stanza-en</c>, <c>stanfordnlp/stanza-de</c>), into a directory
/// <see cref="Pipeline.Load"/> can read. Loading never downloads anything; call this once.
/// </summary>
/// <example>
/// <code>
/// await ModelDownloader.DownloadAsync("models/stanza/en");                            // everything
/// await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
/// await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Package = "default_fast" });
/// await ModelDownloader.DownloadAsync("models/stanza/de", new PipelineOptions { Language = "de" });
/// using var nlp = Pipeline.Load("models/stanza/en");
/// </code>
/// </example>
public static class ModelDownloader
{
    /// <summary>The Stanza release whose models the golden tests verify.</summary>
    public const string StanzaVersion = "1.15.0";

    // The commit of each stanfordnlp/stanza-<lang> repository holding the files Files lists, pinned rather than the v1.15.0
    // tag, which Stanford moved once already: on 2026-10-20 they re-uploaded English's conll03 NER models and German's NER
    // models (germeval2014 replaced by germeval2014_charlm) and moved the tag to these commits (#215).
    internal static readonly Dictionary<string, string> Revisions = new()
    {
        [Pipeline.English] = "810dca30ddb25c30582c019dbaeb3a3e040fb958",
        ["de"] = "a8ecd6c8b4f81298918eade263208fedccddd652",
        ["es"] = "6c5cf2278847a76cd1efd05ee4fddd80594b569d",
        ["fr"] = "caeb7742148cb0368e4a0f2c68a0f7b09bc22880",
        ["it"] = "56584df8218c8db87a8941383cd3ab9741bc28d2",
        ["pt"] = "8ca2d3f186cf37c33271c02fbd8663b328a2d0b5",
        ["nl"] = "ec6376bd712626664e2c3bcfe5101eebe01d8f18",
        [Multilingual] = "e9360ddf1796ec9c18a8260980153c7f269afd63",
    };

    private static string BaseUrl(string language) => $"https://huggingface.co/stanfordnlp/stanza-{language}/resolve/{Revisions[language]}/models/";

    // Every file of each language's supported models (Pipeline.SupportedModels) and of the shared files they read, with its
    // MD5 from Stanza's resources.json (StanzaResources).
    internal static readonly Dictionary<string, (string Path, string Md5)[]> Files =
        Pipeline.SupportedPackages.Keys.ToDictionary(l => l, l => Pipeline.SupportedModels(l)
            .SelectMany(m => Pipeline.Dependencies[l][m].Prepend(m))
            .Distinct()
            .Select(f => (f + ".pt", StanzaResources.Md5(l, f)))
            .ToArray());

    /// <summary>
    /// The files <paramref name="processors"/> (null: all) of <paramref name="package"/> of <paramref name="language"/>
    /// need: their own models, those of the processors they require (e.g. pos needs tokenize and mwt), and the shared
    /// pretrains and charlms they read. Exactly what <see cref="Pipeline.Load"/> reads for the same options, but the
    /// transformers (<see cref="Pipeline.TransformersOf"/>), which come from Hugging Face.
    /// </summary>
    internal static IEnumerable<(string Path, string Md5)> FilesFor(string? processors, string package = Pipeline.DefaultPackage, string paramName = "processors",
        string language = Pipeline.English, IReadOnlyDictionary<string, string>? models = null, bool pretagged = false,
        IReadOnlyDictionary<string, ModelPath>? modelPaths = null)
    {
        var selected = Pipeline.SelectModels(package, processors, addRequired: true, paramName, language, models, pretagged, modelPaths);
        var paths = Pipeline.ModelFiles(selected, language, custom: Pipeline.CustomPaths(modelPaths, paramName)).Select(p => p + ".pt").ToHashSet();
        var files = Files[language].Where(f => paths.Contains(f.Path)).ToList();
        if (files.Count != paths.Count)
            throw new InvalidOperationException($"No checksum for {string.Join(", ", paths.Except(files.Select(f => f.Path)))}");
        return files;
    }

    private static readonly HttpClient Http = new() { DefaultRequestHeaders = { { "User-Agent", "StanzaSharp" } } };

    /// <summary>
    /// Downloads the English <see langword="default"/> package's models (about 600 MB) into <paramref name="modelDir"/>, laid
    /// out as <c>&lt;processor&gt;/&lt;name&gt;.pt</c>. Files already there with the right checksum are kept;
    /// every new file is checked against its MD5 before it is put in place.
    /// </summary>
    /// <param name="progress">Receives one line per file, e.g. for a console.</param>
    public static Task DownloadAsync(string modelDir, IProgress<string>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(modelDir, FilesFor(null), Http, progress, cancellationToken);

    /// <summary>
    /// Downloads only the models <paramref name="processors"/> need in the English <see langword="default"/> package,
    /// e.g. <c>"tokenize,mwt,pos,lemma"</c>: their own, those of the processors they require, and the shared pretrain
    /// and charlms when used. Otherwise like <see cref="DownloadAsync(string, IProgress{string}?, CancellationToken)"/>.
    /// </summary>
    /// <param name="processors">Comma-separated names from <see cref="Pipeline.AllProcessors"/>.</param>
    public static Task DownloadAsync(string modelDir, string processors, IProgress<string>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(modelDir, FilesFor(processors), Http, progress, cancellationToken);

    /// <summary>
    /// Downloads exactly the models <see cref="Pipeline.Load"/> reads with the same <paramref name="options"/>: those
    /// of its <see cref="PipelineOptions.Language"/> and <see cref="PipelineOptions.Package"/> for its
    /// <see cref="PipelineOptions.Processors"/> (all of the package's when null) and the processors they require, with
    /// the models <see cref="PipelineOptions.Models"/> chooses (with <see cref="PipelineOptions.Pretagged"/>, lemma,
    /// depparse and constituency require none), and the transformer a transformer model reads (from Hugging Face, e.g.
    /// google/electra-large-discriminator's 1.3 GB, into <c>transformer/</c>). The other options are ignored. Otherwise like
    /// <see cref="DownloadAsync(string, IProgress{string}?, CancellationToken)"/>.
    /// </summary>
    /// <exception cref="ArgumentException">An unknown language, package or processor, a processor the package lacks, or an
    /// unknown or unsupported model in <see cref="PipelineOptions.Models"/>.</exception>
    /// <example>
    /// <code>
    /// var options = new PipelineOptions { Package = "default_fast" };
    /// await ModelDownloader.DownloadAsync("models/stanza/en", options);
    /// using var nlp = Pipeline.Load("models/stanza/en", options);
    /// </code>
    /// </example>
    public static async Task DownloadAsync(string modelDir, PipelineOptions options, IProgress<string>? progress = null, CancellationToken cancellationToken = default)
    {
        var files = FilesFor(options.Processors, options.Package, nameof(options), options.Language, options.Models, options.Pretagged, options.ModelPaths);
        var selected = Pipeline.SelectModels(options.Package, options.Processors, addRequired: true, nameof(options), options.Language, options.Models, options.Pretagged,
            options.ModelPaths);
        await DownloadAsync(modelDir, files, Http, progress, cancellationToken, options.Language);
        foreach (var name in Pipeline.TransformersOf(selected, options.Language))
            await DownloadTransformerAsync(modelDir, name, progress: progress, cancellationToken: cancellationToken);
    }

    internal static async Task DownloadAsync(string modelDir, IEnumerable<(string Path, string Md5)> files, HttpClient http,
        IProgress<string>? progress, CancellationToken cancellationToken, string language = Pipeline.English)
    {
        foreach (var (path, md5) in files)
            await FetchAsync(Path.Combine(modelDir, path), path, BaseUrl(language) + path, md5, Md5Async, http, progress, cancellationToken);
    }

    /// <summary>The language identifier's file in Stanza's <c>multilingual</c> folder (<see cref="LanguageIdentifier"/>) and its MD5 (Stanza 1.15.0's resources.json).</summary>
    internal static readonly (string Path, string Md5) LanguageIdentifierFile = (LanguageIdentifier.ModelPath + ".pt", "84cd508b9702b31cc236ba2b6d94e41a");

    /// <summary>
    /// Downloads the language identifier's model (9 MB) into <paramref name="modelDir"/>, Stanza's <c>multilingual</c>
    /// folder (e.g. <c>models/stanza/multilingual</c>), as <c>langid/ud.pt</c>, for <see cref="LanguageIdentifier.Load"/>.
    /// A file already there with the right checksum is kept; a new one is checked against its MD5.
    /// </summary>
    /// <param name="progress">Receives one line per file, e.g. for a console.</param>
    public static Task DownloadLanguageIdentifierAsync(string modelDir, IProgress<string>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(modelDir, [LanguageIdentifierFile], Http, progress, cancellationToken, Multilingual);

    // Stanza's code for the language identifier's resources (stanza-multilingual on Hugging Face).
    internal const string Multilingual = "multilingual";

    /// <summary><see cref="LanguageIdentifierOptions.VerifyChecksums"/>: the model file against its MD5, as <see cref="Verify"/>.</summary>
    internal static void VerifyLanguageIdentifier(string modelDir)
    {
        var basePath = Path.Combine(modelDir, LanguageIdentifier.ModelPath);
        if (File.Exists(basePath + ".json"))
        {
            throw new InvalidOperationException($"VerifyChecksums: {basePath}.json is a converted model, which has no published checksum. "
                + "Load Stanza's .pt file or turn VerifyChecksums off.");
        }

        string actual;
        using (var stream = File.OpenRead(basePath + ".pt"))
            actual = Convert.ToHexStringLower(MD5.HashData(stream));
        if (actual != LanguageIdentifierFile.Md5)
        {
            throw new InvalidDataException($"{basePath}.pt: checksum {actual} does not match Stanza {StanzaVersion}'s {LanguageIdentifierFile.Md5}. "
                + "The file is corrupted or from another Stanza version; download it again with ModelDownloader.DownloadLanguageIdentifierAsync.");
        }
    }

    /// <summary>
    /// The transformers that Stanza's transformer models read, which transformers downloads from Hugging Face rather than
    /// Stanza (docs/transformer-plan.md): each repository's pinned revision and the files StanzaSharp reads, with their
    /// sha256 (Hugging Face's own checksum; these files are not in Stanza's resources.json).
    /// </summary>
    internal static readonly Dictionary<string, (string Revision, (string File, string Sha256)[] Files)> Transformers = new()
    {
        ["google/electra-large-discriminator"] = ("c13c3df7efadc2162f42588bd28eb4e187d602a5",
        [
            ("config.json", "9df6e6aeb17dbb95a44b31c3ba081dc84478f168ec0d56d1e94e397a8d876ac6"),
            ("tokenizer.json", "a23e7b62fb92e2cacc11087144ff99699f5575758cec85318a31f5295ae02f3b"),
            ("tokenizer_config.json", "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"),
            ("pytorch_model.bin", "070894f07d07a499d73946378afa5548d64aca2418d17e67d398233692497e63"),
        ]),

        // German default_accurate's (docs/german-accurate-plan.md): a vocab.txt tokenizer, no tokenizer.json.
        ["german-nlp-group/electra-base-german-uncased"] = ("16e66fe6c6d52b459b7de7330bc9db2c62f3620d",
        [
            ("config.json", "e1a432c5209b9f72d53088b972d7a8dc0645859bfcdf2e33a153286c8e7a1955"),
            ("vocab.txt", "ffcfe163e0a45a6c112822fcd026e82dd23065861f1c3f7a71fdd653106a51ba"),
            ("tokenizer_config.json", "3888c4068947ecd386bfac973ec5e348e8503d05699427900698e74d4fdc2889"),
            ("pytorch_model.bin", "a13cf267dfad08dd5c889af59f2e7f74120ff28107619d6c81904c3250efe76e"),
        ]),

        // German sentiment sb10k_dbmdz-bert-german-cased's: BERT (cased) with a
        // vocab.txt WordPiece, like german-nlp-group's.
        ["dbmdz/bert-base-german-cased"] = ("43cce138f63f8bd34b42d8a77321e3262fd7bbba",
        [
            ("config.json", "2b6e682bc4c25157c68b7ce4f2b7305255aaa528203d357ccce0d54e4ee3d984"),
            ("vocab.txt", "bf186f7fb9cb667d811c08b45ce64d1e8e262277bb053f05d1aef3e9f13039a3"),
            ("tokenizer_config.json", "4c052d60b505817149a29e88c5292d7779ddecfadbfc6208729e61403ec558ba"),
            ("pytorch_model.bin", "e482d0e2a09cef20bed85c86d7df9d2f1f5d090e7e53c772ab94ced0e3eafee3"),
        ]),

        // Coref's (docs/coref-plan.md, #176; udcoref_xlm-roberta-lora reads it): XLM-RoBERTa with a SentencePiece Unigram tokenizer.json.
        // pytorch_model.bin is torch's legacy format, as Stanza's own checkpoints.
        ["FacebookAI/xlm-roberta-large"] = ("c23d21b0620b635a76227c604d44e43a9f0ee389",
        [
            ("config.json", "ec7c3a99c58a38ebb702a2297f1c4586944e841b5eb29d201ff87e0f9c9abe0e"),
            ("tokenizer.json", "a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c"),
            ("tokenizer_config.json", "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce"),
            ("pytorch_model.bin", "01e55aa45dbb9164fee19aef60007a1c91d175051c01be1fb15056cfa60f3e53"),
        ]),

        // RoBERTa with a byte-level BPE tokenizer.json (#172), by the old alias Hugging Face serves FacebookAI/roberta-base under.
        // No model reads it since English ner conll03_electra-large was retrained on ELECTRA-large (#215); kept as tested
        // infrastructure for ByteLevelBpe and RoBERTa's config (owner's decision, 2026-10-24).
        ["roberta-base"] = ("e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        [
            ("config.json", "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9"),
            ("tokenizer.json", "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e"),
            ("tokenizer_config.json", "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce"),
            ("pytorch_model.bin", "278b7a95739c4392fae9b818bb5343dde20be1b89318f37a6d939e1e1b9e461b"),
        ]),
    };

    /// <summary>
    /// Where a transformer's files go in a model directory: <c>transformer/&lt;repository&gt;/</c>, e.g.
    /// <c>models/stanza/en/transformer/google/electra-large-discriminator/</c>.
    /// </summary>
    internal static string TransformerDirectory(string modelDir, string name) => Path.Combine(modelDir, "transformer", Path.Combine(name.Split('/')));

    /// <summary>
    /// Downloads <paramref name="files"/> (null: all of them) of the transformer <paramref name="name"/> (a key of
    /// <see cref="Transformers"/>) at its pinned revision into <see cref="TransformerDirectory"/>, checking each against
    /// its sha256 and keeping files that already match.
    /// </summary>
    internal static async Task DownloadTransformerAsync(string modelDir, string name, IReadOnlyCollection<string>? files = null,
        IProgress<string>? progress = null, CancellationToken cancellationToken = default)
    {
        if (!Transformers.TryGetValue(name, out var transformer))
            throw new ArgumentException($"Unknown transformer '{name}'; supported: {string.Join(", ", Transformers.Keys)}", nameof(name));
        var unknown = files?.Except(transformer.Files.Select(f => f.File)).ToList();
        if (unknown is { Count: > 0 })
            throw new ArgumentException($"{name} has no file {string.Join(", ", unknown)}", nameof(files));
        var selected = transformer.Files.Where(f => files == null || files.Contains(f.File)).ToList();
        await DownloadTransformerAsync(modelDir, name, transformer.Revision, selected, Http, progress, cancellationToken);
    }

    internal static async Task DownloadTransformerAsync(string modelDir, string name, string revision, IEnumerable<(string File, string Sha256)> files,
        HttpClient http, IProgress<string>? progress, CancellationToken cancellationToken)
    {
        var directory = TransformerDirectory(modelDir, name);
        foreach (var (file, sha256) in files)
        {
            await FetchAsync(Path.Combine(directory, file), $"{name}/{file}", $"https://huggingface.co/{name}/resolve/{revision}/{file}", sha256,
                Sha256Async, http, progress, cancellationToken);
        }
    }

    // Downloads url to target unless target already has the expected checksum; a new file is checked before it is put
    // in place, so a failed or wrong download never replaces a good one.
    private static async Task FetchAsync(string target, string name, string url, string expected, Func<string, CancellationToken, Task<string>> checksum,
        HttpClient http, IProgress<string>? progress, CancellationToken cancellationToken)
    {
        if (File.Exists(target) && await checksum(target, cancellationToken) == expected)
        {
            progress?.Report($"{name}: up to date");
            return;
        }

        progress?.Report($"{name}: downloading");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        var partial = target + ".download";
        try
        {
            using (var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
            {
                response.EnsureSuccessStatusCode();
                await using var file = File.Create(partial);
                await response.Content.CopyToAsync(file, cancellationToken);
            }

            var actual = await checksum(partial, cancellationToken);
            if (actual != expected)
                throw new InvalidDataException($"{name}: checksum {actual} does not match the expected {expected}");
            File.Move(partial, target, overwrite: true);
        }
        finally
        {
            File.Delete(partial);
        }
    }

    /// <summary>
    /// <see cref="PipelineOptions.VerifyChecksums"/>: checks each file <see cref="Pipeline.Load"/> reads for
    /// <paramref name="models"/> of <paramref name="language"/> against its MD5, and each transformer file against its
    /// sha256 (<see cref="Transformers"/>). Converted models, which have no published checksum, throw; the caller's own
    /// files (<paramref name="custom"/>) are left out.
    /// </summary>
    internal static void Verify(string modelDir, IReadOnlyDictionary<string, string> models, string language, IReadOnlyDictionary<string, ModelPath>? custom = null)
    {
        foreach (var path in Pipeline.ModelFiles(models, language, readOnly: true, custom))
        {
            var basePath = Path.Combine(modelDir, path);

            // Checkpoint.Load reads the converted files whenever the .json exists.
            if (File.Exists(basePath + ".json"))
            {
                throw new InvalidOperationException($"VerifyChecksums: {basePath}.json is a converted model, which has no published checksum. "
                    + "Load Stanza's .pt files (e.g. a ModelDownloader directory) or turn VerifyChecksums off.");
            }

            var file = basePath + ".pt"; // Load checked that it exists
            var expected = Files[language].Single(f => f.Path == path + ".pt").Md5;
            string actual;
            using (var stream = File.OpenRead(file))
                actual = Convert.ToHexStringLower(MD5.HashData(stream));
            if (actual != expected)
            {
                throw new InvalidDataException($"{file}: checksum {actual} does not match Stanza {StanzaVersion}'s {expected}. "
                    + "The file is corrupted or from another Stanza version; download it again with ModelDownloader.DownloadAsync.");
            }
        }

        foreach (var name in Pipeline.TransformersOf(models, language))
        {
            foreach (var (fileName, expected) in Transformers[name].Files)
            {
                var file = Path.Combine(TransformerDirectory(modelDir, name), fileName); // Load checked that it exists
                string actual;
                using (var stream = File.OpenRead(file))
                    actual = Convert.ToHexStringLower(SHA256.HashData(stream));
                if (actual != expected)
                {
                    throw new InvalidDataException($"{file}: sha256 {actual} does not match {name}@{Transformers[name].Revision}'s {expected}. "
                        + "The file is corrupted or of another revision; download it again with ModelDownloader.DownloadAsync.");
                }
            }
        }
    }

    private static async Task<string> Md5Async(string path, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await MD5.HashDataAsync(file, cancellationToken));
    }

    private static async Task<string> Sha256Async(string path, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(file, cancellationToken));
    }
}
