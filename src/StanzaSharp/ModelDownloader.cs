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

    private static string BaseUrl(string language) => $"https://huggingface.co/stanfordnlp/stanza-{language}/resolve/v{StanzaVersion}/models/";

    // Every file of each language's packages in Pipeline.Packages and their MD5s, from Stanza 1.15.0's resources.json.
    internal static readonly Dictionary<string, (string Path, string Md5)[]> Files = new()
    {
        [Pipeline.English] =
        [
            ("tokenize/combined_nocharlm.pt", "764639a5f65cc42d84b294566ffa6215"),
            ("mwt/combined.pt", "dc569dbb26cc72b71847f9d0b8775d43"),
            ("pos/combined_charlm.pt", "a31dbf7269152bc85c3c79c5318ae848"),
            ("lemma/combined_nocharlm.pt", "8d3742b3f507e78a9e605afd72b34c92"),
            ("constituency/ptb3-revised_charlm.pt", "74dadb9e65bc7b88889550889d0a1d16"),
            ("pretrain/conll17.pt", "c339580492002fb1f759537dfbc57fb2"),
            ("forward_charlm/1billion.pt", "468b3377455fa0311565d46865f55afb"),
            ("backward_charlm/1billion.pt", "1405948b125b4264fd17509d1b6175ca"),
            ("depparse/combined_charlm.pt", "993a0eb712cefb881c684df8d9934f15"),
            ("ner/ontonotes-ww-multi_charlm.pt", "3cd5e98549640f448846389169a36812"),
            ("sentiment/sstplus_charlm.pt", "cf7e8e8f2a76c1dc236329dbc0e61cb4"),

            // default_fast's own models
            ("pos/combined_nocharlm.pt", "005fab042873317838e692c650e8e5e9"),
            ("depparse/combined_nocharlm.pt", "d155cf16050ba5fb640dc38424924346"),
            ("ner/ontonotes-ww-multi_nocharlm.pt", "e927ed0ce0960638a158c8838b1e7265"),
        ],
        ["de"] =
        [
            ("tokenize/combined_nocharlm.pt", "a52679e759e3f6c29b749cfa8ca506be"),
            ("mwt/combined.pt", "0efb649b428cabcb89380498bb61c591"),
            ("pos/combined_charlm.pt", "757ff3b6a67c61d919a010f579cefb6c"),
            ("lemma/combined_nocharlm.pt", "207e57ac2df6fee797b163df299e0fa0"),
            ("constituency/spmrl_charlm.pt", "a46f6e39f5f34dd0a8d06c26389dbf2b"),
            ("pretrain/conll17.pt", "c489a3bcc876b2e25f4460e00401c124"),
            ("pretrain/fasttextwiki.pt", "29fe2ec4cf2d8b18c18b6aa9f5d48d78"),
            ("forward_charlm/newswiki.pt", "f2841fa3f1981a849d3ed08a723ccc22"),
            ("backward_charlm/newswiki.pt", "82f0d9a6a12c736648edbe28d87e9016"),
            ("depparse/combined_charlm.pt", "c07c3344d2635c789870b6627c6ad680"),
            ("ner/germeval2014.pt", "e2d33aecdfc079b18dbdbb06ec6d419a"),
            ("sentiment/sb10k_charlm.pt", "278b42bbd2c127d293a8197252dae1eb"),

            // default_fast's own models
            ("pos/combined_nocharlm.pt", "e2ccaf6a99711487adfe8e016592556a"),
            ("depparse/combined_nocharlm.pt", "12824bc61ad28f0377d5136745db005e"),

            // gsd's and hdt's (combined has default's)
            ("tokenize/gsd_nocharlm.pt", "9e519a21b6ca839cf0ae74d5ad6213f9"),
            ("mwt/gsd.pt", "ce401fcf73273ef02cefc757f0d08bec"),
            ("pos/gsd_charlm.pt", "b0335ee5f5ad28511020407945c2a393"),
            ("lemma/gsd_nocharlm.pt", "814edc18ce6743c2bb4acc4867140eab"),
            ("depparse/gsd_charlm.pt", "daeac687146e4dc32f3ca8f14aea56c2"),
            ("tokenize/hdt_nocharlm.pt", "106e9ec0521d8d6b1f8608b467cc32e9"),
            ("mwt/hdt.pt", "73846173795e9de1ee0546f182e0541e"),
            ("pos/hdt_charlm.pt", "5d460fe45205e40dea5b55a38097657b"),
            ("lemma/hdt_nocharlm.pt", "9fd1ffa0dafcfc9cd51b3cecb959be59"),
            ("depparse/hdt_charlm.pt", "d578c1c4614769122f2cb8481025639e"),
        ],
    };

    /// <summary>
    /// The files <paramref name="processors"/> (null: all) of <paramref name="package"/> of <paramref name="language"/>
    /// need: their own models, those of the processors they require (e.g. pos needs tokenize and mwt), and the shared
    /// pretrains and charlms they read. Exactly what <see cref="Pipeline.Load"/> reads for the same options.
    /// </summary>
    internal static IEnumerable<(string Path, string Md5)> FilesFor(string? processors, string package = Pipeline.DefaultPackage, string paramName = "processors",
        string language = Pipeline.English)
    {
        var models = Pipeline.SelectModels(package, processors, addRequired: true, paramName, language);
        var paths = models.Select(m => $"{m.Key}/{m.Value}").Concat(Pipeline.SharedModels(models, language)).Select(p => p + ".pt").ToHashSet();
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
    /// <see cref="PipelineOptions.Processors"/> (all of the package's when null) and the processors they require. The
    /// other options are ignored. Otherwise like <see cref="DownloadAsync(string, IProgress{string}?, CancellationToken)"/>.
    /// </summary>
    /// <exception cref="ArgumentException">An unknown language, package or processor, or a processor the package lacks.</exception>
    /// <example>
    /// <code>
    /// var options = new PipelineOptions { Package = "default_fast" };
    /// await ModelDownloader.DownloadAsync("models/stanza/en", options);
    /// using var nlp = Pipeline.Load("models/stanza/en", options);
    /// </code>
    /// </example>
    public static Task DownloadAsync(string modelDir, PipelineOptions options, IProgress<string>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(modelDir, FilesFor(options.Processors, options.Package, nameof(options), options.Language), Http, progress, cancellationToken,
            options.Language);

    internal static async Task DownloadAsync(string modelDir, IEnumerable<(string Path, string Md5)> files, HttpClient http,
        IProgress<string>? progress, CancellationToken cancellationToken, string language = Pipeline.English)
    {
        foreach (var (path, md5) in files)
        {
            var target = Path.Combine(modelDir, path);
            if (File.Exists(target) && await Md5Async(target, cancellationToken) == md5)
            {
                progress?.Report($"{path}: up to date");
                continue;
            }

            progress?.Report($"{path}: downloading");
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            var partial = target + ".download";
            try
            {
                using (var response = await http.GetAsync(BaseUrl(language) + path, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                {
                    response.EnsureSuccessStatusCode();
                    await using var file = File.Create(partial);
                    await response.Content.CopyToAsync(file, cancellationToken);
                }

                var actual = await Md5Async(partial, cancellationToken);
                if (actual != md5)
                    throw new InvalidDataException($"{path}: checksum {actual} does not match the expected {md5}");
                File.Move(partial, target, overwrite: true);
            }
            finally
            {
                File.Delete(partial);
            }
        }
    }

    /// <summary>
    /// <see cref="PipelineOptions.VerifyChecksums"/>: checks each file <see cref="Pipeline.Load"/> reads for
    /// <paramref name="models"/> of <paramref name="language"/> against its MD5. Converted models, which have no
    /// published checksum, throw.
    /// </summary>
    internal static void Verify(string modelDir, IReadOnlyDictionary<string, string> models, string language)
    {
        foreach (var path in models.Select(m => $"{m.Key}/{m.Value}").Concat(Pipeline.SharedModels(models, language)))
        {
            var basePath = Path.Combine(modelDir, path);

            // Checkpoint.Load reads the converted files whenever the .json exists.
            if (File.Exists(basePath + ".json"))
            {
                throw new InvalidOperationException($"VerifyChecksums: {basePath}.json is a converted model, which has no published checksum. "
                    + "Load Stanza's .pt files (e.g. a ModelDownloader directory) or turn VerifyChecksums off.");
            }

            var file = basePath + ".pt";
            if (!File.Exists(file))
                throw new FileNotFoundException($"Model file not found: {file}", file);
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
    }

    private static async Task<string> Md5Async(string path, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await MD5.HashDataAsync(file, cancellationToken));
    }
}
