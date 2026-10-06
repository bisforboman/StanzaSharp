using System.Security.Cryptography;

namespace StanzaSharp;

/// <summary>
/// Downloads the Stanza English models this version of StanzaSharp is verified against, from Stanza's
/// Hugging Face repository, into a directory <see cref="Pipeline.Load"/> can read. Loading never
/// downloads anything; call this once.
/// </summary>
/// <example>
/// <code>
/// await ModelDownloader.DownloadAsync("models/stanza/en");
/// using var nlp = Pipeline.Load("models/stanza/en");
/// </code>
/// </example>
public static class ModelDownloader
{
    /// <summary>The Stanza release whose models the golden tests verify.</summary>
    public const string StanzaVersion = "1.15.0";

    private const string BaseUrl = $"https://huggingface.co/stanfordnlp/stanza-en/resolve/v{StanzaVersion}/models/";

    // The English default package's files and their MD5s, from Stanza 1.15.0's resources.json.
    internal static readonly (string Path, string Md5)[] Files =
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
    ];

    private static readonly HttpClient Http = new() { DefaultRequestHeaders = { { "User-Agent", "StanzaSharp" } } };

    /// <summary>
    /// Downloads the English models (about 600 MB) into <paramref name="modelDir"/>, laid out as
    /// <c>&lt;processor&gt;/&lt;name&gt;.pt</c>. Files already there with the right checksum are kept;
    /// every new file is checked against its MD5 before it is put in place.
    /// </summary>
    /// <param name="progress">Receives one line per file, e.g. for a console.</param>
    public static Task DownloadAsync(string modelDir, IProgress<string>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(modelDir, Files, Http, progress, cancellationToken);

    internal static async Task DownloadAsync(string modelDir, IEnumerable<(string Path, string Md5)> files, HttpClient http,
        IProgress<string>? progress, CancellationToken cancellationToken)
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
                using (var response = await http.GetAsync(BaseUrl + path, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
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

    private static async Task<string> Md5Async(string path, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await MD5.HashDataAsync(file, cancellationToken));
    }
}
