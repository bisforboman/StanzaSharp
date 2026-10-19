namespace StanzaSharp.Tool;

/// <summary>
/// <c>download [DIR] [--lang CODE] [--package NAME] [--processors LIST] [--model PROCESSOR=NAME]... [--list]</c>, shared by the <c>stanzasharp</c> tool and
/// samples/StanzaSharp.Cli (which compiles this file in).
/// </summary>
internal static class DownloadCommand
{
    internal static readonly string DefaultDir = DirFor("en");

    /// <summary>Where download puts a language's models without a DIR: <c>models/stanza/&lt;lang&gt;</c>.</summary>
    internal static string DirFor(string language) => Path.Combine("models", "stanza", language);

    /// <summary>
    /// Adds <c>--model</c>'s <c>PROCESSOR=NAME</c> pairs (comma-separated, e.g. <c>ner=i2b2,constituency=ptb3_charlm</c>)
    /// to <paramref name="models"/>; false if one is malformed. <see cref="Pipeline.Load"/> checks the names.
    /// </summary>
    internal static bool AddModels(string value, Dictionary<string, string> models)
    {
        var pairs = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var pair in pairs)
        {
            if (pair.Split('=', StringSplitOptions.TrimEntries) is not [{ Length: > 0 } processor, { Length: > 0 } model] || !models.TryAdd(processor, model))
                return false;
        }

        return pairs.Length > 0;
    }

    /// <summary><paramref name="models"/>, or null when empty (no <c>--model</c>).</summary>
    internal static Dictionary<string, string>? ModelsOrNull(Dictionary<string, string> models) => models.Count > 0 ? models : null;

    /// <summary>Runs the command on the arguments after <c>download</c>; returns the exit code (2: bad arguments).</summary>
    internal static async Task<int> RunAsync(string[] args, string usage)
    {
        string? target = null, only = null;
        string package = Pipeline.DefaultPackage, language = "en";
        var models = new Dictionary<string, string>();
        bool list = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--processors" && i + 1 < args.Length)
            {
                only = args[++i];
            }
            else if (args[i] == "--list")
            {
                list = true;
            }
            else if (args[i] == "--model" && i + 1 < args.Length && AddModels(args[i + 1], models))
            {
                i++;
            }
            else if (args[i] == "--package" && i + 1 < args.Length)
            {
                package = args[++i];
            }
            else if (args[i] == "--lang" && i + 1 < args.Length)
            {
                language = args[++i];
            }
            else if (!args[i].StartsWith('-') && target == null)
            {
                target = args[i];
            }
            else
            {
                Console.Error.WriteLine($"Unexpected argument: {args[i]}\n\n{usage}");
                return 2;
            }
        }

        target ??= DirFor(language);
        try
        {
            // --lang multilingual: the language identifier's model (LanguageIdentifier), Stanza's only multilingual resource.
            if (language == ModelDownloader.Multilingual)
            {
                if (list)
                {
                    Console.WriteLine($"{ModelDownloader.LanguageIdentifierFile.Path} {ModelDownloader.LanguageIdentifierFile.Md5}");
                    return 0;
                }

                await ModelDownloader.DownloadLanguageIdentifierAsync(target, new Progress<string>(Console.Error.WriteLine));
                Console.Error.WriteLine($"The language identifier is in {target}");
                return 0;
            }

            // --list (for CI's cache keys, tools/ci-models.sh): each file and its MD5, what a download would fetch; no network.
            if (list)
            {
                foreach (var (path, md5) in ModelDownloader.FilesFor(only, package, "--processors", language, ModelsOrNull(models)).OrderBy(f => f.Path, StringComparer.Ordinal))
                    Console.WriteLine($"{path} {md5}");
                return 0;
            }

            var progress = new Progress<string>(Console.Error.WriteLine);
            await ModelDownloader.DownloadAsync(target, new PipelineOptions { Language = language, Package = package, Processors = only, Models = ModelsOrNull(models) }, progress);
            Console.Error.WriteLine($"Models are in {target}");
            return 0;
        }
        catch (Exception e) when (e is IOException or HttpRequestException or InvalidDataException or UnauthorizedAccessException or ArgumentException)
        {
            Console.Error.WriteLine(e.Message);
            return 1;
        }
    }
}
