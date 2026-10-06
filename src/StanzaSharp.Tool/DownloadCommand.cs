namespace StanzaSharp.Tool;

/// <summary>
/// <c>download [DIR] [--package NAME] [--processors LIST]</c>, shared by the <c>stanzasharp</c> tool and
/// samples/StanzaSharp.Cli (which compiles this file in).
/// </summary>
internal static class DownloadCommand
{
    internal static readonly string DefaultDir = Path.Combine("models", "stanza", "en");

    /// <summary>Runs the command on the arguments after <c>download</c>; returns the exit code (2: bad arguments).</summary>
    internal static async Task<int> RunAsync(string[] args, string usage)
    {
        string? target = null, only = null;
        string package = Pipeline.DefaultPackage;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--processors" && i + 1 < args.Length)
                only = args[++i];
            else if (args[i] == "--package" && i + 1 < args.Length)
                package = args[++i];
            else if (!args[i].StartsWith('-') && target == null)
                target = args[i];
            else
            {
                Console.Error.WriteLine($"Unexpected argument: {args[i]}\n\n{usage}");
                return 2;
            }
        }
        target ??= DefaultDir;
        try
        {
            var progress = new Progress<string>(Console.Error.WriteLine);
            await ModelDownloader.DownloadAsync(target, new PipelineOptions { Package = package, Processors = only }, progress);
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
