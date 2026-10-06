using StanzaSharp;

const string Usage = """
    Usage: StanzaSharp.Cli [--models DIR] [--processors LIST] [FILE]
           StanzaSharp.Cli download [DIR]

    Runs the English pipeline on FILE (or standard input) and writes CoNLL-U to standard output.
    "download" fetches Stanza's English models into DIR (default: models/stanza/en).

      --models DIR        models (default: models/converted/en if present, else models/stanza/en)
      --processors LIST   comma-separated, from tokenize,mwt,pos,lemma,depparse,constituency (default: all)
    """;

string convertedDir = Path.Combine("models", "converted", "en");
string stanzaDir = Path.Combine("models", "stanza", "en");

if (args is ["download", ..])
{
    if (args.Length > 2)
    {
        Console.Error.WriteLine(Usage);
        return 2;
    }
    var target = args.Length == 2 ? args[1] : stanzaDir;
    try
    {
        await ModelDownloader.DownloadAsync(target, new Progress<string>(Console.Error.WriteLine));
        Console.Error.WriteLine($"Models are in {target}");
        return 0;
    }
    catch (Exception e) when (e is IOException or HttpRequestException or InvalidDataException or UnauthorizedAccessException)
    {
        Console.Error.WriteLine(e.Message);
        return 1;
    }
}

string modelDir = Directory.Exists(convertedDir) ? convertedDir : stanzaDir;
string processors = Pipeline.AllProcessors;
string? file = null;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models" when i + 1 < args.Length:
            modelDir = args[++i];
            break;
        case "--processors" when i + 1 < args.Length:
            processors = args[++i];
            break;
        case "-h" or "--help":
            Console.WriteLine(Usage);
            return 0;
        default:
            if (args[i].StartsWith('-') || file != null)
            {
                Console.Error.WriteLine($"Unexpected argument: {args[i]}\n\n{Usage}");
                return 2;
            }
            file = args[i];
            break;
    }
}

try
{
    var text = file != null ? File.ReadAllText(file) : Console.In.ReadToEnd();
    using var nlp = Pipeline.Load(modelDir, processors);
    var doc = nlp.Process(text);
    Console.Out.Write(Conllu.Write(doc));
    return 0;
}
catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
