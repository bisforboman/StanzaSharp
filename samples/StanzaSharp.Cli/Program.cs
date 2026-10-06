using StanzaSharp;

const string Usage = """
    Usage: StanzaSharp.Cli [--models DIR] [--package NAME] [--processors LIST] [FILE]
           StanzaSharp.Cli download [DIR] [--package NAME] [--processors LIST]

    Runs the English pipeline on FILE (or standard input) and writes CoNLL-U to standard output.
    "download" fetches Stanza's English models into DIR (default: models/stanza/en): all of the package's,
    or only what LIST needs.

      --models DIR        models (default: models/converted/en if present, else models/stanza/en)
      --package NAME      Stanza's English package: default, or default_fast (faster, no constituency)
      --processors LIST   comma-separated, from tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner
                          (default: all of the package's)
    """;

string convertedDir = Path.Combine("models", "converted", "en");
string stanzaDir = Path.Combine("models", "stanza", "en");

if (args is ["download", ..])
{
    string? target = null, only = null;
    string downloadPackage = Pipeline.DefaultPackage;
    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] == "--processors" && i + 1 < args.Length)
            only = args[++i];
        else if (args[i] == "--package" && i + 1 < args.Length)
            downloadPackage = args[++i];
        else if (!args[i].StartsWith('-') && target == null)
            target = args[i];
        else
        {
            Console.Error.WriteLine($"Unexpected argument: {args[i]}\n\n{Usage}");
            return 2;
        }
    }
    target ??= stanzaDir;
    try
    {
        var progress = new Progress<string>(Console.Error.WriteLine);
        await ModelDownloader.DownloadAsync(target, new PipelineOptions { Package = downloadPackage, Processors = only }, progress);
        Console.Error.WriteLine($"Models are in {target}");
        return 0;
    }
    catch (Exception e) when (e is IOException or HttpRequestException or InvalidDataException or UnauthorizedAccessException or ArgumentException)
    {
        Console.Error.WriteLine(e.Message);
        return 1;
    }
}

string modelDir = Directory.Exists(convertedDir) ? convertedDir : stanzaDir;
string? processors = null;
string package = Pipeline.DefaultPackage;
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
        case "--package" when i + 1 < args.Length:
            package = args[++i];
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
    using var nlp = Pipeline.Load(modelDir, new PipelineOptions { Package = package, Processors = processors });
    var doc = nlp.Process(text);
    Console.Out.Write(Conllu.Write(doc));
    return 0;
}
catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
