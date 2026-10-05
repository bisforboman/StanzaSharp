using StanzaSharp;

const string Usage = """
    Usage: StanzaSharp.Cli [--models DIR] [--processors LIST] [FILE]

    Runs the English pipeline on FILE (or standard input) and writes CoNLL-U to standard output.

      --models DIR        converted models (default: models/converted/en)
      --processors LIST   comma-separated, from tokenize,mwt,pos,constituency (default: all)
    """;

string modelDir = Path.Combine("models", "converted", "en");
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
