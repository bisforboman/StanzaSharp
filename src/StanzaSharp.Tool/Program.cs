using StanzaSharp;
using StanzaSharp.Tool;

const string Usage = $"""
    Usage: stanzasharp download [DIR] [--package NAME] [--processors LIST]

    Downloads Stanza {ModelDownloader.StanzaVersion}'s English models into DIR (default: models/stanza/en), the layout
    Pipeline.Load reads: all of the package's, or only what LIST needs. Each file is checked against its MD5;
    files already there and correct are kept.

      --package NAME      Stanza's English package: default, or default_fast (faster, no constituency)
      --processors LIST   comma-separated, from tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner
                          (default: all of the package's)
    """;

switch (args)
{
    case ["download", .. var rest]:
        return await DownloadCommand.RunAsync(rest, Usage);
    case ["-h" or "--help"]:
        Console.WriteLine(Usage);
        return 0;
    default:
        Console.Error.WriteLine(args is [] ? Usage : $"Unknown command: {args[0]}\n\n{Usage}");
        return 2;
}
