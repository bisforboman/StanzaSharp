using StanzaSharp;
using StanzaSharp.Tool;

const string Usage = $"""
    Usage: stanzasharp download [DIR] [--lang CODE] [--package NAME] [--processors LIST] [--model P=NAME]... [--list]
           stanzasharp compare FILE [--lang CODE] [--package NAME] [--processors LIST] [--model P=NAME]... [--models DIR]
                               [--python PATH]
           stanzasharp benchmark [--lang CODE] [--package NAME] [--processors LIST] [--model P=NAME]... [--models DIR]
                                 [--threads N] [--words N] [--python PATH] [--quick] [--json FILE]

    download: downloads Stanza {ModelDownloader.StanzaVersion}'s models of a language into DIR (default:
    models/stanza/CODE), the layout Pipeline.Load reads: all of the package's, or only what LIST needs. Each file is
    checked against its MD5; files already there and correct are kept.

    compare: runs Python Stanza {ModelDownloader.StanzaVersion} and StanzaSharp on FILE with the same models (DIR, default
    models/stanza/CODE, as download writes them) and reports whether their CoNLL-U output is identical, or the first
    difference. Needs Python with "pip install stanza=={ModelDownloader.StanzaVersion}". Exit code 0: identical,
    1: different, 2: bad arguments or setup.

    benchmark: times StanzaSharp on a built-in English or German text (load, each processor, words/s, one-sentence calls, peak
    memory) and, if Python Stanza {ModelDownloader.StanzaVersion} is available, Stanza on the same text with the same models
    and threads, alternating runs. Prints a Markdown block with the machine (CPU, cores, RAM, OS, .NET, SIMD) to share.
    Takes about 2 minutes with Python on 4 cores; --quick runs a short smoke test.

      --lang CODE         the language: en (default) or de (German)
      --package NAME      Stanza's package: default, or default_fast (faster, no constituency); the treebank
                          packages (tokenize to depparse only): English combined, ewt, gum, lines, partut, atis, childes, eslspok,
                          craft, genia, mimic;
                          German combined, gsd, hdt
      --processors LIST   comma-separated, from tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner
                          (default: all of the package's)
      --model P=NAME      a model for processor P instead of the package's (repeatable, or P=NAME,P=NAME), as Stanza's
                          processors dict: English constituency=ptb3_charlm; ner=ontonotes_charlm, ontonotes_nocharlm,
                          combined_charlm, i2b2 (clinical); German ner=conll03. Its processor is added if needed
      --list              download: print each file and its MD5 instead of downloading
      --models DIR        compare, benchmark: the models (default: models/stanza/CODE)
      --python PATH       compare, benchmark: the Python with Stanza installed (default: python)
      --threads N         benchmark: threads on both sides (default: each side's own default, StanzaSharp the
                          physical cores from 4 up, else all logical processors; torch the physical cores)
      --words N           benchmark: the text's length in words, counted by whitespace (default: 1500); the block
                          counts words after tokenizing (2,009 for the default)
      --quick             benchmark: 300 words (by whitespace), one run, 10 calls
      --json FILE         benchmark: also write the results as JSON
    """;

switch (args)
{
    case ["download", .. var rest]:
        return await DownloadCommand.RunAsync(rest, Usage);
    case ["compare", .. var rest]:
        return CompareCommand.Run(rest, Usage, Console.Out);
    case ["benchmark", .. var rest]:
        return BenchmarkCommand.Run(rest, Usage, Console.Out);
    case ["-h" or "--help"]:
        Console.WriteLine(Usage);
        return 0;
    default:
        Console.Error.WriteLine(args is [] ? Usage : $"Unknown command: {args[0]}\n\n{Usage}");
        return 2;
}
