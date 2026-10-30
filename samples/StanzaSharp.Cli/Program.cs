using StanzaSharp;

const string Usage = """
    Usage: StanzaSharp.Cli [--lang CODE] [--models DIR] [--package NAME] [--processors LIST] [--model P=NAME]... [FILE]
           StanzaSharp.Cli download [DIR] [--lang CODE] [--package NAME] [--processors LIST] [--model P=NAME]... [--list]
           StanzaSharp.Cli download-transformer DIR NAME [FILE]...

    Runs the pipeline of --lang (en, the default) on FILE (or standard input) and writes CoNLL-U to standard output.
    "download" fetches Stanza's models of --lang into DIR (default: models/stanza/<lang>):
    all of the package's, or only what LIST needs. "download-transformer" fetches a transformer's files (all, or the FILEs)
    from Hugging Face at their pinned revision into DIR/transformer/NAME (google/electra-large-discriminator: tokenizer.json,
    pytorch_model.bin; docs/transformer-plan.md), for tests and tools; "download" fetches them too for a model that reads them.

      --lang CODE         the language: en (default), de or another code of docs/languages.md
      --models DIR        models (default: models/converted/<lang> if present, else models/stanza/<lang>)
      --package NAME      Stanza's package: default, default_fast (faster, no constituency), default_accurate
                          (transformer models: slower, more accurate); the treebank
                          packages (tokenize to depparse): English combined, ewt, gum, lines, partut, atis, childes, eslspok,
                          craft, genia, mimic;
                          German combined, gsd, hdt (other languages: default; fr, it, pt, nl also default_fast)
      --processors LIST   comma-separated, from tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner
                          (default: all of the package's)
      --model P=NAME      a model for processor P instead of the package's (repeatable, or P=NAME,P=NAME), as Stanza's
                          processors dict: English constituency=ptb3_charlm; ner=ontonotes_charlm, ontonotes_nocharlm,
                          combined_charlm, conll03_charlm, i2b2, radiology (clinical),
                          anatem, bc4chemd, bc5cdr, bionlp13cg, jnlpba, linnaeus, ncbi_disease, s800 (biomedical),
                          ontonotes-ww-multi_electra-large, ontonotes_electra-large, conll03_electra-large; pos and
                          depparse=combined_electra-large, ewt, gum, lines, partut, atis, childes, eslspok + _electra-large;
                          constituency=ptb3-revised_electra-large, ptb3_electra-large; sentiment=sstplus_electra-large (these read the
                          1.3 GB google/electra-large-discriminator, which download fetches too); German ner=conll03.
                          Its processor is added if needed
      --list              download: print each file and its MD5 instead of downloading
      --backend NAME      managed (default; no native code) or torch (TorchSharp, with this sample's TorchSharp-cpu)
    """;

if (args is ["download-transformer", var transformerDir, var transformerName, .. var transformerFiles])
{
    await ModelDownloader.DownloadTransformerAsync(transformerDir, transformerName, transformerFiles.Length > 0 ? transformerFiles : null,
        new Progress<string>(Console.WriteLine));
    return 0;
}

if (args is ["download", .. var downloadArgs])
    return await StanzaSharp.Tool.DownloadCommand.RunAsync(downloadArgs, Usage);

string? modelDir = null;
string language = "en";
string? processors = null;
string package = Pipeline.DefaultPackage;
string? file = null;
var models = new Dictionary<string, string>();
var backend = PipelineBackend.Managed;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models" when i + 1 < args.Length:
            modelDir = args[++i];
            break;
        case "--lang" when i + 1 < args.Length:
            language = args[++i];
            break;
        case "--processors" when i + 1 < args.Length:
            processors = args[++i];
            break;
        case "--package" when i + 1 < args.Length:
            package = args[++i];
            break;
        case "--model" when i + 1 < args.Length && StanzaSharp.Tool.DownloadCommand.AddModels(args[i + 1], models):
            i++;
            break;
        case "--backend" when i + 1 < args.Length && args[i + 1] is "managed" or "torch":
            backend = args[++i] == "managed" ? PipelineBackend.Managed : CudaBackend.Cpu;
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
    string convertedDir = Path.Combine("models", "converted", language);
    modelDir ??= Directory.Exists(convertedDir) ? convertedDir : StanzaSharp.Tool.DownloadCommand.DirFor(language);
    var text = file != null ? File.ReadAllText(file) : Console.In.ReadToEnd();
    using var nlp = Pipeline.Load(modelDir, new PipelineOptions { Language = language, Package = package, Processors = processors, Models = StanzaSharp.Tool.DownloadCommand.ModelsOrNull(models), Backend = backend });
    var doc = nlp.Process(text);
    Console.Out.Write(Conllu.Write(doc));
    return 0;
}
catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
{
    Console.Error.WriteLine(e.Message);
    return 1;
}
