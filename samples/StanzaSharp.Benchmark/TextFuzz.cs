using System.Text.Json;
using StanzaSharp;

/// <summary>
/// <c>text-fuzz</c>: the C# side of tools/text_fuzz.py (docs/text-fuzzing.md). Loads one pipeline, then reads one JSON
/// string per stdin line and answers each with one JSON line on stdout: <c>{"conllu": ...}</c> (Conllu.Write of
/// Process(text), UTF-16 offsets) or <c>{"error": ...}</c> when Process throws. Output is ASCII (non-ASCII escaped).
/// The ready line names the processors and models it loaded (<c>{"ready": true, "models": {...}}</c>, Stanza's order),
/// which text_fuzz.py passes to Stanza as its processors dict when <c>--model</c> is given. With <c>--pretagged</c>
/// (<see cref="PipelineOptions.Pretagged"/>) each line is a CoNLL-U document, annotated by Process(Conllu.Read(...)).
/// </summary>
internal static class TextFuzz
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark text-fuzz --models DIR [--lang CODE] [--package NAME] [--processors LIST]
                                     [--model P=NAME]... [--pretagged] [--backend torch|managed]
                                     [--threads N]
        """;

    public static int Run(string[] args)
    {
        string language = Pipeline.English, package = Pipeline.DefaultPackage;
        string? processors = null, modelDir = null;
        var models = new Dictionary<string, string>();
        bool managed = true, pretagged = false;
        int threads = 0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--models" when i + 1 < args.Length: modelDir = args[++i]; break;
                case "--lang" when i + 1 < args.Length: language = args[++i]; break;
                case "--package" when i + 1 < args.Length: package = args[++i]; break;
                case "--processors" when i + 1 < args.Length: processors = args[++i]; break;
                case "--model" when i + 1 < args.Length && AddModels(args[++i], models): break;
                case "--pretagged": pretagged = true; break;
                case "--backend" when i + 1 < args.Length: managed = args[++i] == "managed"; break;
                case "--threads" when i + 1 < args.Length: threads = int.Parse(args[++i]); break;
                default:
                    Console.Error.WriteLine(Usage);
                    return 2;
            }
        }

        if (modelDir == null)
        {
            Console.Error.WriteLine(Usage);
            return 2;
        }

        var chosen = models.Count > 0 ? models : null;
        using var nlp = Pipeline.Load(modelDir, new PipelineOptions
        {
            Language = language,
            Package = package,
            Processors = processors,
            Models = chosen,
            Pretagged = pretagged,
            Backend = managed ? PipelineBackend.Managed : CudaBackend.Cpu,
            Threads = threads > 0 ? threads : null,
        });
        var order = Pipeline.AllProcessors.Split(',');
        var loaded = Pipeline.SelectModels(package, processors, addRequired: false, "--model", language, chosen, pretagged)
            .OrderBy(m => Array.IndexOf(order, m.Key))
            .ToDictionary(m => m.Key, m => m.Value);
        Console.WriteLine(JsonSerializer.Serialize(new { ready = true, models = loaded }));
        while (Console.In.ReadLine() is { } line)
        {
            string reply;
            try
            {
                var input = JsonSerializer.Deserialize<string>(line)!;
                reply = JsonSerializer.Serialize(new { conllu = Conllu.Write(pretagged ? nlp.Process(Conllu.Read(input)) : nlp.Process(input)) });
            }
            catch (Exception e)
            {
                reply = JsonSerializer.Serialize(new { error = e.ToString() });
            }

            Console.WriteLine(reply);
        }

        return 0;
    }

    // --model's PROCESSOR=NAME pairs, repeatable or comma-separated, as the tool's (DownloadCommand.AddModels).
    private static bool AddModels(string value, Dictionary<string, string> models)
    {
        var pairs = value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var pair in pairs)
        {
            if (pair.Split('=', StringSplitOptions.TrimEntries) is not [{ Length: > 0 } processor, { Length: > 0 } model] || !models.TryAdd(processor, model))
                return false;
        }

        return pairs.Length > 0;
    }
}
