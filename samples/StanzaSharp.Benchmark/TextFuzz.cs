using System.Text.Json;
using StanzaSharp;

/// <summary>
/// <c>text-fuzz</c>: the C# side of tools/text_fuzz.py (docs/text-fuzzing.md). Loads one pipeline, then reads one JSON
/// string per stdin line and answers each with one JSON line on stdout: <c>{"conllu": ...}</c> (Conllu.Write of
/// Process(text), UTF-16 offsets) or <c>{"error": ...}</c> when Process throws. Output is ASCII (non-ASCII escaped).
/// </summary>
internal static class TextFuzz
{
    private const string Usage = """
        Usage: StanzaSharp.Benchmark text-fuzz --models DIR [--lang CODE] [--package NAME] [--processors LIST]
                                     [--backend torch|managed] [--threads N]
        """;

    public static int Run(string[] args)
    {
        string language = Pipeline.English, package = Pipeline.DefaultPackage;
        string? processors = null, modelDir = null;
        bool managed = true;
        int threads = 0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--models" when i + 1 < args.Length: modelDir = args[++i]; break;
                case "--lang" when i + 1 < args.Length: language = args[++i]; break;
                case "--package" when i + 1 < args.Length: package = args[++i]; break;
                case "--processors" when i + 1 < args.Length: processors = args[++i]; break;
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

        using var nlp = Pipeline.Load(modelDir, new PipelineOptions
        {
            Language = language,
            Package = package,
            Processors = processors,
            Backend = managed ? PipelineBackend.Managed : CudaBackend.Cpu,
            Threads = threads > 0 ? threads : null,
        });
        Console.WriteLine("{\"ready\": true}");
        while (Console.In.ReadLine() is { } line)
        {
            string reply;
            try
            {
                reply = JsonSerializer.Serialize(new { conllu = Conllu.Write(nlp.Process(JsonSerializer.Deserialize<string>(line)!)) });
            }
            catch (Exception e)
            {
                reply = JsonSerializer.Serialize(new { error = e.ToString() });
            }

            Console.WriteLine(reply);
        }

        return 0;
    }
}
