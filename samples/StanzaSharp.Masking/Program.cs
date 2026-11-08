// Suggests what to mask (maskning) in a Swedish document before it is released: names, places and organisations from
// StanzaSharp's NER, plus personnummer, organisationsnummer, phone numbers, e-mail and addresses from patterns. Prints
// the text with placeholders; --json writes the suggested spans for a reviewing tool. A suggestion, not a decision:
// a person reviews every mask, and indirect identifiers ("hans syster, rektor på X-skolan") are left to them.
//
//   dotnet run --project samples/StanzaSharp.Masking -- FILE [--models DIR] [--types PER,LOC,ORG] [--link] [--json FILE]
//   dotnet run --project samples/StanzaSharp.Masking -- --eval DIR [--models DIR] [--types PER,LOC,ORG] [--verbose]
//
// --link also masks every other capitalized occurrence of the names NER found and gives each person one placeholder
// (NameLinker). --eval scores the labelled texts in DIR (eval/README.md) without and with --link.
//
// DIR defaults to models/stanza/sv; the first run downloads Swedish tokenize and ner there (about 210 MB).

using System.Text.Json;
using StanzaSharp;
using StanzaSharp.Masking;

string? file = null, json = null, eval = null;
bool link = false, verbose = false;
var modelDir = Path.Combine("models", "stanza", "sv");
IReadOnlySet<string> types = Masker.DefaultNerTypes;
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models": modelDir = args[++i]; break;
        case "--json": json = args[++i]; break;
        case "--types": types = args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries).ToHashSet(); break;
        case "--link": link = true; break;
        case "--eval": eval = args[++i]; break;
        case "--verbose": verbose = true; break;
        default: file = args[i]; break;
    }
}
if (file == null && eval == null)
{
    Console.Error.WriteLine("Usage: StanzaSharp.Masking FILE [--models DIR] [--types PER,LOC,ORG,TME] [--link] [--json FILE]");
    Console.Error.WriteLine("       StanzaSharp.Masking --eval DIR [--models DIR] [--types PER,LOC,ORG,TME] [--verbose]");
    return 2;
}

// NER reads only the tokens, so tokenize and ner are all it needs.
var options = new PipelineOptions { Language = "sv", Processors = "tokenize,ner" };
await ModelDownloader.DownloadAsync(modelDir, options, new Progress<string>(line => Console.Error.WriteLine(line)));
using var nlp = Pipeline.Load(modelDir, options);

if (eval != null)
{
    var runs = new List<(string Name, Score Score)> { ("NER + patterns", new Score()), ("with --link", new Score()) };
    foreach (var path in Directory.GetFiles(eval, "*.txt").Order(StringComparer.Ordinal))
    {
        var (plain, gold) = Evaluation.Parse(File.ReadAllText(path));
        var document = nlp.Process(plain);
        Evaluation.Add(runs[0].Score, Path.GetFileName(path), plain, gold, Masker.Find(plain, document, types));
        Evaluation.Add(runs[1].Score, Path.GetFileName(path), plain, gold, Masker.Find(plain, document, types, linkNames: true));
    }

    Console.Out.Write(Evaluation.Report(runs));
    if (verbose)
    {
        foreach (var (name, score) in runs)
        {
            Console.Out.WriteLine($"\n{name}: leaked");
            score.Leaks.ForEach(l => Console.Out.WriteLine("  " + l));
            Console.Out.WriteLine($"{name}: masked, overlapping no label");
            score.OverMasked.ForEach(l => Console.Out.WriteLine("  " + l));
            Console.Out.WriteLine($"{name}: people with more than one placeholder");
            score.Split.ForEach(l => Console.Out.WriteLine("  " + l));
            Console.Out.WriteLine($"{name}: different people under one placeholder");
            score.Joined.ForEach(l => Console.Out.WriteLine("  " + l));
        }
    }

    return 0;
}

var text = File.ReadAllText(file!);
var spans = Masker.Find(text, nlp.Process(text), types, link);
Console.Out.Write(Masker.Apply(text, spans));

if (json != null)
{
    using var stream = File.Create(json);
    using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
    writer.WriteStartArray();
    foreach (var span in spans)
    {
        writer.WriteStartObject();
        writer.WriteNumber("start", span.Start);
        writer.WriteNumber("length", span.Length);
        writer.WriteString("type", span.Type);
        writer.WriteString("source", span.Source);
        writer.WriteString("text", span.Text);
        if (span.Entity != null)
            writer.WriteString("entity", span.Entity);
        writer.WriteEndObject();
    }

    writer.WriteEndArray();
}
Console.Error.WriteLine($"{spans.Count} spans suggested; review them before release.");
return 0;
