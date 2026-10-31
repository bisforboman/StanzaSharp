// Suggests what to mask (maskning) in a Swedish document before it is released: names, places and organisations from
// StanzaSharp's NER, plus personnummer, organisationsnummer, phone numbers, e-mail and addresses from patterns. Prints
// the text with placeholders; --json writes the suggested spans for a reviewing tool. A suggestion, not a decision:
// a person reviews every mask, and indirect identifiers ("hans syster, rektor på X-skolan") are left to them.
//
//   dotnet run --project samples/StanzaSharp.Masking -- FILE [--models DIR] [--types PER,LOC,ORG] [--json FILE]
//
// DIR defaults to models/stanza/sv; the first run downloads Swedish tokenize and ner there (about 210 MB).

using System.Text.Json;
using StanzaSharp;
using StanzaSharp.Masking;

string? file = null, json = null;
var modelDir = Path.Combine("models", "stanza", "sv");
IReadOnlySet<string> types = Masker.DefaultNerTypes;
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--models": modelDir = args[++i]; break;
        case "--json": json = args[++i]; break;
        case "--types": types = args[++i].Split(',', StringSplitOptions.RemoveEmptyEntries).ToHashSet(); break;
        default: file = args[i]; break;
    }
}
if (file == null)
{
    Console.Error.WriteLine("Usage: StanzaSharp.Masking FILE [--models DIR] [--types PER,LOC,ORG,TME] [--json FILE]");
    return 2;
}

// NER reads only the tokens, so tokenize and ner are all it needs.
var options = new PipelineOptions { Language = "sv", Processors = "tokenize,ner" };
await ModelDownloader.DownloadAsync(modelDir, options, new Progress<string>(line => Console.Error.WriteLine(line)));
using var nlp = Pipeline.Load(modelDir, options);

var text = File.ReadAllText(file);
var spans = Masker.Find(text, nlp.Process(text), types);
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
        writer.WriteEndObject();
    }

    writer.WriteEndArray();
}
Console.Error.WriteLine($"{spans.Count} spans suggested; review them before release.");
return 0;
