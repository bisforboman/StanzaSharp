// Checks where an SRT file's two-line subtitles break against the grammar of their sentences (#267): a break that
// separates what the style guides keep on one line (an article and its noun, a preposition and its noun, a first and
// last name, a subject pronoun and its verb, a verb and its particle) is printed with the relation it splits and the
// best-scoring break instead. Sentences running over several cues are parsed whole.
//
//   dotnet run --project samples/StanzaSharp.Subtitles -- FILE.srt [--lang en|sv] [--models DIR] [--max-chars 42] [--json FILE]
//
// DIR defaults to models/stanza/LANG; the first run downloads tokenize, mwt, pos, lemma and depparse there (managed
// backend, no libtorch).

using System.Text;
using System.Text.Json;
using StanzaSharp;
using StanzaSharp.Subtitles;

string? file = null, json = null, modelDir = null;
var lang = "en";
var maxChars = BreakChecker.DefaultMaxChars;
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--lang": lang = args[++i]; break;
        case "--models": modelDir = args[++i]; break;
        case "--max-chars": maxChars = int.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--json": json = args[++i]; break;
        default: file = args[i]; break;
    }
}

if (file == null || lang is not ("en" or "sv"))
{
    Console.Error.WriteLine("Usage: StanzaSharp.Subtitles FILE.srt [--lang en|sv] [--models DIR] [--max-chars 42] [--json FILE]");
    return 2;
}

modelDir ??= Path.Combine("models", "stanza", lang);
var options = new PipelineOptions { Language = lang, Processors = "tokenize,pos,lemma,depparse" };
await ModelDownloader.DownloadAsync(modelDir, options, new Progress<string>(line => Console.Error.WriteLine(line)));
using var nlp = Pipeline.Load(modelDir, options);

var cues = Srt.Parse(File.ReadAllText(file));
var transcript = Transcript.Build(cues);
var findings = BreakChecker.Check(transcript, nlp.Process(transcript.Text), maxChars);

var output = new StringBuilder();
foreach (var f in findings)
{
    output.AppendLine($"#{f.Cue.Number}  {f.Cue.Timing}");
    for (var l = 0; l < f.Lines.Count; l++)
        output.AppendLine($"    {f.Lines[l]}{(l == f.Line - 1 ? "  |" : "")}");
    foreach (var s in f.Splits)
        output.AppendLine($"  splits {s.Relation} ({s.Dependent} → {s.Head}): {s.Rule}");
    if (f.FitsOneLine)
        output.AppendLine($"  fits on one line: {Srt.Join(f.Lines)}");
    if (f.Suggestion is { } s2)
    {
        output.AppendLine($"  suggest: {s2.Top}");
        output.AppendLine($"           {s2.Bottom}");
        foreach (var s in s2.Splits)
            output.AppendLine($"           (still splits {s.Relation}: {s.Dependent} → {s.Head})");
    }

    output.AppendLine();
}

var breaks = transcript.Lines.Sum(l => Math.Max(0, l.Count - 1));
output.AppendLine($"{findings.Count} of {breaks} line breaks in {cues.Count} cues flagged.");
Console.Out.Write(output);

if (json != null)
{
    using var stream = File.Create(json);
    using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
    writer.WriteStartArray();
    foreach (var f in findings)
    {
        writer.WriteStartObject();
        writer.WriteNumber("cue", f.Cue.Number);
        writer.WriteString("timing", f.Cue.Timing);
        writer.WriteNumber("breakAfterLine", f.Line);
        writer.WriteStartArray("lines");
        foreach (var line in f.Lines)
            writer.WriteStringValue(line);
        writer.WriteEndArray();
        WriteSplits(writer, "splits", f.Splits);
        writer.WriteBoolean("fitsOneLine", f.FitsOneLine);
        if (f.Suggestion is { } s)
        {
            writer.WriteStartObject("suggestion");
            writer.WriteString("top", s.Top);
            writer.WriteString("bottom", s.Bottom);
            WriteSplits(writer, "splits", s.Splits);
            writer.WriteEndObject();
        }

        writer.WriteEndObject();
    }

    writer.WriteEndArray();
}

return 0;

static void WriteSplits(Utf8JsonWriter writer, string name, IReadOnlyList<SplitRelation> splits)
{
    writer.WriteStartArray(name);
    foreach (var s in splits)
    {
        writer.WriteStartObject();
        writer.WriteString("relation", s.Relation);
        writer.WriteString("dependent", s.Dependent);
        writer.WriteString("head", s.Head);
        writer.WriteString("rule", s.Rule);
        writer.WriteEndObject();
    }

    writer.WriteEndArray();
}
