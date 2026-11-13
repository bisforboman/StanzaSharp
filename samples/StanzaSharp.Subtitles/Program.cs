// Checks where an SRT file's two-line subtitles break against the grammar of their sentences (#267): a break that
// separates what the style guides keep on one line (an article and its noun, a preposition and its noun, a first and
// last name, a subject pronoun and its verb, a verb and its particle) is printed with the relation it splits and the
// best-scoring break instead. Sentences running over several cues are parsed whole.
//
//   dotnet run --project samples/StanzaSharp.Subtitles -- FILE.srt [--lang en|sv] [--models DIR] [--max-chars 42] [--json FILE]
//   dotnet run --project samples/StanzaSharp.Subtitles -- --measure FILE [--lang en|sv] [--models DIR] [--max-chars 42]
//
// FILE is SRT (.srt) or tagged text (any other extension: one sentence per line, <eol> and <eob> marking the breaks,
// as MuST-Cinema has them). --measure compares the file's own two-line breaks with breaks made at the midpoint of the
// same subtitles (README, Measurement).
// DIR defaults to models/stanza/LANG; the first run downloads tokenize, mwt, pos, lemma and depparse there (managed
// backend, no libtorch).

using System.Text;
using System.Text.Json;
using StanzaSharp;
using StanzaSharp.Subtitles;

string? file = null, json = null, modelDir = null;
var lang = "en";
var maxChars = BreakChecker.DefaultMaxChars;
var measure = false;
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--lang": lang = args[++i]; break;
        case "--models": modelDir = args[++i]; break;
        case "--max-chars": maxChars = int.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "--json": json = args[++i]; break;
        case "--measure": measure = true; break;
        default: file = args[i]; break;
    }
}

if (file == null || lang is not ("en" or "sv"))
{
    Console.Error.WriteLine("Usage: StanzaSharp.Subtitles [--measure] FILE [--lang en|sv] [--models DIR] [--max-chars 42] [--json FILE]");
    return 2;
}

modelDir ??= Path.Combine("models", "stanza", lang);
var options = new PipelineOptions { Language = lang, Processors = "tokenize,pos,lemma,depparse" };
await ModelDownloader.DownloadAsync(modelDir, options, new Progress<string>(line => Console.Error.WriteLine(line)));
using var nlp = Pipeline.Load(modelDir, options);

var cues = file.EndsWith(".srt", StringComparison.OrdinalIgnoreCase) ? Srt.Parse(File.ReadAllText(file)) : Tagged.Parse(File.ReadAllText(file));
var transcript = Transcript.Build(cues);
var document = nlp.Process(transcript.Text);
if (measure)
{
    var m = Measurement.Measure(cues, document, maxChars);
    Console.Out.WriteLine($"{m.Cues} cues; {m.Compared} two-line cues compared, {m.Skipped} dialogues or longer left out.");
    Console.Out.WriteLine($"flagged: own breaks {Percent(m.HumanFlagged, m.Compared)}, midpoint breaks {Percent(m.MidpointFlagged, m.Compared)}");
    Console.Out.WriteLine($"own break is the midpoint one: {Percent(m.SameBreak, m.Compared)}; where not ({m.Differing}): "
        + $"own flagged {Percent(m.HumanFlaggedDiffering, m.Differing)}, midpoint flagged {Percent(m.MidpointFlaggedDiffering, m.Differing)}");
    Console.Out.WriteLine($"best-scoring break is the own one: {Percent(m.BestIsHuman, m.Compared)} (the midpoint: {Percent(m.SameBreak, m.Compared)})");
    Console.Out.WriteLine($"a line over {maxChars} characters: {Percent(m.OverLimit, m.Compared)}");
    Console.Out.WriteLine("relation          own  midpoint");
    foreach (var r in m.HumanRelations.Keys.Union(m.MidpointRelations.Keys).Order(StringComparer.Ordinal))
        Console.Out.WriteLine($"{r,-16} {m.HumanRelations.GetValueOrDefault(r),4} {m.MidpointRelations.GetValueOrDefault(r),9}");
    return 0;
}

var findings = BreakChecker.Check(transcript, document, maxChars);

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

static string Percent(int part, int whole) =>
    whole == 0 ? "-" : string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{part}/{whole} ({100.0 * part / whole:F1}%)");

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
