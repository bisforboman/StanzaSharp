using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace StanzaSharp.Tool;

/// <summary>
/// <c>compare FILE [--package NAME] [--processors LIST] [--models DIR] [--python PATH]</c>: runs Python Stanza and
/// StanzaSharp on FILE with the same model files and reports whether their CoNLL-U is identical. Exit code 0 identical,
/// 1 different, 2 bad arguments or setup (Python, Stanza, models).
/// </summary>
internal static class CompareCommand
{
    internal static int Run(string[] args, string usage, TextWriter output)
    {
        string? file = null, processors = null;
        string package = Pipeline.DefaultPackage, models = DownloadCommand.DefaultDir, python = "python";
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--package" && i + 1 < args.Length)
                package = args[++i];
            else if (args[i] == "--processors" && i + 1 < args.Length)
                processors = args[++i];
            else if (args[i] == "--models" && i + 1 < args.Length)
                models = args[++i];
            else if (args[i] == "--python" && i + 1 < args.Length)
                python = args[++i];
            else if (!args[i].StartsWith('-') && file == null)
                file = args[i];
            else
                return Fail($"Unexpected argument: {args[i]}\n\n{usage}");
        }
        if (file == null)
            return Fail($"No input file given\n\n{usage}");
        if (!File.Exists(file))
            return Fail($"Input file not found: {file}");
        if (!Directory.Exists(models))
            return Fail($"Model directory not found: {models}. Download the models with: stanzasharp download {models} --package {package}");
        // Stanza looks for <dir>/en/<processor>/<name>.pt, so the folder holding the processors must be named en.
        models = Path.GetFullPath(models).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (Path.GetFileName(models) != "en")
            return Fail($"Python Stanza reads English models from a folder named en, not {models}. " +
                "Download them into one, e.g. stanzasharp download models/stanza/en");

        string list;
        HashSet<string> expectedFiles;
        try
        {
            // Both sides get the full list, with the processors the requested ones require (Stanza adds them itself).
            var selected = Pipeline.SelectModels(package, processors, addRequired: true, "--processors");
            list = string.Join(",", Pipeline.AllProcessors.Split(',').Where(selected.ContainsKey));
            expectedFiles = ModelDownloader.FilesFor(list, package).Select(f => f.Path).ToHashSet();
        }
        catch (ArgumentException e)
        {
            return Fail(e.Message);
        }
        var missing = expectedFiles.Where(f => !File.Exists(Path.Combine(models, f))).Order().ToList();
        if (missing.Count > 0)
            return Fail($"Missing in {models}: {string.Join(", ", missing)}. Python Stanza needs Stanza's .pt files; download them with: " +
                $"stanzasharp download {models} --package {package}");

        // Both sides get this string: StanzaSharp directly, Stanza as UTF-8 bytes it decodes again.
        var text = File.ReadAllText(file);
        var stanza = RunStanza(python, text, models, package, list);
        if (stanza == null)
            return 2;
        if (!stanza.Files.ToHashSet().SetEquals(expectedFiles) || string.Join(",", stanza.Processors) != list)
            return Fail($"Stanza and StanzaSharp would not run the same models.\n  Stanza:      {string.Join(",", stanza.Processors)}; " +
                $"{string.Join(", ", stanza.Files.Order())}\n  StanzaSharp: {list}; {string.Join(", ", expectedFiles.Order())}");

        Document doc;
        string conllu;
        double load, process;
        try
        {
            var watch = Stopwatch.StartNew();
            // VerifyChecksums: every file is Stanza's published one, so both sides read the same, unmodified models.
            using var nlp = Pipeline.Load(models, new PipelineOptions { Package = package, Processors = list, VerifyChecksums = true });
            load = watch.Elapsed.TotalSeconds;
            watch.Restart();
            doc = nlp.Process(text);
            process = watch.Elapsed.TotalSeconds;
            conllu = Conllu.Write(doc);
        }
        catch (Exception e) when (e is IOException or InvalidDataException or InvalidOperationException or ArgumentException or UnauthorizedAccessException)
        {
            return Fail(e.Message);
        }

        int nonBmp = text.Count(char.IsHighSurrogate);
        if (nonBmp > 0)
            conllu = ConlluDiff.ToCodePointOffsets(conllu, text);
        var difference = ConlluDiff.Find(stanza.Conllu, conllu);

        output.WriteLine($"Stanza {ModelDownloader.StanzaVersion} (Python) and StanzaSharp on {file}");
        output.WriteLine($"  package:     {package} ({list})");
        output.WriteLine($"  models:      {models}: the same {expectedFiles.Count} files on both sides, MD5s match Stanza {ModelDownloader.StanzaVersion}'s");
        output.WriteLine($"  text:        {doc.Sentences.Count:N0} sentences, {doc.Sentences.Sum(s => s.Words.Count()):N0} words");
        output.WriteLine($"  Stanza:      loaded in {stanza.Load:F1} s, processed in {stanza.Process:F1} s");
        output.WriteLine($"  StanzaSharp: loaded in {load:F1} s, processed in {process:F1} s");
        if (nonBmp > 0)
            output.WriteLine($"  note:        the text has {nonBmp:N0} characters outside the BMP (e.g. emoji). StanzaSharp's start_char/end_char " +
                "count UTF-16 code units like .NET strings, Stanza's count code points like Python; StanzaSharp's were converted to code points before comparing.");
        output.WriteLine();
        if (difference == null)
        {
            output.WriteLine("Identical: the CoNLL-U output is the same, byte for byte" + (nonBmp > 0 ? " (with offsets in code points)." : "."));
            return 0;
        }
        output.WriteLine($"Different: {difference.DifferentSentences:N0} of {difference.Sentences:N0} sentences differ. The first difference, line {difference.Line:N0} of the CoNLL-U:");
        if (difference.SentId != null)
            output.WriteLine($"  # sent_id = {difference.SentId}");
        if (difference.Text != null)
            output.WriteLine($"  # text = {difference.Text}");
        output.WriteLine($"  Stanza:      {Show(difference.Stanza)}");
        output.WriteLine($"  StanzaSharp: {Show(difference.StanzaSharp)}");
        return 1;

        static string Show(string? line) => line switch
        {
            null => "(end of output)",
            "" => "(blank line: end of the sentence)",
            _ => line,
        };
    }

    private sealed record StanzaResult(string Conllu, string[] Processors, string[] Files, double Load, double Process);

    /// <summary>Runs compare.py; null (after printing why) when Python or Stanza is missing or fails.</summary>
    private static StanzaResult? RunStanza(string python, string text, string models, string package, string processors)
    {
        var dir = Directory.CreateTempSubdirectory("stanzasharp-compare-");
        try
        {
            foreach (var name in new[] { "compare.py", "stanza_resources_en.json" })
            {
                using var resource = typeof(CompareCommand).Assembly.GetManifestResourceStream(name)!;
                using var target = File.Create(Path.Combine(dir.FullName, name));
                resource.CopyTo(target);
            }
            var input = Path.Combine(dir.FullName, "input.txt");
            var result = Path.Combine(dir.FullName, "result.json");
            File.WriteAllText(input, text, new UTF8Encoding(false));

            // Standard output and error are the user's console: Stanza's warnings and errors show as they happen.
            var start = new ProcessStartInfo(python);
            foreach (var arg in new[] { Path.Combine(dir.FullName, "compare.py"), input, result, models, package, processors, ModelDownloader.StanzaVersion })
                start.ArgumentList.Add(arg);
            Console.Error.WriteLine($"Running Stanza {ModelDownloader.StanzaVersion} with {python}...");
            Process process;
            try
            {
                process = Process.Start(start)!;
            }
            catch (Win32Exception e)
            {
                Fail($"Could not start Python ({python}: {e.Message}). Install Python 3 and Stanza {ModelDownloader.StanzaVersion} " +
                    $"(pip install stanza=={ModelDownloader.StanzaVersion}), or pass its path with --python.");
                return null;
            }
            using (process)
            {
                process.WaitForExit();
                if (process.ExitCode != 0)
                {
                    if (process.ExitCode != 3) // 3: compare.py already said what to install
                        Fail($"Python Stanza failed (exit code {process.ExitCode}); its error is above.");
                    return null;
                }
            }
            using var json = JsonDocument.Parse(File.ReadAllBytes(result));
            var root = json.RootElement;
            return new StanzaResult(root.GetProperty("conllu").GetString()!,
                [.. root.GetProperty("processors").EnumerateArray().Select(e => e.GetString()!)],
                [.. root.GetProperty("files").EnumerateArray().Select(e => e.GetString()!)],
                root.GetProperty("load").GetDouble(), root.GetProperty("process").GetDouble());
        }
        finally
        {
            dir.Delete(recursive: true);
        }
    }

    private static int Fail(string message)
    {
        Console.Error.WriteLine(message);
        return 2;
    }
}

/// <summary>The first difference between two CoNLL-U documents, and the known offset difference after non-BMP characters.</summary>
internal static class ConlluDiff
{
    /// <summary>
    /// <paramref name="Line"/> (1-based) is the first line that differs; <paramref name="SentId"/> and
    /// <paramref name="Text"/> come from the comments of Stanza's sentence there; a null line is past the end of the output.
    /// </summary>
    internal sealed record Difference(int Line, string? SentId, string? Text, string? Stanza, string? StanzaSharp, int Sentences, int DifferentSentences);

    /// <summary>Null when identical.</summary>
    internal static Difference? Find(string stanza, string stanzaSharp)
    {
        if (stanza == stanzaSharp)
            return null;
        var a = stanza.Split('\n');
        var b = stanzaSharp.Split('\n');
        int i = 0;
        while (i < a.Length && i < b.Length && a[i] == b[i])
            i++;

        // The sentence holding line i in Stanza's output, and its comment lines.
        int s = Math.Min(i, a.Length - 1);
        while (s > 0 && a[s - 1] != "")
            s--;
        string? sentId = null, text = null;
        for (int j = s; j < a.Length && a[j].StartsWith('#'); j++)
        {
            if (a[j].StartsWith("# sent_id = ", StringComparison.Ordinal))
                sentId = a[j]["# sent_id = ".Length..];
            else if (a[j].StartsWith("# text = ", StringComparison.Ordinal))
                text = a[j]["# text = ".Length..];
        }

        var sa = Sentences(stanza);
        var sb = Sentences(stanzaSharp);
        int different = Enumerable.Range(0, Math.Max(sa.Length, sb.Length)).Count(k => k >= sa.Length || k >= sb.Length || sa[k] != sb[k]);
        return new Difference(i + 1, sentId, text, i < a.Length ? a[i] : null, i < b.Length ? b[i] : null, sa.Length, different);

        static string[] Sentences(string conllu) => conllu.TrimEnd('\n').Split("\n\n");
    }

    /// <summary>
    /// StanzaSharp's CoNLL-U with the start_char/end_char values in MISC converted from UTF-16 indices into
    /// <paramref name="text"/> to code point indices, Python's (and so Stanza's) offsets.
    /// </summary>
    internal static string ToCodePointOffsets(string conllu, string text)
    {
        var codePoints = new int[text.Length + 1];
        for (int i = 0; i < text.Length; i++)
            codePoints[i + 1] = codePoints[i] + (char.IsLowSurrogate(text[i]) && i > 0 && char.IsHighSurrogate(text[i - 1]) ? 0 : 1);
        var lines = conllu.Split('\n');
        for (int l = 0; l < lines.Length; l++)
        {
            var columns = lines[l].Split('\t');
            if (columns.Length != 10 || lines[l].StartsWith('#'))
                continue;
            columns[9] = string.Join('|', columns[9].Split('|').Select(field =>
                field.Split('=', 2) is [("start_char" or "end_char") and var key, var value] && int.TryParse(value, out var offset)
                    ? $"{key}={codePoints[offset]}"
                    : field));
            lines[l] = string.Join('\t', columns);
        }
        return string.Join('\n', lines);
    }
}
