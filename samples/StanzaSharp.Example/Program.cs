// A tour of StanzaSharp's public API: download the models, load a pipeline, and read every kind of result.
//
//   dotnet run --project samples/StanzaSharp.Example [MODEL_DIR]
//
// MODEL_DIR defaults to models/stanza/en; the first run downloads the models there (about 600 MB).

using StanzaSharp;

var modelDir = args.Length > 0 ? args[0] : Path.Combine("models", "stanza", "en");

// 1. Models. Pipeline.Load never downloads, so fetch them once. Files already present (and intact) are kept,
//    so running this again is cheap. To fetch less, name the processors:
//    await ModelDownloader.DownloadAsync(modelDir, "tokenize,mwt,pos,lemma");
await ModelDownloader.DownloadAsync(modelDir, new Progress<string>(line => Console.Error.WriteLine(line)));

// 2. Load. The default runs all eight processors, like Stanza's English default. To run fewer, list them:
//    Pipeline.Load(modelDir, new PipelineOptions { Processors = "tokenize,mwt,pos" });
//    or with constants, which catch typos at compile time:
//    Pipeline.Load(modelDir, new PipelineOptions { Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos}" });
//    Other options: Threads (the size of the process-wide thread pool), SplitSentences = false (one sentence per
//    paragraph), VerifyChecksums (check the .pt files' MD5s first) and Logger (an ILogger for load and processing times).
//    Stanza's faster package (no constituency; download it with the same options first):
//    var fast = new PipelineOptions { Package = "default_fast" };
//    await ModelDownloader.DownloadAsync(modelDir, fast);
//    Pipeline.Load(modelDir, fast);
//    The models run on the managed backend (plain C#, no native libraries). The TorchSharp backend gives the same
//    output; it needs the StanzaSharp.Cuda and TorchSharp-cpu packages (TorchSharp-cuda-* for a GPU, see docs/gpu.md):
//    Pipeline.Load(modelDir, new PipelineOptions { Backend = CudaBackend.Cpu });
using var nlp = Pipeline.Load(modelDir);

// 3. Process text. Blank lines separate paragraphs; sentences are found automatically. Process is thread-safe, and
//    every overload also takes a CancellationToken: nlp.Process(text, cancellationToken).
var doc = nlp.Process("""
    Barack Obama was born in Hawaii. He didn't move to Chicago until 1985.

    The new library is fast, and I love it!
    """);

foreach (var sentence in doc.Sentences)
{
    Console.WriteLine($"# {sentence.Text}");

    // Sentence-level results: sentiment (0 negative, 1 neutral, 2 positive) and the constituency parse.
    Console.WriteLine($"sentiment: {sentence.Sentiment}");
    Console.WriteLine($"parse:     {sentence.Constituency}");

    // Tokens are what the text contains; a multi-word token ("didn't") holds several syntactic words.
    foreach (var token in sentence.Tokens.Where(t => t.IsMultiWord))
        Console.WriteLine($"MWT:       {token.Text} -> {string.Join(" + ", token.Words.Select(w => w.Text))}");

    // Words carry the lemma, tags, morphological features and the dependency tree (Head is a word Id, 0 = root);
    // sentence.Dependencies gives each word's arc with the head word itself (null for the root).
    foreach (var (head, relation, word) in sentence.Dependencies)
        Console.WriteLine($"  {word.Id,2} {word.Text,-10} lemma={word.Lemma,-8} {word.Upos,-6} {word.Xpos,-4} {relation,-11} -> {head?.Text ?? "ROOT"}");

    Console.WriteLine();
}

// 4. Named entities, with their character offsets into the original text.
foreach (var entity in doc.Entities)
    Console.WriteLine($"{entity.Type,-8} {entity.Text}  [{entity.StartChar}..{entity.EndChar})");

// 5. CoNLL-U, exactly as Python Stanza writes it; Conllu.Read parses it back.
var conllu = Conllu.Write(doc);
Console.WriteLine();
Console.WriteLine(conllu.Split('\n').Take(8).Aggregate((a, b) => a + "\n" + b));
var roundTrip = Conllu.Read(conllu);
Console.WriteLine($"... round trip: {roundTrip.Sentences.Count} sentences, {roundTrip.Entities.Count()} entities");

// 6. Other languages. Language identification (a 9 MB model in Stanza's multilingual folder, next to en/):
//    await ModelDownloader.DownloadLanguageIdentifierAsync("models/stanza/multilingual");
//    using var identifier = LanguageIdentifier.Load("models/stanza/multilingual");
//    identifier.Identify("Das ist ein deutscher Satz."); // "de"
//    MultilingualPipeline routes each text to its language's pipeline and sets Document.Lang. It loads English and
//    German by default (download German with new PipelineOptions { Language = "de" } into models/stanza/de); a text in
//    another language comes back with only Text and Lang, unless Restrict = true:
//    using var multi = MultilingualPipeline.Load("models/stanza");
//    foreach (var d in multi.Process(["I don't think she's coming to the meeting.", "Die Bundesregierung hat am Mittwoch neue Regeln beschlossen."]))
//        Console.WriteLine($"{d.Lang}: {d.Sentences[0].Text}"); // en: ..., de: ...
