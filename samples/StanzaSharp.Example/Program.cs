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
//    Other options: Threads (libtorch threads, process-wide), SplitSentences = false (one sentence per paragraph),
//    VerifyChecksums (check the .pt files' MD5s first) and Logger (an ILogger for load and processing times).
//    Stanza's faster package (no constituency; download it with the same options first):
//    var fast = new PipelineOptions { Package = "default_fast" };
//    await ModelDownloader.DownloadAsync(modelDir, fast);
//    Pipeline.Load(modelDir, fast);
//    On an NVIDIA GPU (with the TorchSharp-cuda-* package instead of TorchSharp-cpu):
//    Pipeline.Load(modelDir, new PipelineOptions { Device = TorchSharp.torch.CUDA, DisableTf32 = true });
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

    // Words carry the lemma, tags, morphological features and the dependency tree (Head is a word Id, 0 = root).
    var words = sentence.Words.ToList();
    foreach (var word in words)
    {
        var head = word.Head is > 0 ? words[word.Head.Value - 1].Text : "ROOT";
        Console.WriteLine($"  {word.Id,2} {word.Text,-10} lemma={word.Lemma,-8} {word.Upos,-6} {word.Xpos,-4} {word.Deprel,-11} -> {head}");
    }
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
