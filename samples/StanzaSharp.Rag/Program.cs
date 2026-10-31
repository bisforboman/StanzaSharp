// RAG ingestion with StanzaSharp.DataIngestion (#233): Markdown files → SentenceChunker (whole sentences, sized with the
// embedding model's tokenizer) → EntityEnricher (named entities as metadata) → an in-memory vector store, embedded with
// all-MiniLM-L6-v2 on ONNX Runtime. Then a semantic search, unfiltered and filtered to chunks naming an entity. Everything
// runs locally: no service and no API key.
//
//   dotnet run --project samples/StanzaSharp.Rag -- [DOCS] [--query TEXT] [--entity NAME] [--models DIR]
//
// --entity "" skips the filtered search.
//
// DOCS defaults to the sample's own docs/ (fictional notes). DIR defaults to models; the first run downloads English
// tokenize and ner (about 210 MB) to DIR/stanza/en and the embedding model (90 MB) to DIR/all-MiniLM-L6-v2.

using Microsoft.Extensions.DataIngestion;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using StanzaSharp;
using StanzaSharp.DataIngestion;
using StanzaSharp.Rag;

var docs = Path.Combine(AppContext.BaseDirectory, "docs");
var models = "models";

// Similarity alone ranks the notes about the building first and misses the customer's request; the entity filter finds it.
var query = "What does the customer want in the new building?";
var entity = "Nordlys Seafood";
for (var i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--query": query = args[++i]; break;
        case "--entity": entity = args[++i]; break;
        case "--models": models = args[++i]; break;
        default: docs = args[i]; break;
    }
}

// NER reads only the tokens, so tokenize and ner are all both StanzaSharp steps need.
var stanzaDir = Path.Combine(models, "stanza", "en");
var options = new PipelineOptions { Processors = "tokenize,ner" };
await ModelDownloader.DownloadAsync(stanzaDir, options, new Progress<string>(line => Console.Error.WriteLine(line)));
using var nlp = Pipeline.Load(stanzaDir, options);
using var embedder = await MiniLmEmbeddingGenerator.LoadAsync(Path.Combine(models, "all-MiniLM-L6-v2"));

// Chunks are sized with the embedding model's own tokenizer, under its 256-token limit.
var chunker = new SentenceChunker(nlp, new IngestionChunkerOptions(embedder.Tokenizer) { MaxTokensPerChunk = 200, OverlapTokens = 40 });
var store = new InMemoryVectorStore(new InMemoryVectorStoreOptions { EmbeddingGenerator = embedder });
var writer = new VectorStoreWriter<string>(store, MiniLmEmbeddingGenerator.Dimensions, new VectorStoreWriterOptions { CollectionName = "docs" });
using var ingestion = new IngestionPipeline<string>(new MarkdownReader(), chunker, writer);
ingestion.ChunkProcessors.Add(new EntityEnricher(nlp, new HashSet<string> { "PERSON", "ORG", "GPE", "LOC", "FAC" }));

await foreach (var result in ingestion.ProcessAsync(new DirectoryInfo(docs), "*.md"))
    Console.WriteLine(result.Succeeded ? $"Ingested {Path.GetFileName(result.DocumentId)}" : $"Failed {result.DocumentId}: {result.Exception?.Message}");

// The writer disposes the store with the pipeline, so search inside this scope.
var collection = writer.VectorStoreCollection;
var counts = new Dictionary<string, int>();
await foreach (var record in collection.GetAsync(_ => true, 10_000))
{
    foreach (var name in (string[])record[EntityEnricher.MetadataKey]!)
        counts[name] = counts.GetValueOrDefault(name) + 1;
}

Console.WriteLine();
Console.WriteLine("Entities (chunks): " + string.Join(", ", counts.OrderByDescending(c => c.Value).ThenBy(c => c.Key).Take(12).Select(c => $"{c.Key} ({c.Value})")));

await Search($"\"{query}\"", null);
if (entity.Length > 0)
    await Search($"\"{query}\", only chunks naming {entity}", r => ((string[])r[EntityEnricher.MetadataKey]!).Contains(entity));
return 0;

async Task Search(string title, System.Linq.Expressions.Expression<Func<Dictionary<string, object?>, bool>>? filter)
{
    Console.WriteLine();
    Console.WriteLine(title);
    await foreach (var hit in collection.SearchAsync(query, 3, new VectorSearchOptions<Dictionary<string, object?>> { Filter = filter }))
    {
        var content = ((string)hit.Record["content"]!).ReplaceLineEndings(" ");
        Console.WriteLine($"  {hit.Score:F3}  [{Path.GetFileName((string)hit.Record["documentid"]!)}] {(content.Length > 160 ? content[..160] + "..." : content)}");
        Console.WriteLine($"         entities: {string.Join(", ", (string[])hit.Record[EntityEnricher.MetadataKey]!)}");
    }
}
