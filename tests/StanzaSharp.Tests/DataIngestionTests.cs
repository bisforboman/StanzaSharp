using Microsoft.Extensions.DataIngestion;
using Microsoft.ML.Tokenizers;
using StanzaSharp.DataIngestion;

namespace StanzaSharp.Tests;

/// <summary>
/// StanzaSharp.DataIngestion (#233): <see cref="SentenceChunker"/> and <see cref="EntityEnricher"/> on the golden corpus
/// texts, sizes counted with GPT-4o's tokenizer (o200k, offline from its data package).
/// </summary>
public class DataIngestionTests(SharedPipelines pipelines) : IClassFixture<SharedPipelines>
{
    private static readonly Tokenizer Tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");

    [ModelTheory]
    [InlineData(40, 0, "en")]
    [InlineData(40, 15, "en")]
    [InlineData(60, 20, "en")]
    [German(40, 15)]
    public async Task Chunks_AreWholeSentencesWithinTheLimit(int maxTokens, int overlapTokens, string lang)
    {
        var pipeline = pipelines.Get(processors: "tokenize", lang: lang);
        var paragraphs = Paragraphs(lang);
        var chunker = new SentenceChunker(pipeline, Options(maxTokens, overlapTokens));
        var chunks = await Chunk(chunker, Document(paragraphs));

        // Every chunk fits and holds whole sentences; without the overlap, they are the text's sentences in order.
        var sentences = pipeline.Process(paragraphs).SelectMany(d => d.Sentences).Select(s => Text(s)).ToList();
        Assert.All(chunks, c => Assert.InRange(Tokenizer.CountTokens(c.Content), 1, maxTokens));
        var next = 0;
        var overlapping = 0;
        string? previous = null;
        foreach (var chunk in chunks)
        {
            var pieces = Pieces(chunk.Content, sentences);
            var overlap = previous == null ? 0 : pieces.TakeWhile(p => previous.Contains(p, StringComparison.Ordinal)).Count();
            if (overlapTokens == 0)
                Assert.Equal(0, overlap);
            Assert.Equal(sentences[next..(next + pieces.Count - overlap)], pieces[overlap..]);
            next += pieces.Count - overlap;
            overlapping += overlap > 0 ? 1 : 0;
            previous = chunk.Content;
        }

        Assert.Equal(sentences.Count, next);
        Assert.True(chunks.Count >= 4, $"{chunks.Count} chunks");
        if (overlapTokens > 0)
            Assert.True(overlapping >= chunks.Count / 3, $"{overlapping} of {chunks.Count} chunks start with an overlap");
    }

    // With room for every section, the chunks are HeaderChunker's: the same contents and contexts.
    [ModelFact]
    public async Task WithRoomToSpare_ChunksAreHeaderChunkers()
    {
        var document = new IngestionDocument("doc");
        var section = new IngestionDocumentSection();
        section.Elements.Add(new IngestionDocumentHeader("# Weather") { Level = 1 });
        section.Elements.Add(new IngestionDocumentParagraph("I don't think it's going to rain today, but we'll see. Can't you hear the thunder?"));
        section.Elements.Add(new IngestionDocumentHeader("## Travel") { Level = 2 });
        section.Elements.Add(new IngestionDocumentParagraph("Dr. Smith moved to the U.S. in Jan. 1999. He paid $4.50 for coffee."));
        section.Elements.Add(new IngestionDocumentParagraph("\"Where are you going?\" she asked."));
        section.Elements.Add(new IngestionDocumentHeader("# Other") { Level = 1 });
        section.Elements.Add(new IngestionDocumentParagraph("Barack Obama was born in Hawaii."));
        document.Sections.Add(section);

        var options = Options(1000, 0);
        var expected = await Chunk(new HeaderChunker(options), document);
        var actual = await Chunk(new SentenceChunker(pipelines.Get(processors: "tokenize"), options), document);
        Assert.Equal(3, actual.Count);
        Assert.Equal(expected.Select(c => (c.Context, c.Content)), actual.Select(c => (c.Context, c.Content)));
        Assert.Equal("# Weather ## Travel", actual[1].Context);
    }

    [ModelFact]
    public async Task Overlap_IsTheLastSentencesUnderTheSameHeader()
    {
        var text = "The first sentence is here. The second one follows. A third comes now. The fourth ends it.";
        var twoSentences = Tokenizer.CountTokens("The first sentence is here. The second one follows.");
        var chunker = new SentenceChunker(pipelines.Get(processors: "tokenize"), Options(twoSentences + 1, 8));
        var chunks = await Chunk(chunker, Document([text]));
        Assert.Equal(
            [
                "The first sentence is here. The second one follows.",
                "The second one follows. A third comes now.",
                "A third comes now. The fourth ends it.",
            ],
            chunks.Select(c => c.Content));
    }

    [ModelFact]
    public async Task TablesStayWhole_AndALongSentenceIsCut()
    {
        var cells = new IngestionDocumentElement[,]
        {
            { new IngestionDocumentParagraph("Year"), new IngestionDocumentParagraph("Event") },
            { new IngestionDocumentParagraph("2008"), new IngestionDocumentParagraph("Elected. Then sworn in.") },
        };
        var table = new IngestionDocumentTable("| Year | Event |\n| --- | --- |\n| 2008 | Elected. Then sworn in. |", cells);
        var longSentence = string.Join(", ", Enumerable.Repeat("and the rain kept falling over the hills", 12)) + ".";
        var document = new IngestionDocument("doc");
        var section = new IngestionDocumentSection();
        section.Elements.Add(new IngestionDocumentParagraph("Obama was elected in 2008."));
        section.Elements.Add(table);
        section.Elements.Add(new IngestionDocumentParagraph(longSentence));
        document.Sections.Add(section);

        var chunks = await Chunk(new SentenceChunker(pipelines.Get(processors: "tokenize"), Options(40, 10)), document);
        Assert.Contains(chunks, c => c.Content.Contains(table.GetMarkdown(), StringComparison.Ordinal));
        Assert.All(chunks, c => Assert.InRange(Tokenizer.CountTokens(c.Content), 1, 40));
        var cut = chunks.Where(c => longSentence.Contains(c.Content, StringComparison.Ordinal)).Select(c => c.Content);
        Assert.Equal(longSentence, string.Concat(cut));
    }

    [ModelFact]
    public async Task Enricher_AddsTheEntitiesOfEachChunk()
    {
        var pipeline = pipelines.Get(processors: "tokenize,ner");
        var document = Document(["Barack Obama was born in Hawaii. Obama met Angela Merkel in Berlin.", "It rained."]);
        var chunks = await Chunk(new SentenceChunker(pipeline, Options(1000, 0)), document);
        var all = await Enrich(new EntityEnricher(pipeline), chunks);
        Assert.Equal(["Barack Obama", "Hawaii", "Obama", "Angela Merkel", "Berlin"], (string[])all.Single().Metadata[EntityEnricher.MetadataKey]);

        var places = await Enrich(new EntityEnricher(pipeline, new HashSet<string> { "GPE" }, batchSize: 1), chunks);
        Assert.Equal(["Hawaii", "Berlin"], (string[])places.Single().Metadata[EntityEnricher.MetadataKey]);
    }

    [GermanFact]
    public async Task Enricher_TagsGermanEntities()
    {
        var pipeline = pipelines.Get(processors: "tokenize,ner", lang: "de");
        var paragraphs = Paragraphs("de");
        var chunks = await Chunk(new SentenceChunker(pipeline, Options(60, 0)), Document(paragraphs));
        var enriched = await Enrich(new EntityEnricher(pipeline, batchSize: 3), chunks);
        Assert.Equal(chunks.Count, enriched.Count);
        Assert.Contains("Angela Merkel", (string[])enriched[0].Metadata[EntityEnricher.MetadataKey]);
        Assert.All(enriched, c => Assert.IsType<string[]>(c.Metadata[EntityEnricher.MetadataKey]));
    }

    [ModelFact]
    public void Enricher_NeedsNer() =>
        Assert.Throws<ArgumentException>(() => new EntityEnricher(pipelines.Get(processors: "tokenize")));

    private static IngestionChunkerOptions Options(int maxTokens, int overlapTokens) =>
        new(Tokenizer) { MaxTokensPerChunk = maxTokens, OverlapTokens = overlapTokens };

    private static List<string> Paragraphs(string lang) =>
        [.. File.ReadAllText(Path.Combine(Repo.GoldenFor(lang), "corpus.txt")).Split("\n\n").Select(p => p.Trim()).Where(p => p.Length > 0)];

    private static IngestionDocument Document(IEnumerable<string> paragraphs)
    {
        var document = new IngestionDocument("doc");
        var section = new IngestionDocumentSection();
        foreach (var paragraph in paragraphs)
            section.Elements.Add(new IngestionDocumentParagraph(paragraph));
        document.Sections.Add(section);
        return document;
    }

    private static async Task<List<IngestionChunk<string>>> Chunk(IngestionChunker<string> chunker, IngestionDocument document)
    {
        var chunks = new List<IngestionChunk<string>>();
        await foreach (var chunk in chunker.ProcessAsync(document))
            chunks.Add(chunk);
        return chunks;
    }

    private static async Task<List<IngestionChunk<string>>> Enrich(EntityEnricher enricher, List<IngestionChunk<string>> chunks)
    {
        var enriched = new List<IngestionChunk<string>>();
        await foreach (var chunk in enricher.ProcessAsync(chunks.ToAsyncEnumerable()))
            enriched.Add(chunk);
        return enriched;
    }

    private static string Text(Sentence sentence) => sentence.Text ?? string.Join(" ", sentence.Tokens.Select(t => t.Text));

    // The sentences a chunk holds, in order: each the next sentence of the text found at the chunk's current position.
    private static List<string> Pieces(string content, List<string> sentences)
    {
        var pieces = new List<string>();
        var at = 0;
        while (at < content.Length)
        {
            while (at < content.Length && char.IsWhiteSpace(content[at]))
                at++;
            if (at == content.Length)
                break;
            var sentence = sentences.Where(s => string.CompareOrdinal(content, at, s, 0, s.Length) == 0).MaxBy(s => s.Length);
            Assert.True(sentence != null, $"No whole sentence at {at} of \"{content}\"");
            pieces.Add(sentence);
            at += sentence.Length;
        }

        return pieces;
    }
}
