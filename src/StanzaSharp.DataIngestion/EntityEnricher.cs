using System.Runtime.CompilerServices;
using Microsoft.Extensions.DataIngestion;

namespace StanzaSharp.DataIngestion;

/// <summary>
/// Adds each chunk's named entities to its metadata, found by StanzaSharp's NER tagger: offline and deterministic, unlike
/// the enrichers that ask a chat model.
/// </summary>
/// <remarks>
/// <para>The value under <see cref="MetadataKey"/> is a <see cref="string"/> array of the chunk's distinct entity texts
/// (ordinal, in order of first appearance; empty when it has none), for filtered or hybrid search.</para>
/// <para>The entity types are the model's: English's default package tags OntoNotes' 18 (PERSON, ORG, GPE, DATE, ...),
/// German's PER, LOC, ORG and MISC. <c>types</c> keeps only some of them.</para>
/// <para>The whole chunk is tagged, headers included, so a header's entities are in every chunk under it.</para>
/// <para>Chunks are tagged a batch at a time in one bulk pipeline call. The pipeline needs <c>tokenize,ner</c> (NER
/// requires nothing else).</para>
/// </remarks>
public sealed class EntityEnricher : IngestionChunkProcessor<string>
{
    private readonly Pipeline _pipeline;
    private readonly IReadOnlySet<string>? _types;
    private readonly int _batchSize;

    /// <summary>Creates an enricher that tags entities with <paramref name="pipeline"/>.</summary>
    /// <param name="pipeline">A loaded pipeline with the <c>ner</c> processor. The enricher doesn't dispose it.</param>
    /// <param name="types">The entity types to keep; null keeps every type.</param>
    /// <param name="batchSize">How many chunks go into one pipeline call.</param>
    public EntityEnricher(Pipeline pipeline, IReadOnlySet<string>? types = null, int batchSize = 32)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 1);
        if (!pipeline.HasNer)
            throw new ArgumentException("The pipeline has no ner processor; load it with Processors = \"tokenize,ner\".", nameof(pipeline));
        _pipeline = pipeline;
        _types = types;
        _batchSize = batchSize;
    }

    /// <summary>The metadata key the entities are written under.</summary>
    public static string MetadataKey => "entities";

    /// <inheritdoc/>
    public override async IAsyncEnumerable<IngestionChunk<string>> ProcessAsync(IAsyncEnumerable<IngestionChunk<string>> chunks,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        var batch = new List<IngestionChunk<string>>(_batchSize);
        await foreach (var chunk in chunks.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            batch.Add(chunk);
            if (batch.Count < _batchSize)
                continue;
            foreach (var done in Enrich(batch, cancellationToken))
                yield return done;
        }

        foreach (var done in Enrich(batch, cancellationToken))
            yield return done;
    }

    private List<IngestionChunk<string>> Enrich(List<IngestionChunk<string>> batch, CancellationToken cancellationToken)
    {
        var done = new List<IngestionChunk<string>>(batch);
        batch.Clear();
        if (done.Count == 0)
            return done;
        var documents = _pipeline.Process(done.Select(c => c.Content), cancellationToken);
        for (var i = 0; i < done.Count; i++)
        {
            done[i].Metadata[MetadataKey] = documents[i].Entities
                .Where(e => _types == null || _types.Contains(e.Type))
                .Select(e => e.Text)
                .Distinct(StringComparer.Ordinal)
                .ToArray();
        }

        return done;
    }
}
