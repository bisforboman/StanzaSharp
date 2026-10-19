using Microsoft.Extensions.Logging;

namespace StanzaSharp;

/// <summary>Settings for <see cref="Pipeline.Load(string, PipelineOptions?)"/>.</summary>
/// <example>
/// <code>
/// using var fast = Pipeline.Load("models/stanza/en", new PipelineOptions { Package = "default_fast" });
/// using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions
/// {
///     Processors = $"{Processor.Tokenize},{Processor.Mwt},{Processor.Pos}", // or "tokenize,mwt,pos"
///     Threads = 4,
///     CharlmCache = new() { MaxWords = 100_000 },
/// });
/// </code>
/// </example>
public sealed class PipelineOptions
{
    /// <summary>
    /// Stanza's code for the language of the models: <c>"en"</c> (English, the default) or <c>"de"</c> (German). Any
    /// other code throws. <see cref="Pipeline.Load"/> still takes that language's folder (e.g. <c>models/stanza/de</c>).
    /// A language's models are downloaded with <see cref="ModelDownloader.DownloadAsync(string, PipelineOptions, IProgress{string}?, CancellationToken)"/>.
    /// </summary>
    public string Language { get; init; } = Pipeline.English;

    /// <summary>
    /// Stanza's name for the set of models to use. English: <c>"default"</c> (<see cref="Pipeline.DefaultPackage"/>,
    /// all eight processors) or <c>"default_fast"</c> (no constituency; pos, depparse and ner use their own small
    /// character models instead of the large character language models, which makes them faster and smaller),
    /// <c>"default_accurate"</c> (all eight; pos, constituency, depparse, ner and sentiment read the transformer
    /// <c>google/electra-large-discriminator</c>, 1.3 GB, which the downloader fetches too: slower, more accurate), and the
    /// treebank packages <c>"combined"</c>, <c>"ewt"</c>, <c>"gum"</c>, <c>"lines"</c>, <c>"partut"</c>, <c>"atis"</c>,
    /// <c>"childes"</c>, <c>"eslspok"</c> and the biomedical <c>"craft"</c>, <c>"genia"</c> and <c>"mimic"</c> (tokenize to
    /// depparse only; eslspok and the biomedical ones have no mwt, and eslspok's lemma is the word).
    /// German: <c>"default"</c>, <c>"default_fast"</c> (pos and depparse as in English; ner and sentiment unchanged),
    /// <c>"default_accurate"</c> (pos, constituency, depparse and sentiment read the transformer
    /// <c>german-nlp-group/electra-base-german-uncased</c>, 445 MB, which the downloader fetches too) and the treebank
    /// packages <c>"combined"</c>, <c>"gsd"</c> and <c>"hdt"</c> (tokenize to depparse only). Any other name throws.
    /// </summary>
    public string Package { get; init; } = Pipeline.DefaultPackage;

    /// <summary>
    /// Comma-separated subset of <see cref="Pipeline.AllProcessors"/>; each needs the ones before it. Null (the
    /// default) runs every processor of the <see cref="Package"/>. A processor the package lacks throws. Like Stanza,
    /// <c>tokenize</c> brings the package's <c>mwt</c> along, so <c>"tokenize,ner"</c> also expands multi-word tokens.
    /// </summary>
    public string? Processors { get; init; }

    /// <summary>
    /// Models chosen per processor, as with Stanza's <c>processors={"ner": "i2b2"}</c>: <see cref="Processor"/> → a
    /// model's name (e.g. <c>"i2b2"</c>) or a package's name (that package's model for the processor). Every other
    /// processor uses the <see cref="Package"/>'s model. With <see cref="Processors"/> null the pipeline runs the
    /// package's processors plus these (and what they require); with <see cref="Processors"/> set, each key must be
    /// one of them. A replaced tokenizer brings its own treebank's mwt, as in Stanza.
    /// </summary>
    /// <remarks>
    /// Models beyond the packages': English constituency <c>ptb3_charlm</c>; English ner <c>ontonotes_charlm</c>,
    /// <c>ontonotes_nocharlm</c>, <c>combined_charlm</c>, <c>conll03_charlm</c>, <c>conll03_electra-large</c> (on
    /// roberta-base), the clinical <c>i2b2</c> and <c>radiology</c> (with the <c>mimic</c> package's character models) and the biomedical <c>anatem</c>,
    /// <c>bc4chemd</c>, <c>bc5cdr</c>, <c>bionlp13cg</c>, <c>jnlpba</c>, <c>linnaeus</c>, <c>ncbi_disease</c> and
    /// <c>s800</c> (with the <c>pubmed</c> character models); German ner <c>conll03</c> and, on German's transformer,
    /// <c>germeval2014_german-nlp-electra</c>, with pos and depparse <c>gsd_german-nlp-electra</c> and
    /// <c>hdt_german-nlp-electra</c>; the treebanks' taggers and
    /// parsers without character models: English pos and depparse <c>ewt_nocharlm</c>, <c>gum_nocharlm</c>,
    /// <c>lines_nocharlm</c>, <c>partut_nocharlm</c>, <c>atis_nocharlm</c>, <c>childes_nocharlm</c>,
    /// <c>eslspok_nocharlm</c>, depparse <c>mimic_nocharlm</c>, German pos and depparse <c>gsd_nocharlm</c> and
    /// <c>hdt_nocharlm</c>; the treebanks' tokenizers and lemmatizers with character models: English tokenize
    /// <c>combined_charlm</c>, <c>ewt_charlm</c>, <c>gum_charlm</c>, <c>lines_charlm</c>, <c>partut_charlm</c>,
    /// <c>atis_charlm</c>, <c>childes_charlm</c>, <c>eslspok_charlm</c>, lemma <c>ewt_charlm</c>, <c>gum_charlm</c>,
    /// <c>lines_charlm</c>, <c>partut_charlm</c>, <c>atis_charlm</c>, <c>childes_charlm</c>, German tokenize and lemma
    /// <c>combined_charlm</c>, <c>gsd_charlm</c>, <c>hdt_charlm</c>. Character models a chosen
    /// model reads are loaded next to the package's when they differ. <see cref="Pipeline.Load"/> and <see cref="ModelDownloader"/> throw
    /// <see cref="ArgumentException"/> for an unknown processor and for an unknown, transformer or unported model.
    /// </remarks>
    /// <example>
    /// <code>
    /// var options = new PipelineOptions { Package = "mimic", Models = new Dictionary&lt;string, string&gt; { [Processor.Ner] = "i2b2" } };
    /// await ModelDownloader.DownloadAsync("models/stanza/en", options);
    /// using var nlp = Pipeline.Load("models/stanza/en", options);
    /// </code>
    /// </example>
    public IReadOnlyDictionary<string, string>? Models { get; init; }

    /// <summary>
    /// Model files of your own, by processor (<see cref="Processor"/> keys), Stanza's <c>{processor}_model_path</c> and
    /// the dependency paths beside it. A path replaces the package's model for that processor; the word vectors and
    /// character models it reads stay those of the model it replaces unless the <see cref="ModelPath"/> names others. A
    /// path for a processor the package lacks adds it, reading only what its <see cref="ModelPath"/> names (as in Stanza).
    /// A processor in both <see cref="Models"/> and here throws <see cref="ArgumentException"/>, and so does one outside a
    /// set <see cref="Processors"/>. <see cref="ModelDownloader"/> skips these files, and <see cref="VerifyChecksums"/>
    /// checks only the package's (yours have no published checksum).
    /// </summary>
    /// <example>
    /// <code>
    /// using var nlp = Pipeline.Load("models/stanza/en", new PipelineOptions
    /// {
    ///     ModelPaths = new Dictionary&lt;string, ModelPath&gt;
    ///     {
    ///         [Processor.Ner] = "my_models/ner.pt", // reads the default model's word vectors and character models
    ///         [Processor.Pos] = new("my_models/pos.pt", Pretrain: "my_models/vectors.pt"),
    ///     },
    /// });
    /// </code>
    /// </example>
    public IReadOnlyDictionary<string, ModelPath>? ModelPaths { get; init; }

    /// <summary>
    /// The implementation that runs the models: <see cref="PipelineBackend.Managed"/> (the default; no native
    /// dependencies), or TorchSharp (libtorch) on a GPU or the CPU from the <c>StanzaSharp.Cuda</c> package
    /// (<c>CudaBackend.Create()</c>, <c>CudaBackend.Cpu</c>). Output is the same on all of them (on a GPU, with TF32
    /// off).
    /// </summary>
    /// <exception cref="ArgumentNullException">Set to null.</exception>
    public PipelineBackend Backend
    {
        get => _backend;
        init => _backend = value ?? throw new ArgumentNullException(nameof(value));
    }

    private readonly PipelineBackend _backend = PipelineBackend.Managed;

    /// <summary>Sharing of character-model outputs between the tagger and the processors after it.</summary>
    public CharlmCacheOptions CharlmCache { get; init; } = new();

    /// <summary>
    /// The number of threads each operation runs on, set by <see cref="Pipeline.Load"/>. On
    /// <see cref="PipelineBackend.Managed"/> it is the size of StanzaSharp's own thread pool; on TorchSharp
    /// (<c>StanzaSharp.Cuda</c>) it is libtorch's intra-op thread count (<c>torch.set_num_threads</c>).
    /// Null (the default) caps the current count at the default: on <see cref="PipelineBackend.Managed"/> the physical
    /// cores when there are at least 4, else all logical processors; on TorchSharp libtorch's own default (the physical
    /// cores). Neither exceeds <see cref="Environment.ProcessorCount"/>, which .NET limits to a container's CPU quota, and
    /// a lower count set earlier is kept.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown by <see cref="Pipeline.Load"/> for a value below 1.</exception>
    /// <remarks>
    /// <b>Process-wide</b> on both backends, and the last pipeline loaded wins.
    /// <list type="bullet">
    /// <item><b>Managed:</b> one pool for the whole process. On 4 or more cores with SMT, the physical cores give fewer slow
    /// calls and more calls/s than all logical processors, and up to 9% fewer words/s on long texts; with 2 cores and
    /// SMT, all 4 logical processors are faster. Concurrent <see cref="Pipeline.Process(string)"/> calls share
    /// it: each works its own part and idle threads help, so N callers don't start N teams.</item>
    /// <item><b>TorchSharp:</b> libtorch's own default is the host's physical core count, so in a container limited to 2
    /// CPUs on a 32-core node it would start 32 threads per operation. libtorch applies a new count to the loading
    /// thread and to threads that haven't run a tensor operation yet; a thread that already has keeps its count, so
    /// load before processing starts. Each thread that calls Process runs its own operations on up to this many
    /// threads, so with N concurrent calls consider <c>ProcessorCount / N</c>. libtorch's inter-op thread pool is not
    /// used by StanzaSharp and is left alone.</item>
    /// </list>
    /// </remarks>
    public int? Threads { get; init; }

    /// <summary>
    /// At most this many <see cref="Pipeline.Process(string)"/> calls run on this pipeline at once; more callers wait
    /// for a turn (cancellable through the call's token). Null, the default, means no limit.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown by <see cref="Pipeline.Load"/> for a value below 1.</exception>
    /// <remarks>
    /// Output never depends on it. On <see cref="PipelineBackend.Managed"/> concurrent calls evict each other's model
    /// weights from the CPU cache, so beyond about two calls in flight throughput falls (default package, Ryzen 7 5800X,
    /// one sentence per call: 1 caller 29.6 calls/s, 2 callers 36.0, 8 callers 23.6; 8 callers limited to one call at a
    /// time 36.4). A service with many concurrent requests on one pipeline can set 2. Bulk input
    /// (<see cref="Pipeline.Process(IEnumerable{string})"/>) is faster still when requests can be grouped.
    /// </remarks>
    public int? MaxConcurrentCalls { get; init; }

    /// <summary>
    /// Whether the tokenizer splits paragraphs into sentences (true, the default, as Stanza does). False is Stanza's
    /// <c>tokenize_no_ssplit</c>: each paragraph, i.e. text between blank lines, becomes one sentence, for input that
    /// already has one sentence per paragraph. Tokens and multi-word tokens are the same either way. Pretokenized input
    /// (<see cref="Pipeline.Process(IEnumerable{IEnumerable{string}})"/>) keeps its given sentences regardless, as in Stanza.
    /// </summary>
    public bool SplitSentences { get; init; } = true;

    /// <summary>
    /// Stanza's <c>pretagged</c>: lemma, depparse and constituency load without the processors they require, for
    /// documents that already carry tags (and lemmas), annotated with <see cref="Pipeline.Process(Document)"/>. With
    /// <c>Processors = "depparse"</c> the pipeline loads the dependency parser and what it reads, not the tagger or the
    /// lemmatizer, and parses the given UPOS, XPOS and lemmas. Tokenize and the package's mwt are always loaded (about
    /// 1 MB), so the text overloads still run; without pos, depparse and constituency then throw for the untagged
    /// words. pos, ner and sentiment keep their requirements, as in Stanza. Off by default.
    /// </summary>
    /// <remarks>
    /// What the processors read when a word lacks it: depparse throws <see cref="InvalidOperationException"/> for a word
    /// with neither UPOS nor XPOS, and reads a missing XPOS or lemma as <c>_</c> (parses then get worse without lemmas,
    /// in Stanza too); the lemmatizer reads a missing UPOS as <c>_</c>; constituency throws
    /// <see cref="InvalidOperationException"/> for a word without the tag it reads (XPOS for English, UPOS for German;
    /// Stanza parses it with an empty tag).
    /// </remarks>
    /// <example>
    /// <code>
    /// using var nlp = Pipeline.Load(dir, new PipelineOptions { Processors = "depparse", Pretagged = true });
    /// var doc = nlp.Process(Conllu.Read(File.ReadAllText("tagged.conllu")));
    /// </code>
    /// </example>
    public bool Pretagged { get; init; }

    /// <summary>
    /// Stanza's <c>tokenize_postprocessor</c>: called after the tokenizer with every sentence's tokens (their text and
    /// multi-word flag), it returns the sentences and tokens the pipeline goes on with. It can merge, split or regroup
    /// tokens and sentences, flag a token for the MWT model (<see cref="PostprocessorToken.IsMultiWord"/>) or give its words
    /// itself (<see cref="PostprocessorToken.Expansion"/>), as long as each token's text is found in the text after the one
    /// before. Null (the default) runs none. It applies to text and bulk input (once over the joined texts, as in Stanza), not
    /// to pretokenized input or <see cref="Pipeline.Process(Document)"/>.
    /// </summary>
    /// <remarks>
    /// The output equals Stanza's for the same function, with one difference: Stanza shifts a hand-expanded token's word ids
    /// by the automatic MWT expansions earlier in the document, which breaks its dependency trees; here they are numbered
    /// as written. A token not found in the text makes <c>Process</c> throw <see cref="InvalidOperationException"/>.
    /// </remarks>
    /// <example>
    /// <code>
    /// // Keep "New York" one token.
    /// var options = new PipelineOptions
    /// {
    ///     TokenizePostprocessor = sentences => sentences.Select(tokens =>
    ///     {
    ///         var merged = new List&lt;PostprocessorToken&gt;();
    ///         foreach (var token in tokens)
    ///         {
    ///             if (token.Text == "York" &amp;&amp; merged.Count &gt; 0 &amp;&amp; merged[^1].Text == "New")
    ///                 merged[^1] = "New York";
    ///             else
    ///                 merged.Add(token);
    ///         }
    ///
    ///         return (IEnumerable&lt;PostprocessorToken&gt;)merged;
    ///     }),
    /// };
    /// </code>
    /// </example>
    public Func<IReadOnlyList<IReadOnlyList<PostprocessorToken>>, IEnumerable<IEnumerable<PostprocessorToken>>>? TokenizePostprocessor { get; init; }

    /// <summary>
    /// Which of the NER model's tag sets to decode, Stanza's <c>ner_predict_tagset</c>. English's
    /// <c>ontonotes-ww-multi</c> models have two: 0 is OntoNotes (18 entity types, the model's own choice) and 1 WorldWide
    /// (<c>ner=combined_charlm</c> has three). Null (the default): the model's own choice. A tag set the model doesn't have
    /// throws <see cref="ArgumentException"/> at <see cref="Pipeline.Load"/>; without ner it is ignored, as in Stanza.
    /// </summary>
    public int? NerTagSet { get; init; }

    /// <summary>
    /// Stanza's <c>lemma_dict_only</c>: each word's lemma comes from the lemmatizer's dictionary alone (by UPOS and word,
    /// then by word), and a word it lacks keeps its text as the lemma. The character model and the contextual lemma
    /// classifiers don't run, so lemmatizing is faster and the lemmas of unseen words are not guessed.
    /// </summary>
    public bool LemmaDictionaryOnly { get; init; }

    /// <summary>
    /// Stanza's <c>mwt_dict_only</c>: multi-word tokens are split only where the MWT model's dictionary has the token (as
    /// written, or an all-caps or capitalized token by its lowercase entry); every other token stays one word. The model
    /// is not loaded.
    /// </summary>
    public bool MwtDictionaryOnly { get; init; }

    /// <summary>
    /// Paragraphs per tokenizer batch; null (the default) uses the model's own (32). Unlike the other batch sizes this has no
    /// Stanza twin that works: Stanza's tokenizer ignores <c>tokenize_batch_size</c> and always reads its model's value. The
    /// tokens stay the same (at most a score's last bits move); only speed and memory change.
    /// </summary>
    public int? TokenizeBatchSize { get; init; }

    /// <summary>Tokens per MWT batch, Stanza's <c>mwt_batch_size</c>; null uses the model's own. Results stay the same.</summary>
    public int? MwtBatchSize { get; init; }

    /// <summary>
    /// Sentences per POS batch, Stanza's <c>pos_batch_size</c>; null uses the model's own. A batch also stops at
    /// <see cref="PosBatchMaximumTokens"/> words. Results stay the same (at most a score's last bits move); only speed and memory change.
    /// </summary>
    public int? PosBatchSize { get; init; }

    /// <summary>
    /// Words per POS batch, Stanza's <c>pos_batch_maximum_tokens</c>; null uses 5,000, as Stanza does. A longer sentence is a
    /// batch of its own.
    /// </summary>
    public int? PosBatchMaximumTokens { get; init; }

    /// <summary>Words per batch of the lemmatizer's character model, Stanza's <c>lemma_batch_size</c>; null uses the model's own. Results stay the same.</summary>
    public int? LemmaBatchSize { get; init; }

    /// <summary>
    /// The beam of the lemmatizer's character model, Stanza's <c>lemma_beam_size</c>; null uses the model's own (1: greedy,
    /// one candidate per step). A wider beam keeps that many candidate lemmas per word, which can change the lemmas of
    /// words the dictionary lacks, at a cost in speed. The dictionary and the contextual classifiers are unaffected.
    /// </summary>
    public int? LemmaBeamSize { get; init; }

    /// <summary>
    /// Sentences the constituency parser keeps in flight, Stanza's <c>constituency_batch_size</c>; null uses 50, as Stanza
    /// does. Results stay the same.
    /// </summary>
    public int? ConstituencyBatchSize { get; init; }

    /// <summary>
    /// Words per dependency-parser batch, Stanza's <c>depparse_batch_size</c>; null uses the model's own (5,000 for most).
    /// <b>The parses can change with it, as in Stanza:</b> the head scores are normalized over the batch's padded width.
    /// </summary>
    public int? DepparseBatchSize { get; init; }

    /// <summary>
    /// Sentences longer than this many words get a dependency-parser batch of their own, Stanza's
    /// <c>depparse_min_length_to_batch_separately</c>; null uses 150, as Stanza does. Like
    /// <see cref="DepparseBatchSize"/>, it can change the parses.
    /// </summary>
    public int? DepparseMinLengthToBatchSeparately { get; init; }

    /// <summary>
    /// Tokens per sentiment batch, Stanza's <c>sentiment_batch_size</c>; null uses 5,000, as Stanza does. <b>The labels
    /// change with it, as in Stanza:</b> each batch is padded to its longest sentence, and the classifier reads the padding.
    /// </summary>
    public int? SentimentBatchSize { get; init; }

    /// <summary>Sentences per NER batch, Stanza's <c>ner_batch_size</c>; null uses the model's own. Results stay the same.</summary>
    public int? NerBatchSize { get; init; }

    /// <summary>
    /// Checks every model file <see cref="Pipeline.Load"/> reads against the MD5 that Stanza publishes for it (the
    /// checksums <see cref="ModelDownloader"/> verifies downloads with) before loading, which catches corrupted files
    /// or files from another Stanza version. It reads every file once more, which takes a second or two for the
    /// <see langword="default"/> package. Off by default.
    /// </summary>
    /// <remarks>
    /// Only Stanza's <c>.pt</c> files have published checksums. If a model would be read from converted
    /// <c>.json</c> + <c>.safetensors</c> files instead, Load throws rather than skip the check.
    /// </remarks>
    public bool VerifyChecksums { get; init; }

    /// <summary>
    /// Receives timings: each model's load time and the thread count at <see cref="LogLevel.Information"/>, and each
    /// processor's time per <see cref="Pipeline.Process(string)"/> call at <see cref="LogLevel.Debug"/>. Null (the
    /// default) logs nothing and costs nothing.
    /// </summary>
    public ILogger? Logger { get; init; }

    /// <summary>
    /// Linux with glibc: return the native heap's free memory to the OS at the end of each Process call
    /// (<c>malloc_trim(0)</c>, see <see cref="NativeHeap"/>). Always on for users (decided: no public API, its cost is
    /// in the noise); internal so the benchmark can measure without it.
    /// </summary>
    internal bool TrimNativeHeap { get; init; } = true;
}

/// <summary>
/// The tagger keeps its character-model outputs (about 8 KB per word) for the processors after it that need the same
/// ones: the constituency parser, sentiment and NER (and, on <see cref="PipelineBackend.Managed"/>, the dependency
/// parser's backward pass, from about 4 KB more per sentence), when their models read the tagger's character language
/// models (one chosen with <see cref="PipelineOptions.Models"/> may read others, e.g. ner <c>i2b2</c>; it then computes
/// its own). Output is identical either way; the cache trades memory for speed.
/// </summary>
public sealed class CharlmCacheOptions
{
    /// <summary>Whether those processors reuse the tagger's outputs (true) or compute their own.</summary>
    public bool IsEnabled { get; init; } = true;

    /// <summary>
    /// The most words kept per document (the cache holds only the tagger's pair of character language models, so this is
    /// also the total). Sentences past it are recomputed by the processors that read them. The default, 32,768 words, is
    /// about 256 MB.
    /// </summary>
    public int MaxWords { get; init; } = Nn.CharlmCache.DefaultMaxWords;
}
