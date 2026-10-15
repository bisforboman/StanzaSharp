using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Depparse;

/// <summary>
/// Sets <see cref="Word.Head"/> and <see cref="Word.Deprel"/>. Port of stanza/models/depparse/model.py
/// <c>GraphParser</c> (Dozat and Manning's biaffine parser) for the English configuration:
/// - each sentence gets a ROOT word in front
/// - input is projected pretrain + word + lemma + (UPOS + XPOS) embedding + forward/backward charlm
///   (<c>combined_charlm</c>), or the model's own character LSTM projected by <c>trans_char</c> (<c>combined_nocharlm</c>)
/// - a highway biLSTM
/// - deep biaffine arc and label scorers, plus the linearization and distance terms
/// - a maximum spanning tree with one root (Chu-Liu/Edmonds)
/// The network is an <see cref="IDepparseNet"/> per backend; vocab lookups, batching and decoding are here.
/// </summary>
internal sealed class DependencyParser : IDisposable
{
    internal const int RootId = 3, VocabPrefixSize = 4; // vocab.ROOT_ID (in every vocab, the character model's too), VOCAB_PREFIX_SIZE
    private const int SeparateBatchLength = 150;       // depparse_processor.DEFAULT_SEPARATE_BATCH

    private readonly Pretrain _pretrain;
    private readonly Dictionary<string, int> _wordVocab, _lemmaVocab, _uposVocab;
    private readonly XposVocab _xposVocab;
    private readonly string[] _deprels;
    private readonly int _batchSize;
    private readonly IDepparseNet _net;

    internal DependencyParser(Checkpoint ckpt, Pretrain pretrain, Func<bool, IDepparseNet> net)
    {
        _pretrain = pretrain;
        var config = ckpt.Root["config"]!;
        CheckSupported(ckpt.Root, config);
        var vocab = ckpt.Root["vocab"]!;
        _wordVocab = Checkpoint.UnitToId(vocab["word"]);
        _lemmaVocab = Checkpoint.UnitToId(vocab["lemma"]);
        _uposVocab = Checkpoint.UnitToId(vocab["upos"]);
        _xposVocab = XposVocab.Read(vocab);
        _deprels = vocab["deprel"]!["_id2unit"]!.AsArray().Select(x => x!.GetValue<string>()).ToArray();
        _batchSize = config["batch_size"]!.GetValue<int>();
        _net = net(config["charlm"]?.GetValue<bool>() == true);
    }

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static DependencyParser LoadManaged(string basePath, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward)
    {
        var ckpt = Checkpoint.Load(basePath);
        return new DependencyParser(ckpt, pretrain, charlm =>
        {
            CheckCharlms(charlm, charlmForward?.IsForward, charlmBackward?.IsForward);
            return new ManagedDepparseNet(ckpt, pretrain, charlm ? charlmForward : null, charlm ? charlmBackward : null);
        });
    }

    internal static void CheckCharlms(bool charlm, bool? forward, bool? backward)
    {
        if (charlm && (forward != true || backward != false))
            throw new ArgumentException("This parser needs the forward charlm, then the backward one");
    }

    /// <summary>
    /// Sets Head and Deprel on every word. Needs UPOS/XPOS from the tagger and lemmas from the lemmatizer
    /// (a missing lemma is read as "_", as Stanza does).
    /// </summary>
    /// <remarks>
    /// The parser runs the charlms over the words with a "\n" ROOT word in front, which changes every forward state and
    /// the backward ROOT state. Backward, though, ROOT comes last: the tagger's backward pass is this one's prefix, then
    /// two more characters for ROOT. So the managed parser takes each sentence's backward outputs from
    /// <paramref name="charlms"/> and runs on from the tagger's final state over ROOT only (<see cref="CharlmCache.TryGetBackward"/>).
    /// A managed charlm step gives a row the same arithmetic in any batch, so that is the same bits as running it again.
    /// </remarks>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default)
    {
        var keys = doc.Sentences.Where(s => s.Words.Any()).ToList();
        var sentences = keys.Select(s => s.Words.ToList()).ToList();
        if (sentences.Any(s => s.Any(w => w.Upos == null && w.Xpos == null)))
            throw new InvalidOperationException("Run the POS tagger before the dependency parser");

        foreach (var batch in Batches(sentences.Select(s => s.Count + 1).ToList()))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var parsed = Parse(batch.Select(i => (IReadOnlyList<Word>)sentences[i]).ToList(), cancellationToken, charlms,
                batch.Select(i => (Sentence?)keys[i]).ToList());
            for (int b = 0; b < batch.Count; b++)
            {
                for (int j = 0; j < sentences[batch[b]].Count; j++)
                    (sentences[batch[b]][j].Head, sentences[batch[b]][j].Deprel) = parsed[b][j];
            }
        }
    }

    /// <summary>
    /// Stanza's DataLoader in eval mode: sentences sorted longest first (ties: later sentence first),
    /// cut into batches of at most <c>batch_size</c> words counting ROOT, with any sentence over 150 alone.
    /// Each batch is then ordered like collate does (longest first, ties: later position first).
    /// The batches matter beyond speed: the arc log-softmax runs over the padded width.
    /// </summary>
    internal List<List<int>> Batches(IReadOnlyList<int> lengths)
    {
        var order = Enumerable.Range(0, lengths.Count).OrderByDescending(i => lengths[i]).ThenByDescending(i => i);
        var batches = new List<List<int>>();
        var current = new List<int>();
        int currentLen = 0;
        foreach (var i in order)
        {
            if (lengths[i] > SeparateBatchLength)
            {
                if (currentLen > 0)
                    batches.Add(current);
                (current, currentLen) = ([], 0);
                batches.Add([i]);
                continue;
            }

            if (lengths[i] + currentLen > _batchSize && currentLen > 0)
            {
                batches.Add(current);
                (current, currentLen) = ([], 0);
            }

            current.Add(i);
            currentLen += lengths[i];
        }

        if (currentLen > 0)
            batches.Add(current);
        return batches.Select(b => Enumerable.Range(0, b.Count).OrderByDescending(p => lengths[b[p]]).ThenByDescending(p => p)
            .Select(p => b[p])
            .ToList()).ToList();
    }

    /// <summary>Parses one batch: (head, deprel) for each word of each sentence.</summary>
    /// <param name="keys">Each sentence of <paramref name="batch"/>, to find its entry in <paramref name="charlms"/>.</param>
    /// <remarks>
    /// A batch is up to 5000 words (seconds on a slow CPU), so <paramref name="cancellationToken"/> is also checked inside
    /// it: in the network and between the sentences' tree decodes.
    /// </remarks>
    internal List<(int Head, string Deprel)[]> Parse(IReadOnlyList<IReadOnlyList<Word>> batch, CancellationToken cancellationToken = default,
        CharlmCache? charlms = null, IReadOnlyList<Sentence?>? keys = null)
    {
        var output = Scores(batch, labelScores: false, cancellationToken, charlms, rentScores: true, keys);
        int width = output.Width;
        var result = new List<(int, string)[]>(batch.Count);
        var scores = ArrayPool<double>.Shared.Rent(width * width); // each sentence's n × n block, written before it is read
        try
        {
            for (int b = 0; b < batch.Count; b++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int n = batch[b].Count + 1;
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                        scores[i * n + j] = output.ArcLogProbs[(b * width + i) * width + j];
                }

                var tree = ChuLiuEdmonds.OneRoot(scores, n);
                result.Add(Enumerable.Range(1, n - 1)
                    .Select(i => (tree[i], _deprels[output.Labels[(b * width + i) * width + tree[i]] + VocabPrefixSize]))
                    .ToArray());
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(scores);
            if (output.Rented)
            {
                ArrayPool<float>.Shared.Return(output.ArcLogProbs);
                ArrayPool<int>.Shared.Return(output.Labels);
            }
        }

        return result;
    }

    /// <summary>The network's scores for one batch (see <see cref="DepparseScores"/>).</summary>
    /// <param name="labelScores">Also return the label scores of every word pair (tests).</param>
    /// <param name="charlms">The tagger's charlm outputs (see <see cref="Process"/>).</param>
    /// <param name="rentScores">Let the net return <see cref="DepparseScores.Rented"/> scores, which the caller gives back.</param>
    /// <param name="keys">Each sentence of <paramref name="batch"/>, to find its entry in <paramref name="charlms"/>.</param>
    internal DepparseScores Scores(IReadOnlyList<IReadOnlyList<Word>> batch, bool labelScores = false, CancellationToken cancellationToken = default,
        CharlmCache? charlms = null, bool rentScores = false, IReadOnlyList<Sentence?>? keys = null)
    {
        // data.py load_doc: simplify_punct changes the words the parser sees (vocab and charlm alike).
        var texts = batch.Select(s => (IReadOnlyList<string>)s.Select(w => Punct.Simplify(w.Text)).ToList()).ToList();
        int size = batch.Count, width = batch.Max(s => s.Count) + 1, parts = _xposVocab.Sizes.Length;
        var input = new DepparseBatch(texts, width, batch.Select(s => (long)s.Count + 1).ToArray(),
            new long[size * width], new long[size * width], new long[size * width], new long[size * width * parts], new long[size * width], charlms, rentScores, keys,
            parts);
        for (int b = 0; b < size; b++)
        {
            int row = b * width;
            input.Word[row] = input.Lemma[row] = input.Upos[row] = input.Pretrained[row] = RootId;
            input.Xpos.AsSpan(row * parts, parts).Fill(RootId);
            for (int j = 0; j < batch[b].Count; j++)
            {
                var w = batch[b][j];
                var lower = PyString.Lower(texts[b][j]);
                input.Word[row + j + 1] = _wordVocab.GetValueOrDefault(lower, 1);
                input.Lemma[row + j + 1] = _lemmaVocab.GetValueOrDefault(PyString.Lower(w.Lemma ?? "_"), 1);
                input.Upos[row + j + 1] = _uposVocab.GetValueOrDefault(w.Upos ?? "_", 1);
                _xposVocab.Ids(w.Xpos ?? "_", input.Xpos.AsSpan((row + j + 1) * parts, parts));
                input.Pretrained[row + j + 1] = _pretrain.UnitToId(lower);
            }
        }

        return _net.Forward(input, labelScores, cancellationToken);
    }

    private static void CheckSupported(JsonNode root, JsonNode config)
    {
        void Require(bool ok, string what)
        {
            if (!ok) throw new NotSupportedException($"Depparse checkpoint uses {what}, which is not ported");
        }

        bool Flag(string key, bool absent) => config[key]?.GetValue<bool>() ?? absent;
        Require(root["model_type"]?.GetValue<string>() is null or "graph", "a transition or ensemble parser");
        Require(!Flag("use_arc_embedding", false), "use_arc_embedding");
        Require(Flag("char", false) && config["char_emb_dim"]!.GetValue<int>() > 0, "a configuration without character features");
        Require(Flag("pretrain", false), "a configuration without pretrain");
        Require(config["bert_model"] == null, "a transformer");
        Require(config["word_emb_dim"]!.GetValue<int>() > 0 && config["tag_emb_dim"]!.GetValue<int>() > 0, "disabled word or tag embeddings");
        Require(Flag("use_upos", true) && Flag("use_xpos", true) && Flag("use_ufeats", true), "a subset of UPOS/XPOS/UFeats");
        Require(!Flag("reversed", false), "reversed sentences");
        Require(!Flag("resolve_head_constraints", false), "resolve_head_constraints");
        foreach (var key in new[] { "word", "lemma" })
            Require(root["vocab"]![key]!["lower"]!.GetValue<bool>(), $"a cased {key} vocab");
        foreach (var key in new[] { "upos", "xpos" })
            Require(!root["vocab"]![key]!["lower"]!.GetValue<bool>(), $"a lowercased {key} vocab");
    }

    public void Dispose() => _net.Dispose();
}
