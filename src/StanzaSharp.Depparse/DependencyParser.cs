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
/// - a maximum spanning tree with one root (Chu-Liu/Edmonds), then, for checkpoints that don't turn it off (English
///   craft, genia, mimic), head_constraints.py's repair of heads with two subjects or objects (<see cref="ResolveHeadConstraints"/>)
/// The network is an <see cref="IDepparseNet"/> per backend; vocab lookups, batching and decoding are here.
/// </summary>
internal sealed class DependencyParser : IDisposable
{
    internal const int RootId = 3, VocabPrefixSize = 4; // vocab.ROOT_ID (in every vocab, the character model's too), VOCAB_PREFIX_SIZE
    private const int SeparateBatchLength = 150;       // depparse_processor.DEFAULT_SEPARATE_BATCH

    private readonly Pretrain? _pretrain;
    private readonly Dictionary<string, int> _wordVocab, _lemmaVocab, _uposVocab;
    private readonly XposVocab _xposVocab;
    private readonly string[] _deprels;
    private int _batchSize, _separateBatchLength = SeparateBatchLength; // set once: the constructor or WithBatchSize's copy
    private readonly bool _resolveHeadConstraints;
    private readonly IDepparseNet _net;
    private readonly TransformerEmbedder? _transformer;
    private readonly float[]? _layerMix;

    /// <param name="pretrain">The word vectors; ignored by a parser without them (<c>_nopretrain</c>: config <c>pretrain</c> false).</param>
    /// <param name="transformer">The transformer a <c>_electra-large</c> parser reads (the pipeline's, shared); else ignored.</param>
    internal DependencyParser(Checkpoint ckpt, Pretrain? pretrain, Func<bool, IDepparseNet> net, TransformerEmbedder? transformer = null)
    {
        var config = ckpt.Root["config"]!;
        CheckSupported(ckpt.Root, config);
        _pretrain = UsesPretrain(ckpt)
            ? pretrain ?? throw new ArgumentException("This parser reads word vectors (pretrain), but none were given", nameof(pretrain))
            : null;
        _layerMix = TransformerEmbedder.LayerMix(ckpt, transformer, "depparse");
        _transformer = config["bert_model"] == null ? null : transformer;
        var vocab = ckpt.Root["vocab"]!;
        _wordVocab = Checkpoint.UnitToId(vocab["word"]);
        _lemmaVocab = Checkpoint.UnitToId(vocab["lemma"]);
        _uposVocab = Checkpoint.UnitToId(vocab["upos"]);
        _xposVocab = XposVocab.Read(vocab);
        _deprels = Checkpoint.Strings(vocab["deprel"]!["_id2unit"]);
        _batchSize = config["batch_size"]!.GetValue<int>();

        // depparse_processor: config.get('resolve_head_constraints', True), the checkpoint's config (English default's
        // models say false; older ones, like the biomedical packages', lack the key).
        _resolveHeadConstraints = config["resolve_head_constraints"]?.GetValue<bool>() ?? true;
        _net = net(config["charlm"]?.GetValue<bool>() == true);
    }

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static DependencyParser LoadManaged(string basePath, Pretrain? pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward,
        TransformerEmbedder? transformer = null)
    {
        var ckpt = Checkpoint.Load(basePath);
        int transformerDim = ckpt.Root["config"]!["bert_model"] == null ? 0 : transformer?.Net.Config.Hidden ?? 0;
        return new DependencyParser(ckpt, pretrain, charlm =>
        {
            CheckCharlms(charlm, charlmForward?.IsForward, charlmBackward?.IsForward);
            return new ManagedDepparseNet(ckpt, UsesPretrain(ckpt) ? pretrain : null, charlm ? charlmForward : null, charlm ? charlmBackward : null, transformerDim);
        }, transformer);
    }

    /// <summary>Whether the parser reads word vectors: config <c>pretrain</c> (false in the <c>_nopretrain</c> parsers).</summary>
    internal static bool UsesPretrain(Checkpoint ckpt) => ckpt.Root["config"]!["pretrain"]?.GetValue<bool>() == true;

    /// <summary>Whether this parser reads a transformer (<c>_electra-large</c>).</summary>
    internal bool UsesTransformer => _transformer != null;

    /// <summary>The sentences this parser will send its transformer in <paramref name="doc"/>, for <see cref="TransformerEmbedder.Prefill"/> (null without one).</summary>
    internal TransformerReader? TransformerReader(Document doc) => _transformer == null
        ? null
        : new TransformerReader(_transformer, _layerMix, [.. doc.Sentences.Where(s => s.Words.Any()).Select(s => (IReadOnlyList<string>)[.. s.Words.Select(w => Punct.Simplify(w.Text))])]);

    internal static void CheckCharlms(bool charlm, bool? forward, bool? backward)
    {
        if (charlm && (forward != true || backward != false))
            throw new ArgumentException("This parser needs the forward charlm, then the backward one");
    }

    /// <summary>
    /// Sets Head and Deprel on every word. Needs UPOS/XPOS from the tagger and lemmas from the lemmatizer
    /// (a missing lemma is read as "_", as Stanza does).
    /// </summary>
    /// <param name="transformers">The call's transformer states, which the tagger may have filled (same words, so the same ids).</param>
    /// <remarks>
    /// The parser runs the charlms over the words with a "\n" ROOT word in front, which changes every forward state and
    /// the backward ROOT state. Backward, though, ROOT comes last: the tagger's backward pass is this one's prefix, then
    /// two more characters for ROOT. So the managed parser takes each sentence's backward outputs from
    /// <paramref name="charlms"/> and runs on from the tagger's final state over ROOT only (<see cref="CharlmCache.TryGetBackward"/>).
    /// A managed charlm step gives a row the same arithmetic in any batch, so that is the same bits as running it again.
    /// </remarks>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default, TransformerCache? transformers = null)
    {
        var keys = doc.Sentences.Where(s => s.Words.Any()).ToList();
        var sentences = keys.Select(s => s.Words.ToList()).ToList();
        if (sentences.Any(s => s.Any(w => w.Upos == null && w.Xpos == null)))
            throw new InvalidOperationException("Run the POS tagger before the dependency parser");

        foreach (var batch in Batches(sentences.Select(s => s.Count + 1).ToList()))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var parsed = Parse(batch.Select(i => (IReadOnlyList<Word>)sentences[i]).ToList(), cancellationToken, charlms,
                batch.Select(i => (Sentence?)keys[i]).ToList(), transformers);
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
            if (lengths[i] > _separateBatchLength)
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
        CharlmCache? charlms = null, IReadOnlyList<Sentence?>? keys = null, TransformerCache? transformers = null)
    {
        var output = Scores(batch, labelScores: _resolveHeadConstraints, cancellationToken, charlms, rentScores: true, keys, transformers);
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

                if (_resolveHeadConstraints)
                {
                    var (tree, raw) = ResolveHeadConstraints(output, b, n);
                    result.Add(Enumerable.Range(1, n - 1).Select(i => (tree[i], _deprels[raw[i - 1] + VocabPrefixSize])).ToArray());
                }
                else
                {
                    var tree = ChuLiuEdmonds.OneRoot(scores, n);
                    result.Add(Enumerable.Range(1, n - 1)
                        .Select(i => (tree[i], _deprels[output.Labels[(b * width + i) * width + tree[i]] + VocabPrefixSize]))
                        .ToArray());
                }
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

    // head_constraints.py SINGLETON_DEPREL_GROUPS: relations a head may have only one dependent of per group.
    private static readonly Dictionary<string, string> DeprelGroups = new()
    {
        ["nsubj"] = "subj",
        ["csubj"] = "subj",
        ["nsubj:pass"] = "subj",
        ["csubj:pass"] = "subj",
        ["nsubj:outer"] = "subj_outer",
        ["csubj:outer"] = "subj_outer",
        ["obj"] = "obj",
    };

    /// <summary>
    /// decode_graph_predictions with resolve_head_constraints (head_constraints.py resolve_head_constraint_violations, at
    /// most 5 rounds): the tree, then labels by each arc's best label log-prob; while a head has two dependents of one
    /// group, each choice of the one that keeps it reruns Chu-Liu/Edmonds with the others' rows plus their best allowed
    /// label (not that group at that head), and the best tree by summed float32 arc + label log-prob wins.
    /// </summary>
    /// <returns>The tree (ROOT first) and each word's relation, without the vocabulary prefix.</returns>
    private (int[] Tree, int[] Raw) ResolveHeadConstraints(DepparseScores output, int b, int n)
    {
        int width = output.Width, rel = output.Relations;
        var arcs = new float[n * n]; // preds[0][:n, :n], float32
        var labels = new float[n * n * rel]; // log_softmax(deprel_scores, 3)[:n, :n], float32
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                arcs[i * n + j] = output.ArcLogProbs[(b * width + i) * width + j];
                var x = output.LabelScores.AsSpan(((b * width + i) * width + j) * rel, rel);
                float max = float.NegativeInfinity;
                foreach (var v in x)
                    max = Math.Max(max, v);
                double sum = 0;
                foreach (var v in x)
                    sum += Math.Exp(v - max);
                float lse = (float)Math.Log(sum);
                for (int r = 0; r < rel; r++)
                    labels[(i * n + j) * rel + r] = x[r] - max - lse;
            }
        }

        var groupOf = Enumerable.Range(0, rel).Select(r => DeprelGroups.GetValueOrDefault(_deprels[r + VocabPrefixSize])).ToArray();
        var excluded = new Dictionary<(int Dep, int Head), HashSet<string>>();
        (int Id, float Score) LabelFor(int dep, int head, string? extra = null)
        {
            var row = labels.AsSpan((dep * n + head) * rel, rel);
            var exclude = excluded.GetValueOrDefault((dep, head));
            if ((exclude == null || exclude.Count == 0) && extra == null)
            {
                int best = 0; // np.argmax: the first maximum
                for (int r = 1; r < rel; r++)
                {
                    if (row[r] > row[best])
                        best = r;
                }

                return (best, row[best]);
            }

            (int Id, float Score) result = (-1, float.NegativeInfinity);
            for (int r = 0; r < rel; r++)
            {
                if (groupOf[r] is { } g && (g == extra || exclude?.Contains(g) == true))
                    continue;
                if (row[r] > result.Score)
                    result = (r, row[r]);
            }

            return result;
        }

        int[] LabelsFor(int[] t) => [.. Enumerable.Range(1, n - 1).Select(j => LabelFor(j, t[j]).Id)];
        float Combined(int[] t, int[] ids)
        {
            float sum = 0;
            for (int j = 1; j < n; j++)
                sum += arcs[j * n + t[j]] + labels[(j * n + t[j]) * rel + ids[j - 1]];
            return sum;
        }

        int[] Cle(float[] s) => ChuLiuEdmonds.OneRoot([.. s.Select(v => (double)v)], n);

        var tree = Cle(arcs);
        var raw = LabelsFor(tree);
        for (int round = 0; round < 5; round++)
        {
            // find_head_constraint_violations: heads in order of their first grouped dependent, groups likewise.
            var perHead = new Dictionary<int, Dictionary<string, List<int>>>();
            var heads = new List<int>();
            for (int j = 1; j < n; j++)
            {
                if (groupOf[raw[j - 1]] is not { } g)
                    continue;
                if (!perHead.TryGetValue(tree[j], out var groups))
                {
                    perHead[tree[j]] = groups = [];
                    heads.Add(tree[j]);
                }

                if (!groups.TryGetValue(g, out var list))
                    groups[g] = list = [];
                list.Add(j);
            }

            var violation = heads.SelectMany(h => perHead[h].Where(kv => kv.Value.Count > 1).Select(kv => (Head: h, Group: kv.Key, Deps: kv.Value)))
                .Cast<(int Head, string Group, List<int> Deps)?>()
                .FirstOrDefault();
            if (violation is not var (headIdx, group, deps))
                break;

            (float Score, int[] Tree, int[] Ids, int Keep)? best = null;
            for (int keep = 0; keep < deps.Count; keep++)
            {
                var trial = (float[])arcs.Clone();
                for (int k = 0; k < deps.Count; k++)
                {
                    if (k == keep)
                        continue;
                    int dep = deps[k];
                    for (int h = 0; h < n; h++)
                        trial[dep * n + h] = (float)((double)arcs[dep * n + h] + LabelFor(dep, h, h == headIdx ? group : null).Score);
                }

                var trialTree = Cle(trial);
                var trialIds = LabelsFor(trialTree);
                for (int k = 0; k < deps.Count; k++)
                {
                    if (k != keep && trialTree[deps[k]] == headIdx)
                        trialIds[deps[k] - 1] = LabelFor(deps[k], headIdx, group).Id;
                }

                float score = Combined(trialTree, trialIds);
                if (best == null || score > best.Value.Score)
                    best = (score, trialTree, trialIds, keep);
            }

            (tree, raw) = (best!.Value.Tree, best.Value.Ids);
            for (int k = 0; k < deps.Count; k++)
            {
                if (k == best.Value.Keep)
                    continue;
                if (!excluded.TryGetValue((deps[k], headIdx), out var set))
                    excluded[(deps[k], headIdx)] = set = [];
                set.Add(group);
            }
        }

        return (tree, raw);
    }

    /// <summary>The network's scores for one batch (see <see cref="DepparseScores"/>).</summary>
    /// <param name="labelScores">Also return the label scores of every word pair (tests).</param>
    /// <param name="charlms">The tagger's charlm outputs (see <see cref="Process"/>).</param>
    /// <param name="rentScores">Let the net return <see cref="DepparseScores.Rented"/> scores, which the caller gives back.</param>
    /// <param name="keys">Each sentence of <paramref name="batch"/>, to find its entry in <paramref name="charlms"/>.</param>
    /// <param name="transformers">The call's transformer states (a <c>_electra-large</c> parser).</param>
    internal DepparseScores Scores(IReadOnlyList<IReadOnlyList<Word>> batch, bool labelScores = false, CancellationToken cancellationToken = default,
        CharlmCache? charlms = null, bool rentScores = false, IReadOnlyList<Sentence?>? keys = null, TransformerCache? transformers = null)
    {
        // data.py load_doc: simplify_punct changes the words the parser sees (vocab, charlm and transformer alike).
        var texts = batch.Select(s => (IReadOnlyList<string>)s.Select(w => Punct.Simplify(w.Text)).ToList()).ToList();
        int size = batch.Count, width = batch.Max(s => s.Count) + 1, parts = _xposVocab.Sizes.Length;

        // extract_bert_embeddings with keep_endpoints, then x[:-1]: [CLS] is ROOT's vector, [SEP]'s is dropped.
        float[][]? transformer = null;
        if (_transformer != null)
        {
            int hidden = _transformer.Net.Config.Hidden;
            transformer = _transformer.Embed(texts, keepEndpoints: true, _layerMix, transformers, cancellationToken);
            for (int b = 0; b < size; b++)
                transformer[b] = transformer[b][..((batch[b].Count + 1) * hidden)];
        }

        var input = new DepparseBatch(texts, width, batch.Select(s => (long)s.Count + 1).ToArray(),
            new long[size * width], new long[size * width], new long[size * width], new long[size * width * parts], new long[size * width], charlms, rentScores, keys,
            parts, transformer, _transformer?.Net.Config.Hidden ?? 0);
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
                input.Pretrained[row + j + 1] = _pretrain?.UnitToId(lower) ?? 0;
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
        Require(config["word_emb_dim"]!.GetValue<int>() > 0 && config["tag_emb_dim"]!.GetValue<int>() > 0, "disabled word or tag embeddings");
        Require(Flag("use_upos", true) && Flag("use_xpos", true) && Flag("use_ufeats", true), "a subset of UPOS/XPOS/UFeats");
        Require(!Flag("reversed", false), "reversed sentences");
        foreach (var key in new[] { "word", "lemma" })
            Require(root["vocab"]![key]!["lower"]!.GetValue<bool>(), $"a cased {key} vocab");
        foreach (var key in new[] { "upos", "xpos" })
            Require(!root["vocab"]![key]!["lower"]!.GetValue<bool>(), $"a lowercased {key} vocab");
    }

    /// <summary>
    /// A copy batching <paramref name="batchSize"/> words per batch, a sentence over <paramref name="separateLength"/>
    /// words alone (Stanza's <c>depparse_batch_size</c>, <c>depparse_min_length_to_batch_separately</c>); null keeps
    /// the current value. It shares the networks (the pipeline keeps only the copy), so it is made at Load, before any
    /// call.
    /// </summary>
    internal DependencyParser WithBatchSize(int? batchSize, int? separateLength)
    {
        var copy = (DependencyParser)MemberwiseClone();
        copy._batchSize = batchSize ?? _batchSize;
        copy._separateBatchLength = separateLength ?? _separateBatchLength;
        return copy;
    }

    public void Dispose() => _net.Dispose();
}
