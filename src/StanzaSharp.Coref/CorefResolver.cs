using System.Buffers;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;

namespace StanzaSharp.Coref;

/// <summary>
/// Coreference resolution (processor <c>coref</c>; docs/coref-plan.md, PR c): port of stanza/models/coref/model.py
/// <c>CorefModel.run</c> and stanza/pipeline/coref_processor.py for <c>udcoref_xlm-roberta-lora</c>. The whole document is
/// one call:
/// - every word through the transformer's tokenizer, the subwords cut into windows of 510 backed up to a sentence end
///   (bert.get_subwords_batches), each run through XLM-R with the model's LoRA adapters, the last layer kept;
/// - a vector per word (attention over its subwords), a bilinear rough score for every earlier word and its top 50 in
///   torch.topk's own order (<see cref="TopK"/>), the anaphoricity scores of those pairs and a start-of-chain score;
/// - clusters as the connected components of each word's best antecedent, singletons where the start score wins, a span
///   per clustered word, and zero mentions (empty words) where the zeros predictor says so;
/// - <see cref="Document.Coref"/> set from them as coref_processor.process builds the chains.
/// The networks are an <see cref="ICorefNet"/> (per backend); tokenization, windows, top-k, decisions and chains are here.
/// </summary>
internal sealed class CorefResolver : IDisposable
{
    private const float Epsilon = 1e-7f; // coref/const.py EPSILON, the dummy antecedent's score

    private readonly TransformerEmbedder _transformer;
    private readonly ITransformerNet _encoder; // the transformer with the model's LoRA adapters, sharing its weights
    private readonly ICorefNet _net;
    private readonly int _k, _window;
    private readonly float[] _startMap;
    private readonly HashSet<int> _special; // ids _bertify drops: cls, sep, pad (eos is sep for XLM-R)

    /// <param name="transformer">The pipeline's transformer (the checkpoint's <c>bert_model</c>); the model's LoRA adapters go
    /// onto a copy that shares its weights.</param>
    internal CorefResolver(Checkpoint ckpt, TransformerEmbedder transformer, Func<Checkpoint, ICorefNet> net)
    {
        var config = ckpt.Root["config"] ?? throw new InvalidDataException("A coref checkpoint without config");
        CheckSupported(config, transformer);
        _k = config["rough_k"]!.GetValue<int>();
        _window = config["bert_window_size"]!.GetValue<int>() - 2;
        _startMap = ckpt.Tensor<float>(ckpt.Root["a_scorer"]!["start_map.weight"]);
        if (_startMap.Length != _k)
            throw new InvalidDataException($"start_map has {_startMap.Length} weights, not rough_k {_k}");
        _transformer = transformer;
        var (ids, _) = transformer.Tokenizer.Encode([]);
        _special = [ids[0], ids[^1], transformer.Net.Config.PositionOffset - 1];
        _encoder = transformer.Net.WithLora(LoraWeights.Read(ckpt, ckpt.Root["bert_lora"] ?? throw new InvalidDataException("A coref model without bert_lora"), config));
        _net = net(ckpt);
    }

    /// <summary>The configuration this port follows (udcoref_xlm-roberta-lora's); anything else throws.</summary>
    private static void CheckSupported(JsonNode config, TransformerEmbedder transformer)
    {
        bool Flag(string key) => config[key]?.GetValue<bool>() == true;
        int Int(string key) => config[key]?.GetValue<int>() ?? throw new InvalidDataException($"No {key} in the coref config");
        if (config["bert_model"]?.GetValue<string>() != transformer.Name)
            throw new ArgumentException($"The coref model reads {config["bert_model"]}, not {transformer.Name}");
        if (!Flag("lora") || Flag("full_pairwise") || !Flag("singletons") || Int("n_hidden_layers") != 1 || Int("embedding_size") != 20
            || Int("sp_embedding_size") != 64)
        {
            throw new NotSupportedException("A coref configuration StanzaSharp doesn't port (it ports LoRA, no speakers, singletons, one hidden layer)");
        }
    }

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend.</summary>
    public static CorefResolver LoadManaged(string basePath, TransformerEmbedder transformer)
    {
        var ckpt = Checkpoint.Load(basePath);
        return new CorefResolver(ckpt, transformer, c => new ManagedCorefNet(c));
    }

    /// <summary>
    /// Sets <see cref="Document.Coref"/> (and every word's <see cref="Word.CorefChains"/>), adding an empty word for each zero
    /// mention. Reads the words' text, and their UPOS when set (the representative mention's tie-break counts PROPN).
    /// </summary>
    /// <exception cref="InvalidOperationException">A word has no subwords (U+200B alone, say), on which Stanza fails too.</exception>
    public void Process(Document doc, CancellationToken ct = default)
    {
        var words = new List<Word>();
        var sentOf = new List<int>();
        var posOf = new List<int>();
        for (int s = 0; s < doc.Sentences.Count; s++)
        {
            int p = 0;
            foreach (var w in doc.Sentences[s].Words)
            {
                words.Add(w);
                sentOf.Add(s);
                posOf.Add(p++);
            }
        }

        if (words.Count == 0)
        {
            doc.Coref = [];
            return;
        }

        var result = Run(words, sentOf, ct);
        doc.Coref = Chains(doc, words, sentOf, posOf, result);
    }

    /// <summary>CorefModel.run's results: word clusters (sorted), their spans (document word indices), the zero scores.</summary>
    internal sealed record Result(List<List<int>> WordClusters, List<List<(int Start, int End)>> SpanClusters, float[] ZeroScores, int[] TopIndices, float[] Scores,
        int[] Windows, float[] WordsHead);

    /// <summary>CorefModel.build_doc, _bertify and run, for the words of a whole document.</summary>
    internal Result Run(IReadOnlyList<Word> words, IReadOnlyList<int> sentOf, CancellationToken ct = default)
    {
        int n = words.Count;
        var (ids, owners) = _transformer.Tokenizer.Encode([.. words.Select(w => w.Text)]);
        int subwords = ids.Length - 2;
        var starts = new int[n];
        var ends = new int[n];
        for (int i = 0, t = 0; i < n; i++)
        {
            starts[i] = t;
            while (t < subwords && owners[t + 1] == i)
                t++;
            ends[i] = t;
            if (ends[i] == starts[i])
                throw new InvalidOperationException($"Word {i} ({PyString.Repr(words[i].Text)}) has no subwords in {_transformer.Name}'s tokenizer; Stanza's coref fails on it too");
        }

        // bert.get_subwords_batches: windows of 510 subwords, each moved back to a sentence end (cut hard inside one
        // longer sentence), in <s> ... </s>.
        var windows = new List<int[]>();
        var windowStarts = new List<int>();
        for (int start = 0, end = 0; end < subwords;)
        {
            int prev = end;
            end = Math.Min(end + _window, subwords);
            if (end < subwords)
            {
                int sent = sentOf[owners[end + 1]];
                while (end > 0 && sentOf[owners[end]] == sent)
                    end--;
            }

            if (end == prev)
                end = Math.Min(end + _window, subwords);
            windows.Add([ids[0], .. ids.AsSpan(1 + start, end - start), ids[^1]]);
            windowStarts.Add(start);
            start = end;
        }

        int hidden = _transformer.Net.Config.Hidden;
        var states = _encoder.Encode(windows, 1, ct);
        var x = new float[subwords * hidden];
        int row = 0;
        try
        {
            for (int w = 0; w < windows.Count; w++)
            {
                for (int t = 0; t < windows[w].Length; t++)
                {
                    if (_special.Contains(windows[w][t]))
                        continue;
                    if (row == subwords || t != row - windowStarts[w] + 1)
                        throw new NotSupportedException("A word tokenizes to one of the transformer's special tokens (<s>, </s>, <pad>), which Stanza's coref drops");
                    Array.Copy(states[w], t * hidden, x, row * hidden, hidden);
                    row++;
                }
            }
        }
        finally
        {
            if (_encoder.RentsOutputs)
            {
                foreach (var s in states)
                    ArrayPool<float>.Shared.Return(s);
            }
        }

        if (row != subwords)
            throw new NotSupportedException("A word tokenizes to one of the transformer's special tokens (<s>, </s>, <pad>), which Stanza's coref drops");
        ct.ThrowIfCancellationRequested();

        var wordVectors = _net.Words(x, subwords, starts, ends);

        // RoughScorer: the bilinear scores, -inf for every word not before the row's, then the top k in topk's order.
        int k = Math.Min(_k, n);
        var top = new int[n * k];
        var rough = new float[n * k];
        _net.RoughScores(wordVectors, n, (first, count, scores, ld) =>
        {
            for (int r = 0; r < count; r++)
            {
                int i = first + r;
                var line = scores.AsSpan(r * ld, n);
                line[i..].Fill(float.NegativeInfinity);
                TopK.Select(line, k, top.AsSpan(i * k, k), rough.AsSpan(i * k, k));
            }
        }, ct);

        // PairwiseEncoder: the distance bucket (0-3 for 1-4 words, then log2, capped), from 1 for a later candidate.
        var buckets = new int[n * k];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                int d = Math.Max(i - top[i * k + c], 1);
                buckets[i * k + c] = d < 5 ? d - 1 : Math.Min(31 - int.LeadingZeroCount(d), 6) + 2;
            }
        }

        // AnaphoricityScorer: the pair scores (without the rough score), start_map over them in topk's order (the masked
        // candidates too), then [start, dummy, pair + rough].
        var pair = _net.Anaphoricity(wordVectors, n, top, buckets, k, ct);
        var scores = new float[n * (k + 2)];
        for (int i = 0; i < n; i++)
        {
            float start = 0;
            for (int c = 0; c < k; c++)
                start += pair[i * k + c] * _startMap[c];
            scores[i * (k + 2)] = start;
            scores[(i * (k + 2)) + 1] = Epsilon;
            for (int c = 0; c < k; c++)
                scores[(i * (k + 2)) + 2 + c] = pair[i * k + c] + rough[i * k + c];
        }

        var clusters = Clusterize(scores, top, n, k);
        ct.ThrowIfCancellationRequested();
        var spans = Spans(wordVectors, n, sentOf, clusters, ct);
        int[] clustered = [.. clusters.SelectMany(c => c)];
        var zeros = clustered.Length == 0 ? [] : _net.Zeros(wordVectors, clustered);
        return new Result(clusters, spans, zeros, top, scores, [.. windows.SelectMany(w => w)], wordVectors.AsSpan(0, Math.Min(8, n) * hidden).ToArray());
    }

    /// <summary>
    /// CorefModel._clusterize with singletons: each word's antecedent is the best of [dummy, candidates] (the dummy on a tie),
    /// links make an undirected graph whose connected components are the clusters; a word in none whose start score beats
    /// the dummy (or ties it) is a singleton; clusters sorted, as lists.
    /// </summary>
    private static List<List<int>> Clusterize(float[] scores, int[] top, int n, int k)
    {
        var links = new List<int>[n];
        for (int i = 0; i < n; i++)
        {
            var row = scores.AsSpan(i * (k + 2), k + 2);
            int best = 1;
            for (int c = 2; c < k + 2; c++)
            {
                if (row[c] > row[best])
                    best = c;
            }

            if (best == 1)
                continue;
            int j = top[(i * k) + best - 2];
            (links[i] ??= []).Add(j);
            (links[j] ??= []).Add(i);
        }

        var visited = new bool[n];
        var clusters = new List<List<int>>();
        for (int i = 0; i < n; i++)
        {
            if (links[i] == null || visited[i])
                continue;
            var cluster = new List<int>();
            var stack = new Stack<int>([i]);
            visited[i] = true;
            while (stack.Count > 0)
            {
                int node = stack.Pop();
                cluster.Add(node);
                foreach (int next in links[node])
                {
                    if (!visited[next])
                    {
                        visited[next] = true;
                        stack.Push(next);
                    }
                }
            }

            cluster.Sort();
            clusters.Add(cluster);
        }

        for (int i = 0; i < n; i++)
        {
            if (!visited[i] && scores[i * (k + 2)] >= scores[(i * (k + 2)) + 1])
                clusters.Add([i]);
        }

        clusters.Sort(CompareLists);
        return clusters;
    }

    // Python's list order: elementwise, then the shorter first.
    private static int CompareLists(List<int> a, List<int> b)
    {
        for (int i = 0; i < Math.Min(a.Count, b.Count); i++)
        {
            if (a[i] != b[i])
                return a[i].CompareTo(b[i]);
        }

        return a.Count.CompareTo(b.Count);
    }

    /// <summary>
    /// SpanPredictor.predict: for every clustered word (the heads, sorted), the first best start at or before it and the first
    /// best end at or after it within its sentence; each cluster's spans in its own word order.
    /// </summary>
    private List<List<(int Start, int End)>> Spans(float[] words, int n, IReadOnlyList<int> sentOf, List<List<int>> clusters, CancellationToken ct)
    {
        if (clusters.Count == 0)
            return [];
        int[] heads = [.. clusters.SelectMany(c => c).Order()];
        var sentStart = new int[heads.Length];
        var sentLength = new int[heads.Length];
        for (int h = 0; h < heads.Length; h++)
        {
            int s = sentOf[heads[h]], first = heads[h], last = heads[h];
            while (first > 0 && sentOf[first - 1] == s)
                first--;
            while (last + 1 < n && sentOf[last + 1] == s)
                last++;
            (sentStart[h], sentLength[h]) = (first, last - first + 1);
        }

        int width = sentLength.Max();
        var (startScores, endScores) = _net.SpanScores(words, n, heads, sentStart, sentLength, width, ct);
        var span = new Dictionary<int, (int, int)>();
        for (int h = 0; h < heads.Length; h++)
        {
            int head = heads[h] - sentStart[h], bestStart = 0, bestEnd = head;
            for (int t = 1; t <= head; t++)
            {
                if (startScores[(h * width) + t] > startScores[(h * width) + bestStart])
                    bestStart = t;
            }

            for (int t = head + 1; t < sentLength[h]; t++)
            {
                if (endScores[(h * width) + t] > endScores[(h * width) + bestEnd])
                    bestEnd = t;
            }

            span[heads[h]] = (sentStart[h] + bestStart, sentStart[h] + bestEnd + 1);
        }

        return [.. clusters.Select(c => c.Select(head => span[head]).ToList())];
    }

    /// <summary>
    /// coref_processor.process: empty words for the zero mentions (_handle_zero_anaphora), then per cluster its spans sorted,
    /// the representative (the longest; on a tie the one with more PROPN words; else the earliest; never a zero) and the
    /// chain. Like Stanza, the zero lookup pairs the sorted spans with the cluster's words by position.
    /// </summary>
    private static List<CorefChain> Chains(Document doc, List<Word> words, List<int> sentOf, List<int> posOf, Result result)
    {
        // _handle_zero_anaphora: an empty word before each clustered word whose zero score is positive, numbered within its
        // sentence after the ones it already has.
        var zeros = new Dictionary<(int Cluster, int Word), (int Sentence, int Id, int Index)>();
        int flat = 0;
        for (int c = 0; c < result.WordClusters.Count; c++)
        {
            foreach (int word in result.WordClusters[c])
            {
                if (result.ZeroScores[flat++] <= 0)
                    continue;
                var sentence = doc.Sentences[sentOf[word]];
                var empty = new Word { Id = posOf[word], EmptyIndex = sentence.EmptyWords.Count + 1, Text = "_", Lemma = "_" };
                sentence.EmptyWords.Add(empty);
                zeros[(c, word)] = (sentOf[word], empty.Id, empty.EmptyIndex.Value);
            }
        }

        var chains = new List<CorefChain>();
        for (int c = 0; c < result.SpanClusters.Count; c++)
        {
            var spans = result.SpanClusters[c].Order().ToList();
            var cluster = result.WordClusters[c];
            foreach (var (start, end) in spans)
            {
                if (sentOf[end - 1] != sentOf[start])
                    throw new InvalidOperationException("The coref model predicted a span that crosses two sentences");
            }

            int maxLength = 0, maxPropn = 0;
            int? best = null;
            for (int i = 0; i < spans.Count; i++)
            {
                if (zeros.ContainsKey((c, cluster[i])))
                    continue;
                var (start, end) = spans[i];
                int propn = 0;
                for (int w = start; w < end; w++)
                    propn += words[w].Upos == "PROPN" ? 1 : 0;
                if (end - start > maxLength || (end - start == maxLength && propn > maxPropn))
                    (maxLength, best, maxPropn) = (end - start, i, propn);
            }

            var mentions = new List<CorefMention>(spans.Count);
            for (int i = 0; i < spans.Count; i++)
            {
                var (start, end) = spans[i];
                mentions.Add(zeros.TryGetValue((c, cluster[i]), out var zero)
                    ? new CorefMention(zero.Sentence, zero.Id, zero.Id, zero.Index)
                    : new CorefMention(sentOf[start], posOf[start], posOf[end - 1] + 1));
            }

            var text = best is int b ? CorefLinks.MentionText(doc.Sentences[mentions[b].Sentence], mentions[b]) : "_";
            chains.Add(new CorefChain(chains.Count, mentions, text, best));
        }

        return chains;
    }

    public void Dispose()
    {
        _encoder.Dispose();
        _net.Dispose();
    }
}
