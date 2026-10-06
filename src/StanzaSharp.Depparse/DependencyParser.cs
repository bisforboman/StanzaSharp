using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Depparse;

/// <summary>
/// Sets <see cref="Word.Head"/> and <see cref="Word.Deprel"/>. Port of stanza/models/depparse/model.py
/// <c>GraphParser</c> (Dozat and Manning's biaffine parser) for the English configuration:
/// - each sentence gets a ROOT word in front
/// - input is projected pretrain + word + lemma + (UPOS + XPOS) embedding + forward/backward charlm
/// - a highway biLSTM
/// - deep biaffine arc and label scorers, plus the linearization and distance terms
/// - a maximum spanning tree with one root (Chu-Liu/Edmonds)
/// </summary>
public sealed class DependencyParser : IDisposable
{
    private const int RootId = 3, VocabPrefixSize = 4; // vocab.ROOT_ID, VOCAB_PREFIX_SIZE
    private const int SeparateBatchLength = 150;       // depparse_processor.DEFAULT_SEPARATE_BATCH

    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Dictionary<string, int> _wordVocab, _lemmaVocab, _uposVocab, _xposVocab;
    private readonly string[] _deprels;
    private readonly int _batchSize;
    private readonly bool _linearization, _distance;

    private readonly Embedding _wordEmb, _lemmaEmb, _uposEmb, _xposEmb;
    private readonly Linear _transPretrained;
    private readonly HighwayLstm _lstm;
    private readonly DeepBiaffine _unlabeled, _deprel;
    private readonly DeepBiaffine? _linearizationScorer, _distanceScorer;

    private DependencyParser(Checkpoint ckpt, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        _charlmForward = charlmForward;
        _charlmBackward = charlmBackward;
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");

        var config = ckpt.Root["config"]!;
        CheckSupported(ckpt.Root, config);
        var model = ckpt.Root["model"]!;
        var vocab = ckpt.Root["vocab"]!;

        _wordVocab = Checkpoint.UnitToId(vocab["word"]);
        _lemmaVocab = Checkpoint.UnitToId(vocab["lemma"]);
        _uposVocab = Checkpoint.UnitToId(vocab["upos"]);
        _xposVocab = Checkpoint.UnitToId(vocab["xpos"]);
        _deprels = vocab["deprel"]!["_id2unit"]!.AsArray().Select(x => x!.GetValue<string>()).ToArray();
        _batchSize = config["batch_size"]!.GetValue<int>();
        _linearization = config["linearization"]!.GetValue<bool>();
        _distance = config["distance"]!.GetValue<bool>();

        int hidden = config["hidden_dim"]!.GetValue<int>();
        int biaff = config["deep_biaff_hidden_dim"]!.GetValue<int>();
        int wordEmb = config["word_emb_dim"]!.GetValue<int>();
        int tagEmb = config["tag_emb_dim"]!.GetValue<int>();
        int transformed = config["transformed_dim"]!.GetValue<int>();
        // Stanza appends the UPOS+XPOS embedding twice where it means to add the UFeats one, so the
        // UFeats embeddings are loaded by Stanza but never used.
        int inputSize = transformed + 2 * wordEmb + 2 * tagEmb + charlmForward.HiddenDim + charlmBackward.HiddenDim;

        _wordEmb = nn.Embedding(_wordVocab.Count, wordEmb, padding_idx: 0).LoadFrom(ckpt, model, "word_emb.");
        _lemmaEmb = nn.Embedding(_lemmaVocab.Count, wordEmb, padding_idx: 0).LoadFrom(ckpt, model, "lemma_emb.");
        _uposEmb = nn.Embedding(_uposVocab.Count, tagEmb, padding_idx: 0).LoadFrom(ckpt, model, "upos_emb.");
        _xposEmb = nn.Embedding(_xposVocab.Count, tagEmb, padding_idx: 0).LoadFrom(ckpt, model, "xpos_emb.");
        _transPretrained = nn.Linear(pretrain.Dim, transformed, hasBias: false).LoadFrom(ckpt, model, "trans_pretrained.");
        _lstm = new HighwayLstm(ckpt, model, "parserlstm", inputSize, hidden, config["num_layers"]!.GetValue<int>());

        int relations = _deprels.Length - VocabPrefixSize;
        _unlabeled = new DeepBiaffine(ckpt, model, "unlabeled.", 2 * hidden, biaff, 1);
        _deprel = new DeepBiaffine(ckpt, model, "deprel.", 2 * hidden, biaff, relations);
        if (_linearization)
            _linearizationScorer = new DeepBiaffine(ckpt, model, "linearization.", 2 * hidden, biaff, 1);
        if (_distance)
            _distanceScorer = new DeepBiaffine(ckpt, model, "distance.", 2 * hidden, biaff, 1);
    }

    /// <summary>
    /// Loads e.g. <c>models/converted/en/depparse/combined_charlm</c>. The pretrain and charlms are
    /// shared with the tagger and the constituency parser, so the caller owns them.
    /// </summary>
    public static DependencyParser Load(string basePath, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward) =>
        new(Checkpoint.Load(basePath), pretrain, charlmForward, charlmBackward);

    /// <summary>
    /// Sets Head and Deprel on every word. Needs UPOS/XPOS from the tagger and lemmas from the lemmatizer
    /// (a missing lemma is read as "_", as Stanza does).
    /// </summary>
    /// <remarks>
    /// Takes no <see cref="CharlmCache"/>: the parser runs the charlms over the words with a "\n" ROOT word
    /// in front, which changes every forward state and the backward ROOT state, so the tagger's
    /// representations don't apply.
    /// </remarks>
    public void Process(Document doc)
    {
        var sentences = doc.Sentences.Select(s => s.Words.ToList()).Where(w => w.Count > 0).ToList();
        if (sentences.Any(s => s.Any(w => w.Upos == null && w.Xpos == null)))
            throw new InvalidOperationException("Run the POS tagger before the dependency parser");

        foreach (var batch in Batches(sentences.Select(s => s.Count + 1).ToList()))
        {
            var parsed = Parse(batch.Select(i => (IReadOnlyList<Word>)sentences[i]).ToList());
            for (int b = 0; b < batch.Count; b++)
                for (int j = 0; j < sentences[batch[b]].Count; j++)
                    (sentences[batch[b]][j].Head, sentences[batch[b]][j].Deprel) = parsed[b][j];
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
            .Select(p => b[p]).ToList()).ToList();
    }

    /// <summary>Parses one batch: (head, deprel) for each word of each sentence.</summary>
    internal List<(int Head, string Deprel)[]> Parse(IReadOnlyList<IReadOnlyList<Word>> batch)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var (unlabeled, deprel) = Scores(batch);
        var labels = deprel.max(3).indexes;
        int width = (int)unlabeled.shape[1];
        var arcs = unlabeled.data<float>().ToArray();
        var labelIds = labels.data<long>().ToArray();

        var result = new List<(int, string)[]>(batch.Count);
        for (int b = 0; b < batch.Count; b++)
        {
            int n = batch[b].Count + 1;
            var scores = new double[n, n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    scores[i, j] = arcs[(b * width + i) * width + j];
            var tree = ChuLiuEdmonds.OneRoot(scores);
            result.Add(Enumerable.Range(1, n - 1)
                .Select(i => (tree[i], _deprels[labelIds[(b * width + i) * width + tree[i]] + VocabPrefixSize])).ToArray());
        }
        return result;
    }

    /// <summary>
    /// GraphParser.forward_scores, then the log-softmax over heads that predict applies:
    /// arc log-probs [batch, width, width] (dependent, head) and label scores [batch, width, width, relations].
    /// Row and column 0 are ROOT; padding columns count in the log-softmax, as in Stanza.
    /// </summary>
    internal (Tensor Unlabeled, Tensor Deprel) Scores(IReadOnlyList<IReadOnlyList<Word>> batch)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        // data.py load_doc: simplify_punct changes the words the parser sees (vocab and charlm alike).
        var texts = batch.Select(s => s.Select(w => SimplifyPunct(w.Text)).ToList()).ToList();
        int size = batch.Count, width = batch.Max(s => s.Count) + 1;
        var lengths = batch.Select(s => (long)s.Count + 1).ToArray();

        var word = new long[size * width];
        var lemma = new long[size * width];
        var upos = new long[size * width];
        var xpos = new long[size * width];
        var pretrained = new long[size * width];
        for (int b = 0; b < size; b++)
        {
            int row = b * width;
            word[row] = lemma[row] = upos[row] = xpos[row] = pretrained[row] = RootId;
            for (int j = 0; j < batch[b].Count; j++)
            {
                var w = batch[b][j];
                var lower = PyString.Lower(texts[b][j]);
                word[row + j + 1] = _wordVocab.GetValueOrDefault(lower, 1);
                lemma[row + j + 1] = _lemmaVocab.GetValueOrDefault(PyString.Lower(w.Lemma ?? "_"), 1);
                upos[row + j + 1] = _uposVocab.GetValueOrDefault(w.Upos ?? "_", 1);
                xpos[row + j + 1] = _xposVocab.GetValueOrDefault(w.Xpos ?? "_", 1);
                pretrained[row + j + 1] = _pretrain.UnitToId(lower);
            }
        }
        Tensor Ids(long[] ids) => torch.tensor(ids, [size, width]);

        var pos = _uposEmb.forward(Ids(upos)) + _xposEmb.forward(Ids(xpos));
        // "\n" stands in for ROOT in the charlm input.
        var charlmText = texts.Select(t => (IReadOnlyList<string>)t.Prepend("\n").ToList()).ToList();
        var input = cat([
            _transPretrained.forward(_pretrain.Embeddings[Ids(pretrained)]),
            _wordEmb.forward(Ids(word)),
            _lemmaEmb.forward(Ids(lemma)),
            pos,
            pos,
            Rnn.PadSequence(_charlmForward.BuildCharRepresentation(charlmText)),
            Rnn.PadSequence(_charlmBackward.BuildCharRepresentation(charlmText)),
        ], 2);
        var output = _lstm.Forward(input, lengths);
        // pad_packed_sequence leaves zeros past each sentence; the scorers see them in the padding columns.
        var padding = arange(width).unsqueeze(0).ge(torch.tensor(lengths).unsqueeze(1));
        output = output.masked_fill(padding.unsqueeze(2), 0);

        var unlabeled = _unlabeled.Forward(output).squeeze(3);
        var deprel = _deprel.Forward(output);
        var positions = arange(width);
        var headOffset = (positions.view(1, 1, -1) - positions.view(1, -1, 1)).expand(size, -1, -1);
        if (_linearizationScorer != null)
        {
            var lin = _linearizationScorer.Forward(output).squeeze(3);
            unlabeled = unlabeled + F.logsigmoid(lin * headOffset.sign().to_type(ScalarType.Float32));
        }
        if (_distanceScorer != null)
        {
            var dist = _distanceScorer.Forward(output).squeeze(3);
            var predicted = 1 + F.softplus(dist);
            var target = headOffset.abs();
            unlabeled = unlabeled + (-torch.log((target.to_type(ScalarType.Float32) - predicted).pow(2) / 2 + 1));
        }
        unlabeled = unlabeled.masked_fill(eye(width, dtype: ScalarType.Bool).unsqueeze(0), float.NegativeInfinity);
        var logProbs = F.log_softmax(unlabeled, 2);
        return (logProbs.MoveToOuterDisposeScope(), deprel.MoveToOuterDisposeScope());
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
        Require(Flag("charlm", false) && Flag("char", false) && config["char_emb_dim"]!.GetValue<int>() > 0, "a configuration without charlm");
        Require(Flag("pretrain", false), "a configuration without pretrain");
        Require(config["bert_model"] == null, "a transformer");
        Require(config["word_emb_dim"]!.GetValue<int>() > 0 && config["tag_emb_dim"]!.GetValue<int>() > 0, "disabled word or tag embeddings");
        Require(Flag("use_upos", true) && Flag("use_xpos", true) && Flag("use_ufeats", true), "a subset of UPOS/XPOS/UFeats");
        Require(!Flag("reversed", false), "reversed sentences");
        Require(!Flag("resolve_head_constraints", false), "resolve_head_constraints");
        Require(root["vocab"]!["_key2class"]!["xpos"]!.GetValue<string>() == "WordVocab", "a composite XPOS vocab");
        foreach (var key in new[] { "word", "lemma" })
            Require(root["vocab"]![key]!["lower"]!.GetValue<bool>(), $"a cased {key} vocab");
        foreach (var key in new[] { "upos", "xpos" })
            Require(!root["vocab"]![key]!["lower"]!.GetValue<bool>(), $"a lowercased {key} vocab");
    }

    // common/utils.py simplify_punct (as in the tagger): runs like "?!?" or "!!" become "?" or "!".
    private const string QuestionMarks = "?？︖﹖⁇", AllMarks = QuestionMarks + "!！︕﹗‼";
    private static readonly Regex Question = new($"^[{QuestionMarks}][{AllMarks}]+$");
    private static readonly Regex Exclam = new($"^[!！︕﹗‼][{AllMarks}]+$");

    private static string SimplifyPunct(string word) => Exclam.Replace(Question.Replace(word, "?"), "!");

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _wordEmb, _lemmaEmb, _uposEmb, _xposEmb, _transPretrained })
            m.Dispose();
        _lstm.Dispose();
        _unlabeled.Dispose();
        _deprel.Dispose();
        _linearizationScorer?.Dispose();
        _distanceScorer?.Dispose();
    }

    /// <summary>
    /// common/biaffine.py DeepBiaffineScorer with pairwise=True, scoring every word against every word:
    /// ReLU(W1 x) and ReLU(W2 x), each with a 1 appended, through a PairwiseBilinear.
    /// </summary>
    private sealed class DeepBiaffine : IDisposable
    {
        private readonly Linear _w1, _w2;
        private readonly Tensor _weight, _bias;

        public DeepBiaffine(Checkpoint ckpt, JsonNode model, string prefix, int input, int hidden, int output)
        {
            _w1 = nn.Linear(input, hidden).LoadFrom(ckpt, model, prefix + "W1.");
            _w2 = nn.Linear(input, hidden).LoadFrom(ckpt, model, prefix + "W2.");
            _weight = ckpt.ToTensor(model[prefix + "scorer.W_bilin.weight"]); // [hidden + 1, hidden + 1, output]
            _bias = ckpt.ToTensor(model[prefix + "scorer.W_bilin.bias"]);
            if (!_weight.shape.SequenceEqual([hidden + 1, hidden + 1, output]))
                throw new InvalidOperationException($"{prefix}scorer.W_bilin.weight has shape [{string.Join(", ", _weight.shape)}]");
        }

        /// <returns>[batch, width, width, output]: [b, i, j] scores word i against word j.</returns>
        public Tensor Forward(Tensor x)
        {
            using var scope = NewDisposeScope();
            var input1 = AppendOne(F.relu(_w1.forward(x)));
            var input2 = AppendOne(F.relu(_w2.forward(x)));
            var intermediate = einsum("NLI,IJO->NLJO", input1, _weight);
            var output = einsum("NLJO,NMJ->NLMO", intermediate, input2);
            intermediate.Dispose();
            // In place: the label scorer's output is [batch, width, width, relations], too large to copy.
            return output.add_(_bias).MoveToOuterDisposeScope();
        }

        private static Tensor AppendOne(Tensor x)
        {
            var shape = x.shape.ToArray();
            shape[^1] = 1;
            return cat([x, ones(shape, dtype: x.dtype)], -1);
        }

        public void Dispose()
        {
            _w1.Dispose();
            _w2.Dispose();
            _weight.Dispose();
            _bias.Dispose();
        }
    }
}
