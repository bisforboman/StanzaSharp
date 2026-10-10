using System.Text.RegularExpressions;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Pos;

/// <summary>
/// Predicts UPOS, XPOS and UFeats for every word. Port of stanza/models/pos/model.py <c>Tagger</c>
/// for the English configuration:
/// - input is word embedding + projected pretrain vector + forward/backward charlm (<c>combined_charlm</c>),
///   or the model's own character LSTM projected by <c>trans_char</c> (<c>combined_nocharlm</c>)
/// - a highway biLSTM
/// - UPOS from an MLP
/// - XPOS and each UFeats key from biaffine scorers on an embedding of the predicted UPOS
/// </summary>
internal sealed class PosTagger : IDisposable
{
    // pos_processor.py: batch_maximum_tokens, which the English model's config does not set.
    private const int MaximumTokens = 5000;

    private readonly Pretrain _pretrain;
    private readonly Dictionary<string, int> _wordVocab;
    private readonly string[] _upos, _xpos;
    private readonly (string Key, string[] Values)[] _feats;
    private readonly int _wordUnk, _batchSize;
    private readonly IPosNet _net;

    internal PosTagger(Checkpoint ckpt, Pretrain pretrain, Func<bool, int[], IPosNet> net)
    {
        _pretrain = pretrain;
        var config = ckpt.Root["config"]!;
        CheckSupported(config);
        var vocab = ckpt.Root["vocab"]!;
        _wordVocab = Checkpoint.UnitToId(vocab["word"]);
        _wordUnk = _wordVocab["<UNK>"];
        _upos = Strings(vocab["upos"]!["_id2unit"]);
        _xpos = Strings(vocab["xpos"]!["_id2unit"]);
        _feats = vocab["feats"]!["_id2unit"]!.AsObject().Select(kv => (kv.Key, Strings(kv.Value))).ToArray();
        _batchSize = config["batch_size"]!.GetValue<int>();
        UsesCharlm = config["charlm"]?.GetValue<bool>() == true;
        _net = net(UsesCharlm, _feats.Select(f => f.Values.Length).ToArray());
    }

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static PosTagger LoadManaged(string basePath, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward)
    {
        var ckpt = Checkpoint.Load(basePath);
        return new PosTagger(ckpt, pretrain, (charlm, feats) =>
        {
            if (charlm && (charlmForward == null || charlmBackward == null || !charlmForward.IsForward || charlmBackward.IsForward))
                throw new ArgumentException("This tagger needs the forward charlm, then the backward one");
            var (_, upos, xpos) = Counts(ckpt);
            return new ManagedPosNet(ckpt, upos, xpos, feats, pretrain, charlm ? charlmForward : null, charlm ? charlmBackward : null);
        });
    }

    internal static (int Words, int Upos, int Xpos) Counts(Checkpoint ckpt)
    {
        var vocab = ckpt.Root["vocab"]!;
        return (Checkpoint.UnitToId(vocab["word"]).Count, vocab["upos"]!["_id2unit"]!.AsArray().Count, vocab["xpos"]!["_id2unit"]!.AsArray().Count);
    }

    /// <summary>Sets Upos, Xpos and Feats on every word of the document.</summary>
    /// <param name="charlms">If given, receives each sentence's charlm representations for the parser.</param>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default)
    {
        // Batched like Stanza's LengthLimitedBatchSampler: in document order, at most _batchSize sentences
        // and MaximumTokens words per batch, and a longer sentence alone.
        var sentences = doc.Sentences.Select(s => (Sentence: s, Words: s.Words.ToList())).Where(x => x.Words.Count > 0).ToList();
        for (int b = 0, end; b < sentences.Count; b = end)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int words = 0;
            for (end = b; end < sentences.Count && end - b < _batchSize && (end == b || words + sentences[end].Words.Count <= MaximumTokens); end++)
                words += sentences[end].Words.Count;
            var batch = sentences.GetRange(b, end - b);
            var tags = Predict(batch.Select(x => (IReadOnlyList<string>)x.Words.Select(w => w.Text).ToList()).ToList(), out _,
                charlms, charlms == null ? null : batch.Select(x => (Sentence?)x.Sentence).ToList(), cancellationToken);
            for (int i = 0; i < batch.Count; i++)
                for (int j = 0; j < batch[i].Words.Count; j++)
                    (batch[i].Words[j].Upos, batch[i].Words[j].Xpos, batch[i].Words[j].Feats) = tags[i][j];
        }
    }

    /// <summary>Tags for each word, plus the UPOS logits (one [words, upos] array per sentence) for tests.</summary>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: receives sentence i's charlm representations under its key
    /// (if not null), when simplify_punct left its words unchanged (the readers see the words as written).</param>
    /// <param name="cancellationToken">A batch is up to 5000 words (seconds on a slow CPU), so this is checked after each
    /// charlm pass (or the character model), between LSTM layers and between the heads.</param>
    internal List<(string Upos, string Xpos, string? Feats)[]> Predict(IReadOnlyList<IReadOnlyList<string>> sentences, out List<float[]> uposLogits,
        CharlmCache? charlms = null, IReadOnlyList<Sentence?>? cacheKeys = null, CancellationToken cancellationToken = default)
    {
        var original = sentences;
        sentences = sentences.Select(s => (IReadOnlyList<string>)s.Select(SimplifyPunct).ToList()).ToList();
        int batch = sentences.Count;
        var keys = charlms == null || cacheKeys == null || !UsesCharlm ? null
            : Enumerable.Range(0, batch).Select(i => sentences[i].SequenceEqual(original[i]) ? cacheKeys[i] : null).ToList();

        var flat = sentences.SelectMany(s => s).ToArray();
        var wordIds = new long[flat.Length];
        var pretrainIds = new long[flat.Length];
        for (int k = 0; k < flat.Length; k++)
        {
            var lower = PyString.Lower(flat[k]);
            wordIds[k] = _wordVocab.GetValueOrDefault(lower, _wordUnk);
            pretrainIds[k] = _pretrain.UnitToId(lower);
        }
        var output = _net.Forward(sentences, wordIds, pretrainIds, keys == null ? null : charlms, keys, cancellationToken);

        int nUpos = _upos.Length;
        uposLogits = [];
        var result = new List<(string, string, string?)[]>(batch);
        for (int i = 0, k = 0; i < batch; i++)
        {
            var tags = new (string, string, string?)[sentences[i].Count];
            uposLogits.Add(output.UposScores[(k * nUpos)..((k + tags.Length) * nUpos)]);
            for (int j = 0; j < tags.Length; j++, k++)
                tags[j] = (_upos[output.Upos[k]], _xpos[output.Xpos[k]], FeatsString(output.Feats, k));
            result.Add(tags);
        }
        return result;
    }

    /// <summary>CompositeVocab.id2unit for UFeats: "Key=Value|..." skipping &lt;EMPTY&gt;, null for none.</summary>
    private string? FeatsString(long[][] featIds, int k)
    {
        const int EmptyId = 2; // vocab.EMPTY_ID
        var parts = new List<string>();
        for (int f = 0; f < _feats.Length; f++)
            if (featIds[f][k] != EmptyId)
                parts.Add($"{_feats[f].Key}={_feats[f].Values[featIds[f][k]]}");
        return parts.Count == 0 ? null : string.Join('|', parts);
    }

    private static void CheckSupported(JsonNode config)
    {
        void Require(bool ok, string what)
        {
            if (!ok) throw new NotSupportedException($"POS checkpoint uses {what}, which is not ported");
        }
        Require(config["char"]?.GetValue<bool>() == true && config["char_emb_dim"]?.GetValue<int>() > 0, "a configuration without character features");
        Require(config["charlm_transform_dim"] == null, "charlm_transform_dim");
        Require(config["pretrain"]?.GetValue<bool>() == true, "a configuration without pretrain");
        Require(config["bert_model"] == null, "a transformer");
        Require(config["share_hid"]?.GetValue<bool>() != true, "share_hid");
        Require(config["tag_column_link"]?.GetValue<string>() == "tag_emb", "a tag column link other than tag_emb");
        var columns = config["tag_columns"]?.ToJsonString();
        Require(columns == """[["upos","upos",null,"WORD",true,[]],["xpos","xpos",null,"AUTO",true,["upos"]],["feats","feats",null,"FEATURES",true,["upos"]]]""",
            $"tag columns {columns}");
    }

    /// <summary>Whether this tagger runs the shared charlms (<c>_charlm</c>), whose outputs it can then hand to a <see cref="CharlmCache"/>.</summary>
    internal bool UsesCharlm { get; }

    private static string[] Strings(JsonNode? array) => array!.AsArray().Select(x => x!.GetValue<string>()).ToArray();

    // pos/data.py load_doc → common/utils.py simplify_punct: the tagger sees runs like "?!?" or "!!"
    // as a single "?" or "!" (QUESTION_RE / EXCLAM_RE, including full-width and small forms).
    private const string QuestionMarks = "?？︖﹖⁇", AllMarks = QuestionMarks + "!！︕﹗‼";
    private static readonly Regex Question = new($"^[{QuestionMarks}][{AllMarks}]+$");
    private static readonly Regex Exclam = new($"^[!！︕﹗‼][{AllMarks}]+$");

    internal static string SimplifyPunct(string word) => Exclam.Replace(Question.Replace(word, "?"), "!");

    public void Dispose() => _net.Dispose();
}
