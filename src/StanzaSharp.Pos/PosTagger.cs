using System.Text.RegularExpressions;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Pos;

/// <summary>
/// Predicts UPOS, XPOS and UFeats for every word. Port of stanza/models/pos/model.py <c>Tagger</c>
/// for the English configuration:
/// - input is word embedding + projected pretrain vector + forward/backward charlm
/// - a highway biLSTM
/// - UPOS from an MLP
/// - XPOS and each UFeats key from biaffine scorers on an embedding of the predicted UPOS
/// </summary>
public sealed class PosTagger : IDisposable
{
    // pos_processor.py: batch_maximum_tokens, which the English model's config does not set.
    private const int MaximumTokens = 5000;

    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Dictionary<string, int> _wordVocab;
    private readonly string[] _upos, _xpos;
    private readonly (string Key, string[] Values)[] _feats;
    private readonly int _wordUnk, _batchSize;

    private readonly Embedding _wordEmb, _uposEmb;
    private readonly Linear _transPretrained;
    private readonly HighwayLstm _lstm;
    private readonly Linear _uposHid, _uposClf, _xposHid, _featsHid;
    private readonly Biaffine _xposClf;
    private readonly Biaffine[] _featsClf;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    private PosTagger(Checkpoint ckpt, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        _charlmForward = charlmForward;
        _charlmBackward = charlmBackward;
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");

        var config = ckpt.Root["config"]!;
        CheckSupported(config);
        var model = ckpt.Root["model"]!;
        var vocab = ckpt.Root["vocab"]!;

        _wordVocab = Checkpoint.UnitToId(vocab["word"]);
        _wordUnk = _wordVocab["<UNK>"];
        _upos = Strings(vocab["upos"]!["_id2unit"]);
        _xpos = Strings(vocab["xpos"]!["_id2unit"]);
        _feats = vocab["feats"]!["_id2unit"]!.AsObject().Select(kv => (kv.Key, Strings(kv.Value))).ToArray();
        _batchSize = config["batch_size"]!.GetValue<int>();

        int hidden = config["hidden_dim"]!.GetValue<int>();
        int biaff = config["deep_biaff_hidden_dim"]!.GetValue<int>();
        int compositeBiaff = config["composite_deep_biaff_hidden_dim"]!.GetValue<int>();
        int tagEmb = config["tag_emb_dim"]!.GetValue<int>();
        int transformed = config["transformed_dim"]!.GetValue<int>();
        int wordEmb = config["word_emb_dim"]!.GetValue<int>();
        int inputSize = wordEmb + transformed + charlmForward.HiddenDim + charlmBackward.HiddenDim;

        _wordEmb = nn.Embedding(_wordVocab.Count, wordEmb, padding_idx: 0).LoadFrom(ckpt, model, "word_emb.");
        _uposEmb = nn.Embedding(_upos.Length, tagEmb, padding_idx: 0).LoadFrom(ckpt, model, "upos_emb.");
        _transPretrained = nn.Linear(pretrain.Dim, transformed, hasBias: false).LoadFrom(ckpt, model, "trans_pretrained.");
        _lstm = new HighwayLstm(ckpt, model, "taggerlstm", inputSize, hidden, config["num_layers"]!.GetValue<int>());

        _uposHid = nn.Linear(hidden * 2, biaff).LoadFrom(ckpt, model, "upos_hid.");
        _uposClf = nn.Linear(biaff, _upos.Length).LoadFrom(ckpt, model, "upos_clf.");
        _xposHid = nn.Linear(hidden * 2, biaff).LoadFrom(ckpt, model, "tag_hid.xpos.");
        _xposClf = new Biaffine(ckpt, model, "tag_clf.xpos.", biaff, tagEmb, _xpos.Length);
        _featsHid = nn.Linear(hidden * 2, compositeBiaff).LoadFrom(ckpt, model, "tag_hid.feats.");
        _featsClf = _feats.Select((f, i) => new Biaffine(ckpt, model, $"tag_clf.feats.{i}.", compositeBiaff, tagEmb, f.Values.Length)).ToArray();
    }

    /// <summary>
    /// Loads e.g. <c>models/converted/en/pos/combined_charlm</c>. The pretrain and charlms are
    /// shared with the parser, so the caller owns them.
    /// </summary>
    /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
    public static PosTagger Load(string basePath, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward, Device? device = null) =>
        Weights.On(device, () => new PosTagger(Checkpoint.Load(basePath), pretrain, charlmForward, charlmBackward));

    /// <summary>Sets Upos, Xpos and Feats on every word of the document.</summary>
    /// <param name="charlms">If given, receives each sentence's charlm representations for the parser.</param>
    public void Process(Document doc, CharlmCache? charlms = null)
    {
        // Batched like Stanza's LengthLimitedBatchSampler: in document order, at most _batchSize sentences
        // and MaximumTokens words per batch, and a longer sentence alone.
        var sentences = doc.Sentences.Select(s => (Sentence: s, Words: s.Words.ToList())).Where(x => x.Words.Count > 0).ToList();
        for (int b = 0, end; b < sentences.Count; b = end)
        {
            int words = 0;
            for (end = b; end < sentences.Count && end - b < _batchSize && (end == b || words + sentences[end].Words.Count <= MaximumTokens); end++)
                words += sentences[end].Words.Count;
            var batch = sentences.GetRange(b, end - b);
            var tags = Predict(batch.Select(x => (IReadOnlyList<string>)x.Words.Select(w => w.Text).ToList()).ToList(), out _,
                charlms == null ? null : (i, forward, backward) => charlms.Add(batch[i].Sentence, forward, backward));
            for (int i = 0; i < batch.Count; i++)
                for (int j = 0; j < batch[i].Words.Count; j++)
                    (batch[i].Words[j].Upos, batch[i].Words[j].Xpos, batch[i].Words[j].Feats) = tags[i][j];
        }
    }

    /// <summary>Tags for each word, plus the UPOS logits (one [words, upos] array per sentence) for tests.</summary>
    /// <param name="keepCharlm">Called with each sentence's charlm representations, which it then owns.</param>
    internal List<(string Upos, string Xpos, string? Feats)[]> Predict(IReadOnlyList<IReadOnlyList<string>> sentences, out List<float[]> uposLogits,
        Action<int, Tensor, Tensor>? keepCharlm = null)
    {
        var original = sentences;
        sentences = sentences.Select(s => (IReadOnlyList<string>)s.Select(SimplifyPunct).ToList()).ToList();
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        int batch = sentences.Count, width = sentences.Max(s => s.Count);
        var lengths = sentences.Select(s => (long)s.Count).ToArray();

        var wordIds = new long[batch * width];
        var pretrainIds = new long[batch * width];
        for (int i = 0; i < batch; i++)
            for (int j = 0; j < sentences[i].Count; j++)
            {
                var lower = PyString.Lower(sentences[i][j]);
                wordIds[i * width + j] = _wordVocab.GetValueOrDefault(lower, _wordUnk);
                pretrainIds[i * width + j] = _pretrain.UnitToId(lower);
            }

        var words = _wordEmb.forward(torch.tensor(wordIds, [batch, width], device: _device));
        var pretrained = _transPretrained.forward(_pretrain.Embeddings[torch.tensor(pretrainIds, [batch, width], device: _device)]);
        var repsForward = _charlmForward.BuildCharRepresentation(sentences);
        var repsBackward = _charlmBackward.BuildCharRepresentation(sentences);
        if (keepCharlm != null)
            for (int i = 0; i < batch; i++)
                if (sentences[i].SequenceEqual(original[i])) // the parser reads the words before SimplifyPunct
                    keepCharlm(i, scope.Detach(repsForward[i]), scope.Detach(repsBackward[i]));
        var charsForward = Rnn.PadSequence(repsForward);
        var charsBackward = Rnn.PadSequence(repsBackward);
        var output = _lstm.Forward(cat([words, pretrained, charsForward, charsBackward], 2), lengths);

        var uposScores = _uposClf.forward(F.relu(_uposHid.forward(output)));
        var uposIds = uposScores.argmax(2);
        var parent = _uposEmb.forward(uposIds);
        var xposIds = _xposClf.Forward(F.relu(_xposHid.forward(output)), parent).argmax(2);
        var featsHid = F.relu(_featsHid.forward(output));
        var featIds = _featsClf.Select(c => c.Forward(featsHid, parent).argmax(2).ToArray<long>()).ToArray();

        var upos = uposIds.ToArray<long>();
        var xpos = xposIds.ToArray<long>();
        var logits = uposScores.ToArray<float>();
        int nUpos = _upos.Length;
        uposLogits = [];
        var result = new List<(string, string, string?)[]>(batch);
        for (int i = 0; i < batch; i++)
        {
            var tags = new (string, string, string?)[sentences[i].Count];
            for (int j = 0; j < tags.Length; j++)
            {
                int k = i * width + j;
                tags[j] = (_upos[upos[k]], _xpos[xpos[k]], FeatsString(featIds, k));
            }
            result.Add(tags);
            uposLogits.Add(logits[(i * width * nUpos)..((i * width + sentences[i].Count) * nUpos)]);
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
        Require(config["charlm"]?.GetValue<bool>() == true && config["char"]?.GetValue<bool>() == true, "a configuration without charlm");
        Require(config["charlm_transform_dim"] == null, "charlm_transform_dim");
        Require(config["pretrain"]?.GetValue<bool>() == true, "a configuration without pretrain");
        Require(config["bert_model"] == null, "a transformer");
        Require(config["share_hid"]?.GetValue<bool>() != true, "share_hid");
        Require(config["tag_column_link"]?.GetValue<string>() == "tag_emb", "a tag column link other than tag_emb");
        var columns = config["tag_columns"]?.ToJsonString();
        Require(columns == """[["upos","upos",null,"WORD",true,[]],["xpos","xpos",null,"AUTO",true,["upos"]],["feats","feats",null,"FEATURES",true,["upos"]]]""",
            $"tag columns {columns}");
    }

    private static string[] Strings(JsonNode? array) => array!.AsArray().Select(x => x!.GetValue<string>()).ToArray();

    // pos/data.py load_doc → common/utils.py simplify_punct: the tagger sees runs like "?!?" or "!!"
    // as a single "?" or "!" (QUESTION_RE / EXCLAM_RE, including full-width and small forms).
    private const string QuestionMarks = "?？︖﹖⁇", AllMarks = QuestionMarks + "!！︕﹗‼";
    private static readonly Regex Question = new($"^[{QuestionMarks}][{AllMarks}]+$");
    private static readonly Regex Exclam = new($"^[!！︕﹗‼][{AllMarks}]+$");

    internal static string SimplifyPunct(string word) => Exclam.Replace(Question.Replace(word, "?"), "!");

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _wordEmb, _uposEmb, _transPretrained, _uposHid, _uposClf, _xposHid, _featsHid })
            m.Dispose();
        _lstm.Dispose();
        _xposClf.Dispose();
        foreach (var c in _featsClf)
            c.Dispose();
    }
}
