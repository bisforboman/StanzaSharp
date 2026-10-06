using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Ner;

/// <summary>
/// Named-entity recognizer: sets a BIOES tag on every token and builds each sentence's entities.
/// Port of stanza/models/ner/model.py <c>NERTagger</c> and trainer.py <c>predict</c> for the English
/// configuration:
/// - input is the pretrain vector plus a fine-tuned delta embedding (both lowercased) and the
///   forward/backward charlm, through a linear <c>input_transform</c>
/// - a biLSTM, then one linear layer per tag set and CRF Viterbi decoding
/// - <c>fix_singleton_tags</c> on the decoded sequence
/// NER reads tokens, not words, like Stanza: an MWT such as "don't" is tagged once.
/// </summary>
internal sealed class NerTagger : IDisposable
{
    private const int UnkId = 1, PadId = 0; // vocab.UNK_ID / PAD_ID
    private const int VocabPrefixSize = 4;  // <PAD> <UNK> <EMPTY> <ROOT>

    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel _charlmForward, _charlmBackward;
    private readonly Dictionary<string, int> _deltaVocab;
    private readonly string[] _tags;
    private readonly float[] _transitions;
    private readonly int _batchSize;

    private readonly Embedding _deltaEmb;
    private readonly Linear _inputTransform, _tagClf;
    private readonly LSTM _lstm;
    private readonly Tensor _hInit, _cInit;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    private NerTagger(Checkpoint ckpt, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward)
    {
        _pretrain = pretrain;
        _charlmForward = charlmForward;
        _charlmBackward = charlmBackward;
        if (!charlmForward.IsForward || charlmBackward.IsForward)
            throw new ArgumentException("Pass the forward charlm first, then the backward one");

        var config = ckpt.Root["config"]!;
        var vocab = ckpt.Root["vocab"]!;
        var model = ckpt.Root["model"]!;
        CheckSupported(config, vocab, pretrain);

        _deltaVocab = Checkpoint.UnitToId(vocab["delta"]);
        _batchSize = config["batch_size"]!.GetValue<int>();

        // The tag vocab has one column per tag set ("multi" NER: OntoNotes and WorldWide here). Stanza
        // decodes every column but keeps only predict_tagset's, so that is the only head we run.
        int tagset = config["predict_tagset"]?.GetValue<int>() ?? 0;
        var columns = vocab["tag"]!["_id2unit"]!["$dict"]!.AsArray();
        _tags = columns.Single(c => c![0]!.GetValue<int>() == tagset)![1]!.AsArray().Select(x => x!.GetValue<string>()).ToArray();
        using (var t = ckpt.ToTensor(model[$"crits.{tagset}._transitions"]))
            _transitions = t.ToArray<float>();

        int wordDim = config["word_emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["num_layers"]!.GetValue<int>();
        int inputSize = wordDim + charlmForward.HiddenDim + charlmBackward.HiddenDim;

        _deltaEmb = nn.Embedding(_deltaVocab.Count, wordDim, padding_idx: PadId).LoadFrom(ckpt, model, "delta_emb.");
        _inputTransform = nn.Linear(inputSize, inputSize).LoadFrom(ckpt, model, "input_transform.");
        _lstm = nn.LSTM(inputSize, hidden, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "taggerlstm.lstm.");
        _hInit = ckpt.ToTensor(model["taggerlstm_h_init"]);
        _cInit = ckpt.ToTensor(model["taggerlstm_c_init"]);
        _tagClf = nn.Linear(hidden * 2, _tags.Length).LoadFrom(ckpt, model, $"tag_clfs.{tagset}.");
    }

    /// <summary>
    /// Loads e.g. <c>models/converted/en/ner/ontonotes-ww-multi_charlm</c>. The pretrain and charlms are
    /// shared with the other processors, so the caller owns them.
    /// </summary>
    /// <param name="device">Where the model runs; CPU by default. Load the pretrain and charlms on the same device.</param>
    public static NerTagger Load(string basePath, Pretrain pretrain, CharLanguageModel charlmForward, CharLanguageModel charlmBackward, Device? device = null) =>
        Weights.On(device, () => new NerTagger(Checkpoint.Load(basePath), pretrain, charlmForward, charlmBackward));

    /// <summary>Sets <see cref="Token.Ner"/> on every token and rebuilds every sentence's <see cref="Sentence.Entities"/>.</summary>
    /// <param name="charlms">The tagger's charlm outputs, reused for sentences whose tokens are exactly its words (no MWT).</param>
    public void Process(Document doc, CharlmCache? charlms = null)
    {
        // ner/data.py: batches of batch_size sentences in document order.
        var sentences = doc.Sentences.Where(s => s.Tokens.Count > 0).ToList();
        foreach (var batch in sentences.Chunk(_batchSize))
        {
            var texts = batch.Select(s => (IReadOnlyList<string>)s.Tokens.Select(t => t.Text).ToList()).ToList();
            var tags = Predict(texts, out _, charlms == null ? null : i =>
                charlms.TryGet(batch[i], out var reps) && batch[i].Words.Select(w => w.Text).SequenceEqual(texts[i]) ? reps : null);
            for (int i = 0; i < batch.Length; i++)
            {
                for (int j = 0; j < tags[i].Length; j++)
                    batch[i].Tokens[j].Ner = tags[i][j];
                Entity.Build(batch[i], doc.Text);
            }
        }
    }

    /// <summary>BIOES tags for each token, plus the emission scores (one [tokens, tags] array per sentence) for tests.</summary>
    /// <param name="cachedCharlm">Returns a sentence's charlm representations if they were already computed.</param>
    internal List<string[]> Predict(IReadOnlyList<IReadOnlyList<string>> sentences, out List<float[]> emissions,
        Func<int, (Tensor Forward, Tensor Backward)?>? cachedCharlm = null)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        int batch = sentences.Count, width = sentences.Max(s => s.Count);
        var lengths = sentences.Select(s => (long)s.Count).ToArray();

        // extract_static_embeddings: the word is lowercased for both vocabs. A word the pretrain knows
        // but the delta vocab doesn't gets the delta's zero PAD row rather than its UNK row.
        var wordIds = new long[batch * width];
        var deltaIds = new long[batch * width];
        for (int i = 0; i < batch; i++)
            for (int j = 0; j < sentences[i].Count; j++)
            {
                var lower = PyString.Lower(sentences[i][j]);
                int word = _pretrain.UnitToId(lower);
                int delta = _deltaVocab.GetValueOrDefault(lower, UnkId);
                wordIds[i * width + j] = word;
                deltaIds[i * width + j] = delta == UnkId && word != _pretrain.UnkId ? PadId : delta;
            }
        var words = _pretrain.Embeddings[torch.tensor(wordIds, [batch, width], device: _device)]
            + _deltaEmb.forward(torch.tensor(deltaIds, [batch, width], device: _device));

        var cached = Enumerable.Range(0, batch).Select(i => cachedCharlm?.Invoke(i)).ToList();
        var missing = Enumerable.Range(0, batch).Where(i => cached[i] == null).ToList();
        var forward = _charlmForward.BuildCharRepresentation(missing.Select(i => sentences[i]).ToList());
        var backward = _charlmBackward.BuildCharRepresentation(missing.Select(i => sentences[i]).ToList());
        for (int k = 0; k < missing.Count; k++)
            cached[missing[k]] = (forward[k], backward[k]);
        var charsForward = Rnn.PadSequence(cached.Select(c => c!.Value.Forward).ToList());
        var charsBackward = Rnn.PadSequence(cached.Select(c => c!.Value.Backward).ToList());

        var input = _inputTransform.forward(cat([words, charsForward, charsBackward], 2));
        var h0 = _hInit.expand(_hInit.shape[0], batch, _hInit.shape[2]).contiguous();
        var c0 = _cInit.expand(_cInit.shape[0], batch, _cInit.shape[2]).contiguous();
        var output = Rnn.RunPacked(_lstm, input, lengths, (h0, c0));
        var logits = _tagClf.forward(output).ToArray<float>();

        int nTags = _tags.Length;
        emissions = [];
        var result = new List<string[]>(batch);
        for (int i = 0; i < batch; i++)
        {
            var scores = logits[(i * width * nTags)..((i * width + sentences[i].Count) * nTags)];
            emissions.Add(scores);
            // The model can predict a tag below "O" (a vocab prefix); Stanza turns those into "O".
            var tags = Viterbi(scores, _transitions, nTags).Select(id => _tags[Math.Max(id, VocabPrefixSize)]).ToArray();
            result.Add(FixSingletonTags(tags));
        }
        return result;
    }

    /// <summary>
    /// crf.py viterbi_decode in float32, like numpy: the highest-scoring tag sequence for
    /// <paramref name="scores"/> [n, tags] under <paramref name="transitions"/> [from, to]. Ties go to the
    /// lowest tag id, as with np.argmax.
    /// </summary>
    internal static int[] Viterbi(float[] scores, float[] transitions, int tags)
    {
        int n = scores.Length / tags;
        var trellis = new float[scores.Length];
        var backpointers = new int[scores.Length];
        Array.Copy(scores, trellis, tags);
        for (int t = 1; t < n; t++)
            for (int j = 0; j < tags; j++)
            {
                int best = 0;
                float max = trellis[(t - 1) * tags] + transitions[j];
                for (int i = 1; i < tags; i++)
                {
                    float v = trellis[(t - 1) * tags + i] + transitions[i * tags + j];
                    if (v > max)
                        (max, best) = (v, i);
                }
                trellis[t * tags + j] = scores[t * tags + j] + max;
                backpointers[t * tags + j] = best;
            }

        var path = new int[n];
        int last = (n - 1) * tags;
        for (int j = 1; j < tags; j++)
            if (trellis[last + j] > trellis[last + path[n - 1]])
                path[n - 1] = j;
        for (int t = n - 1; t > 0; t--)
            path[t - 1] = backpointers[t * tags + path[t]];
        return path;
    }

    /// <summary>
    /// trainer.py fix_singleton_tags: an I- that cannot continue becomes E-, one that cannot start
    /// becomes B-; then a B- or E- standing alone becomes S-. Each check sees the tags already fixed
    /// to its left and the original ones to its right, as in Stanza.
    /// </summary>
    internal static string[] FixSingletonTags(string[] tags)
    {
        var fixedTags = (string[])tags.Clone();
        int n = fixedTags.Length;
        for (int i = 0; i < n; i++)
        {
            var tag = fixedTags[i];
            if (!tag.StartsWith("I-", StringComparison.Ordinal))
                continue;
            var type = tag[2..];
            if (i == n - 1 || (fixedTags[i + 1] != "I-" + type && fixedTags[i + 1] != "E-" + type))
                fixedTags[i] = "E-" + type;
            if (i == 0 || (fixedTags[i - 1] != "B-" + type && fixedTags[i - 1] != "I-" + type))
                fixedTags[i] = "B-" + type;
        }
        for (int i = 0; i < n; i++)
        {
            var tag = fixedTags[i];
            var type = tag.Length >= 2 ? tag[2..] : "";
            if (tag.StartsWith("B-", StringComparison.Ordinal) &&
                (i == n - 1 || (fixedTags[i + 1] != "I-" + type && fixedTags[i + 1] != "E-" + type)))
                fixedTags[i] = "S-" + type;
            if (tag.StartsWith("E-", StringComparison.Ordinal) &&
                (i == 0 || (fixedTags[i - 1] != "B-" + type && fixedTags[i - 1] != "I-" + type)))
                fixedTags[i] = "S-" + type;
        }
        return fixedTags;
    }

    private static void CheckSupported(JsonNode config, JsonNode vocab, Pretrain pretrain)
    {
        void Require(bool ok, string what)
        {
            if (!ok) throw new NotSupportedException($"NER checkpoint uses {what}, which is not ported");
        }
        Require(config["bert_model"] == null && config["use_peft"]?.GetValue<bool>() != true, "a transformer");
        Require(config["char"]?.GetValue<bool>() == true && config["charlm"]?.GetValue<bool>() == true, "a configuration without charlm");
        Require(config["char_lowercase"]?.GetValue<bool>() != true, "char_lowercase");
        Require(config["lowercase"]?.GetValue<bool>() != false, "a cased word embedding");
        Require(config["word_emb_dim"]?.GetValue<int>() == pretrain.Dim, "a word embedding not matching the pretrain");
        Require(config["input_transform"]?.GetValue<bool>() == true, "a configuration without input_transform");
        Require(config["connect_output_layers"]?.GetValue<bool>() != true, "connect_output_layers");
        Require(vocab["word"] != null && vocab["delta"] != null, "a vocab without word and delta embeddings");
        // The word embedding is the pretrain's matrix (not saved in the checkpoint), indexed by this vocab.
        Require(vocab["word"]!["_id2unit"]?.AsArray().Count == pretrain.Embeddings.shape[0], "a word vocab other than the pretrain's");
        Require(vocab["delta"]!["lower"]?.GetValue<bool>() == true, "a cased delta vocab");
        Require(vocab["delta"]!["ignore"]?.AsArray().Count is null or 0, "emb_finetune_known_only (a delta vocab with ignored words)");
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _deltaEmb, _inputTransform, _lstm, _tagClf })
            m.Dispose();
        _hInit.Dispose();
        _cInit.Dispose();
    }
}
