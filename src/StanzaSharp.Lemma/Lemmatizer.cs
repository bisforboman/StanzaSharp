using System.IO.Compression;
using System.Text;
using System.Text.Json;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Lemma;

/// <summary>
/// Sets <see cref="Word.Lemma"/> from each word's text and UPOS. Port of Stanza's LemmaProcessor for an
/// <c>ensemble_dict</c> checkpoint (stanza/pipeline/lemma_processor.py, models/lemma/trainer.py):
/// a dictionary by (UPOS, word), then by word alone; words it misses go through a character seq2seq model
/// (models/common/seq2seq_model.py) with POS input, soft attention, a copy gate and an edit classifier
/// (identity / lowercase / use the decoded string), decoded greedily.
/// </summary>
internal sealed class Lemmatizer : IDisposable
{
    private const int PadId = 0, UnkId = 1, SosId = 2, EosId = 3; // seq2seq_constant.py
    private const string Unk = "<UNK>";

    private readonly Dictionary<string, Dictionary<string, string?>> _posDict;
    private readonly Dictionary<string, int> _charToId, _posToId;
    private readonly string[] _idToChar;
    private readonly int _vocabSize, _batchSize, _maxDecLen;
    private readonly Embedding _embedding, _posEmbedding;
    private readonly LSTM _encoder;
    private readonly LSTMCell _decoderCell;
    private readonly Linear _attnIn, _attnOut, _dec2vocab, _editHidden, _editOutput, _copyGate;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    private Lemmatizer(Checkpoint ckpt)
    {
        var root = ckpt.Root;
        var config = root["config"]!;
        bool Flag(string key) => config[key]?.GetValue<bool>() == true;
        void Require(bool ok, string what)
        {
            if (!ok)
                throw new NotSupportedException($"Lemmatizer checkpoints with {what} are not ported");
        }
        Require(!Flag("dict_only") && Flag("ensemble_dict"), "dict_only or without ensemble_dict");
        Require(config["beam_size"]?.GetValue<int>() == 1, "beam search (beam_size > 1)");
        Require(config["attn_type"]?.GetValue<string>() == "soft", $"attn_type {config["attn_type"]}");
        Require(Flag("edit") && config["num_edit"]?.GetValue<int>() == 3, "edit off or num_edit != 3");
        Require(Flag("copy"), "copy off");
        Require(Flag("pos") && config["pos_dim"]?.GetValue<int>() > 0, "POS input off");
        Require(!Flag("charlm"), "a charlm");
        Require(!Flag("caseless"), "caseless");
        Require(config["num_layers"]?.GetValue<int>() == 1, "more than one encoder layer");
        Require(root["contextual"]?.AsArray().Count is null or 0, "contextual lemmatizers");
        Require(root["dicts_version"]?.GetValue<int>() == 3, "a dictionary format other than v3 (gzip + JSON)");

        // trainer._unpack_pos_dict: {pos: {word: lemma}}, "*" holding the POS-independent entries.
        using (var gz = new GZipStream(new MemoryStream(Convert.FromBase64String(root["dicts"]!["$bytes"]!.GetValue<string>())), CompressionMode.Decompress))
            _posDict = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string?>>>(gz)!;

        var vocab = root["vocab"]!;
        _charToId = Checkpoint.UnitToId(vocab["char"]);
        _posToId = Checkpoint.UnitToId(vocab["pos"]);
        _idToChar = vocab["char"]!["_id2unit"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        _vocabSize = config["vocab_size"]!.GetValue<int>();
        _batchSize = config["batch_size"]!.GetValue<int>();
        _maxDecLen = config["max_dec_len"]!.GetValue<int>();

        var model = root["model"]!;
        int emb = config["emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        _embedding = nn.Embedding(_vocabSize, emb, padding_idx: PadId).LoadFrom(ckpt, model, "embedding.");
        _posEmbedding = nn.Embedding(config["pos_vocab_size"]!.GetValue<int>(), config["pos_dim"]!.GetValue<int>(), padding_idx: PadId)
            .LoadFrom(ckpt, model, "pos_embedding.");
        _encoder = nn.LSTM(emb, hidden / 2, numLayers: 1, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "encoder.");
        _decoderCell = nn.LSTMCell(emb, hidden).LoadFrom(ckpt, model, "decoder.lstm_cell.");
        _attnIn = nn.Linear(hidden, hidden, hasBias: false).LoadFrom(ckpt, model, "decoder.attention_layer.linear_in.");
        _attnOut = nn.Linear(hidden * 2, hidden, hasBias: false).LoadFrom(ckpt, model, "decoder.attention_layer.linear_out.");
        _dec2vocab = nn.Linear(hidden, _vocabSize).LoadFrom(ckpt, model, "dec2vocab.");
        _editHidden = nn.Linear(hidden, hidden / 2).LoadFrom(ckpt, model, "edit_clf.0.");
        _editOutput = nn.Linear(hidden / 2, 3).LoadFrom(ckpt, model, "edit_clf.2.");
        _copyGate = nn.Linear(hidden, 1).LoadFrom(ckpt, model, "copy_gate.");
    }

    /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/lemma/combined_nocharlm</c>.</summary>
    /// <param name="device">Where the model runs; CPU by default.</param>
    public static Lemmatizer Load(string basePath, Device? device = null) => Weights.On(device, () => new Lemmatizer(Checkpoint.Load(basePath)));

    /// <summary>Sets the lemma of every word in <paramref name="doc"/>; needs UPOS from the tagger.</summary>
    public void Process(Document doc)
    {
        var words = doc.Sentences.SelectMany(s => s.Words).ToList();
        var lemmas = Lemmatize(words);
        for (int i = 0; i < words.Count; i++)
            // Stanza's Word.lemma setter stores "_" as None unless the word itself is "_".
            words[i].Lemma = lemmas[i] == "_" && words[i].Text != "_" ? null : lemmas[i];
    }

    /// <summary>LemmaProcessor.process: dictionary hits, else the seq2seq prediction; "_" for an empty lemma.</summary>
    internal List<string> Lemmatize(IReadOnlyList<Word> words)
    {
        var lemmas = words.Select(w => Lookup(w.Text, w.Upos)).ToList();
        var misses = Enumerable.Range(0, words.Count).Where(i => lemmas[i] == null).ToList();
        var predicted = Postprocess(misses.Select(i => words[i]).ToList());
        for (int k = 0; k < misses.Count; k++)
            lemmas[misses[k]] = predicted[k];
        return lemmas.Select(l => l!.Length == 0 ? "_" : l).ToList();
    }

    /// <summary>trainer._lookup: the (UPOS, word) entry, then the POS-independent one.</summary>
    internal string? Lookup(string word, string? upos)
    {
        if (upos != null && _posDict.TryGetValue(upos, out var byPos) && byPos.GetValueOrDefault(word) is { } lemma)
            return lemma;
        return _posDict.TryGetValue("*", out var any) ? any.GetValueOrDefault(word) : null;
    }

    /// <summary>trainer.postprocess over the seq2seq output: apply the edit, and fall back to the word on empty or &lt;UNK&gt;.</summary>
    internal List<string> Postprocess(IReadOnlyList<Word> words)
    {
        var (decoded, edits) = Predict(words);
        var result = new List<string>(words.Count);
        for (int i = 0; i < words.Count; i++)
        {
            var word = words[i].Text;
            var lemma = edits[i] switch { 1 => word, 2 => PyString.Lower(word), _ => decoded[i] };
            result.Add(lemma.Length == 0 || lemma.Contains(Unk, StringComparison.Ordinal) ? word : lemma);
        }
        return result;
    }

    /// <summary>
    /// The seq2seq model on the words in batches of <c>batch_size</c>, as Stanza's DataLoader feeds it:
    /// the decoded string (before edits) and the edit class of each word.
    /// </summary>
    internal (List<string> Decoded, List<int> Edits) Predict(IReadOnlyList<Word> words)
    {
        // DeltaVocab: characters the vocabulary lacks get new ids past it, in code point order, so the copy
        // gate can still output them. Stanza builds it over the text, UPOS and current lemma of every word.
        var charToId = _charToId;
        var idToChar = _idToChar;
        var unknown = words.SelectMany(w => (w.Text + (w.Upos ?? "_") + (w.Lemma ?? "_")).EnumerateRunes())
            .Where(r => !_charToId.ContainsKey(r.ToString())).Distinct().Order().Select(r => r.ToString()).ToList();
        if (unknown.Count > 0)
        {
            charToId = new Dictionary<string, int>(_charToId);
            foreach (var c in unknown)
                charToId[c] = charToId.Count;
            idToChar = [.. _idToChar, .. unknown];
        }

        var decoded = new List<string>(words.Count);
        var edits = new List<int>(words.Count);
        for (int start = 0; start < words.Count; start += _batchSize)
        {
            var batch = words.Skip(start).Take(_batchSize).ToList();
            var (d, e) = PredictBatch(batch, charToId, idToChar);
            decoded.AddRange(d);
            edits.AddRange(e);
        }
        return (decoded, edits);
    }

    private (string[] Decoded, int[] Edits) PredictBatch(List<Word> words, Dictionary<string, int> charToId, string[] idToChar)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();

        // <SOS> chars <EOS>, one unit per code point; sorted like data.sort_all: longest first, ties by later index first.
        var src = words.Select(w => w.Text.EnumerateRunes().Select(r => charToId[r.ToString()]).Prepend(SosId).Append(EosId).ToArray()).ToList();
        var order = Enumerable.Range(0, words.Count).OrderByDescending(i => src[i].Length).ThenByDescending(i => i).ToArray();
        int batch = words.Count, width = src[order[0]].Length;
        var ids = new long[batch * width];
        for (int r = 0; r < batch; r++)
            for (int k = 0; k < src[order[r]].Length; k++)
                ids[r * width + k] = src[order[r]][k];
        var posIds = order.Select(i => (long)_posToId.GetValueOrDefault(words[i].Upos ?? "_", UnkId)).ToArray();

        var srcT = torch.tensor(ids, [batch, width], device: _device);
        var (decodedIds, editLogits) = Greedy(srcT, torch.tensor(posIds, device: _device), order.Select(i => (long)src[i].Length + 1).ToArray());

        var editValues = editLogits.ToArray<float>();
        var decoded = new string[batch];
        var edits = new int[batch];
        for (int r = 0; r < batch; r++)
        {
            decoded[order[r]] = string.Concat(decodedIds[r].Select(id => idToChar[id]));
            int best = 0; // numpy argmax: the first maximum
            for (int k = 1; k < 3; k++)
                if (editValues[r * 3 + k] > editValues[r * 3 + best])
                    best = k;
            edits[order[r]] = best;
        }
        return (decoded, edits);
    }

    /// <summary>Seq2SeqModel.predict_greedy (embed, encode, then decode one character per step for the whole batch).</summary>
    private (List<long>[] Decoded, Tensor EditLogits) Greedy(Tensor src, Tensor pos, long[] srcLens)
    {
        long batch = src.shape[0];

        // embed: characters past the trained vocabulary embed as <UNK>; the POS embedding goes in front.
        var embedSrc = src.masked_fill(src >= _vocabSize, UnkId);
        var encInputs = cat([_posEmbedding.forward(pos).unsqueeze(1), _embedding.forward(embedSrc)], 1);
        var srcMask = cat([torch.zeros([batch, 1], ScalarType.Bool, device: _device), src.eq(PadId)], 1);

        // encode
        using var lens = torch.tensor(srcLens);
        var packed = nn.utils.rnn.pack_padded_sequence(encInputs, lens, batch_first: true, enforce_sorted: true);
        var (packedOut, hn, cn) = _encoder.call(packed, null);
        var (ctx, _) = nn.utils.rnn.pad_packed_sequence(packedOut, batch_first: true);
        var h = cat([hn[1], hn[0]], 1);
        var c = cat([cn[1], cn[0]], 1);
        var editLogits = _editOutput.forward(F.relu(_editHidden.forward(h)));

        var decInputs = _embedding.forward(torch.tensor(new long[] { SosId }, device: _device));
        decInputs = decInputs.expand(batch, decInputs.shape[0], decInputs.shape[1]);

        var output = Enumerable.Range(0, (int)batch).Select(_ => new List<long>()).ToArray();
        var done = new bool[batch];
        int totalDone = 0;
        for (int step = 0; totalDone < batch && step < _maxDecLen; step++)
        {
            using var stepScope = NewDisposeScope();
            var logProbs = Decode(decInputs, ref h, ref c, ctx, srcMask, src);
            var preds = logProbs.squeeze(1).max(1, keepdim: true).indexes;
            decInputs = _embedding.forward(preds.masked_fill(preds >= _vocabSize, UnkId)).MoveToOuterDisposeScope();
            h.MoveToOuterDisposeScope();
            c.MoveToOuterDisposeScope();
            var values = preds.ToArray<long>();
            for (int i = 0; i < batch; i++)
            {
                if (done[i])
                    continue;
                if (values[i] == EosId)
                {
                    done[i] = true;
                    totalDone++;
                }
                else
                    output[i].Add(values[i]);
            }
        }
        return (output, editLogits);
    }

    /// <summary>
    /// Seq2SeqModel.decode for one step: LSTMAttention with SoftDotAttention, then the vocabulary
    /// distribution mixed with the copy distribution over the source characters.
    /// </summary>
    private Tensor Decode(Tensor decInputs, ref Tensor h, ref Tensor c, Tensor ctx, Tensor ctxMask, Tensor src)
    {
        // LSTMAttention.forward (batch_first) over a single step.
        var input = decInputs.transpose(0, 1);
        (h, c) = _decoderCell.forward(input[0], (h, c));
        var attn = torch.bmm(ctx, _attnIn.forward(h).unsqueeze(2)).squeeze(2).masked_fill(ctxMask, -1e12);
        attn = F.log_softmax(attn, 1);
        var attnW = torch.exp(attn);
        var weighted = torch.bmm(attnW.view(attnW.shape[0], 1, attnW.shape[1]), ctx).squeeze(1);
        var hTilde = torch.tanh(_attnOut.forward(cat([weighted, h], 1)));
        var hOut = cat([hTilde], 0).view(1, hTilde.shape[0], hTilde.shape[1]).transpose(0, 1);
        var logAttn = stack([attn], 0).transpose(0, 1);

        long batch = hOut.shape[0], steps = hOut.shape[1];
        var logits = _dec2vocab.forward(hOut.contiguous().view(batch * steps, -1)).view(batch, steps, -1);
        var logProbs = F.log_softmax(logits.view(-1, _vocabSize), 1).view(batch, steps, _vocabSize);

        // Copy: renormalize attention without the POS position, then scatter it onto the source character ids.
        var copyLogit = _copyGate.forward(hOut);
        logAttn = F.log_softmax(logAttn[.., .., 1..], -1);
        var logCopyProb = F.logsigmoid(copyLogit) + logAttn;
        var mx = logCopyProb.max(-1, keepdim: true).values;
        var copyProb = torch.exp(logCopyProb - mx);
        long vocab = Math.Max(_vocabSize, src.max().ToArray<long>()[0] + 1);
        var scattered = src.unsqueeze(1).expand(src.shape[0], copyProb.shape[1], src.shape[1]);
        var copied = torch.zeros([batch, steps, vocab], device: _device).scatter_add(-1, scattered, copyProb);
        var zeroMask = copied.eq(0);
        var logCopied = (torch.log(copied.masked_fill(zeroMask, 1e-12)) + mx).masked_fill(zeroMask, -1e12);

        var logNoCopy = -torch.log(torch.exp(copyLogit).add(1));
        if (vocab > _vocabSize) // characters new to the vocabulary reuse the <UNK> score
            logProbs = cat([logProbs, logProbs[.., .., UnkId].unsqueeze(2).expand(batch, steps, vocab - _vocabSize)], 2);
        logProbs = logProbs + logNoCopy;
        return torch.logsumexp(stack([logCopied, logProbs]), 0);
    }

    public void Dispose()
    {
        _embedding.Dispose();
        _posEmbedding.Dispose();
        _encoder.Dispose();
        _decoderCell.Dispose();
        _attnIn.Dispose();
        _attnOut.Dispose();
        _dec2vocab.Dispose();
        _editHidden.Dispose();
        _editOutput.Dispose();
        _copyGate.Dispose();
    }
}
