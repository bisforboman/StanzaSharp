using System.IO.Compression;
using System.Text;
using System.Text.Json;
using StanzaSharp.Nn;

namespace StanzaSharp.Lemma;

/// <summary>
/// Sets <see cref="Word.Lemma"/> from each word's text and UPOS. Port of Stanza's LemmaProcessor for an
/// <c>ensemble_dict</c> checkpoint (stanza/pipeline/lemma_processor.py, models/lemma/trainer.py):
/// a dictionary by (UPOS, word), then by word alone; words it misses go through a character seq2seq model
/// (models/common/seq2seq_model.py) with POS input, soft attention, a copy gate and an edit classifier
/// (identity / lowercase / use the decoded string), decoded greedily.
/// The network is an <see cref="ILemmaNet"/> (per backend); the dictionary, DeltaVocab, batching, the greedy
/// loop and the edits are here.
/// </summary>
internal sealed class Lemmatizer : IDisposable
{
    private const int UnkId = LemmaConfig.UnkId, SosId = LemmaConfig.SosId, EosId = LemmaConfig.EosId;
    private const string Unk = "<UNK>";

    private readonly Dictionary<string, Dictionary<string, string?>> _posDict;
    private readonly Dictionary<string, int> _charToId, _posToId;
    private readonly string[] _idToChar;
    private readonly int _batchSize, _maxDecLen;
    private readonly ILemmaNet _net;

    /// <param name="net">Builds the network from the checkpoint and its sizes (a backend's).</param>
    internal Lemmatizer(Checkpoint ckpt, Func<Checkpoint, LemmaConfig, ILemmaNet> net)
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
        _batchSize = config["batch_size"]!.GetValue<int>();
        _maxDecLen = config["max_dec_len"]!.GetValue<int>();
        var sizes = LemmaConfig.From(config);
        _net = net(ckpt, sizes);
    }

    /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/lemma/combined_nocharlm</c>, on the managed backend.</summary>
    public static Lemmatizer LoadManaged(string basePath) => new(Checkpoint.Load(basePath), (ckpt, sizes) => new ManagedLemmaNet(ckpt, sizes));

    /// <summary>Sets the lemma of every word in <paramref name="doc"/>; needs UPOS from the tagger.</summary>
    public void Process(Document doc, CancellationToken cancellationToken = default)
    {
        var words = doc.Sentences.SelectMany(s => s.Words).ToList();
        var lemmas = Lemmatize(words, cancellationToken);
        for (int i = 0; i < words.Count; i++)
            // Stanza's Word.lemma setter stores "_" as None unless the word itself is "_".
            words[i].Lemma = lemmas[i] == "_" && words[i].Text != "_" ? null : lemmas[i];
    }

    /// <summary>LemmaProcessor.process: dictionary hits, else the seq2seq prediction; "_" for an empty lemma.</summary>
    internal List<string> Lemmatize(IReadOnlyList<Word> words, CancellationToken cancellationToken = default)
    {
        var lemmas = words.Select(w => Lookup(w.Text, w.Upos)).ToList();
        var misses = Enumerable.Range(0, words.Count).Where(i => lemmas[i] == null).ToList();
        var predicted = Postprocess(misses.Select(i => words[i]).ToList(), cancellationToken);
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
    internal List<string> Postprocess(IReadOnlyList<Word> words, CancellationToken cancellationToken = default)
    {
        var (decoded, edits) = Predict(words, cancellationToken);
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
    /// <param name="onStep">For tests: called after each decoder step with its log-probs, the row width and which rows were done before it.</param>
    /// <param name="onEdit">For tests: called at the end of each batch with its [batch, 3] edit logits (rows sorted like the steps').</param>
    internal (List<string> Decoded, List<int> Edits) Predict(IReadOnlyList<Word> words, CancellationToken cancellationToken = default,
        Action<float[], int, bool[]>? onStep = null, Action<float[]>? onEdit = null)
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
            cancellationToken.ThrowIfCancellationRequested();
            var batch = words.Skip(start).Take(_batchSize).ToList();
            var (d, e) = PredictBatch(batch, charToId, idToChar, cancellationToken, onStep, onEdit);
            decoded.AddRange(d);
            edits.AddRange(e);
        }
        return (decoded, edits);
    }

    private (string[] Decoded, int[] Edits) PredictBatch(List<Word> words, Dictionary<string, int> charToId, string[] idToChar, CancellationToken ct,
        Action<float[], int, bool[]>? onStep, Action<float[]>? onEdit)
    {
        // <SOS> chars <EOS>, one unit per code point; sorted like data.sort_all: longest first, ties by later index first.
        var src = words.Select(w => w.Text.EnumerateRunes().Select(r => charToId[r.ToString()]).Prepend(SosId).Append(EosId).ToArray()).ToList();
        var order = Enumerable.Range(0, words.Count).OrderByDescending(i => src[i].Length).ThenByDescending(i => i).ToArray();
        int batch = words.Count, width = src[order[0]].Length;
        var ids = new long[batch * width];
        for (int r = 0; r < batch; r++)
            for (int k = 0; k < src[order[r]].Length; k++)
                ids[r * width + k] = src[order[r]][k];
        var posIds = order.Select(i => (long)_posToId.GetValueOrDefault(words[i].Upos ?? "_", UnkId)).ToArray();

        // Seq2SeqModel.predict_greedy: one character per step for the whole batch, the argmax fed back (torch's max: the first maximum).
        using var decoder = _net.Encode(ids, batch, width, posIds, order.Select(i => (long)src[i].Length + 1).ToArray(), ct);
        var output = Enumerable.Range(0, batch).Select(_ => new List<int>()).ToArray();
        var previous = new long[batch];
        Array.Fill(previous, SosId);
        var done = new bool[batch];
        int totalDone = 0, columns = decoder.Columns;
        for (int step = 0; totalDone < batch && step < _maxDecLen; step++)
        {
            ct.ThrowIfCancellationRequested();
            var logProbs = decoder.Step(previous, ct);
            onStep?.Invoke(logProbs, columns, done);
            for (int i = 0; i < batch; i++)
            {
                var row = logProbs.AsSpan(i * columns, columns);
                int best = 0;
                for (int v = 1; v < columns; v++)
                    if (row[v] > row[best])
                        best = v;
                previous[i] = best;
                if (done[i])
                    continue;
                if (best == EosId)
                {
                    done[i] = true;
                    totalDone++;
                }
                else
                    output[i].Add(best);
            }
        }

        var editValues = decoder.EditLogits;
        onEdit?.Invoke(editValues);
        var decoded = new string[batch];
        var edits = new int[batch];
        for (int r = 0; r < batch; r++)
        {
            decoded[order[r]] = string.Concat(output[r].Select(id => idToChar[id]));
            int best = 0; // numpy argmax: the first maximum
            for (int k = 1; k < 3; k++)
                if (editValues[r * 3 + k] > editValues[r * 3 + best])
                    best = k;
            edits[order[r]] = best;
        }
        return (decoded, edits);
    }

    public void Dispose() => _net.Dispose();
}
